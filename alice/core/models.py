import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffractiveLayer(nn.Module):
    # Angular-spectrum propagation layer.
    def __init__(self, wavelength: float, n_pixels: int, pixel_size: float, distance: float):
        super().__init__()
        self.n_pixels = n_pixels
        self.distance = distance

        fx = torch.fft.fftshift(torch.fft.fftfreq(n_pixels, d=pixel_size))
        fy = torch.fft.fftshift(torch.fft.fftfreq(n_pixels, d=pixel_size))
        fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")

        argument = (2.0 * math.pi) ** 2 * ((1.0 / wavelength) ** 2 - fxx**2 - fyy**2)
        kz_real = torch.sqrt(torch.clamp(argument, min=0.0))
        kz_imag = torch.sqrt(torch.clamp(-argument, min=0.0))
        kz = torch.complex(kz_real, kz_imag)
        phase = torch.exp(1j * kz * distance).to(torch.complex64)
        self.register_buffer("phase", phase, persistent=False)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        # field: (B,H,W), complex64
        field_f = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
        propagated = torch.fft.ifft2(torch.fft.ifftshift(field_f * self.phase, dim=(-2, -1)))
        return propagated


class DetectorRegionGrid(nn.Module):
    # Fixed square detector grid.
    def __init__(self, n_pixels: int, det_grid: int = 3, det_size: int = 16, det_gap: int = 8):
        super().__init__()
        self.n_pixels = n_pixels
        self.det_grid = det_grid
        self.det_size = det_size
        self.det_gap = det_gap
        self.n_det = det_grid * det_grid

        span = det_grid * det_size + (det_grid - 1) * det_gap
        start = max(0, (n_pixels - span) // 2)
        boxes = []
        for r in range(det_grid):
            for c in range(det_grid):
                top = start + r * (det_size + det_gap)
                left = start + c * (det_size + det_gap)
                bottom = min(top + det_size, n_pixels)
                right = min(left + det_size, n_pixels)
                boxes.append((top, bottom, left, right))
        self.boxes = boxes

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        # intensity: (B,H,W)
        vals = []
        for top, bottom, left, right in self.boxes:
            vals.append(intensity[:, top:bottom, left:right].sum(dim=(-2, -1), keepdim=True))
        # (B, n_det, 1) -> (B, n_det)
        return torch.cat(vals, dim=1).squeeze(-1)


class DetectorRegionLegacy(nn.Module):
    # Layouts adapted from legacy optical_unit.py (set_det_pos with edge/det_number).
    def __init__(self, n_pixels: int, det_size: int = 16, det_number: int = 9, edge: int = 1):
        super().__init__()
        self.n_pixels = n_pixels
        self.det_size = det_size
        self.det_number = det_number
        self.edge = edge
        self.boxes = self._build_boxes()
        self.n_det = len(self.boxes)

    def _gen_row(self, start_x: int, start_y: int, step: int, n_det: int):
        p = []
        for i in range(n_det):
            left = start_x + i * (step + self.det_size)
            right = left + self.det_size
            up = start_y
            down = up + self.det_size
            p.append((up, down, left, right))
        return p

    def _build_boxes(self):
        s = self.det_size
        start_x = s
        start_y = s
        p = []
        if self.edge == 1:
            if self.det_number == 15:
                p.extend(self._gen_row(start_x, start_y, 5 * s, 3))
                p.extend(self._gen_row(start_x, start_y + 4 * s, 11 * s, 2))
                p.extend(self._gen_row(start_x, start_y + 8 * s, 11 * s, 2))
                p.extend(self._gen_row(start_x, start_y + 12 * s, 5 * s, 3))
            elif self.det_number == 9:
                p.extend(self._gen_row(start_x, start_y, 2 * s, 3))
                p.extend(self._gen_row(start_x, start_y + 2 * s, 5 * s, 2))
                p.extend(self._gen_row(start_x, start_y + 4 * s, 5 * s, 2))
                p.extend(self._gen_row(start_x, start_y + 6 * s, 2 * s, 3))
        elif self.edge == 0:
            if self.det_number == 15:
                p.extend(self._gen_row(start_x, start_y, 5 * s, 3))
                p.extend(self._gen_row(start_x, start_y + 6 * s, 3 * s, 4))
                p.extend(self._gen_row(start_x, start_y + 12 * s, 5 * s, 3))
            elif self.det_number == 9:
                p.extend(self._gen_row(start_x, start_y, 2 * s, 3))
                p.extend(self._gen_row(start_x, start_y + 3 * s, 1 * s, 4))
                p.extend(self._gen_row(start_x, start_y + 6 * s, 2 * s, 3))
        elif self.edge == 3 and self.det_number == 3:
            p.extend(self._gen_row(0, 0, 0 * s, 3))
            p.extend(self._gen_row(0, 1 * s, 0 * s, 3))
            p.extend(self._gen_row(0, 2 * s, 0 * s, 3))
        elif self.edge == 4 and self.det_number == 12:
            p.extend(self._gen_row(start_x, start_y, 8 * s, 2))
            p.extend(self._gen_row(start_x, start_y + 2 * s, 8 * s, 2))
            p.extend(self._gen_row(start_x, start_y + 4 * s, 8 * s, 2))
            p.extend(self._gen_row(start_x, start_y + 6 * s, 8 * s, 2))
            p.extend(self._gen_row(start_x, start_y + 8 * s, 8 * s, 2))

        # Clamp into sensor boundaries.
        out = []
        for top, bottom, left, right in p:
            top = max(0, min(top, self.n_pixels - 1))
            left = max(0, min(left, self.n_pixels - 1))
            bottom = max(top + 1, min(bottom, self.n_pixels))
            right = max(left + 1, min(right, self.n_pixels))
            out.append((top, bottom, left, right))
        if len(out) == 0:
            # Fallback to a simple 3x3 grid to avoid invalid config crashes.
            g = DetectorRegionGrid(self.n_pixels, det_grid=3, det_size=max(4, s), det_gap=max(1, s // 2))
            return g.boxes
        return out

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        vals = []
        for top, bottom, left, right in self.boxes:
            vals.append(intensity[:, top:bottom, left:right].sum(dim=(-2, -1), keepdim=True))
        return torch.cat(vals, dim=1).squeeze(-1)


class OpticalPhaseNet(nn.Module):
    # Diffractive backbone aligned with legacy structure:
    # layer1 -> single phase modulation -> layer2 -> detector regions.
    def __init__(
        self,
        Tin: int,
        Tout: int,
        D: int,
        optical_size: int = 128,
        wavelength: float = 5.32e-7,
        pixel_size: float = 3.6e-5,
        distance_diffractive: float = 0.03,
        distance_sensor: float = 0.03,
        num_optical_layers: int = 1,
        det_grid: int = 3,
        det_size: int = 16,
        det_gap: int = 8,
        det_number: int = 9,
        det_edge: int = 1,
        detector_mode: str = "legacy",
        hidden: int = 512,
        proj_layers: int = 2,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.Tin = Tin
        self.Tout = Tout
        self.D = D
        self.optical_size = optical_size
        self.num_optical_layers = max(1, int(num_optical_layers))

        self.diffractive_in = DiffractiveLayer(wavelength, optical_size, pixel_size, distance_diffractive)
        self.diffractive_mid = nn.ModuleList(
            [DiffractiveLayer(wavelength, optical_size, pixel_size, distance_diffractive) for _ in range(self.num_optical_layers - 1)]
        )
        self.diffractive_out = DiffractiveLayer(wavelength, optical_size, pixel_size, distance_sensor)
        # Legacy-aligned: one trainable phase mask per optical layer.
        self.phase_masks = nn.ParameterList(
            [nn.Parameter(torch.rand(optical_size, optical_size, dtype=torch.float32)) for _ in range(self.num_optical_layers)]
        )
        if detector_mode == "legacy":
            self.detector = DetectorRegionLegacy(
                n_pixels=optical_size, det_size=det_size, det_number=det_number, edge=det_edge
            )
        else:
            self.detector = DetectorRegionGrid(
                n_pixels=optical_size, det_grid=det_grid, det_size=det_size, det_gap=det_gap
            )
        feat_dim = Tin * self.detector.n_det
        proj_layers = max(1, int(proj_layers))
        proj_dropout = max(0.0, min(0.8, float(proj_dropout)))
        layers = []
        in_dim = feat_dim
        if proj_layers == 1:
            layers.append(nn.Linear(in_dim, Tout * D))
        else:
            for _ in range(proj_layers - 1):
                layers.append(nn.Linear(in_dim, hidden))
                layers.append(nn.GELU())
                if proj_dropout > 0:
                    layers.append(nn.Dropout(proj_dropout))
                in_dim = hidden
            layers.append(nn.Linear(in_dim, Tout * D))
        self.proj = nn.Sequential(*layers)

    def constrained_phase(self) -> torch.Tensor:
        # Keep backward-compatible API: return first layer phase.
        return 2.0 * math.pi * torch.sigmoid(self.phase_masks[0])

    def constrained_phases(self):
        return [2.0 * math.pi * torch.sigmoid(p) for p in self.phase_masks]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,Tin,1,H,W)
        B, T, C, H, W = x.shape
        feats = []
        phases = self.constrained_phases()
        ph_exp_list = [torch.exp(1j * ph).to(torch.complex64) for ph in phases]
        for p in range(T):
            img = x[:, p, 0]
            if img.shape[-1] != self.optical_size or img.shape[-2] != self.optical_size:
                img = F.interpolate(
                    img.unsqueeze(1),
                    size=(self.optical_size, self.optical_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            amp = torch.sqrt(torch.clamp(img, min=0.0))
            field = torch.complex(amp, torch.zeros_like(amp))
            field = self.diffractive_in(field)
            field = field * ph_exp_list[0].unsqueeze(0)
            for li in range(1, self.num_optical_layers):
                field = self.diffractive_mid[li - 1](field)
                field = field * ph_exp_list[li].unsqueeze(0)
            field = self.diffractive_out(field)
            intensity = torch.abs(field) ** 2
            det = self.detector(intensity)
            feats.append(det)
        fcat = torch.cat(feats, dim=1)
        out = self.proj(fcat).view(B, self.Tout, self.D)
        return out


class OpticalPhaseTwoStageNet(nn.Module):
    # Notebook-inspired two-stage cascade:
    # propagate -> phase1 -> propagate -> intensity -> zero-phase amplitude
    # -> propagate -> phase2 -> propagate -> detector -> projection.
    def __init__(
        self,
        Tin: int,
        Tout: int,
        D: int,
        optical_size: int = 128,
        wavelength: float = 5.32e-7,
        pixel_size: float = 3.6e-5,
        distance_diffractive: float = 0.03,
        distance_sensor: float = 0.03,
        num_optical_layers: int = 2,
        det_grid: int = 3,
        det_size: int = 16,
        det_gap: int = 8,
        det_number: int = 9,
        det_edge: int = 1,
        detector_mode: str = "legacy",
        hidden: int = 512,
        proj_layers: int = 2,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.Tin = Tin
        self.Tout = Tout
        self.D = D
        self.optical_size = optical_size

        self.diffractive_in = DiffractiveLayer(wavelength, optical_size, pixel_size, distance_diffractive)
        self.diffractive_out = DiffractiveLayer(wavelength, optical_size, pixel_size, distance_sensor)

        self.phase = nn.Parameter(torch.randn(optical_size, optical_size, dtype=torch.float32) * 0.1)
        self.phase1 = nn.Parameter(torch.randn(optical_size, optical_size, dtype=torch.float32) * 0.1)
        self.output_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.output_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        if detector_mode == "legacy":
            self.detector = DetectorRegionLegacy(
                n_pixels=optical_size, det_size=det_size, det_number=det_number, edge=det_edge
            )
        else:
            self.detector = DetectorRegionGrid(
                n_pixels=optical_size, det_grid=det_grid, det_size=det_size, det_gap=det_gap
            )

        feat_dim = Tin * self.detector.n_det
        proj_layers = max(1, int(proj_layers))
        proj_dropout = max(0.0, min(0.8, float(proj_dropout)))
        layers = []
        in_dim = feat_dim
        if proj_layers == 1:
            layers.append(nn.Linear(in_dim, Tout * D))
        else:
            for _ in range(proj_layers - 1):
                layers.append(nn.Linear(in_dim, hidden))
                layers.append(nn.GELU())
                if proj_dropout > 0:
                    layers.append(nn.Dropout(proj_dropout))
                in_dim = hidden
            layers.append(nn.Linear(in_dim, Tout * D))
        self.proj = nn.Sequential(*layers)

    def constrained_phase(self) -> torch.Tensor:
        return 2.0 * math.pi * torch.sigmoid(self.phase)

    def constrained_phases(self):
        return [
            2.0 * math.pi * torch.sigmoid(self.phase),
            2.0 * math.pi * torch.sigmoid(self.phase1),
        ]

    def _resize_frame(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[-1] == self.optical_size and img.shape[-2] == self.optical_size:
            return img
        return F.interpolate(
            img.unsqueeze(1),
            size=(self.optical_size, self.optical_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    def _propagate_stage(self, field: torch.Tensor, phase_mask: torch.Tensor) -> torch.Tensor:
        field = self.diffractive_in(field)
        field = field * torch.exp(1j * phase_mask).to(torch.complex64).unsqueeze(0)
        field = self.diffractive_out(field)
        return field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        feats = []
        phase0, phase1 = self.constrained_phases()
        for p in range(T):
            img = self._resize_frame(x[:, p, 0])

            amp = torch.sqrt(torch.clamp(img, min=0.0))
            field = torch.complex(amp, torch.zeros_like(amp))

            # Stage 1 from the notebook structure.
            field = self._propagate_stage(field, phase0)
            intensity = torch.abs(field) ** 2
            intensity = torch.clamp(intensity, min=1e-8, max=1e8)
            amplitude = torch.sqrt(intensity)

            # Reset phase to zero and feed into stage 2.
            field2 = torch.complex(amplitude, torch.zeros_like(amplitude))
            field2 = self._propagate_stage(field2, phase1)

            final_intensity = torch.abs(field2) ** 2
            final_intensity = torch.clamp(final_intensity, min=0.0, max=1e4)
            final_intensity = self.output_scale * final_intensity + self.output_bias
            final_intensity = torch.clamp(final_intensity, min=0.0)

            det = self.detector(final_intensity)
            feats.append(det)

        fcat = torch.cat(feats, dim=1)
        out = self.proj(fcat).view(B, self.Tout, self.D)
        return out

class BilinearReranker(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.W = nn.Parameter(torch.eye(D))

    def forward(self, pred_vec, cand_vec):
        # pred_vec: (B,T,D), cand_vec: (B,T,K,D)
        # scores: (B,T,K)
        scores = torch.einsum('btd,dd,btkd->btk', pred_vec, self.W, cand_vec)
        return scores


class ElectricContextRefiner(nn.Module):
    # Lightweight electric contextual module over optical token embeddings.
    def __init__(self, D: int, layers: int = 2, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = max(0, int(layers))
        if self.layers <= 0:
            self.net = None
        else:
            nhead = max(1, min(nhead, D))
            while D % nhead != 0 and nhead > 1:
                nhead -= 1
            enc_layer = nn.TransformerEncoderLayer(
                d_model=D,
                nhead=nhead,
                dim_feedforward=max(4 * D, 256),
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.net = nn.TransformerEncoder(enc_layer, num_layers=self.layers)
            self.norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        if self.net is None:
            return x
        h = self.net(x)
        return self.norm(x + h)


class HybridElectricReranker(nn.Module):
    # Bilinear + nonlinear electric scorer for candidate ranking.
    def __init__(self, D: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.bilinear = BilinearReranker(D=D)
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        in_dim = 4 * D
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, pred_vec: torch.Tensor, cand_vec: torch.Tensor) -> torch.Tensor:
        # pred_vec: (B,T,D), cand_vec: (B,T,K,D)
        bscore = self.bilinear(pred_vec, cand_vec)  # (B,T,K)
        p = pred_vec.unsqueeze(2).expand_as(cand_vec)  # (B,T,K,D)
        feat = torch.cat([p, cand_vec, p * cand_vec, torch.abs(p - cand_vec)], dim=-1)
        escore = self.mlp(feat).squeeze(-1)  # (B,T,K)
        return bscore + self.alpha * escore
