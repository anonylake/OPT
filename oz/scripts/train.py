import os, json, argparse, warnings, numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from core.dataset import OpticalWordDataset, collate_keep_words, load_semantic_npz
from core.models import (
    OpticalPhaseTwoStageNet,
    BilinearReranker,
    ElectricContextRefiner,
    HybridElectricReranker,
)

warnings.filterwarnings("ignore", category=UserWarning, module=r"torch(\.|$)")


def pick_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        supported = set(torch.cuda.get_arch_list())
        if arch not in supported:
            return torch.device("cpu")
    except Exception:
        return torch.device("cpu")
    return torch.device("cuda")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_inputs_dir", required=True)
    ap.add_argument("--input_words_npy", required=True)
    ap.add_argument("--target_words_npy", required=True)
    ap.add_argument("--semantic_npz", required=True)
    ap.add_argument("--cand_cache", required=True)
    ap.add_argument("--Tin", type=int, default=4)
    ap.add_argument("--Tout", type=int, default=4)
    ap.add_argument("--k", type=int, default=18)
    ap.add_argument("--cand_k", type=int, default=5)
    ap.add_argument("--model_type", type=str, default="optical_phase_two_stage", choices=["optical_phase_two_stage"])
    ap.add_argument("--optical_size", type=int, default=128)
    ap.add_argument("--wavelength", type=float, default=5.32e-7)
    ap.add_argument("--pixel_size", type=float, default=3.6e-5)
    ap.add_argument("--distance_diffractive", type=float, default=0.03)
    ap.add_argument("--distance_sensor", type=float, default=0.03)
    ap.add_argument("--num_optical_layers", type=int, default=1)
    ap.add_argument("--det_grid", type=int, default=3)
    ap.add_argument("--det_size", type=int, default=16)
    ap.add_argument("--det_gap", type=int, default=8)
    ap.add_argument("--det_number", type=int, default=9)
    ap.add_argument("--det_edge", type=int, default=1)
    ap.add_argument("--detector_mode", type=str, default="legacy", choices=["legacy", "grid"])
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--proj_layers", type=int, default=2)
    ap.add_argument("--proj_dropout", type=float, default=0.0)
    ap.add_argument("--electric_mode", type=str, default="hybrid", choices=["none", "bilinear", "hybrid"])
    ap.add_argument("--electric_layers", type=int, default=2)
    ap.add_argument("--electric_heads", type=int, default=4)
    ap.add_argument("--electric_dropout", type=float, default=0.1)
    ap.add_argument("--electric_hidden", type=int, default=512)
    ap.add_argument("--limit_n", type=int, default=200)
    ap.add_argument("--optical_demo", action="store_true", default=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--electric_lr_ratio", type=float, default=0.3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--ce_weight", type=float, default=1.0)
    ap.add_argument("--mse_weight", type=float, default=0.1)
    ap.add_argument("--cos_weight", type=float, default=0.2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min_epochs", type=int, default=8)
    ap.add_argument("--report_json", type=str, default="")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    device = pick_device()

    ds_full = OpticalWordDataset(args.img_inputs_dir, args.input_words_npy, args.target_words_npy,
                                 args.semantic_npz, Tin=args.Tin, Tout=args.Tout, limit_n=args.limit_n)

    words, vecs, w2i = load_semantic_npz(args.semantic_npz)
    D = vecs.shape[1]

    # load cached candidate words
    cand_words_full = np.load(args.cand_cache, allow_pickle=True)  # (N,Tout,K)
    y_all_full = np.load(args.target_words_npy, allow_pickle=True)
    n_eff = min(len(ds_full), cand_words_full.shape[0], y_all_full.shape[0], int(args.limit_n))
    if n_eff <= 0:
        raise ValueError("No valid samples available after aligning dataset and candidate cache.")
    cand_words = cand_words_full[:n_eff]
    y_all = y_all_full[:n_eff]
    assert cand_words.shape[1] == args.Tout and cand_words.shape[2] == args.cand_k

    # build candidate vectors tensor cache: (N,Tout,K,D)
    cand_vecs = np.zeros((n_eff, args.Tout, args.cand_k, D), dtype=np.float32)
    gt_idx = -np.ones((n_eff, args.Tout), dtype=np.int64)

    for i in range(n_eff):
        for t in range(args.Tout):
            for k in range(args.cand_k):
                w = str(cand_words[i,t,k])
                j = w2i.get(w, None)
                if j is not None:
                    cand_vecs[i,t,k] = vecs[j]
            # gt index within candidates
            gt = str(y_all[i,t])
            for k in range(args.cand_k):
                if str(cand_words[i,t,k]) == gt:
                    gt_idx[i,t] = k
                    break

    cand_vecs_t = torch.tensor(cand_vecs, dtype=torch.float32, device=device)  # fixed
    gt_idx_t = torch.tensor(gt_idx, dtype=torch.long, device=device)

    ds = Subset(ds_full, list(range(n_eff)))
    N = len(ds)
    g = np.random.default_rng(args.seed)
    all_idx = np.arange(N, dtype=np.int64)
    g.shuffle(all_idx)

    if args.optical_demo or args.train_ratio >= 0.999:
        train_idx = all_idx
        val_idx = all_idx
    else:
        split = int(N * args.train_ratio)
        split = min(max(split, 1), N - 1)
        train_idx = all_idx[:split]
        val_idx = all_idx[split:]

    ds_train = Subset(ds, train_idx.tolist())
    ds_val = Subset(ds, val_idx.tolist())
    nw = max(0, int(args.num_workers))
    dl_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": nw,
        "collate_fn": collate_keep_words,
        "pin_memory": bool(args.pin_memory),
    }
    if nw > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = max(2, int(args.prefetch_factor))

    dl_train = DataLoader(ds_train, shuffle=True, **dl_kwargs)
    dl_val = DataLoader(ds_val, shuffle=False, **dl_kwargs)

    opt = OpticalPhaseTwoStageNet(
        Tin=args.Tin,
        Tout=args.Tout,
        D=D,
        optical_size=args.optical_size,
        wavelength=args.wavelength,
        pixel_size=args.pixel_size,
        distance_diffractive=args.distance_diffractive,
        distance_sensor=args.distance_sensor,
        num_optical_layers=args.num_optical_layers,
        det_grid=args.det_grid,
        det_size=args.det_size,
        det_gap=args.det_gap,
        det_number=args.det_number,
        det_edge=args.det_edge,
        detector_mode=args.detector_mode,
        hidden=args.hidden,
        proj_layers=args.proj_layers,
        proj_dropout=args.proj_dropout,
    ).to(device)
    electric_refiner = ElectricContextRefiner(
        D=D,
        layers=args.electric_layers,
        nhead=args.electric_heads,
        dropout=args.electric_dropout,
    ).to(device)
    if args.electric_mode == "hybrid":
        rerank = HybridElectricReranker(D=D, hidden=args.electric_hidden, dropout=args.electric_dropout).to(device)
    else:
        rerank = BilinearReranker(D=D).to(device)

    elec_lr = args.lr * max(0.01, min(args.electric_lr_ratio, 1.0))
    optim = torch.optim.Adam(
        [
            {"params": list(opt.parameters()), "lr": args.lr},
            {"params": list(electric_refiner.parameters()) + list(rerank.parameters()), "lr": elec_lr},
        ],
        weight_decay=args.weight_decay,
    )

    ce = torch.nn.CrossEntropyLoss(
        ignore_index=-1,
        reduction="none",
        label_smoothing=max(0.0, min(0.3, args.label_smoothing)),
    )

    def evaluate(loader):
        opt.eval(); electric_refiner.eval(); rerank.eval()
        total = 0
        correct = 0
        total_all = 0
        loss_sum = 0.0
        pos_correct = np.zeros((args.Tout,), dtype=np.int64)
        pos_total = np.zeros((args.Tout,), dtype=np.int64)
        pos_cov = np.zeros((args.Tout,), dtype=np.int64)
        with torch.no_grad():
            for x, xw, yw, yv, idx in loader:
                B = x.shape[0]
                idxs = idx.to(device, non_blocking=True)
                x = x.to(device, non_blocking=True)
                yv = yv.to(device, non_blocking=True)
                pred = opt(x)
                if args.electric_mode != "none":
                    pred = electric_refiner(pred)
                cvec = cand_vecs_t[idxs]
                gti = gt_idx_t[idxs]
                pred_n = F.normalize(pred, dim=-1)
                cvec_n = F.normalize(cvec, dim=-1)
                scores = rerank(pred_n, cvec_n)
                mask = (gti >= 0)

                if mask.any():
                    loss = ce(scores.view(-1, args.cand_k), gti.view(-1)).view(B, args.Tout)
                    loss = (loss * mask.float()).sum() / mask.float().sum()
                else:
                    loss = torch.tensor(0.0, device=device)
                reg_mse = F.mse_loss(pred, yv)
                reg_cos = 1.0 - F.cosine_similarity(pred, yv, dim=-1).mean()
                total_loss = args.ce_weight * loss + args.mse_weight * reg_mse + args.cos_weight * reg_cos

                predk = scores.argmax(dim=-1)
                corr_mask = ((predk == gti) & mask)
                correct += int(corr_mask.sum().item())
                total += int(mask.sum().item())
                total_all += int(B * args.Tout)
                loss_sum += float(total_loss.item()) * B

                for t in range(args.Tout):
                    mt = mask[:, t]
                    pos_cov[t] += int(mt.sum().item())
                    pos_total[t] += B
                    if mt.any():
                        pos_correct[t] += int(corr_mask[:, t].sum().item())

        metrics = {
            "loss": loss_sum / max(1, len(loader.dataset)),
            "acc_covered": correct / max(1, total),
            "acc_all": correct / max(1, total_all),
            "coverage_at_k": total / max(1, total_all),
            "pos_acc_all": (pos_correct / np.maximum(1, pos_total)).tolist(),
            "pos_acc_covered": (pos_correct / np.maximum(1, pos_cov)).tolist(),
            "pos_coverage": (pos_cov / np.maximum(1, pos_total)).tolist(),
            "n_samples": int(len(loader.dataset)),
        }
        return metrics

    best_val = -1.0
    best_val_loss = 1e18
    best_epoch = 0
    bad_epochs = 0
    best_state = None
    history = []

    for ep in range(1, args.epochs+1):
        opt.train(); electric_refiner.train(); rerank.train()
        tr_loss_sum = 0.0
        tr_n = 0

        for x, xw, yw, yv, idx in dl_train:
            B = x.shape[0]
            idxs = idx.to(device, non_blocking=True)

            x = x.to(device, non_blocking=True)
            yv = yv.to(device, non_blocking=True)
            pred = opt(x)  # (B,T,D)
            if args.electric_mode != "none":
                pred = electric_refiner(pred)

            # gather cached candidates and gt index
            cvec = cand_vecs_t[idxs]        # (B,T,K,D)
            gti = gt_idx_t[idxs]            # (B,T)

            pred_n = F.normalize(pred, dim=-1)
            cvec_n = F.normalize(cvec, dim=-1)
            scores = rerank(pred_n, cvec_n)     # (B,T,K)
            mask = (gti >= 0)

            if mask.any():
                loss = ce(scores.view(-1, args.cand_k), gti.view(-1)).view(B, args.Tout)
                loss = (loss * mask.float()).sum() / mask.float().sum()
            else:
                loss = torch.tensor(0.0, device=device)

            reg_mse = F.mse_loss(pred, yv)
            reg_cos = 1.0 - F.cosine_similarity(pred, yv, dim=-1).mean()
            total_loss = args.ce_weight * loss + args.mse_weight * reg_mse + args.cos_weight * reg_cos

            optim.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(opt.parameters()) + list(electric_refiner.parameters()) + list(rerank.parameters()),
                args.grad_clip,
            )
            optim.step()
            tr_loss_sum += float(total_loss.item()) * B
            tr_n += B

        train_loss = tr_loss_sum / max(1, tr_n)
        val_m = evaluate(dl_val)
        history.append({
            "epoch": ep,
            "val_acc_all": val_m["acc_all"],
        })

        cur_acc = float(val_m["acc_all"])
        cur_loss = float(val_m["loss"])
        better_acc = cur_acc > (best_val + 1e-6)
        tie_better_loss = (abs(cur_acc - best_val) <= 1e-6) and (cur_loss < best_val_loss - 1e-6)
        if better_acc or tie_better_loss:
            best_val = val_m["acc_all"]
            best_val_loss = cur_loss
            best_epoch = ep
            bad_epochs = 0
            best_state = {
                "opt": {k: v.detach().cpu() for k, v in opt.state_dict().items()},
                "electric_refiner": {k: v.detach().cpu() for k, v in electric_refiner.state_dict().items()},
                "rerank": {k: v.detach().cpu() for k, v in rerank.state_dict().items()},
            }
        else:
            bad_epochs += 1

        if ep >= args.min_epochs and bad_epochs >= args.patience:
            break

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    if best_state is None:
        best_state = {
            "opt": {k: v.detach().cpu() for k, v in opt.state_dict().items()},
            "electric_refiner": {k: v.detach().cpu() for k, v in electric_refiner.state_dict().items()},
            "rerank": {k: v.detach().cpu() for k, v in rerank.state_dict().items()},
        }
    torch.save(
        {
            "opt": best_state["opt"],
            "electric_refiner": best_state["electric_refiner"],
            "rerank": best_state["rerank"],
            "best_epoch": best_epoch,
            "best_val_acc_all": best_val,
            "best_val_loss": best_val_loss,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
        },
        args.ckpt,
    )
    # load best state for final report
    opt.load_state_dict(best_state["opt"])
    electric_refiner.load_state_dict(best_state["electric_refiner"])
    rerank.load_state_dict(best_state["rerank"])
    val_final = evaluate(dl_val)
    report = {
        "val_metrics": {
            "acc_all": val_final["acc_all"],
        },
    }
    report_path = args.report_json or (os.path.splitext(args.ckpt)[0] + ".metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"val_metrics.acc_all={val_final['acc_all']:.4f}")

if __name__ == "__main__":
    main()
