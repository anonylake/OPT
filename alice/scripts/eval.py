import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from core.dataset import OpticalWordDataset, collate_keep_words, load_semantic_npz
from core.models import (
    BilinearReranker,
    ElectricContextRefiner,
    HybridElectricReranker,
    OpticalPhaseTwoStageNet,
)

warnings.filterwarnings("ignore", category=UserWarning, module=r"torch(\.|$)")


def load_cfg(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def build_candidate_tensors(target_words_npy, cand_cache, semantic_npz, limit_n, tout, cand_k, device):
    words, vecs, w2i = load_semantic_npz(str(semantic_npz))
    dim = vecs.shape[1]
    cand_words_full = np.load(cand_cache, allow_pickle=True)
    y_all_full = np.load(target_words_npy, allow_pickle=True)
    n_eff = min(cand_words_full.shape[0], y_all_full.shape[0], int(limit_n))
    if n_eff <= 0:
        raise ValueError("No valid samples available for evaluation.")

    cand_words = cand_words_full[:n_eff]
    y_all = y_all_full[:n_eff]
    if cand_words.shape[1] != tout or cand_words.shape[2] != cand_k:
        raise ValueError("Candidate cache shape does not match config Tout/cand_k.")

    cand_vecs = np.zeros((n_eff, tout, cand_k, dim), dtype=np.float32)
    gt_idx = -np.ones((n_eff, tout), dtype=np.int64)
    for i in range(n_eff):
        for t in range(tout):
            for k in range(cand_k):
                w = str(cand_words[i, t, k])
                j = w2i.get(w)
                if j is not None:
                    cand_vecs[i, t, k] = vecs[j]
            gt = str(y_all[i, t])
            for k in range(cand_k):
                if str(cand_words[i, t, k]) == gt:
                    gt_idx[i, t] = k
                    break

    cand_vecs_t = torch.tensor(cand_vecs, dtype=torch.float32, device=device)
    gt_idx_t = torch.tensor(gt_idx, dtype=torch.long, device=device)
    return words, vecs, dim, n_eff, cand_vecs_t, gt_idx_t


def build_split_indices(n_eff, seed, train_ratio, optical_demo):
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_eff, dtype=np.int64)
    rng.shuffle(all_idx)
    if optical_demo or train_ratio >= 0.999:
        return all_idx, all_idx
    split = int(n_eff * train_ratio)
    split = min(max(split, 1), n_eff - 1)
    return all_idx[:split], all_idx[split:]


def build_model(cfg, dim, device):
    model_cfg = cfg["model"]
    if model_cfg["model_type"] != "optical_phase_two_stage":
        raise ValueError("alice_only only supports the notebook-style two-stage optical backbone.")

    opt = OpticalPhaseTwoStageNet(
        Tin=cfg["train"]["Tin"],
        Tout=cfg["train"]["Tout"],
        D=dim,
        optical_size=model_cfg["optical_size"],
        num_optical_layers=model_cfg["num_optical_layers"],
        detector_mode=model_cfg["detector_mode"],
        det_number=model_cfg["det_number"],
        det_edge=model_cfg["det_edge"],
        det_size=model_cfg["det_size"],
        det_grid=model_cfg["det_grid"],
        det_gap=model_cfg["det_gap"],
        hidden=model_cfg["hidden"],
        proj_layers=model_cfg["proj_layers"],
        proj_dropout=model_cfg["proj_dropout"],
    ).to(device)

    electric_refiner = ElectricContextRefiner(
        D=dim,
        layers=model_cfg["electric_layers"],
        nhead=model_cfg["electric_heads"],
        dropout=model_cfg["electric_dropout"],
    ).to(device)

    if model_cfg["electric_mode"] == "hybrid":
        rerank = HybridElectricReranker(
            D=dim,
            hidden=model_cfg["electric_hidden"],
            dropout=model_cfg["electric_dropout"],
        ).to(device)
    else:
        rerank = BilinearReranker(D=dim).to(device)

    return opt, electric_refiner, rerank


def evaluate(loader, opt, electric_refiner, rerank, cand_vecs_t, gt_idx_t, cfg, device):
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    cand_k = int(train_cfg["cand_k"])
    tout = int(train_cfg["Tout"])
    ce = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    opt.eval()
    electric_refiner.eval()
    rerank.eval()

    total = 0
    correct = 0
    total_all = 0
    loss_sum = 0.0
    pos_correct = np.zeros((tout,), dtype=np.int64)
    pos_total = np.zeros((tout,), dtype=np.int64)
    pos_cov = np.zeros((tout,), dtype=np.int64)

    with torch.no_grad():
        for x, _xw, _yw, yv, idx in loader:
            batch = x.shape[0]
            idxs = idx.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            yv = yv.to(device, non_blocking=True)

            pred = opt(x)
            if model_cfg["electric_mode"] != "none":
                pred = electric_refiner(pred)

            cvec = cand_vecs_t[idxs]
            gti = gt_idx_t[idxs]
            pred_n = F.normalize(pred, dim=-1)
            cvec_n = F.normalize(cvec, dim=-1)
            scores = rerank(pred_n, cvec_n)
            mask = gti >= 0

            if mask.any():
                loss = ce(scores.view(-1, cand_k), gti.view(-1)).view(batch, tout)
                loss = (loss * mask.float()).sum() / mask.float().sum()
            else:
                loss = torch.tensor(0.0, device=device)

            reg_mse = F.mse_loss(pred, yv)
            reg_cos = 1.0 - F.cosine_similarity(pred, yv, dim=-1).mean()
            total_loss = loss + 0.1 * reg_mse + 0.2 * reg_cos

            predk = scores.argmax(dim=-1)
            corr_mask = (predk == gti) & mask
            correct += int(corr_mask.sum().item())
            total += int(mask.sum().item())
            total_all += int(batch * tout)
            loss_sum += float(total_loss.item()) * batch

            for t in range(tout):
                mt = mask[:, t]
                pos_cov[t] += int(mt.sum().item())
                pos_total[t] += batch
                if mt.any():
                    pos_correct[t] += int(corr_mask[:, t].sum().item())

    return {
        "loss": loss_sum / max(1, len(loader.dataset)),
        "acc_covered": correct / max(1, total),
        "acc_all": correct / max(1, total_all),
        "coverage_at_k": total / max(1, total_all),
        "pos_acc_all": (pos_correct / np.maximum(1, pos_total)).tolist(),
        "pos_acc_covered": (pos_correct / np.maximum(1, pos_cov)).tolist(),
        "pos_coverage": (pos_cov / np.maximum(1, pos_total)).tolist(),
        "n_samples": int(len(loader.dataset)),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate the Alice optical-electrical rerank checkpoint on accuracy metrics.")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out_json", default="")
    ap.add_argument("--out_md", default="")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_cfg(root / args.config)
    device = pick_device()

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    output_cfg = cfg["output"]

    cache_dir = root / data_cfg["cache_dir"]
    semantic_npz = cache_dir / f"alice_semantic_k{train_cfg['k']}.npz"
    target_words_npy = cache_dir / "alice_target_words.npy"
    cand_cache = cache_dir / f"alice_cands_k{train_cfg['cand_k']}.npy"
    ckpt_path = root / (args.ckpt or output_cfg["ckpt"])

    ds_full = OpticalWordDataset(
        str(root / data_cfg["render_dir"]),
        str(cache_dir / "alice_input_words.npy"),
        str(target_words_npy),
        str(semantic_npz),
        Tin=train_cfg["Tin"],
        Tout=train_cfg["Tout"],
        limit_n=train_cfg["limit_n"],
    )

    _words, _vecs, dim, n_eff, cand_vecs_t, gt_idx_t = build_candidate_tensors(
        target_words_npy=str(target_words_npy),
        cand_cache=str(cand_cache),
        semantic_npz=str(semantic_npz),
        limit_n=train_cfg["limit_n"],
        tout=train_cfg["Tout"],
        cand_k=train_cfg["cand_k"],
        device=device,
    )

    ds = Subset(ds_full, list(range(n_eff)))
    train_idx, val_idx = build_split_indices(
        n_eff=n_eff,
        seed=train_cfg["seed"],
        train_ratio=train_cfg["train_ratio"],
        optical_demo=bool(train_cfg.get("optical_demo", True)),
    )

    dl_kwargs = {
        "batch_size": int(train_cfg.get("batch_size", 32)),
        "num_workers": 0,
        "collate_fn": collate_keep_words,
        "pin_memory": False,
    }
    dl_train = DataLoader(Subset(ds, train_idx.tolist()), shuffle=False, **dl_kwargs)
    dl_val = DataLoader(Subset(ds, val_idx.tolist()), shuffle=False, **dl_kwargs)
    dl_all = DataLoader(ds, shuffle=False, **dl_kwargs)

    opt, electric_refiner, rerank = build_model(cfg, dim, device)
    payload = torch.load(ckpt_path, map_location=device)
    opt.load_state_dict(payload["opt"])
    electric_refiner.load_state_dict(payload["electric_refiner"])
    rerank.load_state_dict(payload["rerank"])

    val_metrics = evaluate(dl_val, opt, electric_refiner, rerank, cand_vecs_t, gt_idx_t, cfg, device)
    report = {
        "val_metrics": {
            "acc_all": val_metrics["acc_all"],
        },
    }

    out_json = root / (
        args.out_json
        or f"{ckpt_path.with_suffix('').relative_to(root)}.eval.json"
    )
    out_md = root / (
        args.out_md
        or f"{ckpt_path.with_suffix('').relative_to(root)}.eval.md"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_lines = [f"val_metrics.acc_all={val_metrics['acc_all']:.4f}"]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"val_metrics.acc_all={val_metrics['acc_all']:.4f}")


if __name__ == "__main__":
    main()
