import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def run(cmd, env=None):
    proc = subprocess.run(cmd, check=False, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    for line in proc.stdout.splitlines():
        if line.startswith("val_metrics.acc_all="):
            print(line)


def load_cfg(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_bmps(path: Path):
    if not path.exists():
        return 0
    return len(list(path.glob("*.bmp")))


def cand_cache_ok(path: Path, need_n: int, tout: int, cand_k: int):
    if not path.exists():
        return False
    try:
        arr = np.load(path, allow_pickle=True)
        return arr.ndim == 3 and arr.shape[0] >= need_n and arr.shape[1] == tout and arr.shape[2] == cand_k
    except Exception:
        return False


def build_parser():
    ap = argparse.ArgumentParser(description="Pipeline runner for Alice optical accuracy evaluation.")
    ap.add_argument("--config", required=True, help="JSON config path, e.g. config.json")
    ap.add_argument("--optical_demo", action="store_true", default=None)
    ap.add_argument("--skip_preprocess", action="store_true")
    ap.add_argument("--skip_render", action="store_true")
    ap.add_argument("--skip_candidates", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    cfg = load_cfg(Path(args.config))

    runtime = cfg["runtime"]
    data = cfg["data"]
    model = cfg["model"]
    train = cfg["train"]
    cand = cfg["candidate"]
    output = cfg["output"]

    if model.get("model_type") != "optical_phase_two_stage":
        raise ValueError("alice_only only supports model_type='optical_phase_two_stage'.")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    if runtime.get("hf_endpoint"):
        env["HF_ENDPOINT"] = runtime["hf_endpoint"]
    if runtime.get("unset_ssl_env", True):
        for k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            env.pop(k, None)

    tin = int(train["Tin"])
    tout = int(train["Tout"])
    k = int(train["k"])
    limit_n = int(train["limit_n"])
    cand_k = int(train["cand_k"])
    reuse_cache = bool(runtime.get("reuse_cache", True))

    cache_dir = Path(data["cache_dir"])
    render_dir = Path(data["render_dir"])
    render_out_hw = int(data.get("render_out_hw", 500))
    semantic_npz = cache_dir / f"alice_semantic_k{k}.npz"
    inputs_int8 = cache_dir / "alice_inputs_int8.npy"
    cand_cache = cache_dir / f"alice_cands_k{cand_k}.npy"

    if not args.skip_preprocess:
        if reuse_cache and semantic_npz.exists() and inputs_int8.exists():
            pass
        else:
            run(
                [
                    sys.executable,
                    "scripts/preprocess.py",
                    "--txt",
                    data["text"],
                    "--out",
                    str(cache_dir),
                    "--Tin",
                    str(tin),
                    "--Tout",
                    str(tout),
                    "--k",
                    str(k),
                    "--gpt2_name",
                    cand["gpt2_name"],
                ],
                env=env,
            )

    if not args.skip_render:
        need_bmps = limit_n * tin
        if reuse_cache and render_dir.exists() and count_bmps(render_dir) >= need_bmps:
            pass
        else:
            run(
                [
                    sys.executable,
                    "scripts/render.py",
                    "--inputs",
                    str(inputs_int8),
                    "--outdir",
                    str(render_dir),
                    "--out_hw",
                    str(render_out_hw),
                    "--n",
                    str(limit_n),
                ],
                env=env,
            )

    if not args.skip_candidates:
        if reuse_cache and cand_cache_ok(cand_cache, need_n=limit_n, tout=tout, cand_k=cand_k):
            pass
        else:
            cmd = [
                sys.executable,
                "scripts/candidates.py",
                "--input_words_npy",
                    str(cache_dir / "alice_input_words.npy"),
                "--target_words_npy",
                    str(cache_dir / "alice_target_words.npy"),
                "--limit_n",
                str(limit_n),
                "--gpt2_name",
                cand["gpt2_name"],
                "--cand_k",
                str(cand_k),
                "--out",
                str(cand_cache),
            ]
            if cand.get("teacher_forcing", False):
                cmd.append("--teacher_forcing")
            if cand.get("use_semantic", False):
                cmd += [
                    "--semantic_npz",
                    str(semantic_npz),
                    "--semantic_mix_ratio",
                    str(cand.get("semantic_mix_ratio", 0.0)),
                ]
            if cand.get("soft_include_target", False):
                cmd += [
                    "--soft_include_target",
                    "--soft_include_margin",
                    str(cand.get("soft_include_margin", 0.5)),
                    "--soft_include_topn",
                    str(cand.get("soft_include_topn", 200)),
                    "--soft_include_min_freq",
                    str(cand.get("soft_include_min_freq", 2)),
                ]
            run(cmd, env=env)

    if not args.skip_train:
        Path(output["ckpt"]).parent.mkdir(parents=True, exist_ok=True)
        train_cmd = [
            sys.executable,
            "-u",
            "scripts/train.py",
            "--img_inputs_dir",
            str(render_dir),
            "--input_words_npy",
            str(cache_dir / "alice_input_words.npy"),
            "--target_words_npy",
            str(cache_dir / "alice_target_words.npy"),
            "--semantic_npz",
            str(semantic_npz),
            "--cand_cache",
            str(cand_cache),
            "--Tin",
            str(tin),
            "--Tout",
            str(tout),
            "--k",
            str(k),
            "--cand_k",
            str(cand_k),
            "--model_type",
            model["model_type"],
            "--optical_size",
            str(model["optical_size"]),
            "--num_optical_layers",
            str(model["num_optical_layers"]),
            "--detector_mode",
            model["detector_mode"],
            "--det_number",
            str(model["det_number"]),
            "--det_edge",
            str(model["det_edge"]),
            "--det_size",
            str(model["det_size"]),
            "--det_grid",
            str(model["det_grid"]),
            "--det_gap",
            str(model["det_gap"]),
            "--hidden",
            str(model["hidden"]),
            "--proj_layers",
            str(model["proj_layers"]),
            "--proj_dropout",
            str(model["proj_dropout"]),
            "--electric_mode",
            model["electric_mode"],
            "--electric_layers",
            str(model["electric_layers"]),
            "--electric_heads",
            str(model["electric_heads"]),
            "--electric_hidden",
            str(model["electric_hidden"]),
            "--electric_dropout",
            str(model["electric_dropout"]),
            "--limit_n",
            str(limit_n),
            "--epochs",
            str(train["epochs"]),
            "--batch_size",
            str(train["batch_size"]),
            "--num_workers",
            str(train.get("num_workers", 0)),
            "--lr",
            str(train["lr"]),
            "--electric_lr_ratio",
            str(train["electric_lr_ratio"]),
            "--label_smoothing",
            str(train["label_smoothing"]),
            "--train_ratio",
            str(train["train_ratio"]),
            "--seed",
            str(train["seed"]),
            "--patience",
            str(train["patience"]),
            "--min_epochs",
            str(train["min_epochs"]),
            "--ckpt",
            output["ckpt"],
        ]
        if bool(train.get("pin_memory", False)):
            train_cmd.append("--pin_memory")
        if int(train.get("num_workers", 0)) > 0:
            train_cmd += ["--prefetch_factor", str(train.get("prefetch_factor", 2))]
        optical_demo = bool(train.get("optical_demo", True))
        if args.optical_demo is True:
            optical_demo = True
        if optical_demo:
            train_cmd.append("--optical_demo")
        run(train_cmd, env=env)


if __name__ == "__main__":
    main()
