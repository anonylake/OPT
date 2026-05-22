import os
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.semantic import build_gpt2_token_embedding_matrix, word_to_gpt2_vector, fit_pca_to_k2
from core.utils import sanitize_ssl_env


def main():
    sanitize_ssl_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_words_npy", required=True)
    ap.add_argument("--target_words_npy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--gpt2_name", type=str, default="gpt2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    xw = np.load(args.input_words_npy, allow_pickle=True)
    yw = np.load(args.target_words_npy, allow_pickle=True)
    vocab = sorted({str(w) for w in np.concatenate([xw.reshape(-1), yw.reshape(-1)])})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.gpt2_name)
    lm = AutoModelForCausalLM.from_pretrained(args.gpt2_name).to(device)
    lm.eval()
    token_emb = build_gpt2_token_embedding_matrix(lm, tok)

    vecs = []
    keep = []
    for w in vocab:
        v = word_to_gpt2_vector(w, tok, token_emb)
        if v is None:
            continue
        vecs.append(v)
        keep.append(w)
    vecs = np.stack(vecs, axis=0).astype(np.float32)
    _pca, z = fit_pca_to_k2(vecs, k2=args.k * args.k, seed=args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, words=np.array(keep, dtype=object), vecs=z)
    print("[Save] semantic ->", args.out)


if __name__ == "__main__":
    main()
