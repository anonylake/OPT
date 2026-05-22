import os, argparse, numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.text import simple_tokenize, build_windows
from core.semantic import build_gpt2_token_embedding_matrix, word_to_gpt2_vector, fit_pca_to_k2, apply_pca
from core.utils import auto_quantize_int8, quantize_int8, sanitize_ssl_env

def vec_to_square(v, k):
    kk = k*k
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if v.size >= kk:
        vv = v[:kk]
    else:
        vv = np.zeros((kk,), dtype=np.float32)
        vv[:v.size] = v
    return vv.reshape(k,k)

def main():
    sanitize_ssl_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Tin", type=int, default=4)
    ap.add_argument("--Tout", type=int, default=4)
    ap.add_argument("--k", type=int, default=18)
    ap.add_argument("--gpt2_name", type=str, default="gpt2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base = os.path.basename(args.out.rstrip("/"))

    with open(args.txt, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = simple_tokenize(text)
    Xw, Yw = build_windows(tokens, args.Tin, args.Tout)
    if Xw is None:
        raise RuntimeError("Text too short for Tin+Tout")

    np.save(os.path.join(args.out, f"{base}_input_words.npy"), Xw, allow_pickle=True)
    np.save(os.path.join(args.out, f"{base}_target_words.npy"), Yw, allow_pickle=True)

    # Build word list (from X and Y) for semantic vectors
    vocab = sorted({str(w) for w in np.concatenate([Xw.reshape(-1), Yw.reshape(-1)])})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.gpt2_name)
    lm = AutoModelForCausalLM.from_pretrained(args.gpt2_name).to(device)
    lm.eval()
    token_emb = build_gpt2_token_embedding_matrix(lm, tok)

    vecs = []
    keep = []
    for w in tqdm(vocab, desc="Word->GPT2-vec"):
        v = word_to_gpt2_vector(w, tok, token_emb)
        if v is None:
            continue
        vecs.append(v)
        keep.append(w)
    vecs = np.stack(vecs, axis=0).astype(np.float32)

    k2 = args.k * args.k
    pca, Z = fit_pca_to_k2(vecs, k2=k2, seed=args.seed)

    # save semantic npz
    sem_path = os.path.join(args.out, f"{base}_semantic_k{args.k}.npz")
    np.savez(sem_path, words=np.array(keep, dtype=object), vecs=Z)
    print("[Save] semantic ->", sem_path)

    # map words to squares using PCA vectors
    w2i = {w:i for i,w in enumerate(keep)}
    N = Xw.shape[0]
    X_sq = np.zeros((N, args.Tin, args.k, args.k), dtype=np.float32)
    Y_sq = np.zeros((N, args.Tout, args.k, args.k), dtype=np.float32)

    for i in tqdm(range(N), desc="Squares"):
        for t in range(args.Tin):
            w = str(Xw[i,t])
            j = w2i.get(w, None)
            if j is not None:
                X_sq[i,t] = Z[j].reshape(args.k, args.k)
        for t in range(args.Tout):
            w = str(Yw[i,t])
            j = w2i.get(w, None)
            if j is not None:
                Y_sq[i,t] = Z[j].reshape(args.k, args.k)

    all_sq = np.concatenate([X_sq.reshape(-1,args.k,args.k), Y_sq.reshape(-1,args.k,args.k)], axis=0)
    q_all, scale = auto_quantize_int8(all_sq)
    X_int8 = quantize_int8(X_sq, scale)
    Y_int8 = quantize_int8(Y_sq, scale)

    np.save(os.path.join(args.out, f"{base}_inputs_int8.npy"), X_int8)
    np.save(os.path.join(args.out, f"{base}_targets_int8.npy"), Y_int8)
    np.savez(os.path.join(args.out, f"{base}_proj.npz"), k=args.k, Tin=args.Tin, Tout=args.Tout, q_scale=scale)

    print("[Done] saved int8 squares and meta. q_scale=", scale)

if __name__ == "__main__":
    main()
