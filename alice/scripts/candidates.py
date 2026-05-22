import os, argparse, numpy as np
from collections import Counter
from tqdm import tqdm
import torch
from core.gpt2_task_vocab import build_task_vocab, GPT2TaskVocabScorer
from core.dataset import load_semantic_npz
from core.utils import sanitize_ssl_env

def main():
    sanitize_ssl_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_words_npy", required=True)
    ap.add_argument("--target_words_npy", required=True)
    ap.add_argument("--limit_n", type=int, default=200)
    ap.add_argument("--gpt2_name", type=str, default="gpt2")
    ap.add_argument("--cand_k", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher_forcing", action="store_true")
    ap.add_argument("--force_include_target", action="store_true")
    ap.add_argument("--soft_include_target", action="store_true")
    ap.add_argument("--soft_include_margin", type=float, default=0.5)
    ap.add_argument("--soft_include_topn", type=int, default=200)
    ap.add_argument("--soft_include_min_freq", type=int, default=2)
    ap.add_argument("--semantic_npz", type=str, default="")
    ap.add_argument("--semantic_mix_ratio", type=float, default=0.0)
    args = ap.parse_args()

    X = np.load(args.input_words_npy, allow_pickle=True)[:args.limit_n]
    Y = np.load(args.target_words_npy, allow_pickle=True)[:args.limit_n]
    Tin = X.shape[1]
    Tout = Y.shape[1]

    vocab = build_task_vocab(args.input_words_npy, args.target_words_npy, limit_n=args.limit_n)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = GPT2TaskVocabScorer(model_name=args.gpt2_name, device=device, task_vocab=vocab)

    # High-frequency target words used by soft target inclusion.
    freq = Counter([str(w) for w in Y.reshape(-1).tolist()])
    soft_topn = max(1, min(args.soft_include_topn, len(freq)))
    high_freq_words = set()
    for w, c in freq.most_common(soft_topn):
        if c >= args.soft_include_min_freq:
            high_freq_words.add(str(w))

    cands = np.empty((args.limit_n, Tout, args.cand_k), dtype=object)

    sem_enabled = bool(args.semantic_npz) and args.semantic_mix_ratio > 0.0
    sem_vocab_words = []
    sem_mat = None
    sem_w2i = {}
    if sem_enabled:
        words, vecs, w2i = load_semantic_npz(args.semantic_npz)
        keep_words = []
        keep_vecs = []
        vocab_set = set(vocab)
        for w in vocab:
            j = w2i.get(w, None)
            if j is not None:
                keep_words.append(w)
                keep_vecs.append(vecs[j])
        if len(keep_words) > 0:
            sem_vocab_words = keep_words
            sem_mat = np.asarray(keep_vecs, dtype=np.float32)
            sem_mat = sem_mat / np.maximum(np.linalg.norm(sem_mat, axis=1, keepdims=True), 1e-12)
            sem_w2i = {str(w): i for i, w in enumerate(words)}
        else:
            sem_enabled = False

    def semantic_topk(prompt_words, k):
        if (not sem_enabled) or sem_mat is None or k <= 0:
            return []
        pvecs = []
        for w in prompt_words:
            j = sem_w2i.get(str(w), None)
            if j is None:
                continue
            pvecs.append(vecs[j])
        if len(pvecs) == 0:
            return []
        q = np.mean(np.asarray(pvecs, dtype=np.float32), axis=0)
        qn = np.linalg.norm(q)
        if qn < 1e-12:
            return []
        q = q / qn
        sims = sem_mat @ q
        top = np.argpartition(-sims, min(k, sims.shape[0]) - 1)[: min(k, sims.shape[0])]
        top = top[np.argsort(-sims[top])]
        return [sem_vocab_words[int(i)] for i in top.tolist()]

    for i in tqdm(range(args.limit_n), desc="precompute"):
        prompt = [str(w) for w in X[i].tolist()]
        for t in range(Tout):
            sem_k = int(round(args.cand_k * min(max(args.semantic_mix_ratio, 0.0), 1.0))) if sem_enabled else 0
            sem_k = min(max(sem_k, 0), args.cand_k)
            gpt_k = max(1, args.cand_k - sem_k)

            gpt_pairs = scorer.topk_words_with_scores(prompt, k=max(args.cand_k * 4, gpt_k * 4))
            gpt_cands = [w for w, _ in gpt_pairs]
            gpt_score_map = {str(w): float(s) for w, s in gpt_pairs}
            sem_cands = semantic_topk(prompt, k=max(sem_k * 3, sem_k))

            merged = []
            gi, si = 0, 0
            while len(merged) < args.cand_k and (gi < len(gpt_cands) or si < len(sem_cands)):
                if gi < len(gpt_cands):
                    wg = str(gpt_cands[gi]); gi += 1
                    if wg not in merged:
                        merged.append(wg)
                        if len(merged) >= args.cand_k:
                            break
                if si < len(sem_cands):
                    ws = str(sem_cands[si]); si += 1
                    if ws not in merged:
                        merged.append(ws)
            cw = merged[:args.cand_k]
            # keep candidates unique to maximize effective coverage
            uniq = []
            for w in cw:
                sw = str(w)
                if sw not in uniq:
                    uniq.append(sw)
            gt = str(Y[i, t])
            if args.force_include_target and gt not in uniq:
                if len(uniq) >= args.cand_k:
                    uniq[-1] = gt
                else:
                    uniq.append(gt)
            elif args.soft_include_target and gt not in uniq:
                # Soft inclusion: replace tail candidate only when gt is frequent
                # and LM score is close to current tail score.
                if gt in high_freq_words:
                    gt_score = gpt_score_map.get(gt, scorer.score_word(prompt, gt))
                    tail_word = uniq[-1] if len(uniq) > 0 else ""
                    tail_score = gpt_score_map.get(str(tail_word), -1e9)
                    if gt_score >= (tail_score - args.soft_include_margin):
                        if len(uniq) >= args.cand_k:
                            uniq[-1] = gt
                        else:
                            uniq.append(gt)
            while len(uniq) < args.cand_k:
                uniq.append(uniq[-1] if uniq else gt)
            cw = uniq[:args.cand_k]
            cands[i,t,:] = np.array(cw, dtype=object)
            # update prompt
            if args.teacher_forcing:
                prompt.append(gt)
            else:
                prompt.append(cw[0])  # greedy
    np.save(args.out, cands, allow_pickle=True)
    print("[Save] cand cache ->", args.out)

if __name__ == "__main__":
    main()
