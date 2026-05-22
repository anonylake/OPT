import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from core.utils import sanitize_ssl_env

def build_task_vocab(input_words_npy: str, target_words_npy: str, limit_n: int = 200):
    X = np.load(input_words_npy, allow_pickle=True)[:limit_n]
    Y = np.load(target_words_npy, allow_pickle=True)[:limit_n]
    vocab = sorted({str(w).strip().lower() for w in np.concatenate([X.reshape(-1), Y.reshape(-1)]) if str(w).strip()})
    return vocab

class GPT2TaskVocabScorer:
    def __init__(self, model_name: str = "gpt2", device: str = "cpu", task_vocab=None, cache_size: int = 200000):
        assert task_vocab is not None and len(task_vocab) > 0
        sanitize_ssl_env()
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.lm.eval()
        self.task_vocab = task_vocab
        self.word_token_ids = []
        for w in self.task_vocab:
            ids1 = self.tok.encode(" " + w, add_special_tokens=False)
            ids2 = self.tok.encode(w, add_special_tokens=False)
            self.word_token_ids.append((ids1, ids2))
        self.cache = {}
        self.cache_size = cache_size

    @torch.no_grad()
    def topk_words(self, prompt_words, k=5):
        pairs = self.topk_words_with_scores(prompt_words, k=k)
        return [w for w, _ in pairs]

    @torch.no_grad()
    def topk_words_with_scores(self, prompt_words, k=5):
        key = " ".join(prompt_words)
        got = self.cache.get((key, k, "pairs"), None)
        if got is not None:
            return got
        # simple cache cap
        if len(self.cache) > self.cache_size:
            self.cache.clear()

        prompt = " ".join(prompt_words)
        inp = self.tok(prompt, return_tensors="pt").to(self.device)
        input_ids = inp["input_ids"]  # (1,L)
        out = self.lm(input_ids=input_ids)
        first_logits = out.logits[0, -1]  # (V,)

        scores = []
        for (ids1, ids2), w in zip(self.word_token_ids, self.task_vocab):
            s = self._score_word(input_ids, first_logits, ids1, ids2)
            scores.append((s, w))
        scores.sort(key=lambda x: x[0], reverse=True)
        pairs = [(w, float(s)) for s, w in scores[:k]]
        self.cache[(key, k, "pairs")] = pairs
        return pairs

    @torch.no_grad()
    def score_word(self, prompt_words, word: str):
        prompt = " ".join(prompt_words)
        inp = self.tok(prompt, return_tensors="pt").to(self.device)
        input_ids = inp["input_ids"]
        out = self.lm(input_ids=input_ids)
        first_logits = out.logits[0, -1]
        w = str(word)
        ids1 = self.tok.encode(" " + w, add_special_tokens=False)
        ids2 = self.tok.encode(w, add_special_tokens=False)
        return float(self._score_word(input_ids, first_logits, ids1, ids2))

    @torch.no_grad()
    def _score_seq(self, base_input_ids, first_logits, seq_ids):
        if len(seq_ids) == 0:
            return -1e9
        logp = F.log_softmax(first_logits, dim=-1)[seq_ids[0]].item()
        if len(seq_ids) == 1:
            return logp
        cur = torch.cat([base_input_ids, torch.tensor([seq_ids], device=self.device)], dim=1)
        out = self.lm(input_ids=cur)
        base_L = base_input_ids.shape[1]
        for j in range(1, len(seq_ids)):
            logits_j = out.logits[0, base_L + j - 1]
            logp += F.log_softmax(logits_j, dim=-1)[seq_ids[j]].item()
        return logp

    def _score_word(self, base_input_ids, first_logits, ids1, ids2):
        s1 = self._score_seq(base_input_ids, first_logits, ids1)
        s2 = self._score_seq(base_input_ids, first_logits, ids2)
        return max(s1, s2)
