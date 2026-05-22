import numpy as np
from sklearn.decomposition import PCA
import torch

def build_gpt2_token_embedding_matrix(model, tokenizer):
    # returns numpy matrix [vocab, dim]
    emb = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)
    return emb

def word_to_gpt2_vector(word: str, tokenizer, token_emb: np.ndarray):
    # Encode both " word" and "word", choose the one with more 'word-like' tokens (heuristic) then average embeddings.
    ids1 = tokenizer.encode(" " + word, add_special_tokens=False)
    ids2 = tokenizer.encode(word, add_special_tokens=False)
    def avg(ids):
        if len(ids) == 0:
            return None
        return token_emb[np.array(ids, dtype=np.int64)].mean(axis=0)
    v1 = avg(ids1)
    v2 = avg(ids2)
    if v1 is None and v2 is None:
        return None
    if v1 is None:
        return v2
    if v2 is None:
        return v1
    # prefer the encoding with fewer tokens (closer to a single word)
    return v1 if len(ids1) <= len(ids2) else v2

def fit_pca_to_k2(word_vecs: np.ndarray, k2: int, seed: int = 0):
    # word_vecs: (M, D)
    pca = PCA(n_components=k2, random_state=seed)
    Z = pca.fit_transform(word_vecs).astype(np.float32)
    return pca, Z

def apply_pca(pca, vecs: np.ndarray):
    return pca.transform(vecs).astype(np.float32)
