import re

_WORD_RE = re.compile(r"[^a-z0-9]+")

def simple_tokenize(text: str):
    text = text.lower()
    text = _WORD_RE.sub(" ", text)
    toks = text.split()
    return toks

def build_windows(tokens, Tin, Tout):
    L = len(tokens)
    if L < Tin + Tout:
        return None, None
    X, Y = [], []
    for i in range(L - Tin - Tout + 1):
        X.append(tokens[i:i+Tin])
        Y.append(tokens[i+Tin:i+Tin+Tout])
    import numpy as np
    return np.array(X, dtype=object), np.array(Y, dtype=object)
