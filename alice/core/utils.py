import numpy as np
import os


def sanitize_ssl_env():
    # Some shells export SSL cert env vars with invalid paths,
    # which breaks huggingface/httpx client initialization.
    for k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        v = os.environ.get(k, "").strip()
        if v and (not os.path.exists(v)):
            os.environ.pop(k, None)

def auto_quantize_int8(x: np.ndarray):
    # global symmetric scale
    x = np.asarray(x, dtype=np.float32)
    m = float(np.max(np.abs(x))) if x.size else 1.0
    scale = m / 127.0 if m > 0 else 1.0
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return q, float(scale)

def quantize_int8(x: np.ndarray, scale: float):
    x = np.asarray(x, dtype=np.float32)
    q = np.clip(np.round(x / float(scale)), -128, 127).astype(np.int8)
    return q

def dequantize_int8(q: np.ndarray, scale: float):
    return q.astype(np.float32) * float(scale)
