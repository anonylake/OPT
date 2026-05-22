import os, re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def load_semantic_npz(path: str):
    d = np.load(path, allow_pickle=True)
    words = d["words"].astype(object).tolist()
    vecs = d["vecs"].astype(np.float32)
    w2i = {str(w): i for i, w in enumerate(words)}
    return words, vecs, w2i

def collate_keep_words(batch):
    xs, xws, yws, yvs, idxs = zip(*batch)
    x = torch.stack(xs, dim=0)
    yv = torch.stack(yvs, dim=0)
    idx = torch.tensor(idxs, dtype=torch.long)
    return x, list(xws), list(yws), yv, idx

class OpticalWordDataset(Dataset):
    def __init__(self, img_inputs_dir: str, input_words_npy: str, target_words_npy: str,
                 semantic_npz: str, Tin: int, Tout: int, limit_n: int = 0):
        self.img_inputs_dir = img_inputs_dir
        self.Xw = np.load(input_words_npy, allow_pickle=True)
        self.Yw = np.load(target_words_npy, allow_pickle=True)
        if limit_n and limit_n > 0:
            self.Xw = self.Xw[:limit_n]
            self.Yw = self.Yw[:limit_n]
        self.Tin = Tin
        self.Tout = Tout
        self.words, self.vecs, self.w2i = load_semantic_npz(semantic_npz)
        self.D = self.vecs.shape[1]

    def __len__(self):
        return self.Xw.shape[0]

    def _load_bmp(self, idx: int, p: int):
        path = os.path.join(self.img_inputs_dir, f"{idx:06d}_p{p}.bmp")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing BMP: {path}")
        im = Image.open(path).convert("L")
        arr = np.array(im, dtype=np.float32) / 255.0
        # (1,H,W)
        t = torch.from_numpy(arr).unsqueeze(0)
        return t

    def __getitem__(self, i: int):
        xs = [self._load_bmp(i, p) for p in range(self.Tin)]
        x = torch.stack(xs, dim=0)  # (Tin,1,H,W)
        xw = [str(w) for w in self.Xw[i].tolist()]
        yw = [str(w) for w in self.Yw[i].tolist()]
        # y semantic vectors
        yv = []
        for w in yw:
            j = self.w2i.get(w, None)
            if j is None:
                yv.append(np.zeros((self.D,), dtype=np.float32))
            else:
                yv.append(self.vecs[j])
        yv = torch.tensor(np.stack(yv, axis=0), dtype=torch.float32)  # (Tout,D)
        return x, xw, yw, yv, i
