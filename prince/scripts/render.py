import os, argparse, numpy as np
from PIL import Image

def upscale_square_to_bmp(square, out_hw):
    # square: (k,k) float/int
    # normalize to 0..255
    s = square.astype(np.float32)
    mn, mx = float(s.min()), float(s.max())
    if mx - mn < 1e-12:
        img = np.zeros_like(s, dtype=np.uint8)
    else:
        img = np.clip((s - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(img, mode="L").resize((out_hw, out_hw), resample=Image.NEAREST)
    return im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out_hw", type=int, default=500)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    X = np.load(args.inputs, allow_pickle=True)  # (N,Tin,k,k) int8
    N = min(args.n, X.shape[0])
    Tin = X.shape[1]
    for i in range(N):
        for p in range(Tin):
            im = upscale_square_to_bmp(X[i,p], args.out_hw)
            im.save(os.path.join(args.outdir, f"{i:06d}_p{p}.bmp"))
    print("[Done] wrote", N*Tin, "bmps to", args.outdir)

if __name__ == "__main__":
    main()
