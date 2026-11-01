"""Tune the graph-cut hyper-parameters (lambda, sigma_d scale, upsampling) on the VALIDATION split.

    python tune.py                 # print the grid
    python tune.py --write         # also store the best setting in params.npz

Region features are computed once per photo and re-used for every grid point.
"""
import common
import argparse
import itertools
import os

import numpy as np

import metrics
import seg

ap = argparse.ArgumentParser()
ap.add_argument("--write", action="store_true")
ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.5, 1, 2, 4, 8, 16])
ap.add_argument("--sigmas", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4])
ap.add_argument("--params", default=common.PARAMS)
ap.add_argument("--n", type=int, default=0, help="limit number of validation photos (0 = all)")
a = ap.parse_args()

params = seg.load_params(a.params)
sg = seg.Segmenter(params)
X = np.load(os.path.join(common.CACHE, "tr_img.npy"), mmap_mode="r")
Y = np.load(os.path.join(common.CACHE, "tr_msk.npy"), mmap_mode="r")
val = np.load(os.path.join(common.CACHE, "split.npz"))["val"]
if a.n:
    val = val[:a.n]
preps = [sg.prepare(np.ascontiguousarray(X[i])) for i in val]
gts = np.stack([Y[i] for i in val])
print(f"{len(val)} validation photos prepared")


def evaluate(lam, sig, up):
    sg.upsample = up
    preds = np.stack([sg.paint(d, sg.decide(d, lam, sig)) for d in preps])
    return metrics.score(preds, gts)


best = None
print(f"{'lam':>6} {'sigma':>6} {'up':>3} | {'mIoU':>6} {'BF1':>6} {'Q':>6}")
for lam, sig in itertools.product(a.lams, a.sigmas):
    m, b, q = evaluate(lam, sig, 1)
    print(f"{lam:6.2f} {sig:6.2f} {1:>3} | {m:6.4f} {b:6.4f} {q:6.4f}", flush=True)
    if best is None or q > best[0]:
        best = (q, lam, sig)
q, lam, sig = best
for up in (0, 1):
    m, b, qq = evaluate(lam, sig, up)
    print(f"best lam={lam} sigma={sig} upsample={up} | mIoU {m:.4f} BF1 {b:.4f} Q {qq:.4f}")
print("selected", best)
if a.write:
    params.update(lam=lam, sigma_scale=sig, upsample=1)
    np.savez(a.params, **{k: np.asarray(v) for k, v in params.items()})
    print("written to", a.params)
