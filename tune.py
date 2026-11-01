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
ap.add_argument("--biases", type=float, nargs="+", default=[0.0])
ap.add_argument("--biases_small", type=float, nargs="+", default=None, help="offset for small predicted objects (default: same as --biases)")
ap.add_argument("--plo", type=float, default=None)
ap.add_argument("--phi", type=float, default=None)
ap.add_argument("--temps", type=float, nargs="+", default=[1.0])
ap.add_argument("--stratified", action="store_true", help="select by Q averaged over 4 object-size groups")
ap.add_argument("--n", type=int, default=0, help="limit number of validation photos (0 = all)")
a = ap.parse_args()

params = seg.load_params(a.params)
sg = seg.Segmenter(params)
if a.plo: sg.p_lo = a.plo
if a.phi: sg.p_hi = a.phi
X = np.load(os.path.join(common.CACHE, "tr_img.npy"), mmap_mode="r")
Y = np.load(os.path.join(common.CACHE, "tr_msk.npy"), mmap_mode="r")
val = np.load(os.path.join(common.CACHE, "split.npz"))["val"]
if a.n:
    val = val[:a.n]
preps = [sg.prepare(np.ascontiguousarray(X[i])) for i in val]
gts = np.stack([Y[i] for i in val])
area = gts.reshape(len(gts), -1).mean(1)
GROUPS = [(area < 0.10), (area >= 0.10) & (area < 0.25), (area >= 0.25) & (area < 0.45), (area >= 0.45)]
print("validation photos per object-size group (<10%, 10-25%, 25-45%, >45%):", [int(g.sum()) for g in GROUPS])
print(f"{len(val)} validation photos prepared")


def evaluate(lam, sig, up, bias=0.0, temp=1.0, bias_s=None):
    sg.upsample = up
    sg.bias, sg.temp = bias, temp
    sg.bias_s = bias if bias_s is None else bias_s
    preds = np.stack([sg.paint(d, sg.decide(d, lam, sig)) for d in preps])
    m, b, q = metrics.score(preds, gts)
    qs = [metrics.score(preds[g], gts[g])[2] for g in GROUPS]
    evaluate.last_groups = qs
    return m, b, (float(np.mean(qs)) if a.stratified else q)


best = None
print(f"{'lam':>6} {'sigma':>6} {'b_lrg':>6} {'b_sml':>6} {'temp':>5} | {'mIoU':>6} {'BF1':>6} {'Q':>6}")
for lam, sig, bias, temp, bias_s in itertools.product(a.lams, a.sigmas, a.biases, a.temps, a.biases_small or [None]):
    m, b, q = evaluate(lam, sig, 1, bias, temp, bias_s)
    print(f"{lam:6.2f} {sig:6.2f} {bias:6.2f} {bias if bias_s is None else bias_s:6.2f} {temp:5.2f} | {m:6.4f} {b:6.4f} {q:6.4f} | per-size-group Q: " + " ".join(f"{x:.3f}" for x in evaluate.last_groups), flush=True)
    if best is None or q > best[0]:
        best = (q, lam, sig, bias, temp, bias if bias_s is None else bias_s)
q, lam, sig, bias, temp, bias_s = best
print("selected (score, lam, sigma, bias_large, temp, bias_small) =", best)
if a.write:
    params.update(lam=lam, sigma_scale=sig, upsample=1, bias=bias, temp=temp, bias_small=bias_s, p_lo=sg.p_lo, p_hi=sg.p_hi)
    np.savez(a.params, **{k: np.asarray(v) for k, v in params.items()})
    print("written to", a.params)
