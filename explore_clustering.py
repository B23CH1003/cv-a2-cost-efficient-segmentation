"""Design-choice study for the clustering stage on the VALIDATION split.

For each (R, G, wpos, iters) it clusters the validation photos and labels every region with
the ground-truth majority ("oracle"). The resulting mIoU/BF1 is the ceiling that any region
probability model could reach with that clustering, so it shows what resolution / number of
regions / spatial weight are worth their time. It also reports the time of the clustering +
feature stage.
"""
import common
import itertools
import os
import sys
import time

import cv2
import numpy as np

import metrics
import seg

N_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 300
X = np.load(os.path.join(common.CACHE, "tr_img.npy"), mmap_mode="r")
Y = np.load(os.path.join(common.CACHE, "tr_msk.npy"), mmap_mode="r")
val = np.load(os.path.join(common.CACHE, "split.npz"))["val"][:N_IMG]

print(f"{'R':>4} {'G':>3} {'wpos':>5} {'it':>3} {'up':>3} | {'mIoU':>6} {'BF1':>6} {'Q':>6} | prepare ms")
for R, G, wpos, iters, up in itertools.product([64], [10, 12], [160, 240, 320], [3, 5], [1]):
    params = dict(R=R, G=G, wpos=wpos, iters=iters, lam=1.0, sigma_scale=1.0, upsample=up,
                  prior=np.full((R, R), 0.3), llr=np.zeros(4096), mu=np.zeros(seg.N_FEAT),
                  sd=np.ones(seg.N_FEAT), w=np.zeros(2 * seg.N_FEAT), b=0.0)
    sg = seg.Segmenter(params)
    preds, gts = [], []
    tp = 0.0
    sg.prepare(np.ascontiguousarray(X[val[0]]))
    for i in val:
        img = np.ascontiguousarray(X[i])
        t0 = time.process_time()
        d = sg.prepare(img)
        tp += time.process_time() - t0
        g = cv2.resize(Y[i].astype(np.float32), (R, R), interpolation=cv2.INTER_AREA).ravel()
        frac = np.bincount(d["lab"], weights=g, minlength=d["n"]) / d["cnt"]
        preds.append(sg.paint(d, frac > 0.5))
        gts.append(Y[i])
    m, b, q = metrics.score(np.stack(preds), np.stack(gts))
    print(f"{R:>4} {G:>3} {wpos:>5} {iters:>3} {up:>3} | {m:6.3f} {b:6.3f} {q:6.3f} | {1000 * tp / len(val):6.1f}", flush=True)
