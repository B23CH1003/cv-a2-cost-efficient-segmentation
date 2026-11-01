"""Average time per pipeline stage (wall clock, ms) over DUTS-TE photos -> results.json['stages'].

Stage boundaries: downscale+Lab | K-Means | region features & adjacency | probability
(both logistic regressions) | graph cut (edge weights + max-flow) | paint & enlarge.
"""
import common
import json
import os
import time

import cv2
import numpy as np

import seg

N = 1000
sg = seg.Segmenter(seg.load_params(common.PARAMS))
X = np.load(os.path.join(common.CACHE, "te_img.npy"), mmap_mode="r")
imgs = [np.ascontiguousarray(X[i]) for i in range(0, len(X), len(X) // N)][:N]
sg.segment(imgs[0])
t = dict(downscale_lab=0.0, kmeans=0.0, features=0.0, probability=0.0, graph_cut=0.0, paint=0.0)
pc = time.perf_counter
for img in imgs:
    t0 = pc()
    small = cv2.resize(img, (sg.R, sg.R), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    t1 = pc()
    sg.Xa[:, :3] = lab.reshape(-1, 3)
    sg.XT[:3] = lab.reshape(-1, 3).T
    sg._kmeans()
    t2 = pc()
    d = sg.prepare(img)
    t3 = pc()
    sg.probability(d)
    t4 = pc()
    obj = sg.decide(d)
    t5 = pc()
    sg.paint(d, obj)
    t6 = pc()
    t["downscale_lab"] += t1 - t0
    t["kmeans"] += t2 - t1
    t["features"] += (t3 - t2) - (t2 - t1) - (t1 - t0)       # prepare() minus the parts timed above
    t["probability"] += t4 - t3
    t["graph_cut"] += (t5 - t4) - (t4 - t3)                  # decide() recomputes probability
    t["paint"] += t6 - t5
out = {k: 1000 * v / len(imgs) for k, v in t.items()}
out["total"] = sum(out.values())
print(json.dumps(out, indent=1))
rp = os.path.join(common.ROOT, "results.json")
res = json.load(open(rp)) if os.path.exists(rp) else {}
res["stages"] = out
json.dump(res, open(rp, "w"), indent=1)
