"""Evaluate the saved pipeline on the validation split, DUTS-TE or the captured photos.

    python evaluate.py --set val
    python evaluate.py --set te        # also measures T_CPU (one photo at a time, process_time)
    python evaluate.py --set captured

Results are merged into results.json (used by make_report.py). T_CPU is measured on DUTS-TE
as required; for the other sets the timing is still recorded but is not the official one.
"""
import common
import argparse
import glob
import json
import os
import re
import time

import cv2
import numpy as np

import metrics
import seg


def load_set(name, captured_dir=None):
    if name == "captured":
        cd = captured_dir or os.path.join(common.ROOT, "captured")
        names = sorted((os.path.splitext(os.path.basename(f))[0]
                        for f in glob.glob(os.path.join(cd, "images_256", "*.png"))),
                       key=lambda n: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", n)])
        if not names:
            raise SystemExit(f"no photos found in {cd}/images_256 - see captured_tools.py")
        X = np.stack([cv2.imread(os.path.join(cd, "images_256", n + ".png")) for n in names])
        Y = np.stack([(cv2.imread(os.path.join(cd, "masks_256", n + ".png"), 0) >= 128).astype(np.uint8)
                      for n in names])
        return names, X, Y
    tag = "te" if name == "te" else "tr"
    X = np.load(os.path.join(common.CACHE, f"{tag}_img.npy"), mmap_mode="r")
    Y = np.load(os.path.join(common.CACHE, f"{tag}_msk.npy"), mmap_mode="r")
    if name == "val":
        idx = np.load(os.path.join(common.CACHE, "split.npz"))["val"]
        return idx, X[idx], Y[idx]
    return np.arange(len(X)), X, Y


def run(sg, X):
    """Segment photos one at a time; returns masks and the mean CPU / wall time per photo in ms."""
    sg.segment(np.ascontiguousarray(X[0]))      # warm-up (not timed)
    preds = np.empty((len(X), common.SIZE, common.SIZE), dtype=np.uint8)
    cpu = 0.0
    wall = 0.0
    for i in range(len(X)):
        img = np.ascontiguousarray(X[i])
        c0, w0 = time.process_time(), time.perf_counter()
        preds[i] = sg.segment(img)
        cpu += time.process_time() - c0
        wall += time.perf_counter() - w0
    return preds, 1000 * cpu / len(X), 1000 * wall / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["val", "te", "captured"], required=True)
    ap.add_argument("--params", default=common.PARAMS)
    ap.add_argument("--captured_dir", default=None)
    ap.add_argument("--results", default=os.path.join(common.ROOT, "results.json"))
    ap.add_argument("--save_preds", default=None, help="optional .npy path for the predicted masks")
    a = ap.parse_args()

    sg = seg.Segmenter(seg.load_params(a.params))
    ids, X, Y = load_set(a.set, a.captured_dir)
    preds, t_cpu, t_wall = run(sg, X)
    miou, bf1, q = metrics.score(preds, np.asarray(Y))
    s_disk = os.path.getsize(a.params)
    ces = metrics.ces(q, t_cpu, s_disk)
    res = dict(n=len(X), mIoU=miou, BF1=bf1, Q=q, T_CPU_ms=t_cpu, T_wall_ms=t_wall, S_disk=s_disk, CES=ces)
    print(json.dumps(res, indent=1))
    rp = a.results
    allres = json.load(open(rp)) if os.path.exists(rp) else {}
    allres[a.set] = res
    json.dump(allres, open(rp, "w"), indent=1)
    if a.save_preds:
        np.save(a.save_preds, preds)


if __name__ == "__main__":
    main()
