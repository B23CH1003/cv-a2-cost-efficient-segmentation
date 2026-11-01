"""python predict.py --image photo.jpg --out mask.png

Resizes the photo (256x256, INTER_AREA), segments it with the saved parameters, writes the
0/255 mask PNG, then times 20 further runs and prints exactly two lines:
    T_CPU_ms: <average CPU ms per run>
    S_disk_bytes: <size of params.npz>
"""
import common  # sets the single-thread environment variables before numpy/cv2/torch load
import argparse
import os
import time

import cv2
import numpy as np
import torch

import seg

cv2.setNumThreads(1)
torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default=common.PARAMS)
    a = ap.parse_args()

    sg = seg.Segmenter(seg.load_params(a.params))
    img = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {a.image}")
    img = common.resize_image(img)             # not timed (resizing the original photo)

    mask = sg.segment(img)                     # first run: slow, not timed
    cv2.imwrite(a.out, mask * 255)

    runs = 20
    total = 0.0
    for _ in range(runs):
        t0 = time.process_time()
        sg.segment(img)
        total += time.process_time() - t0
    print(f"T_CPU_ms: {1000 * total / runs:.2f}")
    print(f"S_disk_bytes: {os.path.getsize(a.params)}")


if __name__ == "__main__":
    main()
