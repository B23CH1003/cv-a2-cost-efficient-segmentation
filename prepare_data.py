"""Unzip DUTS, resize every photo/mask with the Section 1b rule and cache them as .npy.

Run once:  python prepare_data.py
Creates data/cache/{tr,te}_img.npy (N,256,256,3 uint8 BGR), {tr,te}_msk.npy (N,256,256 uint8 0/1),
{tr,te}_names.txt and split.npz (seeded 90/10 train/validation split of DUTS-TR).
DUTS-TE is only ever used for the final evaluation.
"""
import common
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

RAW = os.path.join(common.DATA, "raw")


def extract(split):
    out = os.path.join(RAW, split)
    if not os.path.isdir(out):
        os.makedirs(out)
        with zipfile.ZipFile(os.path.join(common.DATA, f"DUTS-{split}.zip")) as z:
            z.extractall(out)
    imgs, msks = {}, {}
    for dp, _, fs in os.walk(out):
        for f in fs:
            base, ext = os.path.splitext(f)
            full = os.path.join(dp, f)
            if ext.lower() == ".jpg" and "Image" in dp:
                imgs[base] = full
            elif ext.lower() == ".png" and "Mask" in dp:
                msks[base] = full
    names = sorted(set(imgs) & set(msks))
    print(split, "images", len(imgs), "masks", len(msks), "paired", len(names))
    return names, imgs, msks


def build(split, tag):
    names, imgs, msks = extract(split)
    n = len(names)
    X = np.lib.format.open_memmap(os.path.join(common.CACHE, f"{tag}_img.npy"), "w+", np.uint8, (n, 256, 256, 3))
    Y = np.lib.format.open_memmap(os.path.join(common.CACHE, f"{tag}_msk.npy"), "w+", np.uint8, (n, 256, 256))

    def load(i):
        im = cv2.imread(imgs[names[i]], cv2.IMREAD_COLOR)
        mk = cv2.imread(msks[names[i]], cv2.IMREAD_GRAYSCALE)
        X[i] = common.resize_image(im)
        Y[i] = common.resize_mask(mk)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(load, range(n)))
    X.flush()
    Y.flush()
    with open(os.path.join(common.CACHE, f"{tag}_names.txt"), "w") as f:
        f.write("\n".join(names))
    return n


if __name__ == "__main__":
    os.makedirs(common.CACHE, exist_ok=True)
    ntr = build("TR", "tr")
    nte = build("TE", "te")
    rng = np.random.default_rng(0)
    perm = rng.permutation(ntr)
    nval = int(round(0.1 * ntr))
    np.savez(os.path.join(common.CACHE, "split.npz"), val=np.sort(perm[:nval]), train=np.sort(perm[nval:]))
    print(f"DUTS-TR {ntr} -> train {ntr - nval} / val {nval};  DUTS-TE {nte}")
