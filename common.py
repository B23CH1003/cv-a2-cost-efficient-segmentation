"""Shared setup: single-thread environment, paths, and the resizing rule from Section 1b.

Import this module FIRST in every script (before numpy / cv2 / torch) so that the
thread-limiting environment variables are in place before the libraries load.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "cache")
PARAMS = os.path.join(ROOT, "params.npz")
SIZE = 256


def resize_image(img):
    """Photo -> 256x256 with INTER_AREA (squashes non-square photos on purpose)."""
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def resize_mask(mask):
    """Mask -> 256x256 with INTER_NEAREST, then binarise at 128 -> uint8 {0,1}."""
    m = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return (m >= 128).astype(np.uint8)


def set_single_thread_torch():
    """Literal compliance with the timing rules (torch is only imported when asked for)."""
    import torch
    torch.set_num_threads(1)
