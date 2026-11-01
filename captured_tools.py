"""Helpers for the captured-photo test set.

Folder layout (same base name in all three):
    captured/original/item1.jpg      photo as taken
    captured/images_256/item1.png    resized photo (written by `resize`)
    captured/masks_256/item1.png     your mask, object 255 / background 0 (written by `annotate`)

    python captured_tools.py resize      # original -> images_256 (INTER_AREA, 256x256)
    python captured_tools.py annotate    # click polygons around the object on each resized photo

Annotation keys: left click = add vertex | Enter or right click = close polygon (object part)
u = undo vertex | z = remove last polygon | r = reset | s = save mask and go to next | q = quit
Several polygons may be drawn per photo (e.g. object split by an occluder). Everything inside a
polygon becomes white; the mask is saved at 256x256.
"""
import common
import glob
import os
import sys

import cv2
import numpy as np

CAP = os.path.join(common.ROOT, "captured")
ZOOM = 3


def resize_all():
    os.makedirs(os.path.join(CAP, "images_256"), exist_ok=True)
    exts = (".jpg", ".jpeg", ".png")
    files = sorted(os.path.join(CAP, "original", f) for f in os.listdir(os.path.join(CAP, "original"))
                   if f.lower().endswith(exts))
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)           # honours EXIF rotation
        if img is None:
            print("cannot read", f)
            continue
        base = os.path.splitext(os.path.basename(f))[0]
        cv2.imwrite(os.path.join(CAP, "images_256", base + ".png"), common.resize_image(img))
        print("resized", base, img.shape[:2], "->", (common.SIZE, common.SIZE))


def annotate():
    os.makedirs(os.path.join(CAP, "masks_256"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(CAP, "images_256", "*.png")))
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(CAP, "masks_256", base + ".png")
        if os.path.exists(out):
            continue
        img = cv2.imread(f)
        polys, cur = [], []

        def render():
            m = np.zeros((common.SIZE, common.SIZE), np.uint8)
            for p in polys:
                cv2.fillPoly(m, [np.array(p, np.int32)], 255)
            view = cv2.resize(img, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
            tint = view.copy()
            mz = cv2.resize(m, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST) > 0
            tint[mz] = (0.5 * tint[mz] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            for p in polys + [cur]:
                if len(p) > 1:
                    cv2.polylines(tint, [np.array(p, np.int32) * ZOOM], p is not cur, (0, 255, 0), 1)
            for q in cur:
                cv2.circle(tint, (q[0] * ZOOM, q[1] * ZOOM), 3, (255, 255, 0), -1)
            return m, tint

        def mouse(ev, x, y, flags, param):
            if ev == cv2.EVENT_LBUTTONDOWN:
                cur.append((x // ZOOM, y // ZOOM))
            elif ev == cv2.EVENT_RBUTTONDOWN and len(cur) >= 3:
                polys.append(list(cur))
                cur.clear()

        win = f"annotate {base}"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, mouse)
        while True:
            m, view = render()
            cv2.imshow(win, view)
            k = cv2.waitKey(30) & 0xFF
            if k in (13, 10) and len(cur) >= 3:
                polys.append(list(cur))
                cur.clear()
            elif k == ord("u") and cur:
                cur.pop()
            elif k == ord("z") and polys:
                polys.pop()
            elif k == ord("r"):
                polys.clear()
                cur.clear()
            elif k == ord("s"):
                if len(cur) >= 3:
                    polys.append(list(cur))
                m, _ = render()
                cv2.imwrite(out, m)
                print("saved", out, "object pixels", int((m > 0).sum()))
                break
            elif k == ord("q"):
                cv2.destroyAllWindows()
                return
        cv2.destroyWindow(win)


if __name__ == "__main__":
    {"resize": resize_all, "annotate": annotate}[sys.argv[1]]()
