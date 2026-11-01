"""Check the batched BF1 / mIoU against slow, literal pixel-loop implementations."""
import common  # noqa: F401
import numpy as np
import metrics


def naive_boundary(m):
    h, w = m.shape
    pts = set()
    for y in range(h):
        for x in range(w):
            if m[y, x] == 1:
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    yy, xx = y + dy, x + dx
                    out = yy < 0 or yy >= h or xx < 0 or xx >= w
                    if out or m[yy, xx] == 0:
                        pts.add((y, x))
                        break
    return pts


def naive_bf1(p, g):
    bp, bg = naive_boundary(p), naive_boundary(g)
    if not bp and not bg:
        return 1.0
    if not bp or not bg:
        return 0.0
    near = lambda pt, S: any((pt[0] + dy, pt[1] + dx) in S for dy in range(-2, 3) for dx in range(-2, 3))
    prec = sum(near(q, bg) for q in bp) / len(bp)
    rec = sum(near(q, bp) for q in bg) / len(bg)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


rng = np.random.default_rng(1)


def blob(h, w):
    m = (rng.random((h, w)) < 0.5).astype(np.uint8)
    for _ in range(3):  # smooth a bit so boundaries look like real masks
        pad = np.pad(m, 1)
        m = ((pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] + m) >= 3).astype(np.uint8)
    return m


P = np.stack([blob(24, 24) for _ in range(12)] + [np.zeros((24, 24), np.uint8)] * 2)
G = np.stack([blob(24, 24) for _ in range(12)] + [np.zeros((24, 24), np.uint8), blob(24, 24)])
fast = metrics.boundary_f1(P, G).numpy()
slow = np.array([naive_bf1(p, g) for p, g in zip(P, G)])
print("max |fast - naive| BF1:", np.abs(fast - slow).max())
assert np.abs(fast - slow).max() < 1e-5

cm = metrics.confusion(P, G)
print("confusion", cm.tolist(), "mIoU", metrics.miou_from_confusion(cm))
print("example CES from the handout (Q=.70, 20 ms, 0.5 MiB): %.4f" % metrics.ces(0.70, 20, 524288))
