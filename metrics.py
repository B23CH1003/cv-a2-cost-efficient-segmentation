"""mIoU and Boundary-F1, written from scratch (batched with torch tensors for speed)."""
import torch
import torch.nn.functional as F


def confusion(pred, gt):
    """2x2 confusion matrix [[TN, FP], [FN, TP]] for uint8 {0,1} arrays of any shape."""
    p = pred.reshape(-1).astype("int64")
    g = gt.reshape(-1).astype("int64")
    idx = g * 2 + p
    import numpy as np
    return np.bincount(idx, minlength=4).reshape(2, 2)


def miou_from_confusion(cm):
    """Mean of background IoU and object IoU from one confusion matrix summed over a set."""
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    iou_obj = tp / max(tp + fp + fn, 1)
    iou_bg = tn / max(tn + fp + fn, 1)
    return 0.5 * (iou_obj + iou_bg)


def _boundary(m):
    """m: (B,H,W) float {0,1}. Object pixels with a background 4-neighbour (outside = background)."""
    p = F.pad(m, (1, 1, 1, 1), value=0.0)
    inner = p[:, :-2, 1:-1] * p[:, 2:, 1:-1] * p[:, 1:-1, :-2] * p[:, 1:-1, 2:]
    return m * (1.0 - inner)


def boundary_f1(pred, gt):
    """Per-image BF1 (5x5 tolerance window) for a batch. pred, gt: uint8/bool arrays (B,H,W).

    The 5x5 'is there a boundary pixel in my window' test is a 5x5 max-pool (stride 1,
    padding 2) of the other mask's boundary map, so no Python loop over pixels is needed.
    Returns a float tensor of shape (B,).
    """
    p = torch.as_tensor(pred).float()
    g = torch.as_tensor(gt).float()
    bp, bg = _boundary(p), _boundary(g)
    near_g = F.max_pool2d(bg[:, None], 5, 1, 2)[:, 0]
    near_p = F.max_pool2d(bp[:, None], 5, 1, 2)[:, 0]
    n_p = bp.sum((1, 2))
    n_g = bg.sum((1, 2))
    correct = (bp * near_g).sum((1, 2))   # predicted boundary pixels that have a true one nearby
    found = (bg * near_p).sum((1, 2))     # true boundary pixels that have a predicted one nearby
    prec = correct / n_p.clamp(min=1)
    rec = found / n_g.clamp(min=1)
    f1 = torch.where(prec + rec > 0, 2 * prec * rec / (prec + rec).clamp(min=1e-12), torch.zeros_like(prec))
    both_empty = (n_p == 0) & (n_g == 0)
    one_empty = (n_p == 0) != (n_g == 0)
    f1 = torch.where(both_empty, torch.ones_like(f1), f1)
    f1 = torch.where(one_empty, torch.zeros_like(f1), f1)
    return f1


def quality(miou, bf1):
    return 0.70 * miou + 0.30 * bf1


def ces(q, t_cpu_ms, s_disk_bytes):
    if t_cpu_ms > 50.0:
        return 0.0
    return q * (1 - 0.20 * t_cpu_ms / 50.0 - 0.10 * min(1.0, s_disk_bytes / 4194304))


def score(preds, gts, chunk=256):
    """mIoU (one confusion matrix over the set), mean per-image BF1, and Q for stacked masks."""
    import numpy as np
    cm = np.zeros((2, 2), dtype=np.int64)
    bf = []
    for s in range(0, len(preds), chunk):
        p, g = preds[s:s + chunk], gts[s:s + chunk]
        cm += confusion(p, g)
        bf.append(boundary_f1(p, g))
    miou = miou_from_confusion(cm)
    bf1 = float(torch.cat(bf).mean())
    return float(miou), bf1, quality(float(miou), bf1)
