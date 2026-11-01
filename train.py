"""Train the region probability model (Track A) and write params.npz.

What is learned from the training split (never from validation / DUTS-TE):
  * prior : fraction of training images in which each R x R position is object
  * llr   : log-ratio of object / background colour histograms (16x16x16 Lab bins)
  * w, b  : logistic regression over region features (+ squares), standardised by mu / sd
Because the prior and colour histogram are themselves learned from the training images, the
regression is trained on features computed from *cross-fitted* tables (the tables used for an
image never saw that image), otherwise the regression would over-trust those two features.

Usage: python train.py [--R 64 --G 10 --wpos 100 --iters 5]
"""
import common
import argparse
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

import seg

_SEG = None
_STAGE = 0
_X = None
_Y = None


def _init_worker(params, tag):
    global _SEG, _X, _Y, _STAGE
    _SEG = seg.Segmenter(params)
    _STAGE = int(params.get("feature_stage", 0))   # 0: region features, 2/3: auto-context features
    _X = np.load(os.path.join(common.CACHE, f"{tag}_img.npy"), mmap_mode="r")
    _Y = np.load(os.path.join(common.CACHE, f"{tag}_msk.npy"), mmap_mode="r")


def _extract(idx):
    """Features, soft targets (fraction of object pixels in the region) and pixel counts."""
    Fs, Ts, Cs, Is, F2s = [], [], [], [], []
    R = _SEG.R
    for i in idx:
        d = _SEG.prepare(np.ascontiguousarray(_X[i]))
        g = cv2.resize(_Y[i].astype(np.float32), (R, R), interpolation=cv2.INTER_AREA).ravel()
        n = d["n"]
        t = np.bincount(d["lab"], weights=g, minlength=n) / d["cnt"]
        Fs.append(d["F"].astype(np.float32))
        if _STAGE:
            z = _SEG.logit1(d["F"]) if _STAGE == 2 else _SEG.logit2(d)
            F2s.append(_SEG.stage2_features(d, z).astype(np.float32))
        Ts.append(t.astype(np.float32))
        Cs.append(d["cnt"].astype(np.float32))
        Is.append(np.full(n, i, dtype=np.int32))
    return (np.concatenate(F2s) if _STAGE else np.concatenate(Fs)), np.concatenate(Ts), np.concatenate(Cs), np.concatenate(Is)


def _small_stats(X, Y, idx, R):
    """Lab histogram index and soft ground truth of images `idx` at R x R."""
    for i in idx:
        small = cv2.resize(np.ascontiguousarray(X[i]), (R, R), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3) >> 4
        h = lab[:, 0].astype(np.int32) * 256 + lab[:, 1].astype(np.int32) * 16 + lab[:, 2]
        g = cv2.resize(Y[i].astype(np.float32), (R, R), interpolation=cv2.INTER_AREA)
        yield h, g


def learn_tables(X, Y, idx, R):
    """Position prior (R,R) and colour log-likelihood-ratio table (4096,) from images `idx`."""
    prior = np.zeros((R, R))
    ho = np.zeros(4096)
    hb = np.zeros(4096)
    for h, g in _small_stats(X, Y, idx, R):
        prior += g
        gf = g.ravel()
        ho += np.bincount(h, weights=gf, minlength=4096)
        hb += np.bincount(h, weights=1 - gf, minlength=4096)
    prior /= len(idx)
    llr = np.log((ho + 1.0) / (ho.sum() + 4096)) - np.log((hb + 1.0) / (hb.sum() + 4096))
    return prior.astype(np.float32), llr.astype(np.float32)


def base_params(cfg, prior, llr):
    p = dict(cfg)
    p.update(prior=prior, llr=llr, mu=np.zeros(seg.N_FEAT), sd=np.ones(seg.N_FEAT), w=np.zeros(2 * seg.N_FEAT), b=0.0)
    return p


def extract_all(params, tag, idx, workers):
    chunks = np.array_split(idx, workers * 8)
    with Pool(workers, initializer=_init_worker, initargs=(params, tag)) as pool:
        out = pool.map(_extract, chunks)
    return tuple(np.concatenate(z) for z in zip(*out))


def fit_logreg(F, T, C, l2=1e-4, iters=150):
    import torch
    torch.set_num_threads(8)
    mu = F.mean(0)
    sd = F.std(0) + 1e-6
    Z = torch.from_numpy((F - mu) / sd)
    Xt = torch.cat([Z, Z * Z], 1)
    Tt, Wt = torch.from_numpy(T.astype(np.float64)), torch.from_numpy(C.astype(np.float64))
    Wt = Wt / Wt.sum()
    w = torch.zeros(Xt.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xt @ w + b
        loss = (Wt * torch.nn.functional.binary_cross_entropy_with_logits(z, Tt, reduction="none")).sum() + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return mu, sd, w.detach().numpy().astype(np.float64), float(b.item())


def weighted_logloss(F, T, C, mu, sd, w, b):
    Z = (F - mu) / sd
    z = np.concatenate([Z, Z * Z], 1) @ w + b
    p = np.clip(1 / (1 + np.exp(-z)), 1e-3, 1 - 1e-3)
    ll = -(T * np.log(p) + (1 - T) * np.log(1 - p))
    acc = ((p > 0.5) == (T > 0.5))
    return (C * ll).sum() / C.sum(), (C * acc).sum() / C.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--G", type=int, default=10)
    ap.add_argument("--wpos", type=float, default=100.0)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--sigma_scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stage3", action="store_true", help="add a third auto-context logistic regression")
    ap.add_argument("--out", default=common.PARAMS)
    a = ap.parse_args()
    cfg = dict(R=a.R, G=a.G, wpos=a.wpos, iters=a.iters, lam=a.lam, sigma_scale=a.sigma_scale, upsample=1)

    X = np.load(os.path.join(common.CACHE, "tr_img.npy"), mmap_mode="r")
    Y = np.load(os.path.join(common.CACHE, "tr_msk.npy"), mmap_mode="r")
    sp = np.load(os.path.join(common.CACHE, "split.npz"))
    tr, va = sp["train"], sp["val"]
    print(f"train {len(tr)}  val {len(va)}")
    t0 = time.time()

    foldA, foldB = tr[0::2], tr[1::2]
    tabA = learn_tables(X, Y, foldA, a.R)
    tabB = learn_tables(X, Y, foldB, a.R)
    tabAll = learn_tables(X, Y, tr, a.R)
    print(f"tables learned ({time.time() - t0:.0f}s)")

    fa = extract_all(base_params(cfg, *tabB), "tr", foldA, a.workers)   # fold A scored with tables from B
    fb = extract_all(base_params(cfg, *tabA), "tr", foldB, a.workers)
    fv = extract_all(base_params(cfg, *tabAll), "tr", va, a.workers)
    print(f"features extracted ({time.time() - t0:.0f}s): train rows {len(fa[0]) + len(fb[0])}, val rows {len(fv[0])}")
    F = np.concatenate([fa[0], fb[0]]).astype(np.float64)
    T = np.concatenate([fa[1], fb[1]])
    C = np.concatenate([fa[2], fb[2]])

    mu, sd, w, b = fit_logreg(F, T, C)
    print("train  logloss %.4f  region-acc %.4f" % weighted_logloss(F, T, C, mu, sd, w, b))
    print("val    logloss %.4f  region-acc %.4f" % weighted_logloss(fv[0].astype(np.float64), fv[1], fv[2], mu, sd, w, b))
    base_rate = (fv[2] * fv[1]).sum() / fv[2].sum()
    print("val object pixel rate %.3f" % base_rate)

    # ---- stage 2 (auto-context): features built from the stage-1 probabilities
    st1 = dict(mu=mu, sd=sd, w=w, b=b)
    p2 = lambda tab: dict(base_params(cfg, *tab), **st1, feature_stage=2)
    ga = extract_all(p2(tabB), "tr", foldA, a.workers)
    gb = extract_all(p2(tabA), "tr", foldB, a.workers)
    gv = extract_all(p2(tabAll), "tr", va, a.workers)
    F2 = np.concatenate([ga[0], gb[0]]).astype(np.float64)
    mu2, sd2, w2, b2 = fit_logreg(F2, T, C)
    print("stage 2 train  logloss %.4f  region-acc %.4f" % weighted_logloss(F2, T, C, mu2, sd2, w2, b2))
    print("stage 2 val    logloss %.4f  region-acc %.4f" % weighted_logloss(gv[0].astype(np.float64), gv[1], gv[2], mu2, sd2, w2, b2))

    params = base_params(cfg, *tabAll)
    params.update(mu=mu, sd=sd, w=w, b=b, mu2=mu2, sd2=sd2, w2=w2, b2=b2)
    if a.stage3:                      # ---- stage 3: same features, built from the stage-2 probabilities
        st2 = dict(mu=mu, sd=sd, w=w, b=b, mu2=mu2, sd2=sd2, w2=w2, b2=b2)
        p3 = lambda tab: dict(base_params(cfg, *tab), **st2, feature_stage=3)
        ha = extract_all(p3(tabB), "tr", foldA, a.workers)
        hb = extract_all(p3(tabA), "tr", foldB, a.workers)
        hv = extract_all(p3(tabAll), "tr", va, a.workers)
        F3 = np.concatenate([ha[0], hb[0]]).astype(np.float64)
        mu3, sd3, w3, b3 = fit_logreg(F3, T, C)
        print("stage 3 train  logloss %.4f  region-acc %.4f" % weighted_logloss(F3, T, C, mu3, sd3, w3, b3))
        print("stage 3 val    logloss %.4f  region-acc %.4f" % weighted_logloss(hv[0].astype(np.float64), hv[1], hv[2], mu3, sd3, w3, b3))
        params.update(mu3=mu3, sd3=sd3, w3=w3, b3=b3)
    np.savez(a.out, **{k: np.asarray(v) for k, v in params.items()})
    print("saved", a.out, os.path.getsize(a.out), "bytes")


if __name__ == "__main__":
    main()
