"""Track A pipeline: K-Means regions -> learned region probabilities -> graph cut.

Stages (everything between the 256x256 photo and the 256x256 mask is timed):
  1. downscale to R x R (INTER_AREA) and convert to Lab
  2. K-Means (own implementation, grid initialised) on [L, a, b, x, y] -> ~G*G regions
  3. per-region features (position prior, colour histogram ratio, border contact,
     colour contrasts, sharpness ...) -> logistic regression -> P_p, clipped to [0.001, 0.999]
  4. graph over regions, own Dinic max-flow -> minimum cut -> object / background per region
  5. paint region labels back to pixels and enlarge to 256 x 256
"""
import common  # noqa: F401  (must come first: single-thread environment)
import cv2
import numpy as np

from maxflow import min_cut

LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
N_FEAT = 24
N_FEAT2 = 13


def load_params(path):
    """Read the saved parameter file into a plain dict (0-d arrays become Python scalars)."""
    with np.load(path) as z:
        out = {}
        for k in z.files:
            v = z[k]
            out[k] = v.item() if v.ndim == 0 else v
    return out


class Segmenter:
    """Holds the saved parameters and runs the pipeline on one 256x256 BGR uint8 photo."""

    def __init__(self, params):
        self.p = params
        self.R = int(params["R"])
        self.G = int(params["G"])
        self.iters = int(params["iters"])
        self.wpos = float(params["wpos"])
        self.lam = float(params["lam"])
        self.sigma_scale = float(params["sigma_scale"])
        self.upsample = int(params.get("upsample", 0))
        R, G = self.R, self.G
        self.N = R * R
        self.K = G * G

        # fixed geometry, built once (not timed)
        ys, xs = np.mgrid[0:R, 0:R]
        self.pos = np.stack([(xs.ravel() + 0.5) / R, (ys.ravel() + 0.5) / R], 1).astype(np.float32)
        # K-Means buffers: Xa = [L a b wx wy 1] per pixel (the 1 folds |c|^2 into one matmul),
        # XT = the five feature columns as contiguous float64 rows for np.bincount
        self.Xa = np.ones((self.N, 6), dtype=np.float32)
        self.Xa[:, 3:5] = self.pos * self.wpos
        self.XT = np.empty((5, self.N), dtype=np.float64)
        self.XT[3:] = self.Xa[:, 3:5].T
        self.Ca = np.empty((6, self.K), dtype=np.float32)
        step = R / G
        gy, gx = np.mgrid[0:G, 0:G]
        cy = ((gy.ravel() + 0.5) * step).astype(int)
        cx = ((gx.ravel() + 0.5) * step).astype(int)
        self.init_idx = cy * R + cx

        # learned tables (training statistics)
        prior = np.clip(np.asarray(params["prior"], dtype=np.float64), 0.01, 0.99)
        self.prior_logit = np.log(prior / (1 - prior)).astype(np.float32).ravel()
        self.llr = np.asarray(params["llr"], dtype=np.float32)
        # learned logistic regression
        self.mu = np.asarray(params["mu"], dtype=np.float64)
        self.sd = np.asarray(params["sd"], dtype=np.float64)
        self.w = np.asarray(params["w"], dtype=np.float64)
        self.b = float(params["b"])
        self.has2 = "w2" in params
        if self.has2:
            self.mu2 = np.asarray(params["mu2"], dtype=np.float64)
            self.sd2 = np.asarray(params["sd2"], dtype=np.float64)
            self.w2 = np.asarray(params["w2"], dtype=np.float64)
            self.b2 = float(params["b2"])
        self.has3 = "w3" in params          # optional third auto-context stage
        if self.has3:
            self.mu3 = np.asarray(params["mu3"], dtype=np.float64)
            self.sd3 = np.asarray(params["sd3"], dtype=np.float64)
            self.w3 = np.asarray(params["w3"], dtype=np.float64)
            self.b3 = float(params["b3"])

    # ------------------------------------------------------------------ stage 1-2
    def _kmeans(self):
        """K-Means on [L, a, b, x, y]; centres start on a regular grid. Returns pixel labels,
        region pixel counts and the per-region feature sums of the final assignment."""
        K, Xa, XT, Ca = self.K, self.Xa, self.XT, self.Ca
        C = Xa[self.init_idx, :5].T.copy()                        # (5, K)
        for _ in range(self.iters):
            Ca[:5] = -2.0 * C
            Ca[5] = (C * C).sum(0)
            a = (Xa @ Ca).argmin(1)                               # nearest centre per pixel
            cnt = np.bincount(a, minlength=K)
            S = np.stack([np.bincount(a, weights=XT[c], minlength=K) for c in range(5)])
            nz = cnt > 0
            C[:, nz] = S[:, nz] / cnt[nz]                         # empty clusters keep their centre
        return a, cnt, S.T

    # ------------------------------------------------------------------ stage 3
    def prepare(self, img):
        """Cluster the photo and compute region features, adjacency and colours."""
        R, N = self.R, self.N
        small = cv2.resize(img, (R, R), interpolation=cv2.INTER_AREA)
        lab3 = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        labf = lab3.reshape(N, 3)
        self.Xa[:, :3] = labf
        self.XT[:3] = labf.T

        a, cnt, S = self._kmeans()
        ids = np.flatnonzero(cnt > 0)
        n = len(ids)
        remap = np.full(self.K, -1, dtype=np.int64)
        remap[ids] = np.arange(n)
        lab = remap[a]
        cnt = cnt[ids].astype(np.float64)
        means = S[ids] / cnt[:, None]
        M = means[:, :3]                                  # mean Lab colour of each region
        cxy = means[:, 3:] / self.wpos                    # centroid in [0,1]^2

        # sharpness (focus) cue from a Laplacian of the full-resolution grey image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.filter2D(gray, cv2.CV_32F, LAPLACIAN))
        sharp = cv2.resize(lap, (R, R), interpolation=cv2.INTER_AREA).ravel()

        # per-pixel quantities averaged inside each region
        q = labf >> 4
        hidx = q[:, 0].astype(np.int32) * 256 + q[:, 1].astype(np.int32) * 16 + q[:, 2]
        llr_pix = self.llr[hidx]
        ex = np.stack([np.bincount(lab, weights=v, minlength=n)
                       for v in (self.prior_logit, llr_pix, sharp)], 1) / cnt[:, None]

        # contact with each side of the image
        L2 = lab.reshape(R, R)
        sides = np.stack([np.bincount(L2[0], minlength=n), np.bincount(L2[-1], minlength=n),
                          np.bincount(L2[:, 0], minlength=n), np.bincount(L2[:, -1], minlength=n)],
                         1) / R
        touch = (sides.sum(1) > 0).astype(np.float64)

        # adjacency: 4-neighbour pixel pairs with different regions
        ea = np.concatenate([L2[:, :-1].ravel(), L2[:-1, :].ravel()])
        eb = np.concatenate([L2[:, 1:].ravel(), L2[1:, :].ravel()])
        diff = ea != eb
        ea, eb = ea[diff], eb[diff]
        key = np.minimum(ea, eb) * n + np.maximum(ea, eb)
        ck = np.bincount(key, minlength=n * n)
        nz = np.flatnonzero(ck)
        eu, ev, bl = nz // n, nz % n, ck[nz].astype(np.float64)

        # colour distances between regions and to the image border
        m2 = (M * M).sum(1)
        D = np.sqrt(np.maximum(m2[:, None] + m2[None, :] - 2.0 * (M @ M.T), 0.0))
        pxy2 = (cxy * cxy).sum(1)
        DP2 = np.maximum(pxy2[:, None] + pxy2[None, :] - 2.0 * (cxy @ cxy.T), 0.0)
        ring = np.concatenate([lab3[0], lab3[-1], lab3[1:-1, 0], lab3[1:-1, -1]]).astype(np.float64)
        d2r = np.maximum(m2[:, None] - 2.0 * (M @ ring.T) + (ring * ring).sum(1)[None, :], 0.0)
        dr = np.sqrt(d2r)
        dring_min = dr.min(1)
        dring_mean = dr.mean(1)
        dring_avg = np.sqrt(((M - ring.mean(0)) ** 2).sum(1))
        dglob = np.sqrt(((M - labf.mean(0)) ** 2).sum(1))

        c0 = (cnt[None, :] * D).sum(1) / max(N - 1.0, 1.0)
        W1 = cnt[None, :] * np.exp(-DP2 / 0.4)
        c1 = (W1 * D).sum(1) / np.maximum(W1.sum(1) - cnt, 1e-6)
        W2 = cnt[None, :] * np.exp(-DP2 / 0.05)
        c2 = (W2 * D).sum(1) / np.maximum(W2.sum(1) - cnt, 1e-6)
        dcol = D[eu, ev]
        cn = (np.bincount(eu, weights=bl * dcol, minlength=n) + np.bincount(ev, weights=bl * dcol, minlength=n))
        cn = cn / np.maximum(np.bincount(eu, weights=bl, minlength=n) + np.bincount(ev, weights=bl, minlength=n), 1.0)

        wsum = cnt.sum()

        def rel(v):  # value relative to the image-wide (size weighted) average
            return v / ((cnt * v).sum() / wsum + 1e-3)

        sh = ex[:, 2]
        F = np.empty((n, N_FEAT))
        F[:, 0] = ex[:, 0]                                  # position prior (logit)
        F[:, 1] = ex[:, 1]                                  # object/background colour ratio
        F[:, 2:6] = sides                                   # contact with top/bottom/left/right
        F[:, 6] = touch
        F[:, 7] = dring_min
        F[:, 8] = dring_mean
        F[:, 9] = dring_avg
        F[:, 10] = c0
        F[:, 11] = c1
        F[:, 12] = c2
        F[:, 13] = cn
        F[:, 14] = dglob
        F[:, 15] = M[:, 0]
        F[:, 16] = np.hypot(M[:, 1] - 128.0, M[:, 2] - 128.0)
        F[:, 17] = np.hypot(cxy[:, 0] - 0.5, cxy[:, 1] - 0.5)
        F[:, 18] = cxy[:, 1]
        F[:, 19] = np.log(cnt / N)
        F[:, 20] = np.log1p(sh)
        F[:, 21] = np.log((sh + 0.5) / ((cnt * sh).sum() / wsum + 0.5))
        F[:, 22] = rel(c1)
        F[:, 23] = rel(dring_mean)
        return {"lab": lab, "n": n, "cnt": cnt, "M": M, "F": F, "eu": eu, "ev": ev, "bl": bl,
                "D": D, "DP2": DP2, "cxy": cxy}

    def logit1(self, F):
        """Stage 1: logistic regression on standardised region features (plus their squares)."""
        Z = (F - self.mu) / self.sd
        return np.concatenate([Z, Z * Z], 1) @ self.w + self.b

    def stage2_features(self, d, z1):
        """Auto-context features from the stage-1 logits: image-specific object / background
        colour models, colour-similarity propagation and neighbour smoothing of z1."""
        n, cnt, M, D = d["n"], d["cnt"], d["M"], d["D"]
        z1 = np.clip(z1, -8.0, 8.0)
        p1 = 1.0 / (1.0 + np.exp(-z1))
        D2 = D * D
        wf, wb = cnt * p1, cnt * (1.0 - p1)
        sf, sb = wf.sum() + 1e-6, wb.sum() + 1e-6
        F2 = np.empty((n, N_FEAT2))
        F2[:, 0] = z1
        for j, s in ((1, 6.0), (2, 15.0)):                   # propagate z1 between similar colours
            A = np.exp(-D2 / (2 * s * s)) * cnt[None, :]
            F2[:, j] = (A @ z1) / A.sum(1)
        A = np.exp(-D2 / 200.0)                              # object / background colour density ratio
        F2[:, 3] = np.log(((A @ wf) / sf + 1e-3) / ((A @ wb) / sb + 1e-3))
        mf, mb = (wf @ M) / sf, (wb @ M) / sb               # mean object / background colour
        dfg = np.sqrt(((M - mf) ** 2).sum(1))
        dbg = np.sqrt(((M - mb) ** 2).sum(1))
        F2[:, 4] = np.log((dbg + 1.0) / (dfg + 1.0))
        eu, ev, bl = d["eu"], d["ev"], d["bl"]
        num = np.bincount(eu, weights=bl * z1[ev], minlength=n) + np.bincount(ev, weights=bl * z1[eu], minlength=n)
        den = np.bincount(eu, weights=bl, minlength=n) + np.bincount(ev, weights=bl, minlength=n)
        F2[:, 5] = np.where(den > 0, num / np.maximum(den, 1e-9), z1)   # neighbour average
        cxy = d["cxy"]
        Asp = np.exp(-d["DP2"] / 0.045 - D2 / 450.0) * cnt[None, :]       # nearby AND similar colour
        F2[:, 6] = (Asp @ z1) / Asp.sum(1)
        pbar = sf / cnt.sum()
        F2[:, 7] = np.log(pbar + 0.01)
        cf = (wf @ cxy) / sf
        F2[:, 8] = np.sqrt(((cxy - cf) ** 2).sum(1))          # distance to the object's centre of mass
        F2[:, 9] = z1 * np.log(pbar + 0.01)
        F2[:, 10] = F2[:, 3] * F2[:, 7]
        F2[:, 11] = np.log(np.sqrt((cnt * ((M - mf) ** 2).sum(1)).sum() / cnt.sum()) + 1.0)  # spread of colours
        F2[:, 12] = np.log(cnt / cnt.sum() + 1e-4)
        return F2

    def logit2(self, d):
        """Stage 2 logit (stage 1 if no second stage is stored)."""
        z = self.logit1(d["F"])
        if self.has2:
            Z = (self.stage2_features(d, z) - self.mu2) / self.sd2
            z = np.concatenate([Z, Z * Z], 1) @ self.w2 + self.b2
        return z

    def probability(self, d):
        """Region probability P_p, clipped to [0.001, 0.999]."""
        z = self.logit2(d)
        if self.has3:
            Z = (self.stage2_features(d, z) - self.mu3) / self.sd3
            z = np.concatenate([Z, Z * Z], 1) @ self.w3 + self.b3
        return np.clip(1.0 / (1.0 + np.exp(-z)), 0.001, 0.999)

    # ------------------------------------------------------------------ stage 4-5
    def decide(self, d, lam=None, sigma_scale=None):
        """Graph cut on the regions of `d`; returns a boolean object flag per region."""
        lam = self.lam if lam is None else lam
        sigma_scale = self.sigma_scale if sigma_scale is None else sigma_scale
        P = self.probability(d)
        cs = -np.log(1.0 - P) * d["cnt"]
        ct = -np.log(P) * d["cnt"]
        eu, ev = d["eu"], d["ev"]
        if len(eu) == 0:
            return cs > ct
        M = d["M"]
        dc2 = ((M[eu] - M[ev]) ** 2).sum(1)
        sig2 = dc2.mean() * sigma_scale + 1e-6
        ew = lam * np.exp(-dc2 / (2.0 * sig2)) * d["bl"]
        return min_cut(cs, ct, eu, ev, ew)

    def paint(self, d, obj):
        """Region decisions -> 256x256 uint8 mask of 0/1."""
        R = self.R
        m = obj[d["lab"]].reshape(R, R).astype(np.float32)
        if self.upsample == 0:
            return cv2.resize(m, (common.SIZE, common.SIZE), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        return (cv2.resize(m, (common.SIZE, common.SIZE), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)

    def segment(self, img):
        d = self.prepare(img)
        return self.paint(d, self.decide(d))
