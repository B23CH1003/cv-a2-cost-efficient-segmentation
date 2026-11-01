"""Check min_cut against brute force enumeration on small random graphs."""
import common  # noqa: F401
import itertools
import numpy as np
from maxflow import min_cut, cut_value

rng = np.random.default_rng(0)
bad = 0
for trial in range(300):
    n = int(rng.integers(2, 11))
    cs = rng.random(n) * 5
    ct = rng.random(n) * 5
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.4]
    eu = np.array([p[0] for p in pairs], dtype=np.int64)
    ev = np.array([p[1] for p in pairs], dtype=np.int64)
    ew = rng.random(len(pairs)) * 4
    obj = min_cut(cs, ct, eu, ev, ew)
    got = cut_value(cs, ct, eu, ev, ew, obj)
    best = min(cut_value(cs, ct, eu, ev, ew, np.array(l, dtype=bool))
               for l in itertools.product([0, 1], repeat=n))
    if abs(got - best) > 1e-6:
        bad += 1
        print("MISMATCH", trial, got, best)
print("trials: 300, mismatches:", bad)
