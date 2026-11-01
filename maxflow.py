"""Minimum s-t cut via Dinic's max-flow algorithm (written from scratch, pure Python).

The graph has one node per region plus an implicit source s and sink t:
    cs[p] : capacity s -> p   (background cost of region p)
    ct[p] : capacity p -> t   (object cost of region p)
    (eu[i], ev[i], ew[i]) : undirected neighbour edge, capacity ew[i] in both directions.

Speed trick: before the real max flow, every region p first sends min(cs[p], ct[p]) along
the path s -> p -> t.  This is a valid partial flow and afterwards every region keeps
residual capacity to only ONE terminal, which removes most of the work.
"""
import numpy as np

EPS = 1e-7


def min_cut(cs, ct, eu, ev, ew):
    """Return a boolean array: True for regions on the source (object) side of a minimum cut."""
    n = len(cs)
    direct = np.minimum(cs, ct)
    cs = (cs - direct).tolist()
    ct = (ct - direct).tolist()

    # arcs 2i and 2i+1 are the two directions of undirected edge i; each is the other's reverse
    m = len(eu)
    to = [0] * (2 * m)
    cap = [0.0] * (2 * m)
    adj = [[] for _ in range(n)]
    for i, (u, v, w) in enumerate(zip(eu.tolist(), ev.tolist(), ew.tolist())):
        a = 2 * i
        to[a] = v
        to[a + 1] = u
        cap[a] = w
        cap[a + 1] = w
        adj[u].append(a)
        adj[v].append(a + 1)

    sources = [p for p in range(n) if cs[p] > EPS]
    while True:
        # BFS levels from all nodes that still have residual capacity from the source
        level = [-1] * n
        for p in sources:
            level[p] = 0
        queue = list(sources)
        reach_sink = False
        for u in queue:
            if ct[u] > EPS:
                reach_sink = True
            lu = level[u] + 1
            for a in adj[u]:
                if cap[a] > EPS:
                    v = to[a]
                    if level[v] < 0:
                        level[v] = lu
                        queue.append(v)
        if not reach_sink:
            break

        # blocking flow with current-arc pointers
        ptr = [0] * n
        for s0 in sources:
            while cs[s0] > EPS:
                path = []
                u = s0
                found = False
                while True:
                    if ct[u] > EPS:
                        found = True
                        break
                    au = adj[u]
                    k = ptr[u]
                    nxt = -1
                    while k < len(au):
                        a = au[k]
                        if cap[a] > EPS and level[to[a]] == level[u] + 1:
                            nxt = a
                            break
                        k += 1
                    ptr[u] = k
                    if nxt >= 0:
                        path.append(nxt)
                        u = to[nxt]
                    else:
                        if not path:
                            break
                        a = path.pop()
                        u = to[a ^ 1]
                        ptr[u] += 1
                if not found:
                    break
                f = min(cs[s0], ct[u])
                for a in path:
                    if cap[a] < f:
                        f = cap[a]
                cs[s0] -= f
                ct[u] -= f
                for a in path:
                    cap[a] -= f
                    cap[a ^ 1] += f
        sources = [p for p in sources if cs[p] > EPS]
        if not sources:
            level = [-1] * n
            break

    # the last BFS found no path to t: nodes it reached are the source side of the min cut
    return np.array([l >= 0 for l in level], dtype=bool)


def cut_value(cs, ct, eu, ev, ew, obj):
    """Energy of a labelling (for testing): background cost + object cost + broken neighbour edges."""
    obj = np.asarray(obj, dtype=bool)
    # a labelling cuts s->p for background regions, p->t for object regions, and one
    # direction of every neighbour edge whose ends have different labels
    return cs[~obj].sum() + ct[obj].sum() + ew[obj[eu] != obj[ev]].sum()
