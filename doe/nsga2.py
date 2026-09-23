"""NSGA-II (Deb et al., 2002) and NSGA-III (Deb & Jain, 2014) for multi-objective optimization, with
constraint-domination.

All objectives are minimized (maximize = minimize the negative value). Continuous variables use SBX + polynomial
mutation; integer variables (categorical factors) use rounding + random-reset mutation.

NSGA-III replaces the crowding distance with reference points (Das-Dennis) so the solutions stay evenly spread
for 3 or more objectives, when almost all solutions are mutually non-dominated and the crowding distance can no
longer tell them apart.

Convergence is monitored with the hypervolume (area/volume of the objective space dominated by the front, after
normalization). Automatic stop: when the mean hypervolume of the last `stop_window` generations rises by less than
`stop_tol` (relative, or less than its noise level) compared with the previous `stop_window` generations and the
ideal point moves by less than 1% of the objective range, and this holds for `stop_window` consecutive generations,
the search is considered converged.
"""
from math import comb

import numpy as np

ALGORITHMS = {"auto": "Automatic (NSGA-III for 3 or more objectives)", "nsga2": "NSGA-II",
              "nsga3": "NSGA-III (reference points)"}
HV_SAMPLES = 8000        # Monte Carlo hypervolume samples for 3 or more objectives (per-generation monitoring)


def _dominates(Fa, Fb):
    return np.all(Fa <= Fb, axis=-1) & np.any(Fa < Fb, axis=-1)


def non_dominated_sort(F, CV):
    """Non-dominated sorting with constraint-domination. Returns the rank of each individual (0 = best front)."""
    N = len(F)
    feas = CV <= 0
    dom = np.zeros((N, N), bool)                      # dom[i, j] = i dominates j
    both = feas[:, None] & feas[None, :]
    dom |= both & _dominates(F[:, None, :], F[None, :, :])
    dom |= feas[:, None] & ~feas[None, :]
    dom |= (~feas[:, None] & ~feas[None, :]) & (CV[:, None] < CV[None, :])
    n_dom = dom.sum(axis=0)
    rank = np.full(N, -1)
    current = np.where(n_dom == 0)[0]
    r = 0
    while len(current):
        rank[current] = r
        n_dom = n_dom - dom[current].sum(axis=0)
        n_dom[rank >= 0] = -1
        current = np.where(n_dom == 0)[0]
        r += 1
    return rank


def crowding(F, rank):
    N, m = F.shape
    d = np.zeros(N)
    for r in np.unique(rank):
        idx = np.where(rank == r)[0]
        if len(idx) <= 2:
            d[idx] = np.inf
            continue
        for k in range(m):
            o = idx[np.argsort(F[idx, k])]
            span = F[o[-1], k] - F[o[0], k]
            d[o[0]] = d[o[-1]] = np.inf
            if span > 0:
                d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / span
    return d


def _tournament(rank, crowd, rng, n):
    a, b = rng.integers(0, len(rank), n), rng.integers(0, len(rank), n)
    better = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
    return np.where(better, a, b)


def _sbx(p1, p2, lo, hi, eta, pc, rng):
    c1, c2 = p1.copy(), p2.copy()
    do = rng.random(len(p1)) < pc
    for i in np.where(do)[0]:
        for j in range(p1.shape[1]):
            if rng.random() > 0.5 or abs(p1[i, j] - p2[i, j]) < 1e-14 or hi[j] == lo[j]:
                continue
            x1, x2 = sorted((p1[i, j], p2[i, j]))
            u = rng.random()
            beta = 1 + 2 * min(x1 - lo[j], hi[j] - x2) / (x2 - x1)
            alpha = 2 - beta ** (-(eta + 1))
            bq = (u * alpha) ** (1 / (eta + 1)) if u <= 1 / alpha else (1 / (2 - u * alpha)) ** (1 / (eta + 1))
            y1 = 0.5 * ((x1 + x2) - bq * (x2 - x1))
            y2 = 0.5 * ((x1 + x2) + bq * (x2 - x1))
            y1, y2 = np.clip([y1, y2], lo[j], hi[j])
            if rng.random() < 0.5:
                y1, y2 = y2, y1
            c1[i, j], c2[i, j] = y1, y2
    return c1, c2


def _mutate(X, lo, hi, is_int, eta, pm, rng):
    X = X.copy()
    mask = rng.random(X.shape) < pm
    for i, j in zip(*np.where(mask)):
        if hi[j] == lo[j]:
            continue
        if is_int[j]:
            X[i, j] = rng.integers(int(lo[j]), int(hi[j]) + 1)
            continue
        x = X[i, j]
        d1, d2 = (x - lo[j]) / (hi[j] - lo[j]), (hi[j] - x) / (hi[j] - lo[j])
        u = rng.random()
        if u < 0.5:
            dq = (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** (1 / (eta + 1)) - 1
        else:
            dq = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)) ** (1 / (eta + 1))
        X[i, j] = np.clip(x + dq * (hi[j] - lo[j]), lo[j], hi[j])
    return X


# ---------------------------------------------------------------- hypervolume
def _hv_samples(m, n=HV_SAMPLES, ref=1.1):
    return np.random.default_rng(20240607).random((n, m)) * ref


def hypervolume(F, lo, hi, ref=1.1, samples=None):
    """Hypervolume of front F (minimization) after normalization (F - lo) / (hi - lo); reference point = ref on every
    axis. Exact for 1-2 objectives; Monte Carlo (fixed samples, so comparisons between fronts are fair) for >= 3."""
    F = np.asarray(F, float)
    if not len(F):
        return 0.0
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    Z = np.maximum((F - lo) / span, 0.0)
    Z = Z[np.all(Z < ref, axis=1)]
    m = F.shape[1]
    if not len(Z):
        return 0.0
    if m == 1:
        return float(ref - Z.min())
    if m == 2:
        Z = Z[np.lexsort((Z[:, 1], Z[:, 0]))]
        hv, y_best = 0.0, ref
        for x, y in Z:
            if y < y_best:
                hv += (ref - x) * (y_best - y)
                y_best = y
        return float(hv)
    S = _hv_samples(m, ref=ref) if samples is None else samples
    dom = 0
    for c in range(0, len(S), 2000):
        blk = S[c:c + 2000]
        dom += int(np.any(np.all(Z[None, :, :] <= blk[:, None, :], axis=2), axis=1).sum())
    return float(ref ** m * dom / len(S))


def hv_max(m, ref=1.1):
    """Theoretical maximum hypervolume (front reaches the ideal point in all objectives), for a 0-100% scale."""
    return ref ** m


# ---------------------------------------------------------------- NSGA-III
def reference_directions(m, n_max):
    """Das-Dennis reference points on the simplex: the largest number of divisions H with <= n_max points."""
    H = 1
    while comb(H + 1 + m - 1, m - 1) <= n_max:
        H += 1
    out = []

    def rec(prefix, left, k):
        if k == 1:
            out.append(prefix + [left])
            return
        for v in range(left + 1):
            rec(prefix + [v], left - v, k - 1)

    rec([], H, m)
    return np.array(out, float) / H


def _nsga3_select(F, CV, rank, n_sel, refs, rng):
    """NSGA-III environmental selection: fill front by front; the last front that does not fit is selected by niching
    on the reference points (the least represented reference points are filled first)."""
    chosen = []
    last = None
    for r in np.unique(rank):
        fr = np.where(rank == r)[0]
        if len(chosen) + len(fr) <= n_sel:
            chosen += fr.tolist()
            if len(chosen) == n_sel:
                break
        else:
            last = fr
            break
    if last is None:
        return np.array(chosen, int)
    K = n_sel - len(chosen)
    if CV[last[0]] > 0:                              # infeasible front: pick the smallest violations
        return np.array(chosen + last[np.argsort(CV[last], kind="stable")[:K]].tolist(), int)
    St = np.array(chosen + last.tolist(), int)
    Fs = F[St]
    ideal = Fs.min(axis=0)
    Fp = Fs - ideal
    m = F.shape[1]
    ext = np.empty((m, m))
    for i in range(m):
        w = np.full(m, 1e-6)
        w[i] = 1.0
        ext[i] = Fp[np.argmin(np.max(Fp / w, axis=1))]
    try:
        b = np.linalg.solve(ext, np.ones(m))
        a = 1.0 / b
        if not np.all(np.isfinite(a)) or np.any(a <= 1e-10):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        a = Fp.max(axis=0)
    a = np.where(a > 1e-10, a, 1.0)
    Fz = Fp / a
    U = refs / np.linalg.norm(refs, axis=1, keepdims=True)
    proj = Fz @ U.T
    d2 = np.maximum((Fz ** 2).sum(axis=1)[:, None] - proj ** 2, 0.0)
    near = np.argmin(d2, axis=1)
    dist = np.sqrt(d2[np.arange(len(St)), near])
    n_c = len(chosen)
    rho = np.bincount(near[:n_c], minlength=len(refs)).astype(float)
    avail = np.ones(len(St), bool)
    avail[:n_c] = False
    picks = []
    while len(picks) < K:
        has = np.bincount(near[avail], minlength=len(refs)) > 0
        cand = np.where(has)[0]
        rmin = rho[cand].min()
        j = rng.choice(cand[rho[cand] == rmin])
        mem = np.where(avail & (near == j))[0]
        k = mem[np.argmin(dist[mem])] if rho[j] == 0 else rng.choice(mem)
        picks.append(k)
        avail[k] = False
        rho[j] += 1
    return np.array(chosen + St[picks].tolist(), int)


def convergence_check(fronts, w, samples=None):
    """Compare the fronts of the last `w` generations with the `w` generations before them.

    - gain : relative increase of the mean hypervolume (joint normalization over both windows). Uses the window mean,
             not a single generation, because with many objectives the front points keep swapping, so the
             per-generation hypervolume fluctuates even when the front is no longer advancing.
    - noise: uncertainty (2 x standard error) of that difference in means; a gain below this is not real.
    - shift: movement of the ideal point (best value of each objective) relative to the front range.
    """
    new, old = fronts[-w:], fronts[-2 * w:-w]
    allF = [f for f in new + old if len(f)]
    if not allF:
        return {}
    U = np.vstack(allF)
    ideal, nadir = U.min(axis=0), U.max(axis=0)
    span = np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
    step = 1 if U.shape[1] <= 2 else 2
    hn = np.array([hypervolume(f, ideal, nadir, samples=samples) for f in new[::-1][::step]])
    ho = np.array([hypervolume(f, ideal, nadir, samples=samples) for f in old[::-1][::step]])
    mn, mo = hn.mean(), ho.mean()
    if mn <= 0:
        return {}
    se = np.sqrt(hn.var(ddof=1) / len(hn) + ho.var(ddof=1) / len(ho)) if len(hn) > 1 and len(ho) > 1 else 0.0
    i_new = np.vstack([f for f in new if len(f)] or [U]).min(axis=0)
    i_old = np.vstack([f for f in old if len(f)] or [U]).min(axis=0)
    return {"gain": float((mn - mo) / mn), "noise": float(2 * se / mn),
            "shift": float(np.max(np.abs(i_new - i_old) / span)), "conv": True}


def stop_tol_eff(h, tol):
    """Effective limit: the user tolerance, or the noise level if larger (at most 1%)."""
    return max(tol, min(h.get("noise", 0.0), 0.01))


def run(evaluate, lo, hi, is_int=None, repair=None, pop=100, gens=200, pc=0.9, eta_c=15, eta_m=20, pm=None,
        seed=0, initial=None, progress=None, algorithm="nsga2", auto_stop=False, stop_tol=1e-3, stop_window=20,
        min_gens=30, cancel=None):
    """evaluate(X) -> (F [N x m], CV [N]); repair(X) -> X. Returns a result dict (final population).

    algorithm: "nsga2" | "nsga3". auto_stop: stop before `gens` when converged (see the module docstring)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = len(lo)
    is_int = np.zeros(n, bool) if is_int is None else np.asarray(is_int, bool)
    pm = pm if pm is not None else 1.0 / max(n, 1)
    pop = int(pop) + int(pop) % 2

    X = lo + rng.random((pop, n)) * (hi - lo)
    if initial is not None and len(initial):
        k = min(len(initial), pop // 2)
        X[:k] = initial[:k]
    X[:, is_int] = np.rint(X[:, is_int])
    if repair:
        X = repair(X)
    F, CV = evaluate(X)
    m = F.shape[1]
    refs = reference_directions(m, pop) if algorithm == "nsga3" else None
    rank = non_dominated_sort(F, CV)
    crowd = crowding(F, rank) if refs is None else np.zeros(len(F))
    history = []
    samples = _hv_samples(m) if m >= 3 else None
    fronts = []                                    # feasible front of each generation (for the hypervolume curve)
    stop_gen, stop_reason = None, None
    ok_since = None
    for g in range(gens):
        if refs is not None:
            crowd = rng.random(len(F))             # NSGA-III: parents are paired at random within the same rank
        parents = _tournament(rank, crowd, rng, pop)
        p1, p2 = X[parents[0::2]], X[parents[1::2]]
        c1, c2 = _sbx(p1, p2, lo, hi, eta_c, pc, rng)
        C = _mutate(np.vstack([c1, c2]), lo, hi, is_int, eta_m, pm, rng)
        C[:, is_int] = np.rint(C[:, is_int])
        if repair:
            C = repair(C)
        FC, CVC = evaluate(C)
        XA, FA, CVA = np.vstack([X, C]), np.vstack([F, FC]), np.concatenate([CV, CVC])
        rA = non_dominated_sort(FA, CVA)
        if refs is None:
            cA = crowding(FA, rA)
            order = np.lexsort((-cA, rA))[:pop]
        else:
            order = _nsga3_select(FA, CVA, rA, pop, refs, rng)
        X, F, CV, rank = XA[order], FA[order], CVA[order], rA[order]
        rank = non_dominated_sort(F, CV)
        crowd = crowding(F, rank) if refs is None else np.zeros(len(F))
        front = (rank == 0) & (CV <= 0)
        fronts.append(F[front].copy())
        h = {"gen": g + 1, "n_front": int(front.sum()), "n_feasible": int((CV <= 0).sum()),
             "best": F[CV <= 0].min(axis=0).tolist() if (CV <= 0).any() else None, "gain": np.nan,
             "shift": np.nan, "noise": np.nan, "conv": None}
        # convergence monitoring (see convergence_check)
        w = int(stop_window)
        every = 1 if m <= 2 else 5
        if g + 1 >= 2 * w and (g + 1) % every == 0:
            h.update(convergence_check(fronts, w, samples))
        history.append(h)
        if progress:
            progress(g + 1, gens)
        if h.get("conv") is not None:
            if h["gain"] < stop_tol_eff(h, stop_tol) and h["shift"] < 0.01:
                if ok_since is None:
                    ok_since = g + 1
            else:
                ok_since = None
        # stop only when the criterion holds continuously for a full window: with many objectives the front can
        # look stationary for a moment while still advancing slowly (especially NSGA-II)
        if auto_stop and g + 1 >= min_gens and ok_since is not None and g + 1 - ok_since >= w:
            stop_gen = g + 1
            stop_reason = (f"converged: for {w} consecutive generations the mean hypervolume of the last {w} "
                           f"generations changed less than the limit (last {100 * h['gain']:.3f}%, limit "
                           f"{100 * stop_tol_eff(h, stop_tol):.2f}%) and the ideal point moved less than 1%")
            break
        if cancel is not None and cancel():
            stop_gen, stop_reason = g + 1, "stopped by user"
            break
    front = np.where((rank == 0) & (CV <= 0))[0]
    # remove duplicates
    if len(front):
        _, u = np.unique(np.round(X[front], 8), axis=0, return_index=True)
        front = front[np.sort(u)]
    # hypervolume curve normalized by the final front (for the convergence graph)
    hv_curve = []
    if len(front) and fronts:
        Ff = F[front]
        ideal, nadir = Ff.min(axis=0), Ff.max(axis=0)
        step = 1 if m <= 2 else max(1, len(fronts) // 80)
        idx = list(range(0, len(fronts), step))
        if idx[-1] != len(fronts) - 1:
            idx.append(len(fronts) - 1)
        for i in idx:
            hv_curve.append((i + 1, hypervolume(fronts[i], ideal, nadir, samples=samples) / hv_max(m)))
    return {"X": X, "F": F, "CV": CV, "rank": rank, "front": front, "history": history, "hv_curve": hv_curve,
            "algorithm": "nsga3" if refs is not None else "nsga2", "n_refs": 0 if refs is None else len(refs),
            "gens_run": len(history), "stop_gen": stop_gen, "stop_reason": stop_reason,
            "stop_tol": stop_tol, "stop_window": int(stop_window)}


def topsis(F, weights=None):
    """TOPSIS for minimization objectives. Returns closeness scores (larger = better)."""
    F = np.asarray(F, float)
    w = np.ones(F.shape[1]) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    norm = np.sqrt((F ** 2).sum(axis=0))
    V = F / np.where(norm > 0, norm, 1) * w
    best, worst = V.min(axis=0), V.max(axis=0)
    dp = np.sqrt(((V - best) ** 2).sum(axis=1))
    dn = np.sqrt(((V - worst) ** 2).sum(axis=1))
    return dn / np.where(dp + dn > 0, dp + dn, 1)


def knee(F):
    """Point closest to the utopia point after min-max normalization (index)."""
    F = np.asarray(F, float)
    span = np.ptp(F, axis=0)
    Z = (F - F.min(axis=0)) / np.where(span > 0, span, 1)
    return int(np.argmin(np.sqrt((Z ** 2).sum(axis=1))))
