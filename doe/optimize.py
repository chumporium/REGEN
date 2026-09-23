"""Numerical multi-response optimization with desirability functions (Derringer-Suich)."""
import numpy as np
from scipy.optimize import minimize

GOALS = {
    "none": "None",
    "maximize": "Maximize",
    "minimize": "Minimize",
    "target": "Target",
    "range": "In range",
}


def desirability(y, goal, low, high, target=None, w_low=1.0, w_high=1.0):
    y = np.asarray(y, float)
    d = np.ones_like(y)
    if goal == "maximize":
        d = np.clip((y - low) / max(high - low, 1e-12), 0, 1) ** w_low
    elif goal == "minimize":
        d = np.clip((high - y) / max(high - low, 1e-12), 0, 1) ** w_high
    elif goal == "target":
        t = target if target is not None else (low + high) / 2
        d = np.zeros_like(y)
        lo = (y >= low) & (y <= t)
        hi = (y > t) & (y <= high)
        d[lo] = ((y[lo] - low) / max(t - low, 1e-12)) ** w_low
        d[hi] = ((high - y[hi]) / max(high - t, 1e-12)) ** w_high
    elif goal == "range":
        d = ((y >= low) & (y <= high)).astype(float)
    return np.nan_to_num(d, nan=0.0)


def overall_desirability(coded, criteria):
    """criteria: list of dicts {fit, goal, low, high, target, importance}."""
    coded = np.atleast_2d(coded)
    log_sum = np.zeros(coded.shape[0])
    total_w = 0.0
    zero = np.zeros(coded.shape[0], bool)
    for c in criteria:
        if c["goal"] == "none":
            continue
        y = c["fit"].predict(coded)
        d = desirability(y, c["goal"], c["low"], c["high"], c.get("target"))
        r = float(c.get("importance", 3))
        zero |= d <= 0
        log_sum += r * np.log(np.maximum(d, 1e-300))
        total_w += r
    if total_w == 0:
        return np.ones(coded.shape[0])
    D = np.exp(log_sum / total_w)
    D[zero] = 0.0
    return D


def _sample(rng, lo, hi, n, mixture, mix_idx=None):
    k = len(lo)
    pts = lo + rng.random((n * 4 if mixture else n, k)) * (hi - lo)
    if not mixture:
        return pts
    # components: random points on the simplex (sum = 1) that satisfy the lower & upper limits
    idx = list(range(k)) if mix_idx is None else list(mix_idx)
    free = 1.0 - lo[idx].sum()
    pts[:, idx] = lo[idx] + free * rng.dirichlet(np.ones(len(idx)), size=len(pts))
    pts = pts[np.all(pts[:, idx] <= hi[idx] + 1e-12, axis=1)]
    return pts[:n]


def optimize(criteria, bounds, mixture=False, G=None, h=None, n_random=4000, n_starts=30, seed=0,
             mix_idx=None):
    """bounds: list of (low, high), coded / pseudo-components; G z <= h: linear constraints.

    Returns [(D, x)] sorted from the highest desirability.
    """
    if not [c for c in criteria if c["goal"] != "none"]:
        raise ValueError("Select at least one goal for a response.")
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    has_cons = G is not None and len(G) > 0
    idx = list(range(len(lo))) if mix_idx is None else list(mix_idx)
    if mixture and (lo[idx].sum() > 1 + 1e-9 or hi[idx].sum() < 1 - 1e-9):
        raise ValueError("The component limits do not allow the mixture to sum to the total.")

    def ok(z):
        return np.all(np.atleast_2d(z) @ G.T <= h + 1e-9, axis=1) if has_cons else np.ones(len(np.atleast_2d(z)), bool)

    samples = _sample(rng, lo, hi, n_random * (3 if has_cons else 1), mixture, idx)
    samples = samples[ok(samples)][:n_random]
    if len(samples) == 0:
        raise ValueError("No point satisfies all limits and constraints.")
    D = overall_desirability(samples, criteria)
    starts = samples[np.argsort(-D)[:n_starts]]

    def obj(x):
        x = np.clip(x, lo, hi)
        if has_cons and not ok(x)[0]:
            return 0.0
        return -overall_desirability(x[None, :], criteria)[0]

    cons = []
    if mixture:
        cons.append({"type": "eq", "fun": lambda x: x[idx].sum() - 1.0})
    if has_cons:
        cons.append({"type": "ineq", "fun": lambda x: h - G @ x})

    solutions = []
    for x0 in starts:
        if mixture or has_cons:
            res = minimize(obj, x0, method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                           options={"maxiter": 300, "ftol": 1e-10})
            x = np.clip(res.x, lo, hi)
            if mixture and abs(x[idx].sum() - 1) > 1e-9:
                x[idx] = x[idx] / x[idx].sum()
            if -obj(x) < -obj(x0) or (has_cons and not ok(x)[0]):
                x = x0
        else:
            res = minimize(obj, x0, method="Nelder-Mead", bounds=list(zip(lo, hi)),
                           options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 2000})
            x = np.clip(res.x, lo, hi)
        solutions.append((-obj(x), x))

    solutions.sort(key=lambda s: -s[0])
    unique = []
    for d, x in solutions:
        if all(np.linalg.norm(x - ux) > 0.03 for _, ux in unique):
            unique.append((d, x))
    return unique
