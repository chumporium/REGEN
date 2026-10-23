"""Parametric and sensitivity studies on fitted models (RSM or ANN).

- Parametric study: one factor is swept over a range while the others stay at a reference point; an optional
  second factor takes a few fixed values (family of curves). All responses are predicted.
- Global sensitivity: variance-based Sobol indices over the factor box. First-order index S1 with the estimator of
  Saltelli et al. (2010), total index ST with the estimator of Jansen (1999), confidence intervals by bootstrap.
- Local sensitivity: each factor is moved by ±delta of its range from the reference point (tornado chart).
- Pareto front: Spearman rank correlation of each factor with each objective along the front, and between the
  objectives (negative = conflicting objectives).

All work is done in coded units; factor values are reported in actual units by the caller.
"""
import numpy as np
from scipy import stats


def predictors(project, responses=None, source=None):
    """{j: function(coded) -> predicted response} for the responses that have a model."""
    out = {}
    for j in (range(len(project.responses)) if responses is None else responses):
        m = project._model_for(j, source) if source else project.model(j)
        if m is not None:
            out[j] = (lambda m: lambda z: np.asarray(m.predict(np.atleast_2d(z)), float).ravel())(m)
    return out


def variables(project):
    """Factors that can be varied independently: process factors and categoric factors. Mixture components sum to
    a constant and cannot be varied one at a time."""
    comp = set(project.comp_idx)
    return [i for i in range(project.k) if i not in comp]


def factor_range(project, i):
    lo, hi = project.coded_bounds()
    return float(lo[i]), float(hi[i])


# ------------------------------------------------------------------ parametric study
def parametric(project, base, xi, x_values=None, steps=41, zi=None, z_values=None, responses=None, source=None):
    """Sweep factor xi (coded values, default its whole range) with factor zi held at each of z_values.

    Returns {"x": coded x values, "z": list of coded z values (or [None]), "Y": {j: array (len(z), len(x))},
    "feasible": bool array (len(z), len(x))}. Points outside the design region (constraints) are NaN."""
    base = np.asarray(base, float)
    lo, hi = factor_range(project, xi)
    f = project.factors[xi]
    if x_values is None:
        x_values = np.arange(len(f.levels), dtype=float) if f.categoric else np.linspace(lo, hi, steps)
    x_values = np.asarray(x_values, float)
    zs = [None] if zi is None else list(np.asarray(z_values, float))
    pred = predictors(project, responses, source)
    Y = {j: np.full((len(zs), len(x_values)), np.nan) for j in pred}
    feas = np.zeros((len(zs), len(x_values)), bool)
    for r, z in enumerate(zs):
        pts = np.tile(base, (len(x_values), 1))
        pts[:, xi] = x_values
        if zi is not None:
            pts[:, zi] = z
        ok = project.feasible(pts)
        feas[r] = ok
        for j, fn in pred.items():
            y = fn(pts)
            Y[j][r] = np.where(ok, y, np.nan)
    return {"x": x_values, "z": zs, "Y": Y, "feasible": feas, "xi": xi, "zi": zi}


# ------------------------------------------------------------------ Sobol indices
def _sample(project, idx, n, rng_seed):
    """Two independent (n x d) samples of the factors idx, uniform over their coded ranges (Sobol sequence);
    categoric factors take each level with equal probability."""
    from scipy.stats import qmc
    d = len(idx)
    u = qmc.Sobol(2 * d, scramble=True, seed=rng_seed).random(n)
    A, B = u[:, :d].copy(), u[:, d:].copy()
    for c, i in enumerate(idx):
        lo, hi = factor_range(project, i)
        if project.factors[i].categoric:
            nlev = len(project.factors[i].levels)
            A[:, c] = np.minimum(np.floor(A[:, c] * nlev), nlev - 1)
            B[:, c] = np.minimum(np.floor(B[:, c] * nlev), nlev - 1)
        else:
            A[:, c] = lo + A[:, c] * (hi - lo)
            B[:, c] = lo + B[:, c] * (hi - lo)
    return A, B


def sobol_indices(fA, fB, fAB):
    """S1 (Saltelli et al. 2010) and ST (Jansen 1999) from f(A), f(B) and f(A with column i from B). The outputs are
    centered first, which leaves the estimates unchanged in expectation but lowers their variance."""
    f0 = np.mean(np.concatenate([fA, fB]))
    fA, fB, fAB = fA - f0, fB - f0, fAB - f0
    V = np.var(np.concatenate([fA, fB]))
    if not np.isfinite(V) or V <= 0:
        return np.zeros(fAB.shape[1]), np.zeros(fAB.shape[1]), 0.0
    S1 = np.mean(fB[:, None] * (fAB - fA[:, None]), axis=0) / V
    ST = 0.5 * np.mean((fA[:, None] - fAB) ** 2, axis=0) / V
    return S1, ST, V


def sobol(project, base, n=2048, responses=None, source=None, n_boot=200, seed=0, level=0.95):
    """Sobol indices of every response with respect to the independent factors (process and categoric).

    The factors vary uniformly over their full range; other factors (mixture components) stay at `base`. Linear
    constraints are not applied, because the indices assume independent inputs.
    Returns {"idx": factor indices, "n": n, "results": {j: {"S1", "ST", "S1_ci", "ST_ci", "V"}}}."""
    base = np.asarray(base, float)
    idx = variables(project)
    if not idx:
        raise ValueError("Sobol indices need at least one process or categoric factor. Mixture components sum to a "
                         "constant and are not independent inputs.")
    n = int(2 ** np.ceil(np.log2(max(n, 64))))
    A, B = _sample(project, idx, n, seed)
    d = len(idx)

    def full(M):
        P = np.tile(base, (len(M), 1))
        P[:, idx] = M
        return P

    AB = []
    for c in range(d):
        M = A.copy()
        M[:, c] = B[:, c]
        AB.append(full(M))
    pred = predictors(project, responses, source)
    rng = np.random.default_rng(seed + 1)
    boots = [rng.integers(0, n, n) for _ in range(n_boot)]
    a = (1 - level) / 2
    out = {}
    for j, fn in pred.items():
        fA, fB = fn(full(A)), fn(full(B))
        fAB = np.column_stack([fn(P) for P in AB])
        S1, ST, V = sobol_indices(fA, fB, fAB)
        b1, bT = [], []
        for r in boots:
            s1, st, _ = sobol_indices(fA[r], fB[r], fAB[r])
            b1.append(s1)
            bT.append(st)
        b1, bT = np.array(b1), np.array(bT)
        out[j] = {"S1": S1, "ST": ST, "V": V,
                  "S1_ci": np.quantile(b1, [a, 1 - a], axis=0).T, "ST_ci": np.quantile(bT, [a, 1 - a], axis=0).T}
    return {"idx": idx, "n": n, "evaluations": n * (d + 2), "results": out,
            "constraints_ignored": bool(project.constraints)}


# ------------------------------------------------------------------ local sensitivity (tornado)
def tornado(project, base, delta=0.1, responses=None, source=None):
    """Change of each response when each factor moves by ±delta of its range from `base` (clipped to the range).
    Categoric factors take every level (low = smallest, high = largest response).

    Returns {"idx", "delta", "y0": {j: value}, "results": {j: [{"i", "x_lo", "x_hi", "y_lo", "y_hi"}]}}."""
    base = np.asarray(base, float)
    idx = variables(project)
    pred = predictors(project, responses, source)
    y0 = {j: float(fn(base)[0]) for j, fn in pred.items()}
    res = {j: [] for j in pred}
    for i in idx:
        lo, hi = factor_range(project, i)
        if project.factors[i].categoric:
            levels = np.arange(len(project.factors[i].levels), dtype=float)
            pts = np.tile(base, (len(levels), 1))
            pts[:, i] = levels
            for j, fn in pred.items():
                y = fn(pts)
                res[j].append({"i": i, "x_lo": float(levels[np.argmin(y)]), "x_hi": float(levels[np.argmax(y)]),
                               "y_lo": float(np.min(y)), "y_hi": float(np.max(y)), "categoric": True})
            continue
        step = delta * (hi - lo)
        xl, xh = max(lo, base[i] - step), min(hi, base[i] + step)
        pts = np.tile(base, (2, 1))
        pts[:, i] = [xl, xh]
        ok = project.feasible(pts)
        for j, fn in pred.items():
            y = fn(pts)
            res[j].append({"i": i, "x_lo": float(xl), "x_hi": float(xh), "categoric": False,
                           "y_lo": float(y[0]) if ok[0] else np.nan, "y_hi": float(y[1]) if ok[1] else np.nan})
    return {"idx": idx, "delta": delta, "y0": y0, "results": res}


# ------------------------------------------------------------------ Pareto front
def front_analysis(project, result):
    """Spearman correlation of every factor with every objective along a Pareto front, and between objectives.

    result: NSGA result (keys X, Yobj, active). Returns {"factors": idx, "objectives": [response index],
    "rho": (n_factors, n_obj), "p": same, "rho_obj": (n_obj, n_obj), "p_obj": same, "n": number of solutions}."""
    X = np.asarray(result["X"], float)
    Yo = [np.asarray(y, float) for y in result["Yobj"]]
    objs = [o["index"] for o in result["active"]]
    idx = [i for i in range(project.k) if np.ptp(X[:, i]) > 1e-12]
    rho = np.full((len(idx), len(Yo)), np.nan)
    p = np.full_like(rho, np.nan)
    for a, i in enumerate(idx):
        for b, y in enumerate(Yo):
            if np.ptp(y) > 0:
                rho[a, b], p[a, b] = stats.spearmanr(X[:, i], y)
    ro = np.eye(len(Yo))
    po = np.zeros_like(ro)
    for a in range(len(Yo)):
        for b in range(a + 1, len(Yo)):
            r_, p_ = stats.spearmanr(Yo[a], Yo[b])
            ro[a, b] = ro[b, a] = r_
            po[a, b] = po[b, a] = p_
    return {"factors": idx, "objectives": objs, "rho": rho, "p": p, "rho_obj": ro, "p_obj": po, "n": len(X)}


def tradeoff_rates(f1, f2, groups=None):
    """Trade-off along a two-objective front: change of objective 2 per unit change of objective 1.

    The solutions are sorted by objective 1 and split into groups of neighbouring solutions (default: about six
    solutions per group, 2 to 12 groups); the rate is computed between the group means. Neighbouring solutions with
    almost equal objective values would otherwise give very noisy rates."""
    f1, f2 = np.asarray(f1, float), np.asarray(f2, float)
    o = np.argsort(f1)
    f1, f2 = f1[o], f2[o]
    k = groups or int(np.clip(len(f1) // 6, 2, 12))
    parts = [g for g in np.array_split(np.arange(len(f1)), min(k, len(f1))) if len(g)]
    m1 = np.array([f1[g].mean() for g in parts])
    m2 = np.array([f2[g].mean() for g in parts])
    d1 = np.diff(m1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(np.abs(d1) > 1e-12, np.diff(m2) / d1, np.nan)
    return (m1[:-1] + m1[1:]) / 2, rate


# ------------------------------------------------------------------ standardized regression coefficients
def _lhs(project, idx, n, seed):
    """Latin hypercube sample of the factors idx over their coded ranges (categoric: equal shares of the levels)."""
    from scipy.stats import qmc
    u = qmc.LatinHypercube(len(idx), seed=seed).random(n)
    M = np.empty_like(u)
    for c, i in enumerate(idx):
        lo, hi = factor_range(project, i)
        if project.factors[i].categoric:
            nlev = len(project.factors[i].levels)
            M[:, c] = np.minimum(np.floor(u[:, c] * nlev), nlev - 1)
        else:
            M[:, c] = lo + u[:, c] * (hi - lo)
    return M


def regression_significance(project, base, n=120, responses=None, source=None, seed=0):
    """Significance of the decision variables from a Latin hypercube sample of n points over the factor space.

    Every response is predicted with its model at the sample points and regressed linearly on the factors.
    Numeric factors: standardized coefficient beta = b * sd(x) / sd(y) with the t-test p-value. Two-level categoric
    factors: the same for the 0/1 indicator. Categoric factors with more levels: F-test of their indicator columns
    (no single beta). Points that violate the design constraints are replaced by new ones; other factors (mixture
    components) stay at `base`.

    Returns {"idx", "n", "results": {j: {"beta": [...], "p": [...], "r2": float}}}."""
    base = np.asarray(base, float)
    idx = variables(project)
    if not idx:
        raise ValueError("At least one process or categoric factor is needed.")
    rows, tries = [], 0
    while len(rows) < n and tries < 50:
        M = _lhs(project, idx, n, seed + 1000 * tries)
        P = np.tile(base, (n, 1))
        P[:, idx] = M
        rows.extend(P[project.feasible(P)])
        tries += 1
    if len(rows) < max(len(idx) + 3, 10):
        raise ValueError("Too few points of the sample satisfy the design constraints.")
    P = np.array(rows[:n])
    cols, groups = [], []
    for i in idx:
        f = project.factors[i]
        if f.categoric:
            lev = np.round(P[:, i]).astype(int)
            g = []
            for v in range(1, len(f.levels)):
                cols.append((lev == v).astype(float))
                g.append(len(cols) - 1)
            groups.append(g)
        else:
            cols.append(P[:, i])
            groups.append([len(cols) - 1])
    X = np.column_stack([np.ones(len(P))] + cols)
    df = len(P) - X.shape[1]
    out = {}
    for j, fn in predictors(project, responses, source).items():
        y = fn(P)
        sy = np.std(y, ddof=1)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b
        sse = float(res @ res)
        sst = float(np.sum((y - y.mean()) ** 2))
        mse = sse / df if df > 0 else np.nan
        cov = mse * np.linalg.pinv(X.T @ X)
        beta, pv = [], []
        for g in groups:
            c = [gg + 1 for gg in g]
            if len(c) == 1:
                sx = np.std(X[:, c[0]], ddof=1)
                se = np.sqrt(cov[c[0], c[0]])
                t = b[c[0]] / se if se > 0 else np.inf
                beta.append(float(b[c[0]] * sx / sy) if sy > 0 else 0.0)
                pv.append(float(2 * stats.t.sf(abs(t), df)) if df > 0 else np.nan)
            else:
                keep = [q for q in range(X.shape[1]) if q not in c]
                b0, *_ = np.linalg.lstsq(X[:, keep], y, rcond=None)
                r0 = y - X[:, keep] @ b0
                F = ((float(r0 @ r0) - sse) / len(c)) / mse if mse > 0 else np.inf
                beta.append(None)
                pv.append(float(stats.f.sf(F, len(c), df)) if df > 0 else np.nan)
        out[j] = {"beta": beta, "p": pv, "r2": 1 - sse / sst if sst > 0 else np.nan}
    return {"idx": idx, "n": len(P), "results": out}
