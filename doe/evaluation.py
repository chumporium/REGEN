"""Design evaluation before the experiment: degrees of freedom, aliases, standard errors, VIF, power, leverage, FDS."""
import numpy as np
from scipy import stats

from . import models


def evaluate(coded, terms, mixture=False, blocks=None, space=None, delta=2.0, sigma=1.0, alpha=0.05,
             potential_terms=None):
    """delta/sigma = signal-to-noise ratio (response difference between the low & high level divided by σ)."""
    coded = np.asarray(coded, float)
    ok = np.all(np.isfinite(coded), axis=1)
    coded = coded[ok]
    blk = None if blocks is None else np.asarray(blocks)[ok]
    n = len(coded)
    dc = models.design_columns(coded, terms, mixture, blk, space)
    X, p = dc.X, dc.X.shape[1]
    df_resid = n - p
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv), 0))          # in units of σ
    vif = np.full(p, np.nan)
    for pos, ei in zip(dc.term_idx, dc.exp_idx):
        if ei >= 0 and not (mixture and not dc.use_intercept):
            cj = X[:, pos] - X[:, pos].mean()
            vif[pos] = xtx_inv[pos, pos] * float(cj @ cj)
    lev = np.einsum("ij,jk,ik->i", X, xtx_inv, X)

    # pure error & lack of fit
    g = models.replicate_groups(coded, blk)
    df_pe = int(n - len(np.unique(g))) if n else 0
    df_lof = df_resid - df_pe

    # power per coefficient (two-sided t test)
    power = np.full(p, np.nan)
    if df_resid > 0:
        tcrit = stats.t.ppf(1 - alpha / 2, df_resid)
        for pos, t in zip(dc.term_idx, dc.col_terms):
            if models.is_intercept(t):
                continue
            ncp = (delta / 2.0) * (1.0 / sigma) / se[pos] if se[pos] > 0 else 0
            power[pos] = 1 - stats.nct.cdf(tcrit, df_resid, ncp) + stats.nct.cdf(-tcrit, df_resid, ncp)

    # alias structure: potential terms not in the model
    alias = []
    if potential_terms:
        extra = [t for t in (models.normalize_term(x) for x in potential_terms)
                 if t not in dc.col_terms and not models.is_intercept(t)]
        if extra:
            X2, e_term, _ = models.expand(coded, extra, dc.space)
            A = xtx_inv @ X.T @ X2
            for j, (pos, t, lv) in enumerate(zip(dc.term_idx, dc.col_terms, dc.col_levels)):
                links = [(models.term_name(extra[e_term[c]]), float(A[pos, c])) for c in range(A.shape[1])
                         if abs(A[pos, c]) > 1e-6]
                if links:
                    alias.append((models.term_name(t), links))

    return {"n": n, "p": p, "df_resid": df_resid, "df_pe": df_pe, "df_lof": df_lof, "dc": dc, "se": se,
            "vif": vif, "power": power, "leverage": lev, "aliased": dc.aliased, "alias": alias,
            "xtx_inv": xtx_inv, "labels": _labels(dc)}


def _labels(dc):
    out = []
    for t, lv in zip(dc.col_terms, dc.col_levels):
        levels = {i: (dc.space.level_names[i][j] if dc.space.level_names[i] else j + 1) for i, j in lv.items()}
        out.append(models.term_name(t, None, levels))
    return out


def std_error_prediction(ev, coded):
    """Standard error of prediction (units of σ) for coded points."""
    dc = ev["dc"]
    E = models.expand(np.atleast_2d(coded), dc.body, dc.space)[0]
    R = np.zeros((E.shape[0], ev["p"]))
    for pos, ei in zip(dc.term_idx, dc.exp_idx):
        R[:, pos] = 1.0 if ei < 0 else E[:, ei]
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", R, ev["xtx_inv"], R), 0))


def fds_curve(ev, region_points):
    """Fraction of Design Space: curve (fraction, prediction SE) from random points in the design region."""
    se = np.sort(std_error_prediction(ev, region_points))
    frac = (np.arange(1, len(se) + 1)) / len(se)
    return frac, se


def correlations(ev):
    """(labels, coefficient correlations from (X'X)⁻¹, Pearson r between term columns excluding the intercept)."""
    dc = ev["dc"]
    pos = list(dc.term_idx)
    C = ev["xtx_inv"][np.ix_(pos, pos)]
    d = np.sqrt(np.maximum(np.diag(C), 1e-300))
    coef = C / np.outer(d, d)
    labels = [ev["labels"][k] for k in range(len(pos))]
    X = dc.X[:, pos]
    keep = [k for k in range(len(pos)) if np.ptp(X[:, k]) > 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.corrcoef(X[:, keep].T) if len(keep) > 1 else np.ones((len(keep), len(keep)))
    return labels, coef, [labels[k] for k in keep], np.atleast_2d(R)


def matrix_measures(ev, region_points=None):
    """Design matrix quality measures (for comparing designs / assessing historical data)."""
    X = ev["dc"].X
    n, p = X.shape
    XtX = X.T @ X
    eig = np.linalg.eigvalsh(XtX)
    out = {"n": n, "p": p}
    out["cond"] = float(eig.max() / eig.min()) if eig.min() > 1e-12 * eig.max() else np.inf
    sign, logdet = np.linalg.slogdet(XtX)
    out["det_inv"] = float(np.exp(-logdet)) if sign > 0 else np.inf
    out["trace_inv"] = float(np.trace(ev["xtx_inv"]))
    out["d_eff"] = float(100 * np.exp(logdet / p) / n) if sign > 0 else 0.0
    out["a_eff"] = float(100 * p / (n * out["trace_inv"])) if out["trace_inv"] > 0 else 0.0
    lev = ev["leverage"]
    out["lev_max"], out["lev_mean"] = float(lev.max()), float(lev.mean())
    if region_points is not None and len(region_points):
        v = std_error_prediction(ev, region_points) ** 2           # prediction variance (units of σ²)
        out["pv_max"], out["pv_mean"], out["pv_min"] = float(v.max()), float(v.mean()), float(v.min())
        out["g_eff"] = float(100 * p / (n * v.max())) if v.max() > 0 else 0.0
        out["i_scaled"] = float(n * v.mean())                        # scaled average prediction variance
    return out
