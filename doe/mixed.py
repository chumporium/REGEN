"""Split-plot analysis: mixed model y = Xβ + Zu + e, u ~ N(0, σ²_wp), e ~ N(0, σ²), fitted with REML.

Terms are tested with Wald F statistics. Denominator degrees of freedom use the containment method:
whole-plot terms (hard-to-change factors only) are tested against the whole-plot df, other terms against the subplot df.
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

from . import models


def _group_index(groups):
    labels, inv = np.unique(groups, return_inverse=True)
    return labels, inv


def _reml_parts(X, y, inv, n_groups, gamma):
    """Build V0 = I + γ ZZ' block by block; return (-2 profile REML loglik, beta, sigma2, Vinv, XtVinvX_inv)."""
    n, p = X.shape
    Vinv = np.zeros((n, n))
    logdet = 0.0
    for g in range(n_groups):
        idx = np.where(inv == g)[0]
        m = len(idx)
        # (I + γJ)^-1 = I - γ/(1+γm) J ;  det = 1 + γm
        Vg = np.eye(m) - (gamma / (1 + gamma * m)) * np.ones((m, m))
        Vinv[np.ix_(idx, idx)] = Vg
        logdet += np.log1p(gamma * m)
    XtV = X.T @ Vinv
    A = XtV @ X
    A_inv = np.linalg.pinv(A)
    beta = A_inv @ XtV @ y
    r = y - X @ beta
    q = float(r @ Vinv @ r)
    df = n - p
    sigma2 = q / df
    sign, logdetA = np.linalg.slogdet(A)
    m2ll = df * np.log(sigma2) + logdet + logdetA
    return m2ll, beta, sigma2, Vinv, A_inv


class MixedResult(models.FitResult):
    def intervals(self, coded, n_obs=1, level=0.95):
        z = self.predict_t(coded)
        h = self.leverage_at(coded)            # already on the σ² scale (cov β)
        t = self.t_crit(level)
        se_mean = np.sqrt(np.maximum(h, 0))
        se_pred = np.sqrt(np.maximum(h + self.sigma2_wp + self.sigma2 / max(n_obs, 1), 0))
        back = self._back
        return {"pred": models.inverse_transform(z, self.transform), "pred_t": z, "se_mean": se_mean,
                "se_pred": se_pred, "t": t, "ci": back(z - t * se_mean, z + t * se_mean),
                "pi": back(z - t * se_pred, z + t * se_pred)}

    def predict_se(self, coded):
        return np.sqrt(np.maximum(self.leverage_at(coded), 0))


def fit_reml(coded, y, terms, groups, htc, mixture=False, transform=None, space=None):
    """groups: whole-plot id per row; htc: indices of hard-to-change factors."""
    coded = np.asarray(coded, float)
    y_orig = np.asarray(y, float)
    mask = ~np.isnan(y_orig) & np.all(np.isfinite(coded), axis=1)
    coded, y_orig = coded[mask], y_orig[mask]
    y = models.apply_transform(y_orig, transform)
    grp = np.asarray(groups)[mask]
    labels, inv = _group_index(grp)
    n_wp = len(labels)
    n = len(y)
    dc = models.design_columns(coded, terms, mixture, None, space)
    X, p = dc.X, dc.X.shape[1]
    if n - p <= 0:
        raise ValueError("Residual degrees of freedom = 0. Remove model terms.")

    def obj(lg):
        return _reml_parts(X, y, inv, n_wp, np.exp(lg))[0]

    best = minimize_scalar(obj, bounds=(-12, 9), method="bounded")
    gamma = float(np.exp(best.x))
    if obj(-30) <= best.fun:  # lower bound: whole-plot component = 0
        gamma = 0.0
    m2ll, beta, sigma2, Vinv, A_inv = _reml_parts(X, y, inv, n_wp, gamma)
    cov = sigma2 * A_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    sigma2_wp = gamma * sigma2

    # containment df
    htc = set(htc)
    wp_terms = [t for t in dc.fterms if not models.is_intercept(t) and not models.is_delta(t)
                and all(i in htc for i, e in enumerate(t) if e)]
    n_wp_cols = (1 if dc.use_intercept else 0) + sum(len(dc.term_positions[t]) for t in wp_terms)
    df_wp = max(n_wp - n_wp_cols, 0)
    df_sp = max(n - n_wp - (p - n_wp_cols), 0)

    anova = []
    for t in dc.fterms:
        if models.is_intercept(t) or (mixture and not dc.use_intercept and models.degree(t) == 1):
            continue
        pos = dc.term_positions[t]
        b = beta[pos]
        C = cov[np.ix_(pos, pos)]
        F = float(b @ np.linalg.pinv(C) @ b) / len(pos)
        den = df_wp if t in wp_terms else df_sp
        if den <= 0:
            den = max(n - p, 1)
        anova.append(models.AnovaRow(models.term_name(t), np.nan, len(pos), float(den), F,
                                     float(stats.f.sf(F, len(pos), den))))

    # conditional residuals (with BLUPs of the whole-plot effects)
    r = y - X @ beta
    u = np.zeros(n_wp)
    for g in range(n_wp):
        idx = inv == g
        u[g] = gamma / (1 + gamma * idx.sum()) * r[idx].sum()
    cond = r - u[inv]
    yhat = X @ beta + u[inv]
    lev = np.einsum("ij,jk,ik->i", X, A_inv, X)
    with np.errstate(divide="ignore", invalid="ignore"):
        stud = cond / np.sqrt(sigma2 * np.maximum(1 - lev / (1 + gamma), 1e-9))
    ybar = y.mean()
    sst = float(((y - ybar) ** 2).sum())
    sse = float(cond @ cond)
    st = {"n": n, "p": p, "mean": ybar, "std_dev": np.sqrt(sigma2), "sigma2": sigma2, "sigma2_wp": sigma2_wp,
          "gamma": gamma, "n_wp": n_wp, "df_wp": df_wp, "df_sp": df_sp, "reml_m2ll": m2ll,
          "r2_cond": 1 - sse / sst if sst > 0 else np.nan}
    if ybar:
        st["cv"] = 100 * np.sqrt(sigma2) / abs(ybar)
    if 3 <= n:
        st["shapiro_p"] = float(stats.shapiro(cond).pvalue) if np.ptp(cond) > 0 else np.nan
    res = MixedResult(terms=dc.fterms, aliased=dc.aliased, beta=beta, term_idx=dc.term_idx, block_idx=[],
                      exp_terms=dc.body, exp_idx=dc.exp_idx, col_terms=dc.col_terms, col_levels=dc.col_levels,
                      space=dc.space, se=se, vif=np.full(p, np.nan), n=n, p=p, df_resid=max(df_sp, 1),
                      sse=sse, sst=sst, mse=1.0, y=y, y_orig=y_orig, yhat=yhat, resid=cond, leverage=lev,
                      stud_int=stud, stud_ext=stud, cooks=np.zeros(n), dffits=np.zeros(n), xtx_inv=cov,
                      mixture=mixture and not dc.use_intercept, transform=transform, anova=anova, stats=st,
                      rows_used=np.where(mask)[0])
    res.sigma2, res.sigma2_wp, res.wp_terms = sigma2, sigma2_wp, wp_terms
    res.resid_var_for_poe = sigma2 + sigma2_wp
    return res


def whole_plot_groups(coded, htc, max_size=None, seed=None):
    """Group runs with the same hard-to-change factor settings into whole plots.

    Groups larger than max_size are split (so there are >= 2 whole plots per setting).
    Returns (groups, run_order).
    """
    rng = np.random.default_rng(seed)
    coded = np.asarray(coded, float)
    keys = [tuple(np.round(row[list(htc)], 6)) for row in coded]
    uniq = {}
    for i, key in enumerate(keys):
        uniq.setdefault(key, []).append(i)
    plots = []
    for key, rows in uniq.items():
        # split in standard order (consecutive replicates) so each whole plot is balanced,
        # then randomize the run order within each whole plot
        size = max_size or max(1, int(np.ceil(len(rows) / 2)))
        for s in range(0, len(rows), size):
            plots.append(list(rng.permutation(rows[s:s + size])))
    order = rng.permutation(len(plots))
    groups = np.zeros(len(coded), int)
    run = np.zeros(len(coded), int)
    counter = 1
    for gi, pidx in enumerate(order, start=1):
        for r in plots[pidx]:
            groups[r] = gi
            run[r] = counter
            counter += 1
    return groups, run
