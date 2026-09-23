"""Logistic regression (binomial response: number of successes out of n trials) and Poisson regression (counts).

Estimated with IRLS (iteratively reweighted least squares); terms are tested with likelihood-ratio tests.
"""
import numpy as np
from scipy import stats
from scipy.special import expit, gammaln

from . import models

FAMILIES = {"normal": "Normal (continuous)", "binomial": "Binomial (proportion / logistic)",
            "poisson": "Poisson (counts)"}


def _irls(X, y, family, m, max_iter=100, tol=1e-10):
    n, p = X.shape
    beta = np.zeros(p)
    if family == "binomial":
        pbar = np.clip((y.sum() + 0.5) / (m * n + 1.0), 1e-6, 1 - 1e-6)
        eta = np.full(n, np.log(pbar / (1 - pbar)))
    else:
        eta = np.log(np.maximum(y, 0.5))
    for _ in range(max_iter):
        if family == "binomial":
            pr = np.clip(expit(eta), 1e-10, 1 - 1e-10)
            mu = m * pr
            w = m * pr * (1 - pr)
        else:
            mu = np.exp(np.clip(eta, -700, 700))
            w = mu
        w = np.maximum(w, 1e-12)
        z = eta + (y - mu) / w
        XtW = X.T * w
        beta_new = np.linalg.lstsq(XtW @ X, XtW @ z, rcond=None)[0]
        eta_new = X @ beta_new
        if np.max(np.abs(beta_new - beta)) < tol * (1 + np.max(np.abs(beta))):
            beta, eta = beta_new, eta_new
            break
        beta, eta = beta_new, eta_new
    return beta, eta


def _mu_w(eta, family, m):
    if family == "binomial":
        pr = np.clip(expit(eta), 1e-12, 1 - 1e-12)
        return m * pr, m * pr * (1 - pr)
    mu = np.exp(np.clip(eta, -700, 700))
    return mu, mu


def deviance(y, mu, family, m):
    with np.errstate(divide="ignore", invalid="ignore"):
        if family == "binomial":
            t1 = np.where(y > 0, y * np.log(y / mu), 0.0)
            t2 = np.where(m - y > 0, (m - y) * np.log((m - y) / (m - mu)), 0.0)
            return 2 * float(np.sum(t1 + t2))
        t1 = np.where(y > 0, y * np.log(y / mu), 0.0)
        return 2 * float(np.sum(t1 - (y - mu)))


def loglik(y, mu, family, m):
    if family == "binomial":
        pr = np.clip(mu / m, 1e-12, 1 - 1e-12)
        return float(np.sum(gammaln(m + 1) - gammaln(y + 1) - gammaln(m - y + 1)
                            + y * np.log(pr) + (m - y) * np.log(1 - pr)))
    mu = np.maximum(mu, 1e-300)
    return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))


class GlmResult(models.FitResult):
    """GLM result with the same interface as FitResult (predict, intervals, etc.)."""

    def predict(self, coded, original=True):
        eta = self._rows(coded) @ self.beta
        if self.family == "binomial":
            return expit(eta)
        return np.exp(np.clip(eta, -700, 700))

    def predict_t(self, coded):
        return self._rows(coded) @ self.beta

    def t_crit(self, level=0.95):
        return stats.norm.ppf(0.5 + level / 2)

    def _inv(self, eta):
        return expit(eta) if self.family == "binomial" else np.exp(np.clip(eta, -700, 700))

    def confidence_band(self, coded, level=0.95):
        eta = self.predict_t(coded)
        w = self.t_crit(level) * np.sqrt(np.maximum(self.leverage_at(coded), 0))
        return self._inv(eta - w), self._inv(eta + w)

    def intervals(self, coded, n_obs=1, level=0.95):
        eta = self.predict_t(coded)
        se = np.sqrt(np.maximum(self.leverage_at(coded), 0))
        z = self.t_crit(level)
        nan = np.full(len(eta), np.nan)
        return {"pred": self._inv(eta), "pred_t": eta, "se_mean": se, "se_pred": nan, "t": z,
                "ci": (self._inv(eta - z * se), self._inv(eta + z * se)), "pi": (nan, nan)}


def fit_glm(coded, y, terms, family="binomial", trials=1, mixture=False, blocks=None, space=None):
    coded = np.asarray(coded, float)
    y_raw = np.asarray(y, float)
    mask = ~np.isnan(y_raw) & np.all(np.isfinite(coded), axis=1)
    coded, y = coded[mask], y_raw[mask]
    blk = None if blocks is None else np.asarray(blocks)[mask]
    n, k = len(y), coded.shape[1]
    m = float(trials or 1)
    if n < 2:
        raise ValueError("Not enough response data.")
    if family == "binomial":
        if np.any(y < 0) or np.any(y > m + 1e-9):
            raise ValueError(f"A binomial response must be a number of successes between 0 and {m:g} "
                             "(number of trials).")
    elif np.any(y < 0):
        raise ValueError("A Poisson response (counts) cannot be negative.")
    dc = models.design_columns(coded, terms, mixture, blk, space)
    X, p = dc.X, dc.X.shape[1]
    beta, eta = _irls(X, y, family, m)
    mu, w = _mu_w(eta, family, m)
    XtWX = (X.T * w) @ X
    cov = np.linalg.pinv(XtWX)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    dev = deviance(y, mu, family, m)
    ll = loglik(y, mu, family, m)

    # null model (intercept only; blocks included if present)
    X0 = np.column_stack([np.ones(n)] + ([X[:, dc.block_idx]] if dc.block_idx else []))
    b0, e0 = _irls(X0, y, family, m)
    mu0, _ = _mu_w(e0, family, m)
    dev0 = deviance(y, mu0, family, m)
    ll0 = loglik(y, mu0, family, m)

    # likelihood-ratio test for each term
    anova = []
    df_model = X.shape[1] - X0.shape[1]
    chi_model = max(dev0 - dev, 0.0)
    anova.append(models.AnovaRow("Model", chi_model, df_model, np.nan, chi_model,
                                 stats.chi2.sf(chi_model, df_model) if df_model > 0 else np.nan))
    for t in dc.fterms:
        if models.is_intercept(t) or (dc.use_intercept is False and models.degree(t) == 1):
            continue
        pos = dc.term_positions[t]
        Xr = np.delete(X, pos, axis=1)
        br, er = _irls(Xr, y, family, m)
        mur, _ = _mu_w(er, family, m)
        chi = max(deviance(y, mur, family, m) - dev, 0.0)
        anova.append(models.AnovaRow(models.term_name(t), chi, len(pos), np.nan, chi, stats.chi2.sf(chi, len(pos))))
    df_resid = n - p
    anova.append(models.AnovaRow("Residual (deviance)", dev, df_resid, np.nan))

    # residuals & diagnostics
    var_fn = (mu * (1 - mu / m)) if family == "binomial" else mu
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = (y - mu) / np.sqrt(np.maximum(var_fn, 1e-300))
        sign = np.sign(y - mu)
        if family == "binomial":
            d_i = 2 * (np.where(y > 0, y * np.log(y / mu), 0) +
                       np.where(m - y > 0, (m - y) * np.log((m - y) / (m - mu)), 0))
        else:
            d_i = 2 * (np.where(y > 0, y * np.log(y / mu), 0) - (y - mu))
        dres = sign * np.sqrt(np.maximum(d_i, 0))
        Wh = np.sqrt(w)[:, None] * X
        lev = np.einsum("ij,jk,ik->i", Wh, cov, Wh)
        std_dev_res = dres / np.sqrt(np.maximum(1 - lev, 1e-12))
        cooks = pearson ** 2 * lev / (p * np.maximum(1 - lev, 1e-12) ** 2)
        dffits = std_dev_res * np.sqrt(lev / np.maximum(1 - lev, 1e-12))

    st = {"n": n, "p": p, "deviance": dev, "null_deviance": dev0, "loglik": ll, "loglik0": ll0,
          "aic": -2 * ll + 2 * p, "bic": -2 * ll + np.log(n) * p,
          "mcfadden": 1 - ll / ll0 if ll0 != 0 else np.nan,
          "adj_mcfadden": 1 - (ll - p) / ll0 if ll0 != 0 else np.nan,
          "pearson_chi2": float(np.sum(pearson ** 2)), "mean": float(np.mean(y / m if family == "binomial" else y))}
    if df_resid > 0:
        st["deviance_p"] = float(stats.chi2.sf(dev, df_resid))
        st["pearson_p"] = float(stats.chi2.sf(st["pearson_chi2"], df_resid))
        st["dispersion"] = st["pearson_chi2"] / df_resid
    if family == "binomial":
        pr = mu / m
        succ, fail = y.sum(), (m - y).sum()
        if succ > 0 and fail > 0:
            st["tjur"] = float(np.sum(y * pr) / succ - np.sum((m - y) * pr) / fail)
        st["hosmer_lemeshow"] = hosmer_lemeshow(y, pr, m)

    res = GlmResult(terms=dc.fterms, aliased=dc.aliased, beta=beta, term_idx=dc.term_idx, block_idx=dc.block_idx,
                    exp_terms=dc.body, exp_idx=dc.exp_idx, col_terms=dc.col_terms, col_levels=dc.col_levels,
                    space=dc.space, se=se, vif=np.full(p, np.nan), n=n, p=p, df_resid=df_resid, sse=dev, sst=dev0,
                    mse=1.0, y=y, y_orig=(y / m if family == "binomial" else y), yhat=eta, resid=dres,
                    leverage=lev, stud_int=std_dev_res, stud_ext=std_dev_res, cooks=cooks, dffits=dffits,
                    xtx_inv=cov, mixture=mixture and not dc.use_intercept, transform=None, anova=anova, stats=st,
                    rows_used=np.where(mask)[0])
    res.family = family
    res.trials = m
    res.resid_var_for_poe = 0.0
    return res


def hosmer_lemeshow(y, pr, m, groups=10):
    """Hosmer-Lemeshow test (groups by predicted probability)."""
    order = np.argsort(pr)
    g = min(groups, max(2, len(y) // 2))
    chunks = np.array_split(order, g)
    chi = 0.0
    for c in chunks:
        obs = y[c].sum()
        exp = (m * pr[c]).sum()
        tot = m * len(c)
        pbar = exp / tot if tot else 0
        if 0 < pbar < 1:
            chi += (obs - exp) ** 2 / (tot * pbar * (1 - pbar))
    df = max(g - 2, 1)
    return {"chi2": chi, "df": df, "p": float(stats.chi2.sf(chi, df))}


def sequential_summary(order_fits):
    rows, prev = [], None
    for order, fit in order_fits:
        row = {"order": order, "fit": fit, "aliased": bool(fit.aliased), "seq_p": np.nan, "lof_p": np.nan}
        if prev is not None:
            ddf = fit.p - prev.p
            if ddf > 0:
                row["seq_p"] = float(stats.chi2.sf(max(prev.sse - fit.sse, 0), ddf))
        rows.append(row)
        prev = fit
    suggested = "mean"
    for r in rows[1:]:
        if r["aliased"]:
            break
        if not np.isnan(r["seq_p"]) and r["seq_p"] < 0.05:
            suggested = r["order"]
    return rows, suggested
