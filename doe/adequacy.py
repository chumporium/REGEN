"""Data adequacy assessment for ANN modeling.

Each check yields a status "ok" / "warn" / "bad" / "info" with a value and an explanation, which are then summarized
into a verdict and recommendations (algorithm, neuron limit, data split, transform).
"""
import numpy as np
from scipy import stats

from .ann import encode, n_weights_for


def _check(name, value, status, detail):
    return {"name": name, "value": value, "status": status, "detail": detail}


def max_neurons(n_in, n_train, ratio=1.0, depth=1):
    """Largest neuron count (same in every layer) that keeps the number of weights <= ratio x training rows."""
    h = 0
    while n_weights_for(n_in, [h + 1] * depth) <= ratio * n_train and h < 200:
        h += 1
    return h


def pure_error(coded, y):
    """(Pure error SD, df) from replicated points (rows with identical factors)."""
    from .models import pure_error as pe, replicate_groups
    ss, df = pe(coded, y, decimals=8)
    n_rep = int((np.bincount(replicate_groups(coded, decimals=8)) > 1).sum()) if len(y) else 0
    return (np.sqrt(ss / df) if df > 0 else np.nan), df, n_rep


def assess(project, j, split_train=0.85):
    p = project
    y_all = p.model_data[:, j]
    ok_rows = ~np.isnan(y_all) & np.all(np.isfinite(p.coded), axis=1)
    rows = np.where(ok_rows)[0]
    N = len(rows)
    n_missing = int(p.n - N)
    checks, rec = [], []
    res = {"checks": checks, "recommend": rec, "n": N, "rows": rows}
    if p.response_family(j) != "normal":
        checks.append(_check("Response type", p.response_family(j), "bad",
                             "ANN in this application is for continuous responses. Binomial/Poisson responses are "
                             "analyzed with logistic/Poisson regression on the Analysis page."))
        res["verdict"], res["level"] = "Not adequate - the response is not continuous", "bad"
        return res
    if N < 3:
        checks.append(_check("Complete data", N, "bad", "Fill in at least a few complete runs (factors + response)."))
        res["verdict"], res["level"] = "Not adequate - not enough data to assess", "bad"
        return res
    coded, y = p.coded[rows], y_all[rows]
    X, names = encode(coded, p.space)
    n_in = X.shape[1]
    n_train = int(np.floor(N * split_train))
    res.update(n_in=n_in, n_train=n_train, y=y)

    # 1. data size
    st = "ok" if N >= 30 else ("warn" if N >= 12 else "bad")
    checks.append(_check("Complete data", f"{N} runs" + (f" ({n_missing} empty)" if n_missing else ""), st,
                         "An ANN ideally needs ≥ 30 runs. 12–29 runs can still work with Bayesian Regularization "
                         "(trainbr) and cross-validation. < 12 runs: an ANN is hard to train and test honestly - use "
                         "RSM."))

    # 2. data / parameter ratio
    w2 = n_weights_for(n_in, [2])
    ratio = n_train / w2
    h1 = max_neurons(n_in, n_train, 1.0)
    h1br = max_neurons(n_in, n_train, 2.0)
    st = "ok" if ratio >= 3 else ("warn" if ratio >= 1 else "bad")
    checks.append(_check("Training rows / weights ratio", f"{ratio:.2f} ({n_in}-2-1 network, {w2} weights)", st,
                         f"With ±{n_train} training rows, the number of weights should be ≤ {n_train} "
                         f"(≤ {2 * n_train} with trainbr). Safe limit for 1 hidden layer: {h1} neurons (trainlm) or "
                         f"{h1br} neurons (trainbr). A ratio < 1 means even the smallest network has more parameters "
                         "than data."))
    res.update(h1=h1, h1br=h1br)

    # 3. number of levels of each numeric factor
    few = []
    lv_txt = []
    for i in p.num_idx:
        nl = len(np.unique(np.round(coded[:, i], 6)))
        lv_txt.append(f"{p.factor_label(i).split(':')[0]}={nl}")
        if nl < 3:
            few.append(p.factor_label(i))
    if p.num_idx:
        checks.append(_check("Factor levels", ", ".join(lv_txt), "warn" if few else "ok",
                             ("Factors with only 2 levels: " + ", ".join(few) + ". The ANN cannot learn curvature "
                              "in these factors; the result is only as reliable as a linear/interaction model. ")
                             if few else "All numeric factors have ≥ 3 levels, so nonlinear patterns can be learned."))

    # 4. categorical levels
    for i in p.cat_idx:
        counts = np.bincount(np.rint(coded[:, i]).astype(int), minlength=len(p.factors[i].levels))
        st = "ok" if counts.min() >= 5 else ("warn" if counts.min() >= 2 else "bad")
        checks.append(_check(f"Data per level of {p.factors[i].name}", ", ".join(
            f"{lv}={c}" for lv, c in zip(p.factors[i].levels, counts)), st,
            "Each categorical level (one-hot input) should have ≥ 5 runs so its effect can be learned."))

    # 5. noise: pure error from replicated points
    s_pe, df_pe, n_rep = pure_error(coded, y)
    sd_y = float(np.std(y, ddof=1))
    noise = [v for v in (s_pe,) if v is not None and np.isfinite(v) and v > 0]
    res.update(s_pe=s_pe, df_pe=df_pe, sd_y=sd_y, noise=max(noise) if noise else None)
    if noise:
        snr = sd_y / max(noise)
        st = "ok" if snr >= 4 else ("warn" if snr >= 2 else "bad")
        src = []
        if np.isfinite(s_pe):
            src.append(f"pure error SD = {s_pe:.4g} (df {df_pe}, {n_rep} replicated points)")
        checks.append(_check("Signal / noise ratio", f"{snr:.2f}", st,
                             "Response SD divided by the largest noise (" + "; ".join(src) + "). Ratio < 2: most of "
                             "the response variation is noise, and no model (including an ANN) will be accurate. "
                             f"An ANN RMSE below ≈ {max(noise):.4g} indicates overfitting (memorizing noise)."))
    else:
        checks.append(_check("Signal / noise ratio", "cannot be calculated", "info",
                             "There are no replicated points, so the noise level is unknown. Add a few replicates "
                             "(for example center points) so the noise limit and overfitting can be assessed."))

    # 6. response outliers (robust z based on median/MAD)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    rz = 0.6745 * (y - med) / mad if mad > 0 else np.zeros(N)
    out = np.where(np.abs(rz) > 3.5)[0]
    runs = p.run_order[rows]
    checks.append(_check("Response outliers (|robust z| > 3.5)", f"{len(out)} runs", "warn" if len(out) else "ok",
                         ("Run " + ", ".join(str(runs[i]) for i in out) + " is far from the other data. Check the "
                          "records; an ANN is very sensitive to outliers.") if len(out)
                         else "No response deviates extremely."))

    # 7. response distribution
    sk = float(stats.skew(y)) if N > 2 and sd_y > 0 else 0.0
    ratio_mm = (y.max() / y.min()) if y.min() > 0 else np.nan
    st = "warn" if abs(sk) > 1 or (np.isfinite(ratio_mm) and ratio_mm > 10) else "ok"
    checks.append(_check("Response distribution", f"skewness {sk:.2f}" + (f", max/min {ratio_mm:.1f}"
                                                                          if np.isfinite(ratio_mm) else ""), st,
                         "Highly skewed (|skewness| > 1) or max/min > 10: consider a ln transform (Analysis tab) "
                         "because errors at large values will dominate training."
                         if st == "warn" else "The response spread is reasonable."))

    # 8. correlation between factors
    if len(p.num_idx) >= 2 and not p.is_mixture:
        R = np.corrcoef(coded[:, p.num_idx].T)
        off = np.abs(R - np.eye(len(R)))
        rmax = float(np.nanmax(off))
        a, b = np.unravel_index(np.nanargmax(off), off.shape)
        st = "ok" if rmax < 0.5 else ("warn" if rmax < 0.9 else "bad")
        checks.append(_check("Factor correlation", f"max |r| {rmax:.2f} "
                             f"({p.factor_label(p.num_idx[a]).split(':')[0]}–{p.factor_label(p.num_idx[b]).split(':')[0]})",
                             st, "Correlated factors (common in historical data) make their individual effects hard "
                                 "to separate, and factor importance (Garson) cannot be trusted."))

    # 9. factor space coverage
    if p.num_idx and not p.is_mixture:
        from scipy.spatial import cKDTree
        U = (coded[:, p.num_idx] + 1) / 2
        rng = np.random.default_rng(0)
        R = rng.random((3000, len(p.num_idx)))
        d = cKDTree(U).query(R)[0]
        gap = float(d.max() / np.sqrt(len(p.num_idx)))
        spans = (coded[:, p.num_idx].max(axis=0) - coded[:, p.num_idx].min(axis=0)) / 2
        st = "ok" if gap <= 0.35 and spans.min() >= 0.9 else "warn"
        checks.append(_check("Factor space coverage", f"largest empty gap {gap:.2f} (relative to diagonal)", st,
                             "The largest distance from a point in the factor space to the nearest data. > 0.35 means "
                             "there are empty regions - ANN predictions there are unreliable extrapolations."
                             + (" Some factors do not cover their full range." if spans.min() < 0.9 else "")))

    # 10. is RSM already sufficient?
    fit = p.fit(j) if p.analysis_kind(j) == "ols" else None
    if fit is not None:
        pr2 = fit.stats.get("pred_r2", np.nan)
        lof = next((a.p for a in fit.anova if a.source == "Lack of Fit"), np.nan)
        res.update(rsm_pred_r2=pr2, rsm_lof_p=lof)
        if np.isfinite(lof) and lof < 0.05:
            st, txt = "ok", (f"RSM lack of fit is significant (p = {lof:.4f}): there is a pattern the polynomial "
                             "does not capture - an ANN may improve on it.")
        elif np.isfinite(pr2) and pr2 >= 0.95:
            st, txt = "info", (f"RSM is already very good (Predicted R² = {pr2:.4f}). An ANN is unlikely to give a "
                               "meaningful improvement; use it for comparison.")
        else:
            st, txt = "info", (f"RSM Predicted R² = {pr2:.4f}. Compare with the ANN cross-validation R² to judge the "
                               "benefit of the ANN.")
        checks.append(_check("Need for a nonlinear model", f"RSM Pred R² {pr2:.4f}" +
                             (f", LOF p {lof:.4f}" if np.isfinite(lof) else ""), st, txt))

    # 11. range differences (normalization)
    act = p.actual[rows][:, p.num_idx] if p.num_idx else np.zeros((N, 0))
    spans = [np.ptp(act[:, c]) for c in range(act.shape[1])] + [np.ptp(y)]
    spans = [v for v in spans if v > 0]
    if len(spans) > 1:
        rr = max(spans) / min(spans)
        checks.append(_check("Variable range differences", f"largest/smallest range ratio {rr:.3g}",
                             "info", "Factors and responses have different units & ranges. Normalization (default "
                                     "mapminmax [-1, 1]) is applied automatically before training so every variable "
                                     "carries equal weight; see the Data Normalization page."
                             + (" The ranges differ greatly (> 100×) - training would fail without normalization."
                                if rr > 100 else "")))

    # conclusion
    n_bad = sum(c["status"] == "bad" for c in checks)
    n_warn = sum(c["status"] == "warn" for c in checks)
    if n_bad:
        res["verdict"], res["level"] = "Not adequate for ANN (fix or add data first)", "bad"
    elif n_warn >= 3:
        res["verdict"], res["level"] = "Adequate with many caveats - ANN results need strict validation", "warn"
    elif n_warn:
        res["verdict"], res["level"] = "Adequate with caveats", "warn"
    else:
        res["verdict"], res["level"] = "Adequate for ANN modeling", "ok"

    rec.append("Algorithm: Bayesian Regularization (trainbr)" + (" - data < 50 runs, more resistant to overfitting."
                                                                   if N < 50 else " or Levenberg-Marquardt."))
    if N < 30:
        rec.append("Data split: 85% training / 15% test without validation (trainbr), and choose the architecture "
                   "with automatic architecture search (k-fold cross-validation) - more honest than a single random "
                   "split.")
    else:
        rec.append("Data split: 70/15/15 (trainlm) or 85/0/15 (trainbr).")
    h2 = max_neurons(n_in, n_train, 2.0, depth=2)
    rec.append(f"Size limit: 1 hidden layer ≤ {max(h1br, 1)} neurons; 2 hidden layers ≤ {h2} neurons per layer "
               "(trainbr, weights ≤ 2× training rows)." if h2 else
               f"1 hidden layer is enough (≤ {max(h1br, 1)} neurons); not enough data for 2 hidden layers.")
    if res.get("noise"):
        rec.append(f"Realistic RMSE target ≈ {res['noise']:.4g} (equal to the noise). Anything smaller = overfitting.")
    if any(c["name"] == "Response distribution" and c["status"] == "warn" for c in checks):
        rec.append("Consider a response transform (ln) on the Analysis tab.")
    if n_bad or N < 12:
        rec.append("Add runs (e.g. Augment Design on the Data page) before relying on the ANN.")
    return res
