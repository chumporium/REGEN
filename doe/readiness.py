"""Data readiness assessment for modeling (RSM/ANN) and optimization, with recommendations:
which data needs to be completed (suggested additional runs) and which treatment is needed
(transforms, outlier checks, drift handling, etc.).

Each check: {"cat", "name", "value", "status" (ok/warn/bad/info), "detail", "rec": [recommendations]}.
Recommendation: {"kind": "complete" (complete the data) | "treatment", "text", "action": None | (kind, parameter)}.
"""
import numpy as np
from scipy import stats

from . import models

CATS = {"data": "Data Completeness & Cleanliness", "model": "Data Structure for Modeling",
        "resp": "Response & Model Quality", "opt": "Optimization Readiness"}
WEIGHT = {"ok": 1.0, "warn": 0.5, "bad": 0.0}


def _chk(cat, name, value, status, detail, rec=None):
    return {"cat": cat, "name": name, "value": value, "status": status, "detail": detail, "rec": rec or []}


def _rec(kind, text, action=None):
    return {"kind": kind, "text": text, "action": action}


# ------------------------------------------------------------------ suggested runs
def model_matrix(project, coded, terms):
    return models.term_matrix(np.asarray(coded, float), terms, project.space)


def suggest_d_optimal(project, terms, existing, n_new, n_cand=3000, seed=0):
    """Add runs one at a time that add the most model information (D criterion: largest leverage)."""
    cand = project.region_samples(n_cand, seed)
    if not len(cand) or n_new <= 0:
        return np.zeros((0, project.k))
    Xc = model_matrix(project, cand, terms)
    Xe = model_matrix(project, existing, terms) if len(existing) else np.zeros((0, Xc.shape[1]))
    M = Xe.T @ Xe + 1e-6 * np.eye(Xc.shape[1])
    Mi = np.linalg.pinv(M)
    chosen = []
    for _ in range(int(n_new)):
        lev = np.einsum("ij,jk,ik->i", Xc, Mi, Xc)
        if chosen:
            lev[chosen] = -np.inf
        i = int(np.argmax(lev))
        chosen.append(i)
        x = Xc[i]
        Mx = Mi @ x
        Mi = Mi - np.outer(Mx, Mx) / (1.0 + x @ Mx)          # Sherman-Morrison
    return cand[chosen]


def suggest_space_filling(project, existing, n_new, n_cand=3000, seed=1):
    """Runs that fill gaps in the factor space (maximin): each new run as far as possible from the existing data."""
    cand = project.region_samples(n_cand, seed)
    if not len(cand) or n_new <= 0:
        return np.zeros((0, project.k))
    num = [i for i in range(project.k) if not project.factors[i].categoric]
    lo, hi = project.coded_bounds()
    span = np.where(hi[num] > lo[num], hi[num] - lo[num], 1.0)

    def sc(Z):
        return (np.asarray(Z, float)[:, num] - lo[num]) / span

    C = sc(cand)
    E = sc(existing) if len(existing) else np.zeros((0, len(num)))
    if len(E):
        from scipy.spatial import cKDTree
        d = cKDTree(E).query(C)[0]
    else:
        d = np.full(len(C), np.inf)
    chosen = []
    for _ in range(int(n_new)):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.sqrt(((C - C[i]) ** 2).sum(axis=1)))
        d[i] = -np.inf
    return cand[chosen]


def suggest_near(project, point, n_new, radius=0.15, seed=2):
    """Runs around a point (e.g. the optimum): the point itself + random points within a (coded) radius."""
    point = np.asarray(point, float)
    cand = project.region_samples(4000, seed)
    if not len(cand):
        return np.repeat(point[None], n_new, axis=0)
    num = [i for i in range(project.k) if not project.factors[i].categoric]
    lo, hi = project.coded_bounds()
    span = np.where(hi[num] > lo[num], hi[num] - lo[num], 1.0)
    same_cat = np.all(cand[:, project.cat_idx] == point[project.cat_idx], axis=1) if project.cat_idx else \
        np.ones(len(cand), bool)
    d = np.sqrt((((cand[:, num] - point[num]) / span) ** 2).sum(axis=1))
    pool = cand[(d <= radius) & same_cat]
    out = [point]
    if len(pool):
        pick = np.random.default_rng(seed).choice(len(pool), min(n_new - 1, len(pool)), replace=False)
        out += list(pool[pick])
    while len(out) < n_new:
        out.append(point)
    return np.array(out[:n_new])


# ------------------------------------------------------------------ assessment
def _optimum(project, j, fit):
    """Optimum point of response j (from the last optimization result if any, otherwise searched per the goal)."""
    goal = (project.opt_criteria.get(j) or {}).get("goal", "maximize")
    if goal not in ("maximize", "minimize"):
        goal = "maximize"
    if project.solutions:
        return np.asarray(project.solutions[0][1], float), "last optimization result"
    Z = project.region_samples(4000, 3)
    if not len(Z):
        return None, ""
    y = fit.predict(Z)
    i = int(np.argmax(y) if goal == "maximize" else np.argmin(y))
    return Z[i], ("Model-predicted maximum" if goal == "maximize" else "Model-predicted minimum")


def assess(project, j, order=None, n_suggest=None):
    p = project
    y_all = p.model_data[:, j]
    checks = []
    fam = p.response_family(j)
    comp_rows = np.where(np.isfinite(y_all) & np.all(np.isfinite(p.coded), axis=1))[0]
    N = len(comp_rows)
    order = order or p.model_spec(j)["order"]
    terms = p.terms_for_order(order) if p.model_spec(j)["terms"] is None or order != p.model_spec(j)["order"] \
        else p.model_terms(j)
    n_par = len(models.expand(np.zeros((1, p.k)), terms, p.space)[1])
    existing = p.coded[comp_rows]
    sugg = {}

    # ================= A. completeness & cleanliness
    miss_f = int((~np.isfinite(p.coded)).any(axis=1).sum())
    miss_y = int((~np.isfinite(p.data[:, j])).sum())
    n_exc = len(p.excluded)
    st = "ok" if not (miss_f or miss_y) else ("bad" if N < 3 else "warn")
    recs = []
    if miss_f:
        recs.append(_rec("complete", f"Fill in the factor values of the {miss_f} empty runs, or delete those rows "
                                     "on the Data page (incomplete rows are not analyzed)."))
    if miss_y:
        recs.append(_rec("complete", f"Fill in the {miss_y} empty response values (perform those runs) - runs "
                                     "without a response give no information."))
    checks.append(_chk("data", "Missing data", f"{miss_f} runs with empty factors, {miss_y} empty responses", st,
                       f"{N} complete runs are used for the analysis" + (f" ({n_exc} runs deliberately ignored)."
                                                                         if n_exc else "."), recs))
    if N < 3:
        return _finish(p, j, checks, sugg, N, n_par, order)
    y = y_all[comp_rows]
    # exact duplicate rows (same factors & response) = probably pasted twice
    key = np.column_stack([np.round(existing, 9), np.round(y, 9)])
    _, cnt = np.unique(key, axis=0, return_counts=True)
    dup = int((cnt - 1)[cnt > 1].sum())
    checks.append(_chk("data", "Exact duplicate rows", f"{dup} rows", "warn" if dup else "ok",
                       "Rows with exactly the same factors AND response are usually pasted twice, not true "
                       "replicates (real replicates almost always give slightly different responses)." if dup else
                       "No suspicious duplicate rows.",
                       [_rec("treatment", f"Check the {dup} duplicate rows; delete them if they were recorded twice.")]
                       if dup else []))
    const = [p.factor_label(i) for i in p.num_idx if np.ptp(existing[:, i]) < 1e-12]
    if const:
        checks.append(_chk("data", "Constant factors", ", ".join(const), "bad",
                           "A factor with the same value in every run cannot be modeled.",
                           [_rec("treatment", "Remove constant factors from the project, or add runs with different "
                                              "levels for those factors.")]))

    # ================= B. structure for modeling
    df_res = N - n_par
    need = max(0, n_par + 5 - N)
    st = "ok" if df_res >= 5 else ("warn" if df_res >= 1 else "bad")
    recs = []
    if need:
        sugg["dopt"] = suggest_d_optimal(p, terms, existing, need)
        recs.append(_rec("complete", f"Add at least {need} runs (suggested: D-optimal for the "
                                     f"{models.order_label(order)} model) to leave ≥ 5 residual degrees of freedom.",
                         ("add_runs", "dopt")))
    checks.append(_chk("model", "Runs vs model parameters", f"{N} runs, {n_par} parameters "
                       f"({models.order_label(order)}), residual df {df_res}", st,
                       "≥ 5 residual degrees of freedom are needed for reliable significance tests & error estimates.",
                       recs))

    if not p.is_mixture:
        few = []
        for i in p.num_idx:
            if len(np.unique(np.round(existing[:, i], 6))) < 3:
                few.append(p.factor_label(i))
        quad = any(models.degree(t) >= 2 and max(t) >= 2 for t in terms if not models.is_intercept(t)) \
            if not p.is_mixture else False
        st = "ok" if not few else ("bad" if quad else "warn")
        recs = []
        if few:
            sugg.setdefault("dopt_quad", suggest_d_optimal(p, p.terms_for_order("quadratic")
                                                           if "quadratic" in p.available_orders() else terms,
                                                           existing, max(3, len(few) * 2)))
            recs.append(_rec("complete", "Add runs at the middle level (center/axial points) for " + ", ".join(few)
                             + " so curvature (quadratic) can be estimated - important for finding the optimum.",
                             ("add_runs", "dopt_quad")))
        checks.append(_chk("model", "Numeric factor levels", ", ".join(
            f"{p.factor_label(i).split(':')[0]}={len(np.unique(np.round(existing[:, i], 6)))}" for i in p.num_idx),
            st, "A 2-level factor can only give a linear effect; an optimum inside the range needs ≥ 3 levels.",
            recs))

    s2, dfpe = models.pure_error(existing, y)
    recs = []
    if dfpe < 3:
        center = existing[np.argmin(np.abs(existing[:, p.num_idx]).sum(axis=1))] if p.num_idx and not p.is_mixture \
            else existing.mean(axis=0)
        if not p.is_mixture and p.num_idx:
            center = center.copy()
            center[p.num_idx] = 0.0
        sugg["rep"] = np.repeat(center[None], 4, axis=0)
        recs.append(_rec("complete", "Add 3–5 replicates (e.g. at the center point) to measure pure error: "
                                     "lack of fit can then be tested and the response noise is known.",
                         ("add_runs", "rep")))
    checks.append(_chk("model", "Replicates (pure error)", f"pure error df = {dfpe}",
                       "ok" if dfpe >= 3 else ("warn" if dfpe >= 1 else "warn"),
                       "Without replicates, lack of fit cannot be tested and the noise level is unknown.", recs))

    fit, err = p.fit_with_error(j)
    if fit is not None and p.analysis_kind(j) == "ols":
        vif = fit.coef_vif
        vif = vif[np.isfinite(vif)]
        vmax = float(vif.max()) if len(vif) else 1.0
        st = "ok" if vmax <= 5 else ("warn" if vmax <= 10 else "bad")
        recs = []
        if vmax > 5:
            sugg.setdefault("dopt", suggest_d_optimal(p, terms, existing, max(4, n_par // 2)))
            recs.append(_rec("complete", "Add D-optimal runs to break the correlation between factors (common in "
                                         "historical data) so the effect of each factor can be separated.",
                             ("add_runs", "dopt")))
        checks.append(_chk("model", "Multicollinearity (max VIF)", f"{vmax:.2f}", st,
                           "VIF > 10: unstable coefficients and mixed factor effects. Ideal ≈ 1 (orthogonal design).",
                           recs))
        lev = fit.leverage
        hi_lev = np.where(lev > max(2 * fit.p / fit.n, 0.5))[0] if fit.n > fit.p else np.array([], int)
        runs = p.run_order[fit.rows_used]
        recs = []
        if len(hi_lev):
            sugg.setdefault("space", suggest_space_filling(p, existing, max(5, min(20, N // 4))))
            recs.append(_rec("complete", "Run " + ", ".join(map(str, sorted(runs[hi_lev][:15]))) + " stands alone in "
                             "the factor space (high leverage) - add runs around it so the model does not depend on "
                             "a single point.", ("add_runs", "space")))
        checks.append(_chk("model", "High-leverage points", f"{len(hi_lev)} runs", "warn" if len(hi_lev) else "ok",
                           "Leverage > 2p/n (or > 0.5): these points strongly determine the shape of the model.",
                           recs))

    if p.num_idx and not p.is_mixture:
        from scipy.spatial import cKDTree
        lo, hi = p.coded_bounds()
        num = p.num_idx
        span = np.where(hi[num] > lo[num], hi[num] - lo[num], 1.0)
        U = (existing[:, num] - lo[num]) / span
        R = p.region_samples(3000, 5)
        gap = float(cKDTree(U).query((R[:, num] - lo[num]) / span)[0].max() / np.sqrt(len(num))) if len(R) else 0.0
        st = "ok" if gap <= 0.3 else ("warn" if gap <= 0.5 else "bad")
        recs = []
        if gap > 0.3:
            sugg["space"] = suggest_space_filling(p, existing, max(5, min(20, N // 4)))
            recs.append(_rec("complete", "Add space-filling runs in the empty regions of the factor space - "
                                         "prediction & optimization in empty regions are extrapolation.",
                             ("add_runs", "space")))
        checks.append(_chk("model", "Factor space coverage", f"largest gap {gap:.2f} (relative to diagonal)", st,
                           "The largest distance from a point in the design region to the nearest data. ≤ 0.3 is "
                           "good.", recs))
    for i in p.cat_idx:
        c = np.bincount(np.rint(existing[:, i]).astype(int), minlength=len(p.factors[i].levels))
        st = "ok" if c.min() >= max(3, 0.5 * c.max()) else ("warn" if c.min() >= 2 else "bad")
        checks.append(_chk("model", f"Level balance of {p.factors[i].name}",
                           ", ".join(f"{lv}={n}" for lv, n in zip(p.factors[i].levels, c)), st,
                           "Levels with little data make their effect uncertain.",
                           [_rec("complete", f"Add runs at level {p.factors[i].levels[int(np.argmin(c))]}.")]
                           if st != "ok" else []))

    # ================= C. response & model quality
    if fit is not None and fam == "normal":
        runs = p.run_order[fit.rows_used]
        r = np.nan_to_num(fit.stud_ext)
        lim = stats.t.ppf(1 - 0.05 / (2 * fit.n), fit.df_resid - 1) if fit.df_resid > 1 else np.inf
        out = np.where(np.abs(r) > lim)[0]
        cook = np.where(np.nan_to_num(fit.cooks) > 1.0)[0]
        bad_rows = sorted(set(fit.rows_used[out].tolist()) | set(fit.rows_used[cook].tolist()))
        recs = []
        if bad_rows:
            recs.append(_rec("treatment", "Check run " + ", ".join(map(str, sorted(p.run_order[bad_rows][:15])))
                             + " (recording error? different conditions?). If it really deviates, ignore it in the "
                               "analysis without deleting the data.", ("exclude", bad_rows)))
        checks.append(_chk("resp", "Outliers & influential points", f"{len(out)} outliers, {len(cook)} Cook's > 1",
                           "warn" if bad_rows else "ok",
                           f"Outlier: |externally studentized residual| > {lim:.2f} (Bonferroni limit). Influential "
                           "point: Cook's distance > 1.", recs))

        bc = p.box_cox(j) if p.analysis_kind(j) == "ols" else None
        rec_tr = models.recommend_transform(bc) if bc else None
        cur = (p.model_spec(j).get("transform") or {}).get("kind", "none")
        yr = (y.max() / y.min()) if y.min() > 0 else np.nan
        gain = 0.0
        if bc:
            l1 = bc["lnss"][int(np.argmin(np.abs(np.asarray(bc["lambdas"]) - 1.0)))]
            gain = 1 - float(np.exp(bc["min"] - l1))           # reduction in residual SS when transformed
        # large data: the Box-Cox CI is very narrow, so a transform is only suggested when the gain is real (≥ 5%)
        need_tr = rec_tr is not None and rec_tr.get("kind", "none") != "none" and cur == "none" and gain >= 0.05
        recs = []
        if need_tr:
            recs.append(_rec("treatment", f"Response transform: {models.transform_label(rec_tr)} (Box-Cox "
                                          "recommendation) so the residual variance is constant.",
                             ("transform", rec_tr)))
        checks.append(_chk("resp", "Response spread & transform",
                           (f"max/min {yr:.1f}" if np.isfinite(yr) else "has values ≤ 0")
                           + (f", Box-Cox λ = {bc['best']:.2f} (CI {bc['ci'][0]:.2f}…{bc['ci'][1]:.2f}), residual SS "
                              f"drops {100 * gain:.0f}% when transformed" if bc else ""),
                           "warn" if need_tr else "ok",
                           "A transform is recommended when λ = 1 is outside the Box-Cox CI and the residual SS drops "
                           "≥ 5%.", recs))

        a = np.abs(fit.resid)
        if fit.n > 5 and np.ptp(fit.yhat) > 0:
            rho, pv = stats.spearmanr(fit.yhat, a)
            het = pv < 0.05 and rho > 0
            checks.append(_chk("resp", "Constant residual variance", f"ρ(|residual|, predicted) = {rho:.2f}, "
                               f"p = {pv:.3g}", "warn" if het else "ok",
                               "Residuals that grow with larger predictions = non-constant variance.",
                               [_rec("treatment", "Use a ln or square root transform on the response.",
                                     ("transform", {"kind": "ln"} if y.min() > 0 else {"kind": "sqrt"}))]
                               if het and cur == "none" else []))
        o = np.argsort(p.run_order[fit.rows_used])
        e = fit.resid[o]
        dw = float(np.sum(np.diff(e) ** 2) / max(np.sum(e ** 2), 1e-300)) if len(e) > 3 else 2.0
        rr, pr = stats.spearmanr(np.arange(len(e)), e) if len(e) > 5 else (0.0, 1.0)
        drift = dw < 1.4 or dw > 2.6 or pr < 0.01
        checks.append(_chk("resp", "Drift over run order", f"Durbin-Watson = {dw:.2f}, trend ρ = {rr:.2f}",
                           "warn" if drift else "ok",
                           "Residuals correlated with run order indicate that equipment/materials changed during the "
                           "experiment.",
                           [_rec("treatment", "Check equipment calibration & materials during the experiment; group "
                                              "runs by day/batch as blocks, and randomize the run order in the next "
                                              "experiment.")]
                           if drift else []))

        stt = fit.stats
        r2, pr2, ar2 = stt.get("r2", np.nan), stt.get("pred_r2", np.nan), stt.get("adj_r2", np.nan)
        lof = next((a_.p for a_ in fit.anova if a_.source == "Lack of Fit"), np.nan)
        gap_ = ar2 - pr2 if np.isfinite(pr2) else np.nan
        st = "ok"
        recs = []
        if np.isfinite(lof) and lof < 0.05:
            st = "warn"
            higher = [o_ for o_ in p.available_orders()
                      if p.available_orders().index(o_) > p.available_orders().index(order)] \
                if order in p.available_orders() else []
            recs.append(_rec("treatment", "Significant lack of fit: try a higher-order model"
                             + (f" ({models.order_label(higher[0])})" if higher else "") + ", a transform, or an ANN.",
                             ("order", higher[0]) if higher else None))
        if np.isfinite(gap_) and gap_ > 0.2:
            st = "warn"
            recs.append(_rec("treatment", "Adj R² and Pred R² differ by > 0.2: remove non-significant terms "
                                          "(Backward Elimination) or check for outliers.", ("backward", None)))
        if np.isfinite(pr2) and pr2 < 0.5:
            st = "bad" if pr2 < 0 else "warn"
        checks.append(_chk("resp", "Model fit", f"R² {r2:.3f}, Adj {ar2:.3f}, Pred {pr2:.3f}"
                           + (f", LOF p {lof:.3f}" if np.isfinite(lof) else ""), st,
                           "Pred R² shows the ability to predict new data; an Adj–Pred difference > 0.2 indicates "
                           "overfitting.", recs))
        noise = [v for v in (np.sqrt(s2 / dfpe) if dfpe else None,) if v]
        if noise:
            snr = float(np.std(y, ddof=1) / max(noise))
            checks.append(_chk("resp", "Signal / noise", f"{snr:.2f}", "ok" if snr >= 4 else ("warn" if snr >= 2 else "bad"),
                               "Response SD compared with the noise (pure error from replicated points).",
                               [_rec("treatment", "High noise: improve the measuring precision, repeat measurements "
                                                  "(average several repeats), or widen the factor ranges.")]
                               if snr < 4 else []))
    elif err and fam == "normal":
        checks.append(_chk("resp", "Model", "cannot be calculated yet", "bad", err))

    # ================= D. optimization readiness
    if fit is not None:
        mrow = next((a_ for a_ in fit.anova if a_.source == "Model"), None)
        pm = mrow.p if mrow is not None else np.nan
        if np.isfinite(pm):
            checks.append(_chk("opt", "Model significance", f"p = {pm:.4g}", "ok" if pm < 0.05 else "bad",
                               "The model must be significant (p < 0.05) before it is used for optimization.",
                               [] if pm < 0.05 else [_rec("complete", "Add more runs or widen the factor ranges so the "
                                                                      "factor effects are larger than the noise.")]))
        pr2 = fit.stats.get("pred_r2", np.nan)
        if np.isfinite(pr2):
            checks.append(_chk("opt", "Prediction reliability (Pred R²)", f"{pr2:.3f}",
                               "ok" if pr2 >= 0.7 else ("warn" if pr2 >= 0.4 else "bad"),
                               "An optimum from a model with a low Pred R² cannot be trusted yet.",
                               [] if pr2 >= 0.7 else
                               [_rec("complete", "Add D-optimal runs, then analyze again.", ("add_runs", "dopt"))]
                               + ([_rec("treatment", "The pattern looks strongly nonlinear and there is enough data: "
                                                     "try an ANN (Neural Network page), then compare it with RSM.")]
                                  if N >= 30 else [])))
            if pr2 < 0.7 and "dopt" not in sugg:
                sugg["dopt"] = suggest_d_optimal(p, terms, existing, max(4, n_par // 2))
        opt, src = _optimum(p, j, fit)
        if opt is not None:
            lo, hi = p.coded_bounds()
            at_edge = [p.factor_label(i) for i in p.num_idx if i not in p.comp_idx and
                       (abs(opt[i] - lo[i]) < 0.02 * (hi[i] - lo[i]) or abs(opt[i] - hi[i]) < 0.02 * (hi[i] - lo[i]))]
            recs = []
            if at_edge:
                recs.append(_rec("complete", "The optimum is at the range limit of " + ", ".join(at_edge) + " - the "
                                 "true optimum may be outside the range. Extend the range of these factors in that "
                                 "direction (steepest ascent method), then add runs there."))
            checks.append(_chk("opt", "Optimum location", f"{src}: " + ", ".join(
                f"{p.factor_label(i).split(':')[0]}={p.format_value(i, p.factor_to_actual(i, opt[i]), 4)}"
                for i in range(p.k)), "warn" if at_edge else "ok",
                "An optimum at the edge of the design region often means the factor ranges do not yet cover the "
                "best conditions.", recs))
            num = [i for i in p.num_idx]
            if num:
                lo_, hi_ = lo[num], hi[num]
                sp_ = np.where(hi_ > lo_, hi_ - lo_, 1.0)
                dmin = float(np.sqrt((((existing[:, num] - opt[num]) / sp_) ** 2).sum(axis=1)).min()
                             / np.sqrt(len(num)))
                try:
                    iv = fit.intervals(opt[None, :])
                except Exception:  # noqa: BLE001 - model without intervals (e.g. GLM)
                    iv = None
                half = np.nan
                if iv is not None and np.isfinite(iv["ci"][0][0]):
                    half = float((iv["ci"][1][0] - iv["ci"][0][0]) / 2)
                rel = half / max(np.ptp(y), 1e-300) if np.isfinite(half) else np.nan
                sparse = dmin > 0.2 or (np.isfinite(rel) and rel > 0.1)
                sugg["near"] = suggest_near(p, opt, 4)
                checks.append(_chk("opt", "Data support around the optimum",
                                   f"distance to nearest run {dmin:.2f}" + (f", CI ±{half:.4g} ({100 * rel:.0f}% of "
                                                                            "response range)" if np.isfinite(rel)
                                                                            else ""),
                                   "warn" if sparse else "ok",
                                   "An optimum far from the data or with a wide CI is not yet proven. Confirmation "
                                   "runs at the optimum are still recommended before the result is used.",
                                   [_rec("complete", "Perform 3–4 confirmation runs around the optimum (suggested "
                                                     "points available).", ("add_runs", "near"))] if sparse else []))
    return _finish(p, j, checks, sugg, N, n_par, order)


def _finish(p, j, checks, sugg, N, n_par, order):
    scores = {}
    for cat in CATS:
        cs = [c for c in checks if c["cat"] == cat and c["status"] in WEIGHT]
        scores[cat] = 100.0 * sum(WEIGHT[c["status"]] for c in cs) / len(cs) if cs else np.nan
    model_part = [scores[c] for c in ("data", "model", "resp") if np.isfinite(scores[c])]
    overall_model = float(np.mean(model_part)) if model_part else np.nan
    opt_part = [scores[c] for c in ("data", "model", "resp", "opt") if np.isfinite(scores[c])]
    overall_opt = float(np.mean(opt_part)) if opt_part else np.nan
    recs = [dict(r, check=c["name"], status=c["status"]) for c in checks for r in c["rec"]]
    order_rank = {"bad": 0, "warn": 1, "ok": 2, "info": 3}
    recs.sort(key=lambda r: (order_rank.get(r["status"], 3), r["kind"] != "complete"))
    return {"checks": checks, "scores": scores, "score_model": overall_model, "score_opt": overall_opt,
            "recs": recs, "suggest": sugg, "n": N, "n_par": n_par, "order": order, "j": j}


def verdict(score):
    if not np.isfinite(score):
        return "Cannot be assessed yet", "info"
    if score >= 85:
        return "Ready", "ok"
    if score >= 60:
        return "Fair - some things need fixing", "warn"
    return "Not ready yet", "bad"


SUGGEST_LABEL = {"dopt": "D-optimal (adds model information)", "dopt_quad": "Middle level (curvature)",
                 "rep": "Replicates (pure error)", "space": "Factor space gap filling",
                 "near": "Confirmation around the optimum"}
