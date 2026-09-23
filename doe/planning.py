"""Experiment planning: design recommendations by goal and a run-count calculator for RSM and ANN.

The run count of each proposal is computed by actually building the design (doe.designs), and power is computed
with doe.evaluation for a signal/noise ratio of 2 (response difference between the low and high level = 2 sigma).
"""
from math import comb

import numpy as np

from . import designs, evaluation, models

GOALS = {
    "screening": "Screening: find the influential factors among many factors",
    "characterize": "Characterization: main effects and two-factor interactions",
    "optimize": "RSM optimization: find the optimum (curvature present)",
    "ann": "ANN modeling: complex nonlinear patterns",
    "rsm_ann": "RSM and ANN together (for comparison)",
    "robust": "Ruggedness test: confirm that the factors have NO effect",
}
NONLINEAR = {"low": "Low (close to quadratic)", "mid": "Medium", "high": "High (many peaks / sharp)"}

# practical guideline for the number of ANN runs (trainbr + cross-validation) per number of factors
ANN_TABLE = {1: (10, (15, 20), 30), 2: (15, (25, 30), 40), 3: (20, (30, 50), 60), 4: (30, (50, 80), 100),
             5: (40, (70, 120), 150), 6: (50, (100, 150), 200)}


def n_coef(k, order):
    """Number of process model coefficients (including the intercept)."""
    if order == "linear":
        return 1 + k
    if order == "2fi":
        return 1 + k + comb(k, 2)
    if order == "quadratic":
        return (k + 1) * (k + 2) // 2
    if order == "cubic":
        return comb(k + 3, 3)
    return len(models.model_terms(k, order))


def rsm_runs(k, order="quadratic", n_lof=5, n_rep=5):
    p = n_coef(k, order)
    return {"p": p, "n_lof": n_lof, "n_rep": n_rep, "n_min": p + n_lof + n_rep, "n_abs": p + 1}


def ann_table(k):
    """(minimum, (suggested lower, upper), strongly nonlinear pattern)."""
    if k in ANN_TABLE:
        return ANN_TABLE[k]
    return (10 * k - 10, (17 * k, 25 * k), 35 * k)


def ann_runs(k, hidden, n_out=1, algorithm="trainbr", scheme="kfold"):
    """Data requirement for a given architecture. Rule: training rows >= ratio x number of weights
    (ratio 1 for trainbr because Bayesian regularization reduces the effective parameters, 2 for trainlm/Adam),
    then divided by the training fraction (85% with cross-validation, 70% with hold-out)."""
    from .ann import n_weights_for
    w = n_weights_for(k, list(hidden), n_out)
    ratio = 1.0 if algorithm == "trainbr" else 2.0
    frac = 0.85 if scheme == "kfold" else 0.70
    need = int(np.ceil(ratio * w / frac))
    return {"weights": w, "ratio": ratio, "train_frac": frac, "n_need": need}


def ann_target(k, nonlinear="mid"):
    lo, (r1, r2), hi = ann_table(k)
    return {"low": r1, "mid": (r1 + r2) // 2, "high": max(r2, hi)}.get(nonlinear, (r1 + r2) // 2)


# ------------------------------------------------------------------ power
def _power(coded, order):
    k = coded.shape[1]
    terms = models.model_terms(k, order)
    try:
        ev = evaluation.evaluate(coded, terms)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, 0
    pw = ev["power"][np.isfinite(ev["power"])]
    return (float(pw.min()) if len(pw) else np.nan), int(ev["df_resid"])


def _option(key, title, dtype, k, opts, order, why, cons, levels, runs=None, coded=None, note=""):
    if coded is None and dtype == "lhs":           # quick preview (without full maximin optimization)
        coded = np.vstack([designs.latin_hypercube(k, opts["lhs_runs"], 1, iters=300),
                           np.zeros((opts.get("center_points", 0), k))])
    if coded is None:
        try:
            coded, types, _ = designs.build_design(dtype, k, **opts)
        except (ValueError, KeyError) as exc:
            return {"key": key, "error": str(exc)}
    n = len(coded) if runs is None else runs
    pw, df = _power(coded, order) if order else (np.nan, 0)
    return {"key": key, "title": title, "type": dtype, "options": opts, "runs": int(n), "order": order,
            "power": pw, "df_resid": df, "why": why, "cons": cons, "levels": levels, "note": note}


def recommend(goal, k, budget=0, nonlinear="mid"):
    """List of design proposals for a goal and k numeric factors. budget 0 = unlimited.
    Each proposal: title, type, options, runs, order, power (minimum power of the model terms, 2 sigma signal), why,
    cons, levels, over (exceeds the budget)."""
    out = []
    add = out.append
    if goal in ("screening", "robust"):
        pb = next(r for r in (12, 20, 24) if r > k) if k <= 23 else None
        if k >= 2 and pb:
            add(_option("pb", f"Plackett-Burman {pb} runs", "plackett_burman", k, {"pb_runs": pb, "center_points": 0},
                        "linear", "The fewest runs to test many main effects at once.",
                        "Main effects are partially confounded with interactions; interactions cannot be estimated.",
                        "2 levels"))
        r3 = designs.min_runs_for_resolution(k, 3) if k >= 3 else None
        r4 = designs.min_runs_for_resolution(k, 4) if k >= 3 else None
        if goal == "robust" and r3:
            add(_option("ff3", f"Fractional factorial 2^({k}-{r3[1]}) Res {designs.roman(designs.resolution(k, r3[1]))}",
                        "fractional_factorial" if r3[1] else "full_factorial", k,
                        {"fraction_p": r3[1], "center_points": 0} if r3[1] else {"center_points": 0}, "linear",
                        "The most economical way to confirm the factors have no effect over the tested range.",
                        "Main effects are aliased with two-factor interactions.", "2 levels"))
        if goal == "screening" and k >= 3 and r4:
            p = r4[1]
            add(_option("ff4", f"Fractional factorial 2^({k}-{p}) Res {designs.roman(designs.resolution(k, p))}"
                        if p else f"Full factorial 2^{k}", "fractional_factorial" if p else "full_factorial", k,
                        {"fraction_p": p, "center_points": 4} if p else {"center_points": 4}, "linear",
                        "Main effects are clear of two-factor interactions (resolution IV). Center points detect "
                        "curvature.", "Two-factor interactions are still aliased with each other.",
                        "2 levels + center"))
        if goal == "screening" and 3 <= k <= 20:
            add(_option("dsd", "Definitive Screening Design", "dsd", k, {}, "linear",
                        "3 levels: besides main effects it can also detect curvature; main effects are not aliased "
                        "with two-factor interactions.", "A full quadratic model is only possible when few factors "
                        "are active.", "3 levels"))
    if goal == "characterize":
        r5 = designs.min_runs_for_resolution(k, 5)
        if r5:
            p = r5[1]
            add(_option("ff5", (f"Fractional factorial 2^({k}-{p}) Res {designs.roman(designs.resolution(k, p))}"
                                if p else f"Full factorial 2^{k}"),
                        "fractional_factorial" if p else "full_factorial", k,
                        {"fraction_p": p, "center_points": 4} if p else {"center_points": 4}, "2fi",
                        "The fewest runs with main effects and two-factor interactions free of aliasing.",
                        "Cannot model curvature (only detected from the center points).", "2 levels + center"))
        if k <= 7 and (not r5 or r5[0] < 2 ** k):
            add(_option("full", f"Full factorial 2^{k}", "full_factorial", k, {"center_points": 4}, "2fi",
                        "All interactions can be estimated, no aliasing.", "Runs double with each added factor.",
                        "2 levels + center"))
        rr = rsm_runs(k, "2fi")
        add({"key": "opt2", "title": "Optimal (Custom), 2FI model", "type": "optimal_rsm",
             "options": {"model_order": "2fi"}, "runs": rr["n_min"], "order": "2fi", "power": np.nan,
             "df_resid": rr["n_min"] - rr["p"], "why": "The number of runs can be set freely; suitable when there "
             "are constraints or categorical factors.", "cons": "Not balanced like a classic factorial.",
             "levels": "2 levels", "note": f"{rr['p']} coefficients + 5 lack of fit + 5 replicates"})
    if goal in ("optimize", "rsm_ann"):
        if 2 <= k <= 6:
            add(_option("ccd", "Central Composite (rotatable)", "ccd", k,
                        {"alpha_type": "rotatable", "center_points": designs.default_center_points("ccd", k)},
                        "quadratic", "The most common RSM design; predicts equally well in all directions; 5 levels "
                        "per factor.", "Axial points lie outside the low-high range (make sure they can still be run).",
                        "5 levels"))
            add(_option("ccf", "Central Composite face-centered", "ccd", k,
                        {"alpha_type": "face", "center_points": designs.default_center_points("ccd", k)},
                        "quadratic", "All points lie within the low-high range.", "Only 3 levels per factor.",
                        "3 levels"))
        if 3 <= k <= 7:
            add(_option("bbd", "Box-Behnken", "bbd", k, {"center_points": designs.default_center_points("bbd", k)},
                        "quadratic", "No points at the extreme corners (safe when extreme combinations are "
                        "dangerous/expensive).", "Predictions at the corners are less accurate; 3 levels.", "3 levels"))
        rr = rsm_runs(k, "quadratic")
        add({"key": "optq", "title": "Optimal (Custom), quadratic model", "type": "optimal_rsm",
             "options": {"model_order": "quadratic"}, "runs": rr["n_min"], "order": "quadratic",
             "power": np.nan, "df_resid": rr["n_min"] - rr["p"],
             "why": "The fewest runs for a quadratic model; works with constraints and categorical factors.",
             "cons": "Points are chosen by an algorithm, so less symmetric.", "levels": "3 levels",
             "note": f"{rr['p']} coefficients + 5 lack of fit + 5 replicates"})
    if goal in ("ann", "rsm_ann"):
        target = ann_target(k, nonlinear)
        if goal == "ann":
            n_lhs = max(target, 10)
            add(_option("lhs", f"Latin Hypercube {n_lhs} runs + 4 center", "lhs", k,
                        {"lhs_runs": n_lhs, "center_points": 4}, "quadratic",
                        f"Evenly spread points, {n_lhs} levels per factor: best for ANN. Replicated center points "
                        "give a noise estimate.", "Not balanced/orthogonal like classic RSM designs.",
                        f"{n_lhs} levels", note=f"ANN target for {k} factors, nonlinearity {NONLINEAR[nonlinear]}"))
            lo = ann_table(k)[0]
            add(_option("lhs_min", f"Latin Hypercube minimum {lo} runs + 3 center", "lhs", k,
                        {"lhs_runs": lo, "center_points": 3}, "quadratic",
                        "Practical lower limit for a small ANN (trainbr) with cross-validation.",
                        "The ANN may not yet beat RSM; check with the learning curve.", f"{lo} levels"))
        bases = [("ccd_face", "CCD face-centered")] + ([("bbd", "Box-Behnken")] if k >= 3 else []) + \
            [("ccd", "CCD rotatable")]
        for bkey, bname in bases:
            if bkey != "bbd" and not 2 <= k <= 6:
                continue
            base, _, _ = designs.build_design({"ccd_face": "ccd", "ccd": "ccd", "bbd": "bbd"}[bkey], k,
                                              alpha_type="face" if bkey == "ccd_face" else "rotatable")
            n_extra = max(target - len(base) - 4, 6)
            add(_option(f"hyb_{bkey}", f"Hybrid {bname} + {n_extra} space-filling points + 4 center", "hybrid", k,
                        {"hybrid_base": bkey, "extra_points": n_extra, "center_points": 4}, "quadratic",
                        "One data set for RSM (classic design structure) and ANN (extra points fill the space).",
                        "More runs than an RSM design alone.", "many levels",
                        note=f"Total target of about {target} runs for ANN"))
    if goal in ("optimize", "rsm_ann") and k >= 6:
        out.append({"key": "advice", "title": f"Suggestion: screen first ({k} factors is quite many)", "type": "dsd",
                    "options": {}, "runs": len(designs.definitive_screening(k)), "order": "linear", "power": np.nan,
                    "df_resid": 0, "why": "A Definitive Screening Design finds the important factors and detects "
                    "curvature at the same time. The subsequent optimization then only needs the 3 to 4 most "
                    "important factors, so far fewer runs.", "cons": "Requires two experiment stages.",
                    "levels": "3 levels", "note": ""})
    budget = int(budget or 0)
    for o in out:
        if "error" in o:
            continue
        o["over"] = bool(budget and o["runs"] > budget)
    out = [o for o in out if "error" not in o]
    if goal in ("ann", "rsm_ann") and budget:
        lo = ann_table(k)[0]
        if budget < lo:
            out.append({"key": "warn", "title": f"Budget of {budget} runs is below the ANN minimum ({lo} runs)",
                        "type": None, "options": {}, "runs": budget, "order": None, "power": np.nan, "df_resid": 0,
                        "why": "Consider RSM only, or reduce the factors by screening first.", "cons": "",
                        "levels": "", "over": False, "note": ""})
        elif goal == "ann" and not any(not o["over"] for o in out if o["type"] == "lhs"):
            out.append(_option("lhs_budget", f"Latin Hypercube {budget - 3} runs + 3 center (fits budget)", "lhs", k,
                               {"lhs_runs": budget - 3, "center_points": 3}, "quadratic",
                               "Uses the whole budget with evenly spread points.", "Below the suggested number; "
                               "check with the learning curve once the data is in.", f"{budget - 3} levels"))
            out[-1]["over"] = False
    return out


def ann_adequacy_text(k, n):
    lo, (r1, r2), hi = ann_table(k)
    if n < lo:
        return "low", f"{n} runs is below the practical minimum ({lo}) for {k} factors."
    if n < r1:
        return "minimum", f"{n} runs is enough for a small ANN, below the suggested number ({r1}-{r2})."
    if n < hi:
        return "good", f"{n} runs matches the suggested number for {k} factors."
    return "very good", f"{n} runs is enough for strongly nonlinear patterns."
