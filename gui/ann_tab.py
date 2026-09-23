"""ANN page: data adequacy, separate/combined models, early stopping, automatic architecture search (for one
or all responses at once), training, underfitting/overfitting diagnosis, RSM comparison."""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLineEdit, QGroupBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox,
                               QSplitter, QTabWidget, QVBoxLayout, QWidget)

from doe import adequacy, ann, parallel

from . import plots, theme
from .common import REPORT_CSS, MplCanvas, Task, fmt
from .copying import ReportBrowser

LEARN_STATUS = {"enough": "the data is sufficient", "more": "more data would still help",
                "far": "the data is still far from sufficient", "few": "too little data to assess",
                "?": "cannot be assessed yet"}
ALG_LABEL = {"trainbr": "Bayesian Regularization (trainbr)", "trainlm": "Levenberg-Marquardt (trainlm)",
             "adam": "Adam mini-batch"}


def parse_hidden(text):
    """'20, 10, 5' -> [20, 10, 5]. Any number of layers and neurons (no limit)."""
    parts = [t for t in text.replace(";", ",").replace(" ", ",").split(",") if t.strip()]
    try:
        vals = [int(float(t)) for t in parts]
    except ValueError:
        raise ValueError("Enter the neurons per hidden layer as numbers separated by commas. Example: 20, 10, 5") from None
    if not vals or min(vals) < 1:
        raise ValueError("At least 1 hidden layer is required, and each layer needs at least 1 neuron.")
    return vals
BIG_DATA = 5000             # above this, the number of final training restarts is limited to keep it fast
STATUS_LABEL = {"ok": "Good", "warn": "Caution", "bad": "Not adequate", "info": "Info"}
FIT_CLASS = {"good": "ok", "under": "warn", "over": "bad", "?": "info"}

SET_COLORS = {"train": "#1f5f99", "val": "#1a7f37", "test": "#b42318", "all": "#475569"}
SET_NAMES = {"train": "Training", "val": "Validation", "test": "Test", "all": "All"}


def _runs_text(runs, limit=60):
    runs = sorted(int(v) for v in runs)
    if not runs:
        return "-"
    if len(runs) <= limit:
        return ", ".join(map(str, runs))
    return ", ".join(map(str, runs[:limit])) + f", … (total {len(runs)} runs)"


def group_label(project, js):
    return ", ".join(project.response_label(j).split(":")[0] for j in js)


def fit_status(m):
    """Main fit status: from architecture-search cross-validation if available, else from the final model's split."""
    if m.cv_diag:
        return m.cv_diag["status"]
    return (m.diagnosis or {}).get("status", "?")


def diagnosis_html(m):
    d = m.diagnosis or {}
    st = d.get("status", "?")
    nm = getattr(m, "nmse", {}) or {}
    h = ["<h3>Fit Diagnosis</h3>"]
    cv = m.cv_diag
    if cv:
        h.append(f"<p class='{FIT_CLASS[cv['status']]}' style='font-size: 11pt'>{ann.STATUS[cv['status']]} "
                 f"<span class='note'>(result of {cv['scheme']} during architecture search)</span></p>"
                 f"<p>Training R² {fmt(cv['r2_train'], 4)}, validation R² {fmt(cv['r2_val'], 4)}, test R² "
                 f"{fmt(cv['q2'], 4)} - average over {cv['n_splits']} data splits, so more reliable than "
                 "a single random split.</p><p><b>Final model data split</b> (one random split): "
                 f"<span class='{FIT_CLASS[st]}'>{ann.STATUS[st]}</span></p>")
    else:
        h.append(f"<p class='{FIT_CLASS[st]}' style='font-size: 11pt'>{ann.STATUS[st]}</p>")
    n_check = len(m.idx_val) + len(m.idx_test)
    if n_check < 10:
        h.append(f"<p class='note'>The final model has only {n_check} validation + test runs, so the diagnosis "
                 "from this single split is less certain"
                 + (". Rely on the cross-validation result above." if cv else
                    " - run Search Architecture for a cross-validated diagnosis.") + "</p>")
    if d.get("notes"):
        h.append("<p>" + "; ".join(d["notes"]) + ".</p>")
    h.append("<table><tr><th class='l'>Data</th><th>R² (combined over outputs)</th><th>NMSE</th></tr>")
    for key in ("train", "val", "test"):
        v = nm.get(key, np.nan)
        if v is not None and np.isfinite(v):
            h.append(f"<tr><td class='l'>{SET_NAMES[key]}</td><td>{fmt(1 - v, 4)}</td><td>{fmt(v, 4)}</td></tr>")
    if d.get("noise_nmse"):
        h.append(f"<tr><td class='l'>Measurement noise limit</td><td>{fmt(1 - d['noise_nmse'], 4)}</td>"
                 f"<td>{fmt(d['noise_nmse'], 4)}</td></tr>")
    h.append("</table>")
    es = "enabled" if m.early_stop else "disabled"
    h.append(f"<p class='note'>Early stopping {es}"
             + (f": weights taken from epoch {m.best_epoch} (lowest validation error), training stopped at epoch "
                f"{m.epochs - 1} - {m.stop_reason}." if m.early_stop else f" ({m.stop_reason}).")
             + " NMSE = MSE / response variance (0 = perfect, 1 = no better than the mean). Criteria: "
               "<b>overfitting</b> if the error on new data > 2× the training error or the training error is below "
               "measurement noise; <b>underfitting</b> if training R² is low and the error on new data is similar "
               "(insufficient capacity).</p>")
    if st == "over":
        h.append("<p class='warn'>Suggestion: reduce neurons/hidden layers, enable early stopping, use trainbr, or "
                 "run Automatic Architecture Search.</p>")
    elif st == "under":
        h.append("<p class='warn'>Suggestion: add neurons or hidden layers, increase epochs/restarts, or run "
                 "Automatic Architecture Search.</p>")
    return "".join(h)


def ann_summary_html(project, j, m, search=None):
    p = project
    fit = p.fit(j) if p.analysis_kind(j) == "ols" else None
    alg = ALG_LABEL.get(m.algorithm, m.algorithm)
    if getattr(m, "auto_adam", False):
        alg += " (selected automatically: network too large for Levenberg-Marquardt)"
    h = [f"<h2>ANN - {p.response_label(j)}</h2>"]
    if m.combined:
        h.append(f"<div class='box'><b>Combined model</b>: one network with {m.n_out} outputs for "
                 f"{group_label(p, m.resp_idx)}. This page shows output {m.out + 1} "
                 f"({p.response_label(j)}).</div>")
    h.append(f"<p>Architecture <b>{m.arch}</b> ({m.n_in} inputs, {len(m.layers)} hidden layers with "
             f"{' / '.join(str(v) for v in m.layers)} neurons, activation {ann.ACTIVATIONS[m.activation].split()[0]}, "
             f"{m.n_out} linear outputs). Algorithm: <b>{alg}</b>. Weights & biases: {m.n_weights}. Normalization: "
             f"{ann.NORMS[getattr(m, 'norm', 'mapminmax')].split(' - ')[0]}.</p>")
    h.append(f"<p class='note'>Data split: {len(m.idx_train)} training / {len(m.idx_val)} validation / "
             f"{len(m.idx_test)} test (seed {m.seed}). Epochs: {m.epochs}, stopped: {m.stop_reason}"
             + (f", effective parameters γ = {m.gamma:.1f} of {m.n_weights}" if m.gamma is not None else "")
             + f". Restarts: {m.restarts}"
             + (f" (parallel on {m.workers} CPU cores)" if getattr(m, "workers", 1) > 1 else "") + ".</p>")
    h.append(diagnosis_html(m))
    runs = p.run_order[m.rows_used]
    h.append("<h3>Data Split (Random, Automatic)</h3><table><tr><th class='l'>Set</th><th>n</th>"
             "<th class='l'>Run numbers</th></tr>")
    for key, idx in (("train", m.idx_train), ("val", m.idx_val), ("test", m.idx_test)):
        nums = _runs_text(runs[np.array(idx, int)]) if len(idx) else "-"
        h.append(f"<tr><td class='l'>{SET_NAMES[key]}</td><td>{len(idx)}</td><td class='l'>{nums}</td></tr>")
    h.append(f"</table><p class='note'>The split is determined by the random seed ({m.seed}). Test data is never used "
             "during training" + ("; validation data is used for early stopping." if m.early_stop else ".") + "</p>")
    h.append("<h3>Performance</h3><table><tr><th class='l'>Data</th><th>n</th><th>R</th><th>R²</th><th>RMSE</th>"
             "<th>MAE</th><th>AAD (%)</th></tr>")
    for key in ("train", "val", "test", "all"):
        mt = m.metrics[key]
        if not mt.get("n"):
            continue
        h.append(f"<tr><td class='l'>{SET_NAMES[key]}</td><td>{mt['n']}</td><td>{fmt(mt['r'], 4)}</td>"
                 f"<td>{fmt(mt['r2'], 4)}</td><td>{fmt(mt['rmse'], 4)}</td><td>{fmt(mt['mae'], 4)}</td>"
                 f"<td>{fmt(mt['aad'], 4)}</td></tr>")
    h.append("</table>")
    if fit is not None:
        rows = m.rows_used
        yr = fit.predict(p.coded[rows])
        ya = m.predict(p.coded[rows])
        y = m.y_orig
        te = np.array(m.idx_test, int)
        comp = [("All data", ann.metrics(y, yr), ann.metrics(y, ya))]
        if len(te):
            comp.append(("ANN test data", ann.metrics(y[te], yr[te]), ann.metrics(y[te], ya[te])))
        h.append("<h3>RSM vs ANN Comparison</h3><table><tr><th class='l'>Data</th><th class='l'>Model</th>"
                 "<th>R²</th><th>RMSE</th><th>MAE</th><th>AAD (%)</th></tr>")
        for lab, a, b in comp:
            for name, mt in (("RSM", a), ("ANN", b)):
                h.append(f"<tr><td class='l'>{lab}</td><td class='l'>{name}</td><td>{fmt(mt['r2'], 4)}</td>"
                         f"<td>{fmt(mt['rmse'], 4)}</td><td>{fmt(mt['mae'], 4)}</td><td>{fmt(mt['aad'], 4)}</td></tr>")
        h.append("</table><p class='note'>RSM is fitted on all data, so for 'ANN test data' RSM has already "
                 "seen those points. A fair comparison is on the Architecture Search page (the same test data "
                 f"for both) or RSM Predicted R² = {fmt(fit.stats.get('pred_r2', np.nan), 4)}.</p>")
    imp = m.importance()
    method = "Garson" if len(m.layers) == 1 else "Garson extended to multiple hidden layers"
    h.append(f"<h3>Relative Factor Importance ({method})</h3><table><tr><th class='l'>Factor</th><th>%</th></tr>")
    for i, v in sorted(imp.items(), key=lambda kv: -kv[1]):
        h.append(f"<tr><td class='l'>{p.factor_label(i)}</td><td>{fmt(v, 4)}</td></tr>")
    h.append("</table>")
    if search:
        h.append(search_html(p, search, top=12))
    if m.n_weights > len(m.idx_train) * m.n_out:
        h.append(f"<p class='ns'>The number of weights ({m.n_weights}) exceeds the number of training targets "
                 f"({len(m.idx_train) * m.n_out}). High risk of overfitting - reduce neurons or use trainbr.</p>")
    return "".join(h)


def input_labels(project, m):
    cols = []
    for i, lv in m.input_names:
        f = project.factors[i]
        short = project.factor_label(i).split(":")[0]
        cols.append(f"{short}[{f.levels[lv]}]" if lv is not None else short)
    return cols


def output_labels(project, m):
    return [project.responses[j].name if j < len(project.responses) else f"Output {k + 1}"
            for k, j in enumerate(m.resp_idx)]


def weights_html(project, m):
    layers, W_out, b_out = m._unpack(m.weights)
    cols = input_labels(project, m)
    outs = output_labels(project, m)
    h = []
    prev = cols
    for li, (W, b) in enumerate(layers, 1):
        src = "Input" if li == 1 else f"Hidden {li - 1}"
        h.append(f"<h3>Weights {src} → Hidden {li} (IW{'' if li == 1 else li}) and Bias (b{li})</h3>"
                 "<table><tr><th>Neuron</th>" + "".join(f"<th>{c}</th>" for c in prev) + f"<th>b{li}</th></tr>")
        for k in range(W.shape[0]):
            h.append(f"<tr><td>H{li}.{k + 1}</td>" + "".join(f"<td>{fmt(v, 5)}</td>" for v in W[k])
                     + f"<td>{fmt(b[k], 5)}</td></tr>")
        h.append("</table>")
        prev = [f"H{li}.{k + 1}" for k in range(W.shape[0])]
    h.append(f"<h3>Weights Hidden {len(layers)} → Output (LW) and Output Bias</h3><table><tr><th class='l'>Output</th>"
             + "".join(f"<th>{c}</th>" for c in prev) + "<th>b output</th></tr>")
    for k, name in enumerate(outs):
        h.append(f"<tr><td class='l'>{name}</td>" + "".join(f"<td>{fmt(v, 5)}</td>" for v in W_out[k])
                 + f"<td>{fmt(b_out[k], 6)}</td></tr>")
    h.append("</table>")
    norm = getattr(m, "norm", "mapminmax")
    h.append(f"<h3>Normalization - {ann.NORMS[norm].split(' - ')[0]}</h3><table><tr><th class='l'>Variable</th>"
             "<th>min (training data)</th><th>max (training data)</th><th>offset</th><th>scale</th><th>base</th></tr>")
    for c, a, b, o, sc in zip(cols, m.xmin, m.xmax, m.x_off, m.x_scale):
        h.append(f"<tr><td class='l'>{c}</td><td>{fmt(a, 6)}</td><td>{fmt(b, 6)}</td><td>{fmt(o, 6)}</td>"
                 f"<td>{fmt(sc, 6)}</td><td>{fmt(m.x_base, 3)}</td></tr>")
    for k, name in enumerate(outs):
        h.append(f"<tr><td class='l'>Output: {name}</td><td>{fmt(m.ymin[k], 6)}</td><td>{fmt(m.ymax[k], 6)}</td>"
                 f"<td>{fmt(m.y_off[k], 6)}</td><td>{fmt(m.y_scale[k], 6)}</td><td>{fmt(m.y_base, 3)}</td></tr>")
    h.append("</table>")
    act = "tanh(a)" if m.activation == "tansig" else "1/(1+exp(-a))"
    h.append(f"<p class='note'>Network input = factors in coded units ({'pseudo-components' if project.is_mixture else '-1..+1'}"
             "; categorical = one-hot), then normalized: xₙ = base + (x − offset)/scale (parameters computed from "
             f"training + validation data only). Each hidden layer: h = f(IW·h_previous + b) with f(a) = {act}. Output: "
             "yₙ = LW·h_last + b; y = (yₙ − base)·scale + offset.</p>")
    return "".join(h)


def draw_network(fig, project, m, resp_label=None):
    """Network architecture diagram: teal = positive weight, red = negative; thickness ∝ |weight|."""
    ax = fig.add_subplot(111)
    layers, W_out, _ = m._unpack(m.weights)
    sizes = m.sizes + [m.n_out]
    names = [input_labels(project, m)] + [[""] * n for n in m.layers] + [output_labels(project, m)]
    xs = np.linspace(0, 1, len(sizes))
    ymax = max(sizes)
    pos = []
    step = 0.9 / max(ymax - 1, 1)
    for x, n in zip(xs, sizes):
        ys = 0.5 + ((n - 1) / 2 - np.arange(n)) * step      # equal node spacing in every layer
        pos.append([(x, y) for y in ys])
    mats = [W for W, _ in layers] + [W_out]
    wmax = max(np.abs(W).max() for W in mats) or 1
    for l, W in enumerate(mats):
        for k in range(W.shape[0]):
            for i in range(W.shape[1]):
                v = W[k, i]
                (x0, y0), (x1, y1) = pos[l][i], pos[l + 1][k]
                ax.plot([x0, x1], [y0, y1], color="#1f5f99" if v > 0 else "#b42318",
                        lw=0.3 + 3.2 * abs(v) / wmax, alpha=0.35 + 0.5 * abs(v) / wmax, zorder=1)
    colors = ["#e3edf9"] + ["#fff4d6"] * len(m.layers) + ["#d3f0dc"]
    r = min(0.035, 0.38 / ymax)
    for l, pts in enumerate(pos):
        for n, (x, y) in enumerate(pts):
            last = l == len(pos) - 1
            ax.add_patch(plots_circle((x, y), r, "#a9c8e8" if last and n == m.out and m.n_out > 1 else colors[l]))
            lab = names[l][n] if n < len(names[l]) else ""
            if l == 0:
                ax.text(x - r * 1.6, y, lab, ha="right", va="center", fontsize=9)
            elif last:
                ax.text(x + r * 1.6, y, lab, ha="left", va="center", fontsize=9,
                        fontweight="bold" if n == m.out and m.n_out > 1 else "normal")
    act = m.activation
    titles = ["Input"] + [f"Hidden {i + 1}\n({n} neurons, {act})" for i, n in enumerate(m.layers)] + \
        [f"Output\n({m.n_out}, linear)"]
    for x, t in zip(xs, titles):
        ax.text(x, 1.04, t, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.28, 1.3)
    ax.set_ylim(-0.02, 1.14)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title(f"ANN Architecture {m.arch}  -  teal = positive weight, red = negative, thickness ∝ |weight|",
                 fontsize=10, pad=18)


def plots_circle(center, r, color):
    from matplotlib.patches import Ellipse
    return Ellipse(center, 2 * r, 2 * r * 1.35, facecolor=color, edgecolor="#333", lw=1, zorder=3)


def draw_regression(fig, m):
    axs = fig.subplots(2, 2)
    pred = m.predict(m.coded_used)
    y = m.y_orig
    sets = [("train", m.idx_train), ("val", m.idx_val), ("test", m.idx_test), ("all", list(range(len(y))))]
    for ax, (key, idx) in zip(axs.flat, sets):
        idx = np.array(idx, int)
        ax.set_title(SET_NAMES[key], fontsize=10)
        if not len(idx):
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes, color="#777")
            continue
        yt, yp = y[idx], pred[idx]
        ax.scatter(yt, yp, s=plots.msize(len(yt), 26), color=SET_COLORS[key], edgecolor="none" if len(yt) > 400
                   else "white", zorder=3, rasterized=len(yt) > 2000)
        lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], color="#999", ls=":", lw=1)
        if len(idx) > 1 and np.ptp(yt) > 0:
            a, b = np.polyfit(yt, yp, 1)
            ax.plot([lo, hi], [a * lo + b, a * hi + b], color=SET_COLORS[key], lw=1.5)
        mt = ann.metrics(yt, yp)
        ax.set_title(f"{SET_NAMES[key]}: R = {mt['r']:.4f}" if np.isfinite(mt.get("r", np.nan)) else SET_NAMES[key],
                     fontsize=10)
        ax.set_xlabel("Target (actual)")
        ax.set_ylabel("Output ANN")
        ax.grid(alpha=0.3)


def draw_performance(fig, m):
    ax = fig.add_subplot(111)
    for key in ("train", "val", "test"):
        v = np.array(m.history.get(key, []), float)
        if len(v) and np.isfinite(v).any():
            ax.semilogy(np.arange(len(v)), v, color=SET_COLORS[key], lw=1.8, label=SET_NAMES[key])
    lab = f"Weights used (epoch {m.best_epoch})" + (" - early stopping" if m.early_stop else "")
    ax.axvline(m.best_epoch, color="#333", ls=":", lw=1.2, label=lab)
    if m.early_stop and len(m.history.get("val", [])):
        ax.axvspan(m.best_epoch, len(m.history["val"]) - 1, color="#e11d48", alpha=0.06,
                   label="Validation not improving")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalized scale)" + (", average of all outputs" if m.combined else ""))
    ax.set_title("Training Performance")
    ax.grid(alpha=0.3, which="both")
    ax.legend()


def draw_importance(fig, project, m):
    ax = fig.add_subplot(111)
    imp = sorted(m.importance().items(), key=lambda kv: kv[1])
    ax.barh([project.factor_label(i) for i, _ in imp], [v for _, v in imp], color=plots.BLUE)
    for n, (_, v) in enumerate(imp):
        ax.text(v + 0.5, n, f"{v:.1f}%", va="center")
    ax.set_xlabel("Relative importance (%)")
    ax.set_title("Factor Importance - Garson Method")
    ax.grid(axis="x", alpha=0.3)


def draw_compare(fig, project, j, m):
    fit = project.fit(j) if project.analysis_kind(j) == "ols" else None
    rows = m.rows_used
    y = m.y_orig
    ya = m.predict(project.coded[rows])
    ax1, ax2 = fig.subplots(1, 2)
    lo, hi = min(y.min(), ya.min()), max(y.max(), ya.max())
    if fit is not None:
        yr = fit.predict(project.coded[rows])
        lo, hi = min(lo, yr.min()), max(hi, yr.max())
        ax1.scatter(y, yr, s=plots.msize(len(y), 30), marker="s", color="#e08e0b", rasterized=len(y) > 2000,
                    label=f"RSM (R² {ann.metrics(y, yr)['r2']:.4f})")
        ax2.plot(np.arange(1, len(y) + 1), y - yr, "s-" if len(y) <= 400 else ".", color="#e08e0b",
                 ms=4 if len(y) <= 400 else 2, lw=0.8, label="RSM", rasterized=len(y) > 2000)
    ax1.scatter(y, ya, s=plots.msize(len(y), 30), color=plots.BLUE, rasterized=len(y) > 2000,
                label=f"ANN (R² {ann.metrics(y, ya)['r2']:.4f})")
    ax1.plot([lo, hi], [lo, hi], color="#999", ls=":")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Predicted vs Actual")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(np.arange(1, len(y) + 1), y - ya, "o-" if len(y) <= 400 else ".", color=plots.BLUE,
             ms=4 if len(y) <= 400 else 2, lw=0.8, label="ANN", rasterized=len(y) > 2000)
    ax2.axhline(0, color="#888", lw=0.8)
    ax2.set_xlabel("Run (standard order)")
    ax2.set_ylabel("Residual (actual − predicted)")
    ax2.set_title("Residual")
    ax2.legend()
    ax2.grid(alpha=0.3)


# ------------------------------------------------------------------ data adequacy
def adequacy_html(project, j, a):
    p = project
    lvl = a.get("level", "bad")
    h = [f"<h2>ANN Data Adequacy - {p.response_label(j)}</h2>",
         f"<div class='box'><p style='font-size: 12pt; margin: 2px' class='{lvl}'>{a['verdict']}</p>"
         f"<p class='note' style='margin: 2px'>{a['n']} complete runs"
         + (f", {a.get('n_in', 0)} network inputs (after one-hot)" if a.get("n_in") else "") + ".</p></div>",
         "<h3>Check Results</h3><table><tr><th class='l'>Check</th><th class='l'>Value</th>"
         "<th class='l'>Status</th><th class='l'>Explanation</th></tr>"]
    for c in a["checks"]:
        h.append(f"<tr><td class='l'>{c['name']}</td><td class='l'>{c['value']}</td>"
                 f"<td class='l'><span class='{c['status']}'>{STATUS_LABEL[c['status']]}</span></td>"
                 f"<td class='l'>{c['detail']}</td></tr>")
    h.append("</table>")
    if a["recommend"]:
        h.append("<h3>Recommendations</h3><ul>" + "".join(f"<li>{r}</li>" for r in a["recommend"]) + "</ul>")
    h.append("<p class='note'>Rule of thumb: number of network weights ≤ number of training data (≤ 2× with "
             "trainbr); ideally data ≥ 10× the number of weights. Noise (pure error) is the lower limit of a "
             "reasonable RMSE.</p>")
    return "".join(h)


def draw_adequacy(fig, project, j, a):
    p = project
    if "y" not in a:
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        return
    y = a["y"]
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.1, 1]})
    ax1.hist(y, bins=min(15, max(5, len(y) // 3)), color=theme.PLOT_COLORS[0], alpha=0.75, edgecolor="white")
    ax1.set_xlabel(p.response_label(j))
    ax1.set_ylabel("Frequency")
    ax1.set_title("Response distribution")
    if a.get("noise"):
        ax1.axvspan(np.mean(y) - a["noise"], np.mean(y) + a["noise"], color=theme.PLOT_COLORS[1], alpha=0.15,
                    label="± noise around the mean")
        ax1.legend()
    counts = {"ok": 0, "warn": 0, "bad": 0, "info": 0}
    for c in a["checks"]:
        counts[c["status"]] += 1
    labels = [STATUS_LABEL[k] for k in counts]
    colors = [theme.GOOD, theme.WARN, theme.BAD, "#94a3b8"]
    ax2.bar(labels, list(counts.values()), color=colors)
    for n, v in enumerate(counts.values()):
        ax2.text(n, v + 0.05, str(v), ha="center", va="bottom", fontweight="bold")
    ax2.set_title("Check summary")
    ax2.set_ylabel("Number of checks")
    ax2.set_ylim(0, max(counts.values()) + 1)


# ------------------------------------------------------------------ architecture search
def search_html(project, res, top=None):
    p = project
    js = res.get("resp", [])
    tbl = sorted(res["table"], key=lambda r: r["test"])
    best = res["best"]
    single = res["n_out"] == 1
    scheme = (f"{res['folds']}-fold cross-validation (each fold: 15% of the remaining data for early stopping)"
              if res["scheme"] == "kfold" else "hold-out 70% training / 15% validation / 15% test")
    h = [f"<h3>Automatic Architecture Search - {group_label(p, js) if js else ''}"
         + (" (combined model)" if not single else "") + "</h3>",
         f"<p>{len(res['table'])} architectures evaluated on {res['n']} runs"
         + (f" (sampled from {res['n_all']})" if res.get("n_all", res["n"]) > res["n"] else "")
         + f" with {scheme}. Weight limit: "
         + ("no limit" if not np.isfinite(res["limit"]) else
            f"{res['limit']:.0f} (= {res['ratio']:g} × {res['n_train']} training data)")
         + ". Max hidden layers: " + (str(res["max_layers"]) if res.get("max_layers") else "no limit")
         + ", max neurons per layer: " + (str(res["max_neurons"]) if res.get("max_neurons") else "no limit") + ". "
         f"Algorithm {res['algorithm']}, activation {res['activation']}, normalization "
         f"{ann.NORMS[res['norm']].split(' - ')[0]}."
         + (f" Run in parallel on {res['workers']} CPU cores." if res.get("workers", 1) > 1 else "") + "</p>",
         "<p class='note'><b>Three-level early stopping:</b> (1) each training run stops when the validation error "
         "no longer improves and its best weights are used; (2) neurons are added to each layer step by step, "
         f"stopping when the test error fails to improve by ≥ 1% {res['patience']} times in a row; (3) hidden layers "
         "are only added while they improve the test error by ≥ 1%.</p>",
         "<table><tr><th class='l'>Hidden layers</th><th>Neurons per layer</th><th>Best architecture</th>"
         "<th>Weights</th><th>Training R²</th><th>Test R²</th>" + ("<th>Test RMSE</th>" if single else "")
         + "<th class='l'>Status</th></tr>"]
    for d, r in sorted(res["depth_best"].items()):
        h.append(f"<tr><td class='l'>{d}</td><td>{' / '.join(map(str, r['layers']))}</td><td>{r['arch']}</td>"
                 f"<td>{r['n_weights']}</td><td>{fmt(r['r2_train'], 4)}</td><td>{fmt(r['q2'], 4)}</td>"
                 + (f"<td>{fmt(r['rmse'][0], 4)}</td>" if single else "")
                 + f"<td class='l'><span class='{FIT_CLASS[r['status']]}'>{ann.STATUS[r['status']]}</span></td></tr>")
    rcv = res.get("rsm_cv")
    if rcv:
        h.append(f"<tr><td class='l'>RSM (current model)</td><td>-</td><td>polynomial</td><td>-</td>"
                 f"<td>{fmt(1 - rcv['train'], 4)}</td><td>{fmt(rcv['q2'], 4)}</td>"
                 + (f"<td>{fmt(rcv['rmse'][0], 4)}</td>" if single else "") + "<td></td></tr>")
    h.append("</table>")
    chosen = next(r for r in res["table"] if r["layers"] == best)
    raw = next(r for r in res["table"] if r["layers"] == res["best_raw"])
    h.append(f"<p><b>Selected: {len(best)} hidden layers, neurons {' / '.join(map(str, best))} (architecture "
             f"{chosen['arch']}, {chosen['n_weights']} weights), training R² {fmt(chosen['r2_train'], 4)}, test R² "
             f"{fmt(chosen['q2'], 4)} - {ann.STATUS[chosen['status']]}.</b></p>")
    if best != res["best_raw"]:
        h.append(f"<p class='note'>The architecture with the lowest test error is {raw['arch']} ({raw['n_weights']} "
                 "weights), but the difference is within 1 standard error, so the simpler architecture was chosen "
                 "(1-SE rule) - lower risk of overfitting.</p>")
    n_over = sum(r["status"] == "over" for r in res["table"])
    n_under = sum(r["status"] == "under" for r in res["table"])
    h.append(f"<p class='note'>Only architectures with status <b>Good</b> can be selected. {n_over} overfitting "
             f"architectures (test error far above training error) and {n_under} underfitting (training error also "
             "high) were excluded.</p>")
    for n in res["notes"]:
        h.append(f"<p class='note'>{n}</p>")
    if res.get("cancelled"):
        h.append("<p class='warn'>The search was stopped before finishing; the results above are from the "
                 "architectures evaluated so far.</p>")
    if rcv:
        better = chosen["test"] < rcv["test"]
        h.append(f"<p>{'ANN predicts new data better than RSM' if better else 'RSM predicts new data as well as or better than ANN'} "
                 "(assessed on the same test data).</p>")
        if rcv["q2"] < 0 and res["scheme"] == "kfold":
            h.append("<p class='note'>Negative RSM test R²: small designs (e.g. Box-Behnken/CCD) lose their "
                     "structure when some runs are left out in each fold, so RSM cross-validation is very pessimistic. "
                     "Compare also with RSM Predicted R² on the Analysis page.</p>")
    h.append("<h4>All Architectures (Sorted by Test Error)</h4><table><tr><th>No</th><th>Hidden layers</th>"
             "<th>Neurons per layer</th><th>Weights</th><th>Training R²</th><th>Validation R²</th><th>Test R²</th>"
             "<th>SE</th><th>Best epoch</th><th class='l'>Status</th></tr>")
    for n, r in enumerate(tbl[:top] if top else tbl, 1):
        cls = "model" if r["layers"] == best else ""
        h.append(f"<tr class='{cls}'><td>{n}</td><td>{r['depth']}</td><td>{' / '.join(map(str, r['layers']))}</td>"
                 f"<td>{r['n_weights']}</td><td>{fmt(r['r2_train'], 4)}</td><td>{fmt(1 - r['val'], 4)}</td>"
                 f"<td>{fmt(r['q2'], 4)}</td><td>{fmt(r['se'], 3)}</td><td>{fmt(r['epochs'], 3)}</td>"
                 f"<td class='l'><span class='{FIT_CLASS[r['status']]}'>{ann.STATUS[r['status']]}</span></td></tr>")
    h.append("</table>")
    if top and len(tbl) > top:
        h.append(f"<p class='note'>Showing {top} of {len(tbl)} architectures.</p>")
    return "".join(h)


def draw_learning(fig, project, j, lc):
    """Learning curve: test R2 (median and quartiles across repeats) versus the number of training data."""
    ax = fig.add_subplot(111)
    n = np.array(lc["sizes"])
    r2 = 1 - np.array(lc["ann"])
    lo, hi = 1 - np.array(lc["ann_hi"]), 1 - np.array(lc["ann_lo"])
    ax.fill_between(n, lo, hi, color=plots.BLUE, alpha=0.15, lw=0)
    ax.plot(n, r2, "o-", color=plots.BLUE, lw=2, label="ANN: test R²")
    ax.plot(n, 1 - np.array(lc["ann_train"]), "--", color=plots.BLUE, lw=1, alpha=0.7, label="ANN: training R²")
    rs = np.array(lc["rsm"], float)
    if np.isfinite(rs).any():
        ax.plot(n, 1 - rs, "s-", color="#d9480f", lw=1.5, ms=5, label="RSM: test R²")
    fit = lc.get("fit")
    if fit is not None and lc.get("status") in ("more", "enough"):
        a, b, c, _ = fit
        n_end = max(n[-1], lc.get("n_star_fit") or n[-1]) * 1.1
        xx = np.linspace(n[0], n_end, 200)
        ax.plot(xx, 1 - (a + b * xx ** -c), ":", color="#444", lw=1.2, label="Fitted curve")
        ax.axhline(1 - a, color="#1a7f37", lw=1, ls="--", label=f"Estimated R² upper limit = {1 - a:.3f}")
        if lc.get("n_star_fit") and lc["status"] == "more":
            ax.axvline(lc["n_star_fit"], color="#1a7f37", lw=1, ls=":")
            ax.text(lc["n_star_fit"], ax.get_ylim()[0], f"  {lc['n_star_fit']} training data", color="#1a7f37",
                    fontsize=8, va="bottom")
    ax.axvline(n[-1], color="#888", lw=0.8)
    ymin = np.nanmin(np.concatenate([lo, 1 - rs[np.isfinite(rs)]])) if np.isfinite(rs).any() else np.nanmin(lo)
    ax.set_ylim(max(ymin - 0.05, -0.5), 1.02)
    ax.set_xlabel("Number of training data (runs)")
    ax.set_ylabel("R²")
    ax.set_title(f"Learning curve {project.response_label(j)}: {lc['N']} runs, {lc['n_test']} fixed test runs, "
                 f"{lc['repeats']} repeats (band = quartiles)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")


def learning_html(project, j, lc):
    N = lc["N"]
    h = [f"<h2>Learning Curve - {project.response_label(j)}</h2>",
         f"<p>The ANN (hidden layers {'/'.join(map(str, lc['layers']))} neurons) is trained with 30% to 100% of the "
         "training data, then assessed on "
         f"{lc['n_test']} test runs never used for training. Repeated {lc['repeats']} times with different random "
         "splits. If test R² still rises as data is added, more data would still help.</p>"]
    st = lc.get("status")
    r2_now = 1 - lc["ann"][-1] if np.isfinite(lc["ann"][-1]) else np.nan
    if st == "enough":
        h.append(f"<p class='good'><b>The data is sufficient.</b> The curve has flattened: test R² {fmt(r2_now, 3)} "
                 f"and its upper limit is about {fmt(lc['r2_limit'], 3)}. Adding data is expected to raise test R² by "
                 f"less than {lc['tol_r2']:.2f}. If accuracy is still insufficient, the problem lies in noise or the "
                 "model, not the amount of data.</p>")
    elif st == "more":
        extra = max(lc["total_needed"] - N, 1)
        lo_, hi_ = lc.get("total_range", (lc["total_needed"], lc["total_needed"]))
        h.append(f"<p class='warn'><b>More data would still help.</b> Test R² is now {fmt(r2_now, 3)} and still "
                 f"rising. Estimate: about <b>{lc['total_needed']} runs in total</b> (range {lo_} to {hi_}; add "
                 f"about {extra} runs) for test R² to come within {lc['tol_r2']:.2f} of its upper limit "
                 f"({fmt(lc['r2_limit'], 3)}). This is an extrapolation: add runs gradually (e.g. half first), then "
                 "repeat the learning curve.</p>")
    elif st == "far":
        h.append(f"<p class='bad'><b>The data is still far from sufficient.</b> Test R² {fmt(r2_now, 3)} is still "
                 f"rising sharply; the requirement is estimated at more than 5 times the current data (more than "
                 f"{5 * N} runs). Consider RSM, fewer factors, or narrower factor ranges.</p>")
    elif st == "few":
        h.append(f"<p class='warn'><b>Too little data ({N} runs) for a reliable learning curve.</b> Each point of "
                 "the curve uses only a dozen or so runs, so the results vary widely. Treat it as a rough indication; "
                 "ANN generally needs at least 20 to 30 runs.</p>")
    rs = lc["rsm"][-1]
    if np.isfinite(rs):
        better = "RSM" if rs < lc["ann"][-1] else "ANN"
        h.append(f"<p>With the full data: ANN test R² {fmt(r2_now, 3)}, RSM {fmt(1 - rs, 3)}. {better} is more "
                 "accurate at this data size" + (" (RSM usually wins with little data; ANN needs more data to "
                                                 "catch up)." if better == "RSM" else ".") + "</p>")
    h.append("<table><tr><th>Training data</th><th>ANN test R² (median)</th><th>Quartiles</th><th>ANN training R²</th>"
             "<th>RSM test R²</th></tr>")
    for i, n in enumerate(lc["sizes"]):
        h.append(f"<tr><td>{int(n)}</td><td>{fmt(1 - lc['ann'][i], 3)}</td><td>{fmt(1 - lc['ann_hi'][i], 3)} to "
                 f"{fmt(1 - lc['ann_lo'][i], 3)}</td><td>{fmt(1 - lc['ann_train'][i], 3)}</td>"
                 f"<td>{fmt(1 - lc['rsm'][i], 3)}</td></tr>")
    h.append("</table><p class='note'>To add data: the Data Readiness page suggests new runs "
             "(space-filling fills gaps in the factor space, well suited for ANN).</p>")
    return "".join(h)


def draw_search(fig, res):
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.35, 1]})
    colors = {"good": theme.PLOT_COLORS[0], "over": theme.BAD, "under": theme.WARN}
    tbl = res["table"]
    lo = min(-0.05, min(min(r["q2"], r["r2_train"]) for r in tbl))
    for st in ("good", "under", "over"):
        rows = [r for r in tbl if r["status"] == st]
        if not rows:
            continue
        x = [r["n_weights"] for r in rows]
        ax1.scatter(x, [r["r2_train"] for r in rows], s=34, facecolor="none", edgecolor=colors[st], lw=1.3)
        ax1.scatter(x, [r["q2"] for r in rows], s=34, color=colors[st], label=f"{ann.STATUS[st]}")
        for r in rows:
            ax1.plot([r["n_weights"]] * 2, [r["q2"], r["r2_train"]], color=colors[st], lw=0.7, alpha=0.5)
    chosen = next(r for r in tbl if r["layers"] == res["best"])
    ax1.scatter([chosen["n_weights"]], [chosen["q2"]], s=240, facecolor="none", edgecolor=theme.INK, lw=2, zorder=5,
                label=f"Selected {'/'.join(map(str, chosen['layers']))}")
    ax1.scatter([], [], s=34, facecolor="none", edgecolor="#64748b", label="○ training R²   ● test R²")
    if res.get("rsm_cv"):
        ax1.axhline(res["rsm_cv"]["q2"], color=theme.PLOT_COLORS[2], ls=":", lw=1.5, label="RSM (test R²)")
    ax1.set_ylim(max(lo, -1.0) - 0.02, 1.02)
    ax1.set_xlabel("Number of network weights (complexity)")
    ax1.set_ylabel("R²")
    ax1.set_title("Training vs test: large gap = overfitting, both low = underfitting")
    ax1.grid(alpha=0.7)
    ax1.legend(fontsize=8, loc="best")
    db = sorted(res["depth_best"].items())
    labels = [f"{d} layer{'s' if d > 1 else ''}\n{'/'.join(map(str, r['layers']))}" for d, r in db]
    xs = np.arange(len(db))
    ax2.bar(xs - 0.18, [r["r2_train"] for _, r in db], 0.36, color="#a9c8e8", label="Training R²")
    ax2.bar(xs + 0.18, [r["q2"] for _, r in db], 0.36, color=theme.PLOT_COLORS[0], label="Test R²")
    ax2.set_xticks(xs, labels)
    if res.get("rsm_cv"):
        ax2.axhline(res["rsm_cv"]["q2"], color=theme.PLOT_COLORS[2], ls=":", lw=1.5, label="RSM (test R²)")
    ax2.set_ylim(max(lo, -1.0) - 0.02, 1.3)
    ax2.set_title("Best per number of hidden layers")
    ax2.legend(fontsize=8, loc="upper center", ncol=3)
    ax2.grid(axis="y", alpha=0.7)


class AnnTab(QWidget):
    model_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._loading = False
        self._search = {}
        self._adequacy = {}
        self.task = None

        split = QSplitter(Qt.Horizontal)
        lay = QVBoxLayout(self)
        lay.addWidget(split)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        left = QWidget()
        scroll.setWidget(left)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 6, 0)

        self.cb_resp = QComboBox()
        row = QHBoxLayout()
        row.addWidget(QLabel("Show response:"))
        row.addWidget(self.cb_resp, 1)
        ll.addLayout(row)

        # --- 1. data adequacy
        g = QGroupBox("1. Data Adequacy")
        gl = QVBoxLayout(g)
        self.lbl_verdict = QLabel("-")
        self.lbl_verdict.setWordWrap(True)
        self.lbl_verdict.setMinimumHeight(40)
        self.btn_adequacy = QPushButton("Assess Data Adequacy")
        self.btn_adequacy.setIcon(theme.glyph_icon("eval", theme.ACCENT))
        self.btn_learn = QPushButton("Learning Curve (Enough Data?)")
        self.btn_learn.setToolTip("Train the ANN with 30% to 100% of the data and check whether test accuracy\n"
                                  "still rises. If it does, more data would still help; the application estimates "
                                  "the number of runs needed.")
        gl.addWidget(self.lbl_verdict)
        gl.addWidget(self.btn_adequacy)
        gl.addWidget(self.btn_learn)
        ll.addWidget(g)

        # --- 2. model type
        g = QGroupBox("2. ANN Model")
        gl = QVBoxLayout(g)
        self.cb_mode = QComboBox()
        self.cb_mode.addItem("Separate - one network per response", "separate")
        self.cb_mode.addItem("Combined - one network for several responses", "combined")
        self.cb_mode.setToolTip("Separate: each response has its own network & architecture (usually more accurate).\n"
                                "Combined: one multi-output network; more compact, suitable when the responses are "
                                "related.")
        gl.addWidget(self.cb_mode)
        self.lst_resp = QListWidget()
        self.lst_resp.setMaximumHeight(110)
        self.lbl_group = QLabel("Responses combined in one network:")
        gl.addWidget(self.lbl_group)
        gl.addWidget(self.lst_resp)
        ll.addWidget(g)

        # --- 3. network settings
        g = QGroupBox("3. Network Settings")
        f = QFormLayout(g)
        self.cb_alg = QComboBox()
        for key, label in ann.ALGORITHMS.items():
            self.cb_alg.addItem(label.split(" - ")[0], key)
        self.cb_norm = QComboBox()
        for key, label in ann.NORMS.items():
            self.cb_norm.addItem(label, key)
        self.cb_norm.setToolTip("Brings factors & responses with different ranges to a common scale before training.\n"
                                "Normalization parameters are computed from the training data only.")
        self.cb_act = QComboBox()
        for key, label in ann.ACTIVATIONS.items():
            self.cb_act.addItem(label, key)
        self.chk_es = QCheckBox("Early stopping (use validation data)")
        self.chk_es.setChecked(True)
        self.chk_es.setToolTip("Training stops when the validation error no longer improves; the weights with the\n"
                               "lowest validation error are used. Prevents overfitting.")
        self.ed_hidden = QLineEdit("5")
        self.ed_hidden.setToolTip("Number of neurons in each hidden layer, separated by commas. The count of numbers\n"
                                  "= the number of hidden layers. Example: '10' (1 layer), '20, 10' (2 layers),\n"
                                  "'64, 32, 16, 8' (4 layers). There is no limit on the number of layers or neurons.")
        self.lbl_hidden = QLabel()
        self.lbl_hidden.setStyleSheet(f"color: {theme.MUTED};")
        self.sp_tr, self.sp_va, self.sp_te = QSpinBox(), QSpinBox(), QSpinBox()
        for sp in (self.sp_tr, self.sp_va, self.sp_te):
            sp.setRange(0, 100)
            sp.setSuffix(" %")
        self.sp_restart = QSpinBox()
        self.sp_restart.setRange(1, 100)
        self.sp_restart.setValue(10)
        self.sp_epoch = QSpinBox()
        self.sp_epoch.setRange(10, 5000)
        self.sp_epoch.setValue(500)
        self.sp_fail = QSpinBox()
        self.sp_fail.setRange(1, 100)
        self.sp_fail.setValue(6)
        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 999999)
        self.sp_seed.setValue(1)
        self.sp_cpu = QSpinBox()
        self.sp_cpu.setRange(1, parallel.cpu_count())
        self.sp_cpu.setValue(parallel.default_workers())
        self.sp_cpu.setSuffix(f" of {parallel.cpu_count()}")
        self.sp_cpu.setToolTip("Number of CPU cores for training and architecture search. Restarts, cross-validation\n"
                               "folds and candidate architectures run simultaneously on different cores.\n"
                               "Results are identical to 1 core, just faster. Reduce this if the computer is busy\n"
                               "with other heavy programs.")
        for cb in (self.cb_alg, self.cb_norm, self.cb_act, self.cb_resp, self.cb_mode):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(14)
        f.addRow("Algorithm:", self.cb_alg)
        f.addRow("Normalization:", self.cb_norm)
        f.addRow("Activation function:", self.cb_act)
        f.addRow(self.chk_es)
        f.addRow("Max validation failures:", self.sp_fail)
        f.addRow("Neurons per hidden layer:", self.ed_hidden)
        f.addRow("", self.lbl_hidden)
        for lab, w in (("Training data:", self.sp_tr), ("Validation data:", self.sp_va), ("Test data:", self.sp_te),
                       ("Restarts (initializations):", self.sp_restart), ("Maximum epochs:", self.sp_epoch),
                       ("Random seed:", self.sp_seed), ("CPU cores (parallel):", self.sp_cpu)):
            f.addRow(lab, w)
        self.f = f
        self.btn_train = QPushButton("Train ANN")
        self.btn_train.setObjectName("Primary")
        self.btn_train.setIcon(theme.glyph_icon("play", "white"))
        f.addRow(self.btn_train)
        ll.addWidget(g)

        # --- 4. automatic architecture
        g = QGroupBox("4. Optimize Layers && Neurons (Automatic)")
        f2 = QFormLayout(g)
        self.sp_max_layers = QSpinBox()
        self.sp_max_layers.setRange(0, 1000)
        self.sp_max_layers.setSpecialValueText("no limit")
        self.sp_max_layers.setValue(0)
        self.sp_max_neurons = QSpinBox()
        self.sp_max_neurons.setRange(0, 1000000)
        self.sp_max_neurons.setSpecialValueText("no limit")
        self.sp_max_neurons.setValue(0)
        self.sp_folds = QSpinBox()
        self.sp_folds.setRange(3, 10)
        self.sp_folds.setValue(5)
        self.sp_patience = QSpinBox()
        self.sp_patience.setRange(1, 6)
        self.sp_patience.setValue(2)
        self.sp_patience.setToolTip("Adding neurons to a layer stops when the test error has not improved this\n"
                                    "many times in a row (architecture early stopping).")
        f2.addRow("Maximum hidden layers:", self.sp_max_layers)
        f2.addRow("Maximum neurons per layer:", self.sp_max_neurons)
        f2.addRow("Cross-validation (k-fold):", self.sp_folds)
        f2.addRow("Early stopping patience:", self.sp_patience)
        self.sp_ratio = QDoubleSpinBox()
        self.sp_ratio.setRange(0.0, 1000.0)
        self.sp_ratio.setSingleStep(0.5)
        self.sp_ratio.setDecimals(1)
        self.sp_ratio.setSpecialValueText("no limit")
        self.sp_ratio.setValue(0.0)
        self.sp_ratio.setToolTip("Maximum number of network weights = this value x the number of training data.\n"
                                 "No limit: network size is limited only by early stopping (neurons & layers\n"
                                 "stop being added when test predictions no longer improve).")
        f2.addRow("Weight limit (x training data):", self.sp_ratio)
        self.btn_search = QPushButton("Search Architecture (This Response)")
        self.btn_search.setIcon(theme.glyph_icon("search", theme.ACCENT))
        self.btn_search.setToolTip("Tries 1..N hidden layers with different numbers of neurons in each layer, with\n"
                                   "early stopping; the simplest architecture with status Good is selected (1-SE rule),\n"
                                   "then trained right away.")
        self.btn_all = QPushButton("Optimize All Responses (One Click)")
        self.btn_all.setObjectName("Primary")
        self.btn_all.setIcon(theme.glyph_icon("optim", "white"))
        self.btn_all.setToolTip("Finds the best architecture and trains an ANN for every continuous response, one by\n"
                                "one, without having to select each response.")
        self.chk_use_after = QCheckBox("When finished, use ANN for graphs/optimization")
        self.btn_cancel = QPushButton("Stop")
        self.btn_cancel.setIcon(theme.glyph_icon("stop", theme.BAD))
        self.btn_cancel.setVisible(False)
        f2.addRow(self.btn_search)
        f2.addRow(self.btn_all)
        f2.addRow(self.chk_use_after)
        f2.addRow(self.btn_cancel)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        f2.addRow(self.progress)
        ll.addWidget(g)

        self.lbl_status = QLabel()
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(f"color: {theme.ACCENT}; font-weight: bold;")
        ll.addWidget(self.lbl_status)
        self.chk_use = QCheckBox("Use this response's ANN for graphs, optimization, NSGA-II && prediction")
        ll.addWidget(self.chk_use)
        note = QLabel("Suggested workflow: assess data adequacy → Optimize All Responses (or Search Architecture) → "
                      "check the Fit Diagnosis (good / underfitting / overfitting) and compare with RSM.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.MUTED};")
        ll.addWidget(note)
        ll.addStretch()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(350)
        split.addWidget(scroll)

        self.tabs = QTabWidget()
        self.adq_txt = ReportBrowser()
        self.adq_fig = MplCanvas(figsize=(8, 3.6))
        adq = QSplitter(Qt.Vertical)
        adq.addWidget(self.adq_txt)
        adq.addWidget(self.adq_fig)
        adq.setSizes([520, 260])
        self.txt = ReportBrowser()
        self.srch_txt = ReportBrowser()
        self.srch_fig = MplCanvas(figsize=(9, 4))
        srch = QSplitter(Qt.Vertical)
        srch.addWidget(self.srch_fig)
        srch.addWidget(self.srch_txt)
        srch.setSizes([340, 420])
        self.reg = MplCanvas(figsize=(8, 6))
        self.perf = MplCanvas(figsize=(7, 5))
        self.imp = MplCanvas(figsize=(7, 5))
        self.cmp = MplCanvas(figsize=(9, 5))
        self.wts = ReportBrowser()
        self.net = MplCanvas(figsize=(8, 6))
        self.lc_fig = MplCanvas(figsize=(8, 4))
        self.lc_txt = ReportBrowser()
        lcw = QSplitter(Qt.Vertical)
        lcw.addWidget(self.lc_fig)
        lcw.addWidget(self.lc_txt)
        lcw.setSizes([360, 380])
        self.tab_learn = lcw
        for w, lab in ((adq, "Data Adequacy"), (lcw, "Learning Curve"), (self.txt, "Summary && Diagnosis"),
                       (srch, "Architecture Search"), (self.reg, "Regression Plots"), (self.perf, "Training Performance"),
                       (self.imp, "Factor Importance"), (self.cmp, "RSM vs ANN"), (self.net, "Network Diagram"),
                       (self.wts, "Weights && Biases")):
            self.tabs.addTab(w, lab)
        self.tab_search = srch
        split.addWidget(self.tabs)
        split.setSizes([370, 900])

        self.cb_resp.currentIndexChanged.connect(self.on_resp_changed)
        self.cb_alg.currentIndexChanged.connect(self.on_split_defaults)
        self.chk_es.toggled.connect(self.on_split_defaults)
        self.cb_mode.currentIndexChanged.connect(self.on_mode_changed)
        self.ed_hidden.textChanged.connect(self.on_layers_changed)
        self.cb_alg.currentIndexChanged.connect(self.on_layers_changed)
        self.btn_train.clicked.connect(self.train)
        self.btn_search.clicked.connect(lambda: self.search(False))
        self.btn_all.clicked.connect(lambda: self.search(True))
        self.btn_cancel.clicked.connect(self.cancel_search)
        self.btn_adequacy.clicked.connect(lambda: (self.assess(), self.tabs.setCurrentIndex(0)))
        self.btn_learn.clicked.connect(self.learning)
        self.chk_use.toggled.connect(self.on_use)
        self.on_split_defaults()
        self.on_layers_changed()
        self.on_mode_changed()

    # ---------------------------------------------------------------- controls
    def on_layers_changed(self):
        try:
            lay = parse_hidden(self.ed_hidden.text())
        except ValueError:
            self.lbl_hidden.setText("format: 20, 10, 5")
            return
        text = f"{len(lay)} hidden layer" + ("s" if len(lay) > 1 else "")
        p = self.project
        if p is not None:
            try:
                n_in = ann.encode(p.coded[:1], p.space)[0].shape[1]
                n_out = len(self.checked_responses()) if self.mode() == "combined" else 1
                w = ann.n_weights_for(n_in, lay, max(n_out, 1))
                text += ", " + f"{w:,}".replace(",", ".") + " weights"
                if w > ann.LM_MAX_WEIGHTS and self.cb_alg.currentData() != "adam":
                    text += " (large: trained automatically with Adam)"
            except Exception:  # noqa: BLE001 - informational label only
                pass
        self.lbl_hidden.setText(text)

    def on_split_defaults(self):
        es = self.chk_es.isChecked()
        tr, va, te = (70, 15, 15) if es else (85, 0, 15)
        self.sp_tr.setValue(tr)
        self.sp_va.setValue(va)
        self.sp_te.setValue(te)
        self.f.setRowVisible(self.sp_fail, es)

    def mode(self):
        return self.cb_mode.currentData()

    def on_mode_changed(self):
        comb = self.mode() == "combined"
        self.lst_resp.setVisible(comb)
        self.lbl_group.setVisible(comb)
        self.btn_all.setVisible(not comb)
        self.btn_train.setText("Train Combined ANN" if comb else "Train ANN (This Response)")
        self.btn_search.setText("Search Architecture (Combined Model)" if comb else "Search Architecture (This Response)")

    def fill_resp_list(self, checked=None):
        p = self.project
        self.lst_resp.clear()
        if p is None:
            return
        ok = p.ann_responses()
        for j in range(len(p.responses)):
            it = QListWidgetItem(p.response_label(j))
            it.setData(Qt.UserRole, j)
            if j in ok:
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if (checked is None or j in checked) else Qt.Unchecked)
            else:
                it.setFlags(Qt.NoItemFlags)
                it.setText(it.text() + " (not continuous)")
            self.lst_resp.addItem(it)

    def checked_responses(self):
        return [self.lst_resp.item(r).data(Qt.UserRole) for r in range(self.lst_resp.count())
                if self.lst_resp.item(r).checkState() == Qt.Checked]

    def set_project(self, project):
        self.cancel_search(wait=True)
        self.project = project
        self._search = {}
        self._adequacy = {}
        self._learning = project.learning if isinstance(getattr(project, "learning", None), dict) else {}
        self._loading = True
        self.cb_resp.clear()
        for j in range(len(project.responses)):
            self.cb_resp.addItem(project.response_label(j))
        self.fill_resp_list()
        self._loading = False
        self.on_resp_changed()

    def refresh(self):
        if self.project is not None:
            self._adequacy = {}
            self.on_resp_changed()

    def refresh_responses(self):
        self.set_project(self.project)

    def resp(self):
        return self.cb_resp.currentIndex()

    def on_resp_changed(self):
        if self._loading or self.project is None or self.resp() < 0:
            return
        self.assess()
        self.show_search()
        self.show_learning()
        self.show_model()

    def _kwargs(self):
        tot = self.sp_tr.value() + self.sp_va.value() + self.sp_te.value()
        if tot != 100:
            raise ValueError("Training + validation + test percentages must add up to 100%.")
        es = self.chk_es.isChecked()
        if es and self.sp_va.value() == 0:
            raise ValueError("Early stopping requires validation data (> 0%).")
        return {"hidden": parse_hidden(self.ed_hidden.text()),
                "activation": self.cb_act.currentData(),
                "split": (self.sp_tr.value() / 100, self.sp_va.value() / 100, self.sp_te.value() / 100),
                "seed": self.sp_seed.value(), "restarts": self.sp_restart.value(),
                "max_epochs": self.sp_epoch.value(), "max_fail": self.sp_fail.value(),
                "algorithm": self.cb_alg.currentData(), "norm": self.cb_norm.currentData(), "early_stop": es,
                "n_jobs": self.sp_cpu.value()}

    def targets(self, all_responses=False):
        """List of response groups to train: [[j]] (separate) or [[j1, j2, ...]] (combined)."""
        p = self.project
        if self.mode() == "combined":
            js = [j for j in self.checked_responses() if j in p.ann_responses()]
            if len(js) < 2:
                raise ValueError("A combined model requires at least 2 checked continuous responses.")
            return [js]
        if all_responses:
            js = p.ann_responses()
            if not js:
                raise ValueError("No continuous responses available for ANN.")
            return [[j] for j in js]
        j = self.resp()
        if p.response_family(j) != "normal":
            raise ValueError("ANN here is for continuous (normal) responses. For binomial/Poisson responses use "
                             "logistic/Poisson regression on the Analysis page.")
        return [[j]]

    def _restarts_for(self, kw):
        n = len(ann.complete_rows(self.project.coded, self.project.data[:, self.project.ann_responses() or [0]]))
        cap = max(3, kw.get("n_jobs", 1))           # parallel restarts: up to the number of CPU cores adds no time
        if n > BIG_DATA and kw["restarts"] > cap:
            return cap, f"large data ({n} runs): restarts limited to {cap}"
        return kw["restarts"], ""

    # ---------------------------------------------------------------- learning curve
    def learning(self):
        p, j = self.project, self.resp()
        if j not in p.ann_responses():
            QMessageBox.information(self, "ANN", "The learning curve is only available for continuous responses.")
            return
        m = p.ann.get(j)
        if m is not None:
            layers, act, alg, norm = list(m.layers), m.activation, m.algorithm, getattr(m, "norm", "mapminmax")
        else:
            try:
                layers = parse_hidden(self.ed_hidden.text())
            except ValueError as exc:
                QMessageBox.warning(self, "ANN", str(exc))
                return
            act, alg, norm = self.cb_act.currentData(), self.cb_alg.currentData(), self.cb_norm.currentData()
        if alg == "adam" and m is not None and getattr(m, "auto_adam", False):
            alg = "trainbr"
        coded, y, space = p.coded.copy(), p.model_data[:, j].copy(), p.space
        rsm = p.rsm_eval(j)
        n_jobs = self.sp_cpu.value()
        seed = self.sp_seed.value()
        label = p.response_label(j)

        def job(report, cancelled):
            return ann.learning_curve(coded, y, space, layers, act, alg, norm, seed=seed, rsm_eval=rsm,
                                      n_jobs=n_jobs, cancel=cancelled,
                                      progress=lambda i, n: report(f"Learning curve {label}: training {i}/{n}", i, n))

        def done(lc):
            self._learning[j] = lc
            p.learning = self._learning
            self.show_learning()
            self.tabs.setCurrentWidget(self.tab_learn)
            self.lbl_status.setText(f"Learning curve finished: {LEARN_STATUS.get(lc['status'], '')}")

        self._start(job, done)

    def show_learning(self):
        j = self.resp()
        lc = self._learning.get(j)
        if lc is None:
            self.lc_txt.setHtml(REPORT_CSS + "<h2>Learning Curve</h2><p>Not run yet. Click <b>Learning Curve</b> "
                                "to find out whether the data is sufficient for ANN, or how many more runs are "
                                "needed. Architecture used: this response's ANN model (if already trained) or "
                                "the neuron entry in Network Settings.</p>")
            self.lc_fig.message("No learning curve yet")
            return
        self.lc_txt.setHtml(REPORT_CSS + learning_html(self.project, j, lc))
        self.lc_fig.figure.clear()
        draw_learning(self.lc_fig.figure, self.project, j, lc)
        self.lc_fig.draw()

    # ---------------------------------------------------------------- adequacy
    def assess(self):
        p, j = self.project, self.resp()
        try:
            a = adequacy.assess(p, j, split_train=max(self.sp_tr.value() + self.sp_va.value(), 50) / 100)
        except Exception as exc:  # noqa: BLE001
            self.adq_txt.setHtml(REPORT_CSS + f"<p class='bad'>Assessment failed: {exc}</p>")
            return
        self._adequacy[j] = a
        color = {"ok": theme.GOOD, "warn": theme.WARN, "bad": theme.BAD}.get(a["level"], theme.MUTED)
        self.lbl_verdict.setText(a["verdict"])
        self.lbl_verdict.setStyleSheet(f"color: {color}; font-weight: 600;")
        self.adq_txt.setHtml(REPORT_CSS + adequacy_html(p, j, a))
        self.adq_fig.figure.clear()
        draw_adequacy(self.adq_fig.figure, p, j, a)
        self.adq_fig.draw()

    # ---------------------------------------------------------------- background work
    def _set_busy(self, busy):
        for w in (self.btn_train, self.btn_search, self.btn_all, self.btn_adequacy, self.btn_learn, self.cb_resp,
                  self.cb_mode,
                  self.lst_resp, self.ed_hidden, self.cb_alg, self.cb_norm, self.cb_act, self.chk_es,
                  self.sp_max_layers, self.sp_max_neurons, self.sp_folds, self.sp_patience, self.sp_cpu):
            w.setEnabled(not busy)
        self.btn_cancel.setVisible(busy)
        self.progress.setVisible(busy)

    def _start(self, job, on_done):
        proj = self.project
        self.task = Task(job, self)
        self.task.progressed.connect(self._on_progress)
        self.task.done.connect(lambda res: self._finish(res, on_done, proj))
        self.task.failed.connect(self._on_failed)
        self._set_busy(True)
        self.progress.setRange(0, 0)
        self.task.start()

    def _on_progress(self, text, i, n):
        self.lbl_status.setText(text)
        if n:
            self.progress.setRange(0, n)
            self.progress.setValue(i)
        else:
            self.progress.setRange(0, 0)

    def _finish(self, res, on_done, proj):
        self._set_busy(False)
        self.task = None
        if self.project is not proj:
            self.lbl_status.setText("")
            return                                  # the project changed while the task was running
        on_done(res)

    def _on_failed(self, msg):
        self._set_busy(False)
        self.task = None
        self.lbl_status.setText("")
        QMessageBox.warning(self, "ANN", msg)

    def cancel_search(self, wait=False):
        if self.task is not None and self.task.isRunning():
            self.lbl_status.setText("Stopping...")
            self.task.cancel()
            if wait:
                self.task.wait(30000)

    # ---------------------------------------------------------------- training
    def train(self):
        p = self.project
        try:
            kw = self._kwargs()
            groups = self.targets()
        except ValueError as exc:
            QMessageBox.warning(self, "ANN", str(exc))
            return
        kw["restarts"], note = self._restarts_for(kw)
        js = groups[0]

        def job(report, cancelled):
            return p.build_ann(js, **kw, cancel=cancelled,
                               progress=lambda i, n: report(f"Training {group_label(p, js)}: restart {i + 1}/{n}"
                                                            + (f" ({note})" if note else ""), i, n))

        def done(m):
            p.set_ann(m)
            self.lbl_status.setText(f"Done: {m.arch}, {ann.STATUS[m.diagnosis['status']]}")
            self.show_model()
            self.tabs.setCurrentWidget(self.txt)
            self.model_changed.emit()

        self._start(job, done)

    # ---------------------------------------------------------------- architecture search
    def search(self, all_responses=False):
        p = self.project
        try:
            kw = self._kwargs()
            groups = self.targets(all_responses)
        except ValueError as exc:
            QMessageBox.warning(self, "ANN", str(exc))
            return
        kw["restarts"], note = self._restarts_for(kw)
        opts = {"max_layers": self.sp_max_layers.value() or None, "max_neurons": self.sp_max_neurons.value() or None,
                "activation": kw["activation"], "algorithm": kw["algorithm"], "folds": self.sp_folds.value(),
                "restarts": 3, "max_epochs": 300, "seed": kw["seed"], "norm": kw["norm"],
                "patience": self.sp_patience.value(), "max_fail": kw["max_fail"],
                "weight_ratio": self.sp_ratio.value() or float("inf"), "n_jobs": kw["n_jobs"]}
        coded, data, space = p.coded.copy(), p.model_data.copy(), p.space
        noise = {j: None for j in range(len(p.responses))}
        final_kw = {**kw, "early_stop": True if kw["split"][1] > 0 else kw["early_stop"]}

        def job(report, cancelled):
            def one(g, js):
                head = (f"[{g + 1}/{len(groups)}] " if len(groups) > 1 else "") + group_label(p, js)

                def prog(count, depth, arch):
                    report(f"{head} - {depth} hidden layer{'s' if depth > 1 else ''}: architecture {arch} "
                           f"(candidate {count})", 0, 0)

                res = ann.architecture_search(coded, data[:, js], space, progress=prog, cancel=cancelled,
                                              noise=[noise[j] for j in js], **opts)
                res["resp"] = list(js)
                report(f"{head} - training the selected architecture {'/'.join(map(str, res['best']))}"
                       + (f" ({note})" if note else ""), 0, 0)
                m = ann.train(coded, data[:, js], space, **{**final_kw, "hidden": res["best"]}, resp_idx=js,
                              noise=[noise[j] for j in js], cancel=cancelled)
                ch = next(r for r in res["table"] if r["layers"] == res["best"])
                m.cv_diag = {"status": ch["status"], "r2_train": ch["r2_train"], "r2_val": 1 - ch["val"],
                             "q2": ch["q2"], "n_splits": res["folds"],
                             "scheme": f"{res['folds']}-fold cross-validation" if res["scheme"] == "kfold"
                             else "hold-out"}
                return js, res, m

            # several separate responses: searched simultaneously, each sharing its work across the CPU core pool
            n_par = min(len(groups), max(1, kw["n_jobs"] // 8))
            if n_par > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(n_par) as ex:
                    futs = [ex.submit(one, g, js) for g, js in enumerate(groups)]
                    out = []
                    for f in futs:
                        try:
                            out.append(f.result())
                        except ValueError:
                            if not cancelled():         # stopped before this response finished: skip
                                raise
            else:
                out = []
                for g, js in enumerate(groups):
                    out.append(one(g, js))
                    if cancelled():
                        break
            return out

        def done(results):
            lines = []
            for js, res, m in results:
                res["rsm_cv"] = p.rsm_cv(js, res["rows"], res["splits"])
                for j in js:
                    self._search[j] = res
                p.set_ann(m)
                if self.chk_use_after.isChecked():
                    for j in js:
                        p.model_source[j] = "ann"
                lines.append(f"{group_label(p, js)}: {len(res['best'])} layer{'s' if len(res['best']) > 1 else ''} "
                             f"({'/'.join(map(str, res['best']))}) - "
                             f"{ann.STATUS[fit_status(m)]}")
            self.lbl_status.setText("Done. " + "; ".join(lines))
            self.show_search()
            self.show_model()
            self.tabs.setCurrentWidget(self.tab_search)
            self.model_changed.emit()

        self._start(job, done)

    def show_search(self):
        j = self.resp()
        res = self._search.get(j)
        if res is None:
            self.srch_txt.setHtml(REPORT_CSS + "<h2>Architecture Search</h2><p>Not run yet. Click <b>Search "
                                  "Architecture</b> for this response, or <b>Optimize All Responses</b> for all "
                                  "responses at once. The application adds neurons and hidden layers step by step (the "
                                  "number of neurons may differ per layer) and stops on its own when additions no "
                                  "longer improve test predictions (early stopping), avoiding both underfitting and "
                                  "overfitting.</p>")
            self.srch_fig.message("No search results yet")
            return
        self.srch_txt.setHtml(REPORT_CSS + "<h2>Architecture Search - " + self.project.response_label(j) + "</h2>"
                              + search_html(self.project, res))
        self.srch_fig.figure.clear()
        draw_search(self.srch_fig.figure, res)
        self.srch_fig.draw()

    # ---------------------------------------------------------------- model display
    def on_use(self, on):
        if self._loading or self.project is None:
            return
        self.project.model_source[self.resp()] = "ann" if on else "rsm"
        self.project.dirty = True
        self.model_changed.emit()

    def show_model(self):
        p = self.project
        if p is None or self._loading or self.resp() < 0:
            return
        j = self.resp()
        m = p.ann.get(j)
        self._loading = True
        self.chk_use.setEnabled(m is not None)
        self.chk_use.setChecked(p.source(j) == "ann")
        self._loading = False
        if m is None:
            msg = "No ANN model for this response yet. Set the parameters and click <b>Train ANN</b>, or let the " \
                  "application choose layers and neurons with <b>Search Architecture</b> / " \
                  "<b>Optimize All Responses</b>."
            self.txt.setHtml(REPORT_CSS + f"<h2>ANN - {p.response_label(j)}</h2><p>{msg}</p>")
            self.wts.setHtml("")
            for c in (self.reg, self.perf, self.imp, self.cmp, self.net):
                c.message("No ANN model yet")
            return
        self._loading = True
        self.ed_hidden.setText(", ".join(str(v) for v in m.layers))
        i = self.cb_norm.findData(getattr(m, "norm", "mapminmax"))
        if i >= 0:
            self.cb_norm.setCurrentIndex(i)
        if m.combined:
            self.cb_mode.setCurrentIndex(self.cb_mode.findData("combined"))
            self.fill_resp_list(m.resp_idx)
        self._loading = False
        self.on_layers_changed()
        self.txt.setHtml(REPORT_CSS + ann_summary_html(p, j, m))
        self.wts.setHtml(REPORT_CSS + weights_html(p, m))
        for canvas, fn in ((self.reg, lambda fig: draw_regression(fig, m)),
                           (self.perf, lambda fig: draw_performance(fig, m)),
                           (self.imp, lambda fig: draw_importance(fig, p, m)),
                           (self.cmp, lambda fig: draw_compare(fig, p, j, m)),
                           (self.net, lambda fig: draw_network(fig, p, m))):
            canvas.figure.clear()
            fn(canvas.figure)
            canvas.draw()
