"""Information pages: Project Summary, Notes, Column Graphs (data exploration), and the Coefficient Table of all
responses."""
import datetime

import matplotlib
import numpy as np
from matplotlib import colors as mcolors
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit,
                               QPushButton, QSplitter, QVBoxLayout, QWidget)
from scipy import stats

from doe import designs, models

from . import plots, theme
from .common import REPORT_CSS, MplCanvas, fmt
from .copying import ReportBrowser


# ================================================================== Summary
def summary_html(p):
    name = p.path.replace("\\", "/").split("/")[-1] if p.path else "Untitled project"
    data = p.model_data
    complete = int(np.all(np.isfinite(p.coded), axis=1).sum())
    h = [f"<h2>Project Summary - {name}</h2>",
         f"<div class='box'><p style='margin: 2px'><b>{p.design_description()}</b></p>"
         f"<p class='note' style='margin: 2px'>{p.n} runs ({complete} with complete factors"
         + (f", {len(p.excluded)} excluded from the analysis" if p.excluded else "") + ")"
         + (f" · {p.n_blocks} blocks" if p.n_blocks > 1 else "")
         + (f" · split-plot, {int(p.groups.max()) + 1} whole plots" if p.is_split_plot else "")
         + (f" · mixture total {fmt(p.mixture_total)}" if p.is_mixture else "") + "</p></div>"]
    if p.constraints:
        h.append("<p><b>Constraints:</b> " + "; ".join(c["text"] for c in p.constraints) + "</p>")
    act = p.actual
    h.append("<h3>Factors</h3><table><tr><th class='l'>Factor</th><th class='l'>Name</th><th class='l'>Units</th>"
             "<th class='l'>Type</th><th>Low</th><th>High</th><th>Data min</th><th>Data max</th>"
             "<th>Mean</th><th>SD</th><th>Unique levels</th><th>SD / u (POE)</th></tr>")
    for i, f in enumerate(p.factors):
        v = act[:, i]
        v = v[np.isfinite(v)]
        kind = "Categoric" if f.categoric else ("Component" if i in p.comp_idx else "Numeric")
        if f.categoric:
            cnt = np.bincount(np.rint(v).astype(int), minlength=len(f.levels)) if len(v) else []
            h.append(f"<tr><td class='l'>{designs.LETTERS[i]}</td><td class='l'>{f.name}</td><td class='l'>{f.unit}</td>"
                     f"<td class='l'>{kind}</td><td colspan='6' class='l'>"
                     + ", ".join(f"{lv} ({c})" for lv, c in zip(f.levels, cnt)) + f"</td><td>{len(f.levels)}</td>"
                     "<td></td></tr>")
            continue
        h.append(f"<tr><td class='l'>{designs.LETTERS[i]}</td><td class='l'>{f.name}</td><td class='l'>{f.unit}</td>"
                 f"<td class='l'>{kind}</td><td>{fmt(f.low)}</td><td>{fmt(f.high)}</td>"
                 + (f"<td>{fmt(v.min())}</td><td>{fmt(v.max())}</td><td>{fmt(v.mean())}</td>"
                    f"<td>{fmt(v.std(ddof=1)) if len(v) > 1 else '-'}</td><td>{len(np.unique(np.round(v, 8)))}</td>"
                    if len(v) else "<td colspan='5'>-</td>")
                 + f"<td>{fmt(f.sd) if f.sd else ''}</td></tr>")
    h.append("</table>")
    h.append("<h3>Responses</h3><table><tr><th class='l'>Response</th><th class='l'>Type</th><th>n</th><th>Min</th>"
             "<th>Max</th><th>Mean</th><th>SD</th><th>Max/Min</th><th class='l'>Model</th>"
             "<th class='l'>Transform</th><th>R²</th><th>Adj R²</th><th>Pred R²</th><th class='l'>Used</th>"
             "</tr>")
    for j, r in enumerate(p.responses):
        y = data[:, j]
        y = y[np.isfinite(y)]
        spec = p.model_spec(j)
        fit = p.fit(j)
        st = fit.stats if fit is not None else {}
        r2 = st.get("r2", st.get("mcfadden", st.get("r2_cond", np.nan)))
        src = "ANN " + p.ann[j].arch if p.source(j) == "ann" else "RSM"
        ratio = y.max() / y.min() if len(y) and y.min() > 0 else np.nan
        h.append(f"<tr><td class='l'>{p.response_label(j)}</td><td class='l'>{p.response_family(j)}</td><td>{len(y)}</td>"
                 + (f"<td>{fmt(y.min())}</td><td>{fmt(y.max())}</td><td>{fmt(y.mean())}</td>"
                    f"<td>{fmt(y.std(ddof=1)) if len(y) > 1 else '-'}</td><td>{fmt(ratio, 3)}</td>" if len(y)
                    else "<td colspan='5'>-</td>")
                 + f"<td class='l'>{models.order_label(spec['order'])}{' (selected terms)' if spec['terms'] else ''}</td>"
                 f"<td class='l'>{models.transform_label(spec.get('transform'))}</td><td>{fmt(r2, 4)}</td>"
                 f"<td>{fmt(st.get('adj_r2', np.nan), 4)}</td><td>{fmt(st.get('pred_r2', np.nan), 4)}</td>"
                 f"<td class='l'>{src}</td></tr>")
    h.append("</table>")
    extra = []
    if p.ann:
        extra.append(f"{len({m.gid for m in p.ann.values()})} ANN models")
    if p.solutions:
        extra.append(f"desirability optimization results ({len(p.solutions)} solutions)")
    if p.nsga_result:
        extra.append(f"NSGA-II results ({len(p.nsga_result['X'])} Pareto solutions)")
    if extra:
        h.append("<p><b>Other project contents:</b> " + ", ".join(extra) + ".</p>")
    if (p.notes or "").strip():
        h.append("<h3>Notes</h3><div class='box'><pre style='white-space: pre-wrap'>"
                 + p.notes.replace("&", "&amp;").replace("<", "&lt;") + "</pre></div>")
    return "".join(h)


class SummaryTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        lay = QVBoxLayout(self)
        self.txt = ReportBrowser()
        lay.addWidget(self.txt)

    def set_project(self, project):
        self.project = project
        self.refresh()

    def refresh(self):
        if self.project is not None:
            self.txt.setHtml(REPORT_CSS + summary_html(self.project))


# ================================================================== Notes
class NotesTab(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        lbl = QLabel("Free-form project notes: experiment goals, equipment and material conditions, events during the "
                     "experiment, analysis decisions. Saved in the .doe file and included in the PDF report.")
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {theme.MUTED};")
        bar.addWidget(lbl, 1)
        self.btn_date = QPushButton("Insert Date")
        self.btn_date.setIcon(theme.glyph_icon("add", theme.ACCENT))
        bar.addWidget(self.btn_date)
        lay.addLayout(bar)
        self.edit = QPlainTextEdit()
        self.edit.setStyleSheet("font-size: 10.5pt;")
        self.edit.setPlaceholderText("Write notes here...")
        lay.addWidget(self.edit, 1)
        self._loading = False
        self.edit.textChanged.connect(self.on_text)
        self.btn_date.clicked.connect(lambda: self.edit.insertPlainText(
            datetime.datetime.now().strftime("[%d-%m-%Y %H:%M] ")))

    def set_project(self, project):
        self.project = project
        self._loading = True
        self.edit.setPlainText(project.notes or "")
        self._loading = False

    def on_text(self):
        if self._loading or self.project is None:
            return
        self.project.notes = self.edit.toPlainText()
        self.project.dirty = True
        self.changed.emit()


# ================================================================== Column Graphs
def column_table(p):
    """-> list of (name, value per run (standard order), category labels or None)."""
    cols = [("Std Order", np.arange(1, p.n + 1, dtype=float), None), ("Run", p.run_order.astype(float), None)]
    if p.n_blocks > 1:
        cols.append(("Block", p.blocks.astype(float), None))
    act = p.actual
    for i, f in enumerate(p.factors):
        cols.append((p.factor_label(i), act[:, i], f.levels if f.categoric else None))
    data = p.model_data
    for j in range(len(p.responses)):
        cols.append((p.response_label(j), data[:, j], None))
    for j in range(len(p.responses)):
        fit = p.fit(j) if p.analysis_kind(j) == "ols" else None
        if fit is not None:
            res = np.full(p.n, np.nan)
            res[fit.rows_used] = data[fit.rows_used, j] - fit.predict(p.coded[fit.rows_used])
            cols.append((f"Residual {p.response_label(j).split(':')[0]}", res, None))
    return cols


def corr_matrix(cols):
    k = len(cols)
    R = np.full((k, k), np.nan)
    for a in range(k):
        for b in range(a, k):
            x, y = cols[a][1], cols[b][1]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 2 and np.ptp(x[ok]) > 0 and np.ptp(y[ok]) > 0:
                R[a, b] = R[b, a] = np.corrcoef(x[ok], y[ok])[0, 1]
    return R


class GraphColumnsTab(QWidget):
    KINDS = [("scatter", "Scatterplot"), ("hist", "Histogram"), ("box", "Box plot"), ("matrix", "Scatter matrix")]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.cols = []
        self.R = None
        self._loading = False
        split = QSplitter(Qt.Horizontal)
        QVBoxLayout(self).addWidget(split)
        self.canvas = MplCanvas(figsize=(8, 6))
        split.addWidget(self.canvas)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(6, 0, 0, 0)
        g = QGroupBox("Column Graphs")
        f = QFormLayout(g)
        self.cb_kind = QComboBox()
        for k, lab in self.KINDS:
            self.cb_kind.addItem(lab, k)
        self.cb_color, self.cb_x, self.cb_y, self.cb_z = QComboBox(), QComboBox(), QComboBox(), QComboBox()
        for cb in (self.cb_color, self.cb_x, self.cb_y, self.cb_z):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(16)
        self.chk_trend = QCheckBox("Trend line (linear regression)")
        f.addRow("Type:", self.cb_kind)
        f.addRow("Color by:", self.cb_color)
        f.addRow("X axis:", self.cb_x)
        f.addRow("Y axis:", self.cb_y)
        f.addRow("Z axis (3D):", self.cb_z)
        f.addRow(self.chk_trend)
        self.form = f
        sl.addWidget(g)
        self.lbl_corr = QLabel()
        self.lbl_corr.setStyleSheet("font-weight: 600;")
        self.lbl_corr.setWordWrap(True)
        sl.addWidget(self.lbl_corr)
        hint = QLabel("Correlation heatmap of all columns - click a cell to plot that pair of columns:")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.MUTED};")
        sl.addWidget(hint)
        self.heat = MplCanvas(figsize=(4, 4), toolbar=False)
        sl.addWidget(self.heat, 1)
        split.addWidget(side)
        split.setSizes([860, 400])
        for w in (self.cb_kind, self.cb_color, self.cb_x, self.cb_y, self.cb_z):
            w.currentIndexChanged.connect(self.draw)
        self.chk_trend.toggled.connect(self.draw)
        self.heat.canvas.mpl_connect("button_press_event", self.on_heat_click)

    def set_project(self, project):
        self.project = project
        self.refresh()

    def refresh(self):
        p = self.project
        if p is None:
            return
        self.cols = column_table(p)
        names = [c[0] for c in self.cols]
        self._loading = True
        prev = {cb: cb.currentText() for cb in (self.cb_color, self.cb_x, self.cb_y, self.cb_z)}
        for cb, extra in ((self.cb_color, "None"), (self.cb_x, None), (self.cb_y, None), (self.cb_z, "None")):
            cb.clear()
            if extra:
                cb.addItem(extra, -1)
            for n, name in enumerate(names):
                cb.addItem(name, n)
        defaults = {self.cb_color: "Run", self.cb_x: p.factor_label(0),
                    self.cb_y: p.response_label(0) if p.responses else "Run", self.cb_z: "None"}
        for cb in (self.cb_color, self.cb_x, self.cb_y, self.cb_z):
            k = cb.findText(prev[cb]) if prev[cb] else -1
            cb.setCurrentIndex(k if k >= 0 else max(cb.findText(defaults[cb]), 0))
        self._loading = False
        self.R = corr_matrix(self.cols)
        self.draw_heat()
        self.draw()

    def draw_heat(self):
        fig = self.heat.figure
        fig.clear()
        ax = fig.add_subplot(111)
        short = {"Std Order": "Std", "Run": "Run", "Block": "Block"}
        names = [short.get(c[0]) or (c[0].split(":")[0] if not c[0].startswith("Residual")
                                     else "Res " + c[0].split()[-1]) for c in self.cols]
        ax.imshow(np.nan_to_num(self.R), cmap="bwr", vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.grid(False)
        self.heat_ax = ax
        self.heat.draw()

    def on_heat_click(self, event):
        if event.inaxes is not getattr(self, "heat_ax", None) or event.xdata is None:
            return
        b, a = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= a < len(self.cols) and 0 <= b < len(self.cols):
            self._loading = True
            self.cb_kind.setCurrentIndex(0)
            self.cb_x.setCurrentIndex(self.cb_x.findData(b))
            self.cb_y.setCurrentIndex(self.cb_y.findData(a))
            self.cb_z.setCurrentIndex(0)
            self._loading = False
            self.draw()

    def draw(self):
        if self._loading or self.project is None or not self.cols:
            return
        kind = self.cb_kind.currentData()
        for w, vis in ((self.cb_y, kind in ("scatter", "box")), (self.cb_z, kind == "scatter"),
                       (self.chk_trend, kind == "scatter"), (self.cb_x, kind != "matrix")):
            self.form.setRowVisible(w, vis)
        fig = self.canvas.figure
        fig.clear()
        xi, yi, zi, ci = (self.cb_x.currentData(), self.cb_y.currentData(), self.cb_z.currentData(),
                          self.cb_color.currentData())
        getattr(self, f"_draw_{kind}")(fig, xi, yi, zi, ci)
        self.canvas.draw()

    def _colors(self, ci, mask):
        if ci is None or ci < 0:
            return theme.PLOT_COLORS[0], None
        name, v, levels = self.cols[ci]
        v = v[mask]
        if levels is not None:
            cmap = matplotlib.colormaps["tab10"]
            return [cmap(int(x) % 10) if np.isfinite(x) else "#cbd5e1" for x in v], ("cat", levels, name)
        ok = np.isfinite(v)
        lo, hi = (np.nanmin(v), np.nanmax(v)) if ok.any() else (0, 1)
        norm = mcolors.Normalize(lo, hi if hi > lo else lo + 1)
        cmap = matplotlib.colormaps["jet"]
        return cmap(norm(np.nan_to_num(v, nan=lo))), ("num", cmap, norm, name)

    def _legend(self, fig, ax, info):
        if info is None:
            return
        if info[0] == "num":
            cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=info[2], cmap=info[1]), ax=ax, pad=0.01)
            cb.set_label(info[3])
        else:
            cmap = matplotlib.colormaps["tab10"]
            for n, lv in enumerate(info[1]):
                ax.scatter([], [], color=cmap(n % 10), label=lv)
            ax.legend(title=info[2], fontsize=8)

    def _axis(self, ax, which, idx):
        name, v, levels = self.cols[idx]
        getattr(ax, f"set_{which}label")(name)
        if levels is not None:
            getattr(ax, f"set_{which}ticks")(range(len(levels)), levels)

    def _draw_scatter(self, fig, xi, yi, zi, ci):
        x, y = self.cols[xi][1], self.cols[yi][1]
        z = self.cols[zi][1] if zi is not None and zi >= 0 else None
        ok = np.isfinite(x) & np.isfinite(y) & (np.isfinite(z) if z is not None else True)
        col, info = self._colors(ci, ok)
        n = int(ok.sum())
        if z is not None:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x[ok], y[ok], z[ok], c=col if not isinstance(col, str) else None,
                       color=col if isinstance(col, str) else None, s=plots.msize(n, 20), depthshade=False)
            self._axis(ax, "x", xi)
            self._axis(ax, "y", yi)
            ax.set_zlabel(self.cols[zi][0])
        else:
            ax = fig.add_subplot(111)
            ax.scatter(x[ok], y[ok], c=col if not isinstance(col, str) else None,
                       color=col if isinstance(col, str) else None, s=plots.msize(n, 26),
                       edgecolor="none" if n > 400 else "#333", linewidths=0.4, rasterized=n > 2000)
            if self.chk_trend.isChecked() and n > 2 and np.ptp(x[ok]) > 0:
                a, b = np.polyfit(x[ok], y[ok], 1)
                xx = np.linspace(x[ok].min(), x[ok].max(), 50)
                ax.plot(xx, a * xx + b, color=theme.BAD, lw=1.5, label=f"y = {a:.4g}·x + {b:.4g}")
                ax.legend(fontsize=8, loc="upper left")
            self._axis(ax, "x", xi)
            self._axis(ax, "y", yi)
            plots.fit_limits(ax, x[ok], y[ok])
            ax.grid(alpha=0.35)
        self._legend(fig, ax, info)
        r = self.R[xi, yi]
        rho = stats.spearmanr(x[ok], y[ok])[0] if n > 2 else np.nan
        ax.set_title(f"{self.cols[yi][0]} vs {self.cols[xi][0]}", fontsize=10)
        self.lbl_corr.setText(f"Correlation (Pearson r) = {fmt(r, 4)}   ·   Spearman ρ = {fmt(rho, 4)}   ·   n = {n}")

    def _draw_hist(self, fig, xi, yi, zi, ci):
        name, v, levels = self.cols[xi]
        v = v[np.isfinite(v)]
        ax = fig.add_subplot(111)
        if levels is not None:
            cnt = np.bincount(np.rint(v).astype(int), minlength=len(levels))
            ax.bar(levels, cnt, color=theme.PLOT_COLORS[0])
        else:
            ax.hist(v, bins=min(50, max(8, int(np.sqrt(len(v))))), color=theme.PLOT_COLORS[0], alpha=0.85,
                    edgecolor="white")
        ax.set_xlabel(name)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram {name}", fontsize=10)
        ax.grid(axis="y", alpha=0.35)
        if len(v) > 2 and levels is None:
            self.lbl_corr.setText(f"n = {len(v)}, mean = {fmt(v.mean(), 5)}, SD = {fmt(v.std(ddof=1), 4)}, "
                                  f"median = {fmt(np.median(v), 5)}, skewness = {stats.skew(v):.3f}")
        else:
            self.lbl_corr.setText(f"n = {len(v)}")

    def _draw_box(self, fig, xi, yi, zi, ci):
        xname, x, levels = self.cols[xi]
        yname, y, _ = self.cols[yi]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        ax = fig.add_subplot(111)
        uniq = np.unique(np.round(x, 8))
        if levels is not None or len(uniq) <= 12:
            groups = [y[np.round(x, 8) == u] for u in uniq]
            labels = [levels[int(u)] for u in uniq] if levels is not None else [fmt(u, 4) for u in uniq]
        else:
            edges = np.quantile(x, np.linspace(0, 1, 6))
            idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 4)
            groups = [y[idx == k] for k in range(5)]
            labels = [f"{fmt(edges[k], 3)}–{fmt(edges[k + 1], 3)}" for k in range(5)]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.55)
        for box in bp["boxes"]:
            box.set_facecolor(theme.PLOT_COLORS[0])
            box.set_alpha(0.5)
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=30 if len(labels) > 6 else 0)
        ax.set_xlabel(xname + ("" if levels is not None or len(uniq) <= 12 else " (5 quantile groups)"))
        ax.set_ylabel(yname)
        ax.set_title(f"Box plot {yname} per {xname}", fontsize=10)
        ax.grid(axis="y", alpha=0.35)
        if len(groups) > 1 and all(len(g) > 1 for g in groups):
            pk = stats.kruskal(*groups).pvalue
            self.lbl_corr.setText(f"Kruskal-Wallis test between groups: p = {pk:.4g}"
                                  + (" (significantly different)" if pk < 0.05 else " (not significantly different)"))
        else:
            self.lbl_corr.setText("")

    def _draw_matrix(self, fig, xi, yi, zi, ci):
        p = self.project
        pick = [n for n, c in enumerate(self.cols) if ":" in c[0]][:8]
        k = len(pick)
        mask = np.all([np.isfinite(self.cols[n][1]) for n in pick], axis=0)
        rows = np.where(mask)[0]
        if len(rows) > 2000:
            rows = np.sort(np.random.default_rng(0).choice(rows, 2000, replace=False))
        axs = fig.subplots(k, k, squeeze=False)
        col, info = self._colors(ci, np.isin(np.arange(p.n), rows))
        for a in range(k):
            for b in range(k):
                ax = axs[a, b]
                va, vb = self.cols[pick[a]][1][rows], self.cols[pick[b]][1][rows]
                if a == b:
                    ax.hist(va, bins=15, color=theme.PLOT_COLORS[0], alpha=0.8)
                else:
                    ax.scatter(vb, va, s=4, c=col if not isinstance(col, str) else None,
                               color=col if isinstance(col, str) else None, rasterized=True)
                    ax.text(0.03, 0.9, f"r={self.R[pick[a], pick[b]]:.2f}", transform=ax.transAxes, fontsize=7,
                            color=theme.BAD if abs(self.R[pick[a], pick[b]]) >= 0.5 else "#334155")
                ax.set_xticks([])
                ax.set_yticks([])
                if a == k - 1:
                    ax.set_xlabel(self.cols[pick[b]][0].split(":")[0], fontsize=8)
                if b == 0:
                    ax.set_ylabel(self.cols[pick[a]][0].split(":")[0], fontsize=8)
        fig.suptitle("Scatter matrix of factors & responses" + (f" (sample of {len(rows)} runs)" if mask.sum() > len(rows)
                                                         else ""), fontsize=10)
        self.lbl_corr.setText(f"{k} columns, {int(mask.sum())} complete runs. Red r = |r| ≥ 0.5.")


# ================================================================== Coefficient Table
P_COLORS = [(0.01, "#86efac"), (0.05, "#d9f99d"), (0.10, "#fef08a")]


def coef_pvalues(p, j, fit):
    se = fit.coef_se
    beta = fit.beta[fit.term_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    if p.analysis_kind(j) == "glm":
        return beta, 2 * stats.norm.sf(np.abs(t))
    df = max(int(fit.df_resid), 1)
    return beta, 2 * stats.t.sf(np.abs(t), df)


def coef_table_html(p, show_p=True):
    fits = {j: p.fit(j) for j in range(len(p.responses))}
    cols = [j for j, f in fits.items() if f is not None]
    if not cols:
        return "<h2>Coefficient Table</h2><p>No model can be computed yet.</p>"
    order, cells = [], {}
    for j in cols:
        f = fits[j]
        beta, pv = coef_pvalues(p, j, f)
        for lab, b, pp in zip(f.coef_labels(), beta, pv):
            if lab not in order:
                order.append(lab)
            cells[(lab, j)] = (b, pp)
    h = ["<h2>Coefficient Table (coded factors)</h2>",
         "<p class='note'>Cell color by coefficient p-value: "
         + " ".join(f"<span style='background:{c}'>&nbsp;p &lt; {a:g}&nbsp;</span>" for a, c in P_COLORS)
         + ". Empty cell = the term is not in that response's model.</p>",
         "<table><tr><th class='l'>Term</th>" + "".join(f"<th>{p.response_label(j)}</th>" for j in cols) + "</tr>"]
    for lab in order:
        h.append(f"<tr><td class='l'>{lab}</td>")
        for j in cols:
            if (lab, j) not in cells:
                h.append("<td></td>")
                continue
            b, pp = cells[(lab, j)]
            bg = next((c for a, c in P_COLORS if np.isfinite(pp) and pp < a), None)
            style = f" style='background:{bg}'" if bg and lab != "Intercept" else ""
            h.append(f"<td{style}>{fmt(b, 5)}" + (f"<br><span class='note'>p {'&lt; 0.0001' if pp < 1e-4 else f'{pp:.4f}'}"
                                                   f"</span>" if show_p and np.isfinite(pp) else "") + "</td>")
        h.append("</tr>")
    footer = [("Transform", lambda j, f: models.transform_label(p.model_spec(j).get("transform"))),
              ("Model", lambda j, f: models.order_label(p.model_spec(j)["order"])),
              ("p model", lambda j, f: next((f"{a.p:.4g}" for a in f.anova if a.source == "Model" and np.isfinite(a.p)),
                                             "-")),
              ("R²", lambda j, f: fmt(f.stats.get("r2", f.stats.get("mcfadden", f.stats.get("r2_cond", np.nan))), 4)),
              ("Adj R²", lambda j, f: fmt(f.stats.get("adj_r2", np.nan), 4)),
              ("Pred R²", lambda j, f: fmt(f.stats.get("pred_r2", np.nan), 4)),
              ("Used for optimization", lambda j, f: "ANN " + p.ann[j].arch if p.source(j) == "ann" else "RSM")]
    for lab, fn in footer:
        h.append(f"<tr class='model'><td class='l'>{lab}</td>" + "".join(f"<td>{fn(j, fits[j])}</td>" for j in cols)
                 + "</tr>")
    h.append("</table><p class='note'>Coefficients are in coded units (−1…+1 / pseudo-components), so their sizes "
             "can be compared across factors. For responses that use ANN, the table still shows the RSM "
             "coefficients.</p>")
    return "".join(h)


class CoefTableTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.chk_p = QCheckBox("Show p-values")
        self.chk_p.setChecked(True)
        bar.addWidget(self.chk_p)
        bar.addStretch()
        lay.addLayout(bar)
        self.txt = ReportBrowser()
        lay.addWidget(self.txt)
        self.chk_p.toggled.connect(self.refresh)

    def set_project(self, project):
        self.project = project
        self.refresh()

    def refresh(self):
        if self.project is not None:
            self.txt.setHtml(REPORT_CSS + coef_table_html(self.project, self.chk_p.isChecked()))
