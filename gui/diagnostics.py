"""Interactive diagnostics panel: 4 selectable plots, point colors by factor, click a point to identify its
run (highlighted in all plots), exclude runs from the analysis, and a diagnostics report table + DFBETAS."""
import matplotlib
import numpy as np
from matplotlib import cm, colors as mcolors
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QTableView, QTabWidget,
                               QVBoxLayout, QWidget)
from scipy import stats

from doe import models

from . import plots, theme
from .common import ArrayModel, MplCanvas, fmt
from .copying import install_table_copy

BLUE = plots.BLUE
RED = plots.RED
DEFAULT_PANELS = ["normal", "resid_pred", "pred_actual", "cook"]
LAYOUTS = {"2x2": "4 plots (2 × 2)", "1x2": "2 plots side by side", "1": "1 large plot"}


def dfbetas(project, fit):
    """DFBETAS of each run for each term coefficient (ordinary least squares regression only)."""
    rows = fit.rows_used
    X = fit._rows(project.coded[rows])
    if fit.block_idx:
        Z = models.block_columns(project.blocks[rows])[0]
        if Z.shape[1]:
            X[:, fit.block_idx] = Z
    h = fit.leverage
    e = fit.resid
    n, p = X.shape
    if n - p - 1 <= 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        s_i = np.sqrt(np.maximum((fit.sse - e ** 2 / (1 - h)) / (n - p - 1), 0))
        C = (fit.xtx_inv @ X.T).T                                  # n x p
        D = C * (e / (1 - h))[:, None] / (s_i[:, None] * np.sqrt(np.diag(fit.xtx_inv))[None, :])
    D[~np.isfinite(D)] = np.nan
    return D[:, fit.term_idx]


def bonferroni_limit(fit):
    return stats.t.ppf(1 - 0.05 / (2 * fit.n), fit.df_resid - 1) if fit.df_resid > 1 else np.inf


class Context:
    """Shared drawing data: fit, run numbers, point colors, and selected points."""

    def __init__(self, project, fit, j):
        self.p, self.fit, self.j = project, fit, j
        self.rows = fit.rows_used
        self.run = project.run_order[self.rows]
        self.n = fit.n
        self.cvals = None
        self.cat_levels = None
        self.color_label = ""
        self.sel = set()
        self.dfb = None
        self.dfb_labels = []

    def set_color(self, key):
        p = self.p
        self.cvals, self.cat_levels, self.color_label = None, None, ""
        if key is None:
            return
        if key == "run":
            self.cvals, self.color_label = self.run.astype(float), "Run order"
        elif key == "pred":
            self.cvals, self.color_label = np.asarray(self.fit.yhat, float), "Predicted"
        else:
            i = int(key)
            v = p.actual[self.rows, i]
            self.color_label = p.factor_label(i)
            if p.factors[i].categoric:
                self.cat_levels = p.factors[i].levels
                self.cvals = np.rint(v)
            else:
                self.cvals = v

    def point_colors(self):
        if self.cvals is None:
            return BLUE, None
        if self.cat_levels is not None:
            cmap = matplotlib.colormaps["tab10"]
            return [cmap(int(v) % 10) for v in self.cvals], None
        ok = np.isfinite(self.cvals)
        lo, hi = (np.nanmin(self.cvals), np.nanmax(self.cvals)) if ok.any() else (0, 1)
        norm = mcolors.Normalize(lo, hi if hi > lo else lo + 1)
        cmap = matplotlib.colormaps["jet"]
        return cmap(norm(np.nan_to_num(self.cvals, nan=lo))), (cmap, norm)


def _scatter(ax, ctx, x, y, idx, base=22):
    col, mapping = ctx.point_colors()
    c = col if isinstance(col, str) else np.asarray(col)[idx]
    big = len(x) > 2000
    ax.scatter(x, y, s=plots.msize(len(x), base), c=c, edgecolor="none" if big or ctx.cvals is None else "#333",
               linewidths=0.4, zorder=3, rasterized=big)
    sel = [k for k, i in enumerate(idx) if i in ctx.sel]
    if sel:
        ax.scatter(np.asarray(x)[sel], np.asarray(y)[sel], s=160, facecolor="none", edgecolor="#e11d48", lw=2.2,
                   zorder=6)
        for k in sel:
            ax.annotate(f"run {ctx.run[idx[k]]}", (np.asarray(x)[k], np.asarray(y)[k]), textcoords="offset points",
                        xytext=(8, 6), color="#e11d48", fontsize=8, fontweight="bold", zorder=7)
    return mapping


def _limits(ax, ctx):
    lim = bonferroni_limit(ctx.fit)
    ax.axhline(0, color="#64748b", lw=0.8)
    if np.isfinite(lim):
        for s in (-lim, lim):
            ax.axhline(s, color=RED, lw=1.2)
        ax.text(1.0, lim, f" {lim:.3f}", transform=ax.get_yaxis_transform(), color=RED, fontsize=7, va="bottom",
                ha="right")


def p_normal(ax, ctx):
    r = ctx.fit.stud_ext
    valid = np.where(np.isfinite(r))[0]
    if len(valid) < 3:
        ax.text(0.5, 0.5, "Not enough data", ha="center", transform=ax.transAxes)
        return None
    o = valid[np.argsort(r[valid])]
    probs = (np.arange(1, len(o) + 1) - 0.5) / len(o)
    x, y = r[o], stats.norm.ppf(probs)
    m = _scatter(ax, ctx, x, y, o)
    ax.plot([x.min(), x.max()], [x.min(), x.max()], color="#f87171", lw=1.5)
    ticks = [0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99] if len(o) < 200 else [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    ax.set_yticks(stats.norm.ppf(ticks), [f"{100 * t:g}" for t in ticks])
    ax.set_xlabel("Externally studentized residuals")
    ax.set_ylabel("Normal probability (%)")
    ax.set_title("Normal Plot of Residuals")
    return x, y, o, m


def p_resid_pred(ax, ctx):
    r = ctx.fit.stud_ext
    idx = np.where(np.isfinite(r))[0]
    x, y = np.asarray(ctx.fit.yhat)[idx], r[idx]
    m = _scatter(ax, ctx, x, y, idx)
    _limits(ax, ctx)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ext. studentized residuals")
    ax.set_title("Residuals vs Predicted")
    return x, y, idx, m


def p_resid_run(ax, ctx):
    r = ctx.fit.stud_ext
    idx = np.array([i for i in np.argsort(ctx.run) if np.isfinite(r[i])], int)
    x, y = ctx.run[idx], r[idx]
    if len(idx) <= 400:
        ax.plot(x, y, "-", color="#cbd5e1", lw=0.8, zorder=1)
    m = _scatter(ax, ctx, x, y, idx)
    _limits(ax, ctx)
    ax.set_xlabel("Run number")
    ax.set_ylabel("Ext. studentized residuals")
    ax.set_title("Residuals vs Run Order")
    return x, y, idx, m


def p_resid_factor(ax, ctx, i):
    p = ctx.p
    r = ctx.fit.stud_ext
    idx = np.where(np.isfinite(r))[0]
    v = p.actual[ctx.rows, i][idx]
    if p.factors[i].categoric:
        jit = np.random.default_rng(0).uniform(-0.12, 0.12, len(v))
        x = v + jit
        ax.set_xticks(range(len(p.factors[i].levels)), p.factors[i].levels)
    else:
        x = v
    y = r[idx]
    m = _scatter(ax, ctx, x, y, idx)
    _limits(ax, ctx)
    ax.set_xlabel(p.factor_label(i))
    ax.set_ylabel("Ext. studentized residuals")
    ax.set_title(f"Residuals vs {p.factor_label(i).split(':')[0]}")
    return x, y, idx, m


def p_pred_actual(ax, ctx):
    f = ctx.fit
    idx = np.arange(ctx.n)
    x, y = np.asarray(f.y, float), np.asarray(f.yhat, float)
    m = _scatter(ax, ctx, x, y, idx)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color="#334155", lw=1)
    ax.set_xlabel("Actual" + (" (transformed scale)" if f.transform and f.transform.get("kind", "none") != "none"
                              else ""))
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    return x, y, idx, m


def _stem(ax, ctx, vals, title, ylabel, lines=()):
    idx = np.argsort(ctx.run)
    x, y = ctx.run[idx], np.nan_to_num(vals[idx])
    if len(idx) <= 1500:
        ax.vlines(x, 0, y, color="#94a3b8", lw=1, zorder=1)
    m = _scatter(ax, ctx, x, y, idx, base=14)
    for val, style, lab in lines:
        ax.axhline(val, color=RED if style == "-" else "#e08e0b", lw=1.2, ls=style)
        ax.text(1.0, val, f" {lab}", transform=ax.get_yaxis_transform(), color=RED, fontsize=7, va="bottom", ha="right")
    ax.axhline(0, color="#64748b", lw=0.8)
    ax.set_xlabel("Run number")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return x, y, idx, m


def p_cook(ax, ctx):
    return _stem(ax, ctx, ctx.fit.cooks, "Cook's Distance", "Cook's distance",
                 [(1.0, "-", "1.0"), (4 / ctx.n, ":", "4/n")])


def p_leverage(ax, ctx):
    f = ctx.fit
    return _stem(ax, ctx, f.leverage, "Leverage", "Leverage", [(2 * f.p / f.n, "-", "2p/n")])


def p_dffits(ax, ctx):
    f = ctx.fit
    lim = 2 * np.sqrt(f.p / f.n)
    out = _stem(ax, ctx, f.dffits, "DFFITS", "DFFITS", [(lim, "-", f"{lim:.3f}")])
    ax.axhline(-lim, color=RED, lw=1.2)
    return out


def p_dfbetas(ax, ctx, k):
    if ctx.dfb is None:
        ax.text(0.5, 0.5, "DFBETAS is only for ordinary regression (OLS)", ha="center", transform=ax.transAxes)
        return None
    lim = 2 / np.sqrt(ctx.n)
    out = _stem(ax, ctx, ctx.dfb[:, k], f"DFBETAS - {ctx.dfb_labels[k]}", "DFBETAS", [(lim, "-", f"{lim:.3f}")])
    ax.axhline(-lim, color=RED, lw=1.2)
    return out


def p_hist(ax, ctx):
    r = ctx.fit.stud_ext
    r = r[np.isfinite(r)]
    ax.hist(r, bins=min(40, max(8, len(r) // 5)), color=theme.PLOT_COLORS[0], alpha=0.8, edgecolor="white",
            density=True)
    xx = np.linspace(min(r.min(), -3), max(r.max(), 3), 200)
    ax.plot(xx, stats.norm.pdf(xx), color=RED, lw=1.5, label="Standard normal")
    ax.set_xlabel("Ext. studentized residuals")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of Residuals")
    ax.legend(fontsize=8)
    return None


def catalog(project, ctx):
    """Plot list: (key, label, function)."""
    items = [("normal", "Normal plot of residuals", p_normal), ("resid_pred", "Residuals vs Predicted", p_resid_pred),
             ("resid_run", "Residuals vs Run Order", p_resid_run)]
    for i in range(project.k):
        items.append((f"resid_f{i}", f"Residuals vs {project.factor_label(i)}", lambda ax, c, i=i: p_resid_factor(ax, c, i)))
    items += [("pred_actual", "Predicted vs Actual", p_pred_actual), ("cook", "Cook's Distance", p_cook),
              ("leverage", "Leverage", p_leverage), ("dffits", "DFFITS", p_dffits), ("hist", "Histogram of residuals", p_hist)]
    if ctx is not None and ctx.dfb is not None:
        for k, lab in enumerate(ctx.dfb_labels):
            items.append((f"dfbetas{k}", f"DFBETAS - {lab}", lambda ax, c, k=k: p_dfbetas(ax, c, k)))
    return items


class Panel(QWidget):
    def __init__(self, owner, default):
        super().__init__()
        self.owner = owner
        self.key = default
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.cb = QComboBox()
        self.cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb.setMinimumContentsLength(18)
        lay.addWidget(self.cb)
        self.canvas = MplCanvas(figsize=(5, 3.6))
        lay.addWidget(self.canvas, 1)
        self.pts = None
        self.cb.currentIndexChanged.connect(self.on_choice)
        self.canvas.canvas.mpl_connect("button_press_event", self.on_click)

    def fill(self, items):
        self.cb.blockSignals(True)
        self.cb.clear()
        for key, label, _ in items:
            self.cb.addItem(label, key)
        i = self.cb.findData(self.key)
        self.cb.setCurrentIndex(i if i >= 0 else 0)
        self.key = self.cb.currentData()
        self.cb.blockSignals(False)

    def on_choice(self):
        self.key = self.cb.currentData()
        self.draw()

    def draw(self):
        ctx = self.owner.ctx
        fig = self.canvas.figure
        fig.clear()
        self.pts = None
        if ctx is None:
            self.canvas.message(self.owner.empty_msg)
            return
        fn = next((f for k, _, f in self.owner.items if k == self.key), None)
        if fn is None:
            return
        ax = fig.add_subplot(111)
        out = fn(ax, ctx)
        ax.grid(alpha=0.35)
        ax.title.set_fontsize(10)
        if out is not None:
            x, y, idx, mapping = out
            plots.fit_limits(ax, x, y)
            self.pts = (np.asarray(x, float), np.asarray(y, float), np.asarray(idx, int), ax)
            if mapping is not None:
                cmap, norm = mapping
                cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01, fraction=0.05)
                cb.set_label(ctx.color_label, fontsize=8)
                cb.ax.tick_params(labelsize=7)
            elif ctx.cat_levels is not None:
                cmap = matplotlib.colormaps["tab10"]
                for n, lv in enumerate(ctx.cat_levels):
                    ax.scatter([], [], color=cmap(n % 10), label=lv, s=20)
                ax.legend(fontsize=7, title=ctx.color_label.split(":")[0], title_fontsize=7)
        self.canvas.draw()

    def on_click(self, event):
        tb = self.canvas.toolbar
        if tb is not None and tb.mode:
            return                                           # zooming/panning
        if self.pts is None or event.inaxes is not self.pts[3] or event.x is None:
            return
        x, y, idx, ax = self.pts
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            return
        disp = ax.transData.transform(np.column_stack([x[ok], y[ok]]))
        d = np.hypot(disp[:, 0] - event.x, disp[:, 1] - event.y)
        k = int(np.argmin(d))
        if d[k] > 12:
            self.owner.select(None, add=False)
            return
        add = bool(event.key and ("control" in event.key or "shift" in event.key))
        self.owner.select(int(idx[ok][k]), add=add)


class DiagnosticsPanel(QWidget):
    """Interactive residual diagnostics for one response."""
    excluded_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.ctx = None
        self.items = []
        self.empty_msg = "No model yet"
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Color points by:"))
        self.cb_color = QComboBox()
        bar.addWidget(self.cb_color)
        bar.addSpacing(10)
        bar.addWidget(QLabel("Layout:"))
        self.cb_layout = QComboBox()
        for k, lab in LAYOUTS.items():
            self.cb_layout.addItem(lab, k)
        bar.addWidget(self.cb_layout)
        bar.addStretch()
        self.btn_ex = QPushButton("Exclude Selected Runs")
        self.btn_ex.setIcon(theme.glyph_icon("remove", theme.BAD))
        self.btn_ex.setToolTip("Remove the runs from the analysis (the data is not deleted and can be restored).")
        self.btn_restore = QPushButton("Restore All Runs")
        self.btn_restore.setIcon(theme.glyph_icon("add", theme.ACCENT))
        bar.addWidget(self.btn_ex)
        bar.addWidget(self.btn_restore)
        lay.addLayout(bar)
        self.lbl = QLabel("Click a point on any plot to identify its run (Ctrl+click = select several). "
                          "The same point is highlighted in all plots.")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet(f"color: {theme.MUTED};")
        lay.addWidget(self.lbl)

        self.tabs = QTabWidget()
        grid_w = QWidget()
        self.grid = QGridLayout(grid_w)
        self.grid.setContentsMargins(0, 4, 0, 0)
        self.panels = [Panel(self, k) for k in DEFAULT_PANELS]
        for n, pn in enumerate(self.panels):
            self.grid.addWidget(pn, n // 2, n % 2)
        self.table = QTableView()
        self.table_model = ArrayModel()
        self.table.setModel(self.table_model)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setResizeContentsPrecision(60)
        install_table_copy(self.table)
        rep_w = QWidget()
        rl = QVBoxLayout(rep_w)
        rl.setContentsMargins(0, 4, 0, 0)
        self.lbl_rep = QLabel()
        self.lbl_rep.setWordWrap(True)
        self.lbl_rep.setStyleSheet(f"color: {theme.MUTED};")
        rl.addWidget(self.lbl_rep)
        rl.addWidget(self.table, 1)
        self.tabs.addTab(grid_w, "Graphs")
        self.tabs.addTab(rep_w, "Diagnostics Report (Table)")
        lay.addWidget(self.tabs, 1)

        self.cb_color.currentIndexChanged.connect(self.on_color)
        self.cb_layout.currentIndexChanged.connect(self.apply_layout)
        self.btn_ex.clicked.connect(self.exclude_selected)
        self.btn_restore.clicked.connect(self.restore_all)
        self.table.doubleClicked.connect(lambda ix: self.select(int(ix.row()), add=False, from_table=True))

    # ---------------------------------------------------------------- data
    def set_fit(self, project, fit, j, message=None):
        self.project = project
        cur_color = self.cb_color.currentData()
        self.cb_color.blockSignals(True)
        self.cb_color.clear()
        self.cb_color.addItem("None (single color)", None)
        for i in range(project.k):
            self.cb_color.addItem(project.factor_label(i), str(i))
        self.cb_color.addItem("Run order", "run")
        self.cb_color.addItem("Predicted value", "pred")
        k = self.cb_color.findData(cur_color)
        self.cb_color.setCurrentIndex(k if k >= 0 else 0)
        self.cb_color.blockSignals(False)
        n_ex = len(project.excluded)
        self.btn_restore.setText(f"Restore Excluded Runs ({n_ex})")
        self.btn_restore.setEnabled(n_ex > 0)
        if fit is None:
            self.ctx = None
            self.empty_msg = message or "No model yet"
            self.items = catalog(project, None)
        else:
            old_sel = set()
            if self.ctx is not None and self.ctx.p is project:
                old_sel = {int(self.ctx.rows[i]) for i in self.ctx.sel if i < len(self.ctx.rows)}
            self.ctx = Context(project, fit, j)
            pos = {int(r): n for n, r in enumerate(fit.rows_used)}
            self.ctx.sel = {pos[r] for r in old_sel if r in pos}
            if project.analysis_kind(j) == "ols":
                try:
                    self.ctx.dfb = dfbetas(project, fit)
                    self.ctx.dfb_labels = fit.coef_labels()
                except (ValueError, np.linalg.LinAlgError):
                    self.ctx.dfb = None
            self.ctx.set_color(self.cb_color.currentData())
            self.items = catalog(project, self.ctx)
        for pn in self.panels:
            pn.fill(self.items)
        self.apply_layout()
        self.fill_table()
        self.update_label()

    def apply_layout(self):
        mode = self.cb_layout.currentData()
        show = {"2x2": 4, "1x2": 2, "1": 1}[mode]
        for n, pn in enumerate(self.panels):
            pn.setVisible(n < show)
        for n, pn in enumerate(self.panels[:show]):
            pn.draw()

    def redraw(self):
        for pn in self.panels:
            if not pn.isHidden():            # isHidden: hidden by the layout (not because another tab is active)
                pn.draw()

    def on_color(self):
        if self.ctx is not None:
            self.ctx.set_color(self.cb_color.currentData())
            self.redraw()

    # ---------------------------------------------------------------- selection & exclusion
    def select(self, i, add=False, from_table=False):
        if self.ctx is None:
            return
        if i is None:
            if self.ctx.sel:
                self.ctx.sel = set()
                self.redraw()
                self.update_label()
            return
        if from_table:
            i = int(self.table_rows[i])
        if add:
            self.ctx.sel.symmetric_difference_update({i})
        else:
            self.ctx.sel = set() if self.ctx.sel == {i} else {i}
        self.redraw()
        self.update_label()
        if from_table:
            self.tabs.setCurrentIndex(0)

    def update_label(self):
        c = self.ctx
        if c is None or not c.sel:
            self.lbl.setText("Click a point on any plot to identify its run (Ctrl+click = select several). "
                             "The same point is highlighted in all plots. Double-click a table row to highlight it.")
            self.btn_ex.setEnabled(False)
            return
        f = c.fit
        parts = []
        for i in sorted(c.sel)[:6]:
            parts.append(f"<b>Run {c.run[i]}</b> (Std {c.rows[i] + 1}): actual {fmt(f.y_orig[i], 6)}, predicted "
                         f"{fmt(float(np.asarray(f.yhat)[i]), 6)}, ext. stud. residual {fmt(f.stud_ext[i], 3)}, "
                         f"leverage {fmt(f.leverage[i], 3)}, Cook's {fmt(f.cooks[i], 3)}")
        more = f" … (+{len(c.sel) - 6})" if len(c.sel) > 6 else ""
        self.lbl.setText("<br>".join(parts) + more)
        self.lbl.setStyleSheet(f"color: {theme.INK};")
        self.btn_ex.setEnabled(True)

    def exclude_selected(self):
        c = self.ctx
        if c is None or not c.sel:
            return
        rows = [int(c.rows[i]) for i in c.sel]
        self.project.set_excluded(rows, True)
        c.sel = set()
        self.excluded_changed.emit()

    def restore_all(self):
        if self.project is None:
            return
        self.project.excluded = []
        self.project.dirty = True
        self.excluded_changed.emit()

    # ---------------------------------------------------------------- report table
    def fill_table(self):
        c = self.ctx
        if c is None:
            self.table_model.set_data([], [])
            self.lbl_rep.setText("")
            return
        f, p = c.fit, c.p
        o = np.argsort(c.run)
        self.table_rows = o
        lim_r = bonferroni_limit(f)
        lim_h = 2 * f.p / f.n
        lim_d = 2 * np.sqrt(f.p / f.n)
        lim_b = 2 / np.sqrt(f.n)
        heads = ["Run", "Std", "Type", "Actual", "Predicted", "Residual", "Leverage", "Int. stud. resid.",
                 "Ext. stud. resid.", "Cook's", "DFFITS"]
        cols = [c.run[o], c.rows[o] + 1, np.array([p.point_types[r] for r in c.rows[o]], object),
                np.asarray(f.y_orig, float)[o], np.asarray(f.yhat, float)[o], np.asarray(f.resid, float)[o],
                f.leverage[o], f.stud_int[o], f.stud_ext[o], f.cooks[o], f.dffits[o]]
        limits = {6: lim_h, 8: lim_r, 9: 1.0, 10: lim_d}
        if c.dfb is not None:
            for k, lab in enumerate(c.dfb_labels):
                heads.append(f"DFBETAS {lab}")
                cols.append(c.dfb[o, k])
                limits[len(heads) - 1] = lim_b

        def fmt_cell(col, v):
            if col in (0, 1):
                return str(int(v))
            if col == 2:
                return str(v)
            return "" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt(float(v), 5)

        def bg(row, col):
            lim = limits.get(col)
            if lim is None:
                return None
            v = cols[col][row]
            return "#fde2e1" if np.isfinite(v) and abs(v) > lim else None

        self.table_model.set_data(heads, cols, fmt_cell, bg)
        self.table.resizeColumnsToContents()
        flagged = {k: int(np.sum(np.abs(np.nan_to_num(cols[k])) > lim)) for k, lim in limits.items()}
        self.lbl_rep.setText(
            f"{f.n} runs analyzed" + (f", {len(p.excluded)} runs excluded" if p.excluded else "")
            + f". Red cells exceed the limit: |ext. stud. resid.| > {lim_r:.3f} (Bonferroni): {flagged[8]} runs; "
            f"leverage > 2p/n = {lim_h:.3f}: {flagged[6]}; Cook's > 1: {flagged[9]}; |DFFITS| > {lim_d:.3f}: "
            f"{flagged[10]}" + (f"; |DFBETAS| > {lim_b:.3f}." if c.dfb is not None else ".")
            + " Double-click a row to highlight the run in the plots. The table can be copied to Excel (right-click).")
