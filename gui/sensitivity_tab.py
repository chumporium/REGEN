"""Parametric study and sensitivity analysis of the fitted models (RSM or ANN)."""
import html

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)

from doe import constraints as cons_mod
from doe import sensitivity as sens

from . import plotdata, plots, theme
from .common import REPORT_CSS, MplCanvas, Task, fmt, fmt_p
from .copying import ReportBrowser

SOURCES = [("Selected model of each response", None), ("RSM", "rsm"), ("ANN", "ann")]
# color and marker of each response in the one-factor-at-a-time panels
SERIES = [("#000000", "D"), ("#1f5f99", "o"), ("#d9480f", "s"), ("#16a34a", "^"), ("#9333ea", "v"),
          ("#ca8a04", "P"), ("#475569", "X"), ("#e11d48", "*")]
REFS = [("center", "Design center"), ("desirability", "Best desirability solution"),
        ("topsis", "NSGA: TOPSIS choice"), ("knee", "NSGA: knee point"), ("custom", "Custom values")]


def _grid(fig, n):
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    axs = np.atleast_1d(fig.subplots(nrow, ncol, squeeze=False)).ravel()
    for ax in axs[n:]:
        ax.axis("off")
    return axs[:n]


class SensitivityTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.task = None
        self._sobol = None

        lay = QVBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        lay.addWidget(split)

        # ---- left: reference point and model
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        g = QGroupBox("Reference Point")
        gl = QVBoxLayout(g)
        f = QFormLayout()
        self.cb_ref = QComboBox()
        self.cb_src = QComboBox()
        for label, key in SOURCES:
            self.cb_src.addItem(label, key)
        f.addRow("Start from:", self.cb_ref)
        f.addRow("Model:", self.cb_src)
        gl.addLayout(f)
        self.tbl_ref = QTableWidget(0, 2)
        self.tbl_ref.setHorizontalHeaderLabels(["Factor", "Value"])
        self.tbl_ref.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_ref.verticalHeader().setVisible(False)
        gl.addWidget(self.tbl_ref)
        note = QLabel("Factors not varied in a study stay at these values (actual units). Mixture components cannot "
                      "be varied one at a time; they stay at the reference blend.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.MUTED};")
        gl.addWidget(note)
        ll.addWidget(g)
        split.addWidget(left)

        # ---- right: studies
        self.tabs = QTabWidget()
        split.addWidget(self.tabs)
        split.setSizes([330, 1000])
        self._build_parametric()
        self._build_sobol()
        self._build_tornado()
        self._build_front()

        self.cb_ref.currentIndexChanged.connect(self.load_reference)

    # ------------------------------------------------------------------ layout helpers
    def _study(self, title, controls, run_text, slot):
        w = QWidget()
        v = QVBoxLayout(w)
        bar = QHBoxLayout()
        for lab, wid in controls:
            if lab:
                bar.addWidget(QLabel(lab))
            bar.addWidget(wid)
        bar.addStretch()
        btn = QPushButton(run_text)
        btn.setStyleSheet("font-weight: bold; padding: 5px 16px;")
        btn.clicked.connect(slot)
        bar.addWidget(btn)
        v.addLayout(bar)
        sp = QSplitter(Qt.Vertical)
        canvas = MplCanvas(figsize=(7, 4))
        txt = ReportBrowser()
        sp.addWidget(canvas)
        sp.addWidget(txt)
        sp.setSizes([520, 260])
        v.addWidget(sp)
        self.tabs.addTab(w, title)
        return canvas, txt, btn

    def _build_parametric(self):
        self.cb_layout = QComboBox()
        self.cb_layout.addItem("All factors, one panel each", "all")
        self.cb_layout.addItem("One factor, one panel per response", "one")
        self.cb_layout.setToolTip("All factors: each factor is varied over its range in its own panel, with every "
                                  "response on its own vertical axis.\nOne factor: the chosen factor is varied, "
                                  "optionally with curves for a second factor, in one panel per response.")
        self.cb_px = QComboBox()
        self.sp_from = QDoubleSpinBox()
        self.sp_to = QDoubleSpinBox()
        for s in (self.sp_from, self.sp_to):
            s.setRange(-1e12, 1e12)
            s.setDecimals(4)
        self.sp_steps = QSpinBox()
        self.sp_steps.setRange(3, 401)
        self.sp_steps.setValue(11)
        self.cb_pz = QComboBox()
        self.sp_nz = QSpinBox()
        self.sp_nz.setRange(2, 7)
        self.sp_nz.setValue(3)
        self.sp_nz.setToolTip("Number of values of the second factor, evenly spaced over its range")
        self.can_par, self.txt_par, _ = self._study(
            "Parametric Study",
            [("Layout:", self.cb_layout), ("Vary:", self.cb_px), ("from", self.sp_from), ("to", self.sp_to),
             ("points", self.sp_steps), ("Curves for:", self.cb_pz), ("values", self.sp_nz)], "Run",
            self.run_parametric)
        self.cb_px.currentIndexChanged.connect(self._range_of_x)
        self.cb_layout.currentIndexChanged.connect(self._layout_changed)
        self._layout_changed()

    def _layout_changed(self):
        one = self.cb_layout.currentData() == "one"
        for w in (self.cb_px, self.cb_pz, self.sp_nz):
            w.setEnabled(one)
        self._range_of_x()

    def _build_sobol(self):
        self.cb_gmethod = QComboBox()
        self.cb_gmethod.addItem("Standardized coefficients (Latin hypercube)", "src")
        self.cb_gmethod.addItem("Sobol indices (variance-based)", "sobol")
        self.cb_gmethod.setToolTip("Standardized coefficients: linear regression of each response on the factors "
                                   "over a Latin hypercube sample, with p-values.\nSobol indices: share of the "
                                   "response variance due to each factor, including nonlinear effects and "
                                   "interactions.")
        self.sp_n = QSpinBox()
        self.sp_n.setRange(20, 65536)
        self.sp_n.setValue(120)
        self.sp_n.setToolTip("Standardized coefficients: number of Latin hypercube points.\nSobol indices: base "
                             "sample size N (rounded up to a power of 2); the model is evaluated N × (number of "
                             "factors + 2) times.")
        self.can_sob, self.txt_sob, self.btn_sob = self._study(
            "Global Sensitivity", [("Method:", self.cb_gmethod), ("Sample size:", self.sp_n)], "Run",
            self.run_global)
        self.cb_gmethod.currentIndexChanged.connect(
            lambda: self.sp_n.setValue(120 if self.cb_gmethod.currentData() == "src" else 8192))

    def _build_tornado(self):
        self.sp_delta = QSpinBox()
        self.sp_delta.setRange(1, 50)
        self.sp_delta.setValue(10)
        self.sp_delta.setSuffix(" % of the range")
        self.can_tor, self.txt_tor, _ = self._study(
            "Local Sensitivity (Tornado)", [("Change each factor by ±", self.sp_delta)], "Run", self.run_tornado)

    def _build_front(self):
        self.can_front, self.txt_front, _ = self._study(
            "Pareto Front Trade-off", [("", QLabel("Uses the last NSGA-II / NSGA-III result."))], "Analyze",
            self.run_front)

    # ------------------------------------------------------------------ project
    def set_project(self, project):
        self.project = project
        self._sobol = None
        for c in (self.can_par, self.can_sob, self.can_tor, self.can_front):
            c.clear()
            c.draw()
        for t in (self.txt_par, self.txt_sob, self.txt_tor, self.txt_front):
            t.setHtml("")
        p = project
        self.cb_px.blockSignals(True)
        self.cb_px.clear()
        self.cb_pz.clear()
        self.cb_pz.addItem("(none)", None)
        for i in sens.variables(p):
            self.cb_px.addItem(p.factor_label(i), i)
            self.cb_pz.addItem(p.factor_label(i), i)
        self.cb_px.blockSignals(False)
        self._range_of_x()
        self.tbl_ref.setRowCount(p.k)
        for i, fct in enumerate(p.factors):
            it = QTableWidgetItem(p.factor_label(i))
            it.setFlags(Qt.ItemIsEnabled)
            self.tbl_ref.setItem(i, 0, it)
            if fct.categoric:
                w = QComboBox()
                w.addItems([str(v) for v in fct.levels])
            else:
                w = QDoubleSpinBox()
                w.setDecimals(4)
                lo, hi = p.coded_bounds()
                a, b = sorted(float(v) for v in p.factor_to_actual(i, np.array([lo[i], hi[i]])))
                w.setRange(a, b)
            self.tbl_ref.setCellWidget(i, 1, w)
            (w.currentIndexChanged if fct.categoric else w.valueChanged).connect(self._edited)
        self.cb_ref.blockSignals(True)
        self.cb_ref.clear()
        self.cb_ref.blockSignals(False)
        self.refresh()

    def _edited(self, *_):
        """A value typed by the user makes the reference point custom."""
        k = self.cb_ref.findData("custom")
        if k >= 0 and self.cb_ref.currentIndex() != k:
            self.cb_ref.blockSignals(True)
            self.cb_ref.setCurrentIndex(k)
            self.cb_ref.blockSignals(False)

    def refresh(self):
        if self.project is None:
            return
        p = self.project
        cur = self.cb_ref.currentData()
        self.cb_ref.blockSignals(True)
        self.cb_ref.clear()
        for key, label in REFS:
            if key == "desirability" and not p.solutions:
                continue
            if key in ("topsis", "knee") and not p.nsga_result:
                continue
            self.cb_ref.addItem(label, key)
        k = self.cb_ref.findData(cur)
        self.cb_ref.setCurrentIndex(k if k >= 0 else 0)
        self.cb_ref.blockSignals(False)
        self.load_reference()

    def _reference_coded(self, key):
        p = self.project
        if key == "desirability" and p.solutions:
            return np.asarray(p.solutions[0][1], float)
        if key in ("topsis", "knee") and p.nsga_result:
            r = p.nsga_result
            return np.asarray(r["X"][r["i_topsis" if key == "topsis" else "i_knee"]], float)
        z = np.zeros(p.k)
        if p.comp_idx:
            comp = p.comp_idx
            lo, hi = p.coded_bounds()
            G, h = p.constraint_system(cols=comp)
            z[comp] = cons_mod.region_centroid(lo[comp], hi[comp], G, h, True)
        return z

    def load_reference(self):
        key = self.cb_ref.currentData()
        if self.project is None or key == "custom" or key is None:
            return
        z = self._reference_coded(key)
        p = self.project
        for i, fct in enumerate(p.factors):
            w = self.tbl_ref.cellWidget(i, 1)
            if w is None:
                continue
            w.blockSignals(True)
            if fct.categoric:
                w.setCurrentIndex(int(round(z[i])))
            else:
                w.setValue(float(p.factor_to_actual(i, z[i])))
            w.blockSignals(False)

    def base(self):
        """Reference point in coded units (from the table)."""
        p = self.project
        z = np.zeros(p.k)
        for i, fct in enumerate(p.factors):
            w = self.tbl_ref.cellWidget(i, 1)
            z[i] = w.currentIndex() if fct.categoric else float(p.factor_to_coded(i, w.value()))
        return z

    def _check_base(self, z):
        if not self.project.feasible(z)[0]:
            QMessageBox.warning(self, "Reference point", "The reference point lies outside the design region "
                                "(component bounds or constraints). Change the values or choose another start.")
            return False
        return True

    def _source(self):
        return self.cb_src.currentData()

    def _label_value(self, i, coded):
        p = self.project
        if p.factors[i].categoric:
            return str(p.factors[i].levels[int(round(coded))])
        return fmt(float(p.factor_to_actual(i, coded)), 5)

    def _range_of_x(self):
        p = self.project
        i = self.cb_px.currentData()
        if p is None or i is None:
            return
        cat = p.factors[i].categoric
        one = self.cb_layout.currentData() == "one"
        for s in (self.sp_from, self.sp_to):
            s.setEnabled(one and not cat)
        self.sp_steps.setEnabled(not (one and cat))
        if not cat:
            lo, hi = p.coded_bounds()
            a, b = sorted(float(v) for v in p.factor_to_actual(i, np.array([lo[i], hi[i]])))
            self.sp_from.setValue(a)
            self.sp_to.setValue(b)

    def _no_models(self, canvas):
        if not sens.predictors(self.project, source=self._source()):
            canvas.message("No model is available yet. Fit an RSM model or train an ANN first.")
            return True
        return False

    def _header(self, title, z):
        p = self.project
        ref = ", ".join(f"{p.factor_label(i)} = {self._label_value(i, z[i])}" for i in range(p.k))
        src = self.cb_src.currentText()
        return (REPORT_CSS + f"<h3>{title}</h3><p class='note'>Reference point ({self.cb_ref.currentText()}): "
                f"{html.escape(ref)}. Model: {html.escape(src)}.</p>")

    # ------------------------------------------------------------------ parametric
    def run_parametric(self):
        p = self.project
        if p is None or self._no_models(self.can_par):
            return
        z0 = self.base()
        if not self._check_base(z0):
            return
        if self.cb_layout.currentData() == "all":
            idx = sens.variables(p)
            if not idx:
                self.can_par.message("There are no process or categoric factors to vary.")
                return
            lo, hi = p.coded_bounds()
            res = [sens.parametric(p, z0, i, None if p.factors[i].categoric
                                   else np.linspace(lo[i], hi[i], self.sp_steps.value()), source=self._source())
                   for i in idx]
            self.draw_parametric_all(res, z0)
            return
        xi, zi = self.cb_px.currentData(), self.cb_pz.currentData()
        if xi is None:
            return
        if zi == xi:
            zi = None
        if p.factors[xi].categoric:
            xv = None
        else:
            a, b = (float(p.factor_to_coded(xi, v)) for v in (self.sp_from.value(), self.sp_to.value()))
            xv = np.linspace(a, b, self.sp_steps.value())
        zv = None
        if zi is not None:
            if p.factors[zi].categoric:
                zv = np.arange(len(p.factors[zi].levels), dtype=float)
            else:
                lo, hi = p.coded_bounds()
                zv = np.linspace(lo[zi], hi[zi], self.sp_nz.value())
        res = sens.parametric(p, z0, xi, xv, zi=zi, z_values=zv, source=self._source())
        self.draw_parametric(res, z0)

    def draw_parametric(self, res, z0):
        p = self.project
        fig = self.can_par.figure
        fig.clear()
        js = list(res["Y"])
        xi, zi = res["xi"], res["zi"]
        cat = p.factors[xi].categoric
        xa = res["x"] if cat else p.factor_to_actual(xi, res["x"])
        for ax, j in zip(_grid(fig, len(js)), js):
            for r, zc in enumerate(res["z"]):
                lab = None if zc is None else f"{p.factors[zi].name} = {self._label_value(zi, zc)}"
                col = plots.COLORS[r % len(plots.COLORS)]
                if cat:
                    ax.plot(xa, res["Y"][j][r], "o-", color=col, label=lab)
                else:
                    ax.plot(xa, res["Y"][j][r], color=col, lw=2, label=lab)
            if cat:
                ax.set_xticks(xa, [str(v) for v in p.factors[xi].levels])
            else:
                ax.axvline(float(p.factor_to_actual(xi, z0[xi])), color="#94a3b8", ls=":", lw=1,
                           label="Reference")
            ax.set_xlabel(p.factor_label(xi))
            ax.set_ylabel(p.response_label(j))
            ax.set_title(p.response_label(j))
            ax.grid(alpha=0.3)
            if zi is not None:
                ax.legend(fontsize=7)
        fig.suptitle(f"Parametric study: {p.factors[xi].name}" + (f" with curves for {p.factors[zi].name}"
                                                                  if zi is not None else ""), fontsize=10)
        self.can_par.draw()
        h = [self._header("Parametric Study", z0), "<table><tr>"]
        h.append(f"<th>{html.escape(p.factor_label(xi))}</th>")
        if zi is not None:
            h.append(f"<th>{html.escape(p.factor_label(zi))}</th>")
        h += [f"<th>{html.escape(p.response_label(j))}</th>" for j in js] + ["</tr>"]
        for r, zc in enumerate(res["z"]):
            for c, x in enumerate(res["x"]):
                row = [self._label_value(xi, x)] + ([self._label_value(zi, zc)] if zi is not None else [])
                row += ["" if np.isnan(res["Y"][j][r, c]) else fmt(res["Y"][j][r, c], 6) for j in js]
                h.append("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
        h.append("</table><p class='note'>Empty cells: the point lies outside the design region (bounds or "
                 "constraints).</p>")
        self.txt_par.setHtml("".join(h))

    def draw_parametric_all(self, results, z0):
        """One panel per factor; every response on its own vertical axis (colors and markers match the legend)."""
        p = self.project
        fig = self.can_par.figure
        fig.clear()
        n = len(results)
        ncol = min(2, n)
        nrow = int(np.ceil(n / ncol))
        axs = np.atleast_1d(fig.subplots(nrow, ncol, squeeze=False)).ravel()
        for ax in axs[n:]:
            ax.axis("off")
        h = [self._header("Parametric Study (one factor at a time)", z0)]
        for k, (ax, res) in enumerate(zip(axs, results)):
            i = res["xi"]
            letter = chr(ord("a") + k) if k < 26 else str(k + 1)
            cat = p.factors[i].categoric
            xa = res["x"] if cat else p.factor_to_actual(i, res["x"])
            js = list(res["Y"])
            handles = []
            cols = {p.factor_label(i): xa if not cat else [str(v) for v in p.factors[i].levels]}
            for m, j in enumerate(js):
                col, mk = SERIES[m % len(SERIES)]
                a = ax if m == 0 else ax.twinx()
                if m >= 2:
                    a.spines["right"].set_position(("axes", 1 + 0.2 * (m - 1)))
                if m:
                    a._regen_only = True            # the data of every axis are exported with the panel
                y = res["Y"][j][0]
                cols[p.response_label(j)] = y
                handles += a.plot(xa, y, "-", marker=mk, ms=5, color=col, lw=1.6, label=p.response_label(j))
                a.set_ylabel(p.response_label(j), color=col)
                a.tick_params(axis="y", colors=col)
                if m:
                    a.spines["right"].set_color(col)
            if cat:
                ax.set_xticks(xa, [str(v) for v in p.factors[i].levels])
            else:
                ax.axvline(float(p.factor_to_actual(i, z0[i])), color="#94a3b8", ls=":", lw=1)
            ax.set_xlabel(p.factor_label(i))
            ax.set_title(f"({letter}) {p.factors[i].name}", fontsize=10, pad=20)
            ax.grid(alpha=0.3)
            ax.legend(handles=handles, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                      ncol=min(len(handles), 3), frameon=False, handlelength=2.5)
            plotdata.register_columns(ax, f"({letter}) {p.factors[i].name}", cols, only=True)
            h.append(f"<h4>({letter}) {html.escape(p.factor_label(i))}</h4><table><tr>"
                     f"<th>{html.escape(p.factor_label(i))}</th>"
                     + "".join(f"<th>{html.escape(p.response_label(j))}</th>" for j in js) + "</tr>")
            for c, x in enumerate(res["x"]):
                h.append(f"<tr><td>{self._label_value(i, x)}</td>" + "".join(
                    f"<td>{'' if np.isnan(res['Y'][j][0, c]) else fmt(res['Y'][j][0, c], 6)}</td>" for j in js)
                    + "</tr>")
            h.append("</table>")
        fig.suptitle("Parametric study: each factor varied over its range, the others at the reference point "
                     "(dotted line)", fontsize=10)
        self.can_par.draw()
        h.append("<p class='note'>Empty cells: the point lies outside the design region (bounds or "
                 "constraints).</p>")
        self.txt_par.setHtml("".join(h))

    # ------------------------------------------------------------------ global sensitivity
    def run_global(self):
        if self.cb_gmethod.currentData() == "sobol":
            self.run_sobol()
        else:
            self.run_src()

    def run_src(self):
        p = self.project
        if p is None or self._no_models(self.can_sob):
            return
        z0 = self.base()
        if not self._check_base(z0):
            return
        try:
            res = sens.regression_significance(p, z0, n=self.sp_n.value(), source=self._source())
        except ValueError as exc:
            self.can_sob.message(str(exc))
            return
        idx, js = res["idx"], list(res["results"])
        fig = self.can_sob.figure
        fig.clear()
        ax = fig.add_subplot(111)
        pos = np.arange(len(idx))
        w = 0.8 / max(len(js), 1)
        for m, j in enumerate(js):
            r = res["results"][j]
            beta = np.array([np.nan if b is None else b for b in r["beta"]])
            yy = pos + (m - (len(js) - 1) / 2) * w
            ax.barh(yy, beta, w, color=plots.COLORS[m % len(plots.COLORS)], label=p.response_label(j))
            for c in np.where(np.isnan(beta))[0]:
                ax.text(0, yy[c], f"  F-test, p = {fmt_p(r['p'][c])}", va="center", fontsize=7, color="#555")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(pos, [p.factors[i].name for i in idx])
        ax.invert_yaxis()
        ax.set_xlabel("Standardized coefficient β")
        ax.set_title(f"Significance of the decision variables ({res['n']}-point Latin hypercube)")
        ax.grid(alpha=0.3, axis="x")
        ax.legend(fontsize=7)
        self.can_sob.draw()
        h = [self._header("Significance of the Decision Variables", z0),
             f"<p class='note'>Standardized coefficient β (p-value) from a {res['n']}-point Latin hypercube "
             "design of experiments over the factor space.</p><table><tr><th class='l'>Decision variable (range)</th>"]
        h += [f"<th>{html.escape(p.response_label(j))}</th>" for j in js] + ["</tr>"]
        for c, i in enumerate(idx):
            h.append(f"<tr><td class='l'>{html.escape(p.factor_label(i))} ({html.escape(self._range_text(i))})</td>")
            for j in js:
                b, pv = res["results"][j]["beta"][c], res["results"][j]["p"][c]
                cls = "sig" if pv < 0.05 else "ns"
                txt = f"{b:+.3f} ({fmt_p(pv)})" if b is not None else f"F-test ({fmt_p(pv)})"
                h.append(f"<td class='{cls}'>{txt}</td>")
            h.append("</tr>")
        h.append("<tr><td class='l'>R² of the linear regression</td>"
                 + "".join(f"<td>{res['results'][j]['r2']:.3f}</td>" for j in js) + "</tr></table>")
        h.append("<p class='note'><b>How to read:</b> β is the change of the response, in standard deviations, when "
                 "the factor changes by one standard deviation over the sample; its sign gives the direction and "
                 "|β| the strength. p &lt; 0.05 (green): the linear effect is significant. Categoric factors with more "
                 "than two levels are tested with an F-test of their level indicators (no single β). The responses "
                 "are predicted with the fitted models, so the p-values measure how clearly the linear trend stands "
                 "out from the nonlinear part over this sample (they depend on the sample size), not the "
                 "experimental error. β describes the response well only when R² is high (for example above 0.7); "
                 "for strongly nonlinear responses use the Sobol indices.</p>")
        if p.constraints:
            h.append("<p class='note'>Sample points that violate the design constraints were replaced by new "
                     "points.</p>")
        self.txt_sob.setHtml("".join(h))

    def _range_text(self, i):
        p = self.project
        f = p.factors[i]
        if f.categoric:
            return ", ".join(str(v) for v in f.levels)
        lo, hi = sens.factor_range(p, i)
        a, b = sorted(float(v) for v in p.factor_to_actual(i, np.array([lo, hi])))
        return f"{fmt(a, 5)}–{fmt(b, 5)}"

    def run_sobol(self):
        p = self.project
        if p is None or self._no_models(self.can_sob) or self.task is not None:
            return
        z0 = self.base()
        if not self._check_base(z0):
            return
        if not sens.variables(p):
            self.can_sob.message("Sobol indices need at least one process or categoric factor.\n"
                                 "Mixture components sum to a constant and are not independent inputs.")
            return
        n, src = self.sp_n.value(), self._source()
        self.btn_sob.setEnabled(False)
        self.can_sob.message("Computing Sobol indices...")
        self.task = Task(lambda report, cancelled: sens.sobol(p, z0, n=n, source=src), self)
        self.task.done.connect(lambda res: self._sobol_done(res, z0))
        self.task.failed.connect(self._sobol_failed)
        self.task.start()

    def _sobol_failed(self, msg):
        self.task = None
        self.btn_sob.setEnabled(True)
        self.can_sob.message(f"Sobol indices could not be computed:\n{msg}")

    def _sobol_done(self, res, z0):
        self.task = None
        self.btn_sob.setEnabled(True)
        self._sobol = res
        p = self.project
        idx = res["idx"]
        names = [p.factors[i].name for i in idx]
        fig = self.can_sob.figure
        fig.clear()
        js = list(res["results"])
        x = np.arange(len(idx))
        for ax, j in zip(_grid(fig, len(js)), js):
            r = res["results"][j]
            for k, (key, lab, col) in enumerate((("S1", "First order S1", plots.COLORS[0]),
                                                 ("ST", "Total ST", plots.COLORS[1]))):
                v, ci = r[key], r[key + "_ci"]
                ax.bar(x + (k - 0.5) * 0.38, v, 0.38, color=col, label=lab,
                       yerr=[np.maximum(v - ci[:, 0], 0), np.maximum(ci[:, 1] - v, 0)], capsize=3)
            ax.set_xticks(x, names)
            ax.set_ylim(min(0, ax.get_ylim()[0]), max(1.0, ax.get_ylim()[1]))
            ax.set_ylabel("Sobol index")
            ax.set_title(p.response_label(j))
            ax.grid(alpha=0.3, axis="y")
            ax.legend(fontsize=7)
        fig.suptitle("Global sensitivity: share of the response variance (bars: 95% bootstrap interval)", fontsize=10)
        self.can_sob.draw()
        h = [self._header("Global Sensitivity (Sobol Indices)", z0)]
        for j in js:
            r = res["results"][j]
            h.append(f"<h4>{html.escape(p.response_label(j))}</h4><table><tr><th class='l'>Factor</th>"
                     "<th>S1</th><th>S1 95% interval</th><th>ST</th><th>ST 95% interval</th><th>ST − S1</th></tr>")
            order = np.argsort(-r["ST"])
            for c in order:
                h.append(f"<tr><td class='l'>{html.escape(p.factor_label(idx[c]))}</td><td>{r['S1'][c]:.3f}</td>"
                         f"<td>{r['S1_ci'][c, 0]:.3f} to {r['S1_ci'][c, 1]:.3f}</td><td>{r['ST'][c]:.3f}</td>"
                         f"<td>{r['ST_ci'][c, 0]:.3f} to {r['ST_ci'][c, 1]:.3f}</td>"
                         f"<td>{r['ST'][c] - r['S1'][c]:.3f}</td></tr>")
            h.append(f"</table><p class='note'>Sum of S1 = {np.sum(r['S1']):.3f} (1 = purely additive response; the "
                     "rest is due to interactions).</p>")
        h.append(f"<p class='note'><b>How to read:</b> S1 is the share of the response variance caused by the factor "
                 f"alone; ST also includes all its interactions. A factor with ST close to 0 has no influence over "
                 f"the range and can be fixed. ST − S1 is the part due to interactions. The factors vary uniformly "
                 f"over their whole range (Sobol sequence, N = {res['n']:,}, {res['evaluations']:,} model "
                 f"evaluations per response); other factors stay at the reference point. Estimators: Saltelli et al. "
                 f"(2010) for S1 and Jansen (1999) for ST.</p>")
        if res["constraints_ignored"]:
            h.append("<p class='warn'>The linear constraints of the design are not applied here, because Sobol "
                     "indices assume independent factors; the indices describe the whole factor box.</p>")
        self.txt_sob.setHtml("".join(h))

    # ------------------------------------------------------------------ tornado
    def run_tornado(self):
        p = self.project
        if p is None or self._no_models(self.can_tor):
            return
        z0 = self.base()
        if not self._check_base(z0):
            return
        if not sens.variables(p):
            self.can_tor.message("There are no process or categoric factors to vary.")
            return
        res = sens.tornado(p, z0, self.sp_delta.value() / 100, source=self._source())
        fig = self.can_tor.figure
        fig.clear()
        js = list(res["results"])
        for ax, j in zip(_grid(fig, len(js)), js):
            rows = sorted(res["results"][j], key=lambda r: np.nan_to_num(abs(r["y_hi"] - r["y_lo"])))
            y0 = res["y0"][j]
            pos = np.arange(len(rows))
            lo = np.array([r["y_lo"] for r in rows]) - y0
            hi = np.array([r["y_hi"] for r in rows]) - y0
            ax.barh(pos, lo, left=y0, color=plots.COLORS[0], label="Low setting")
            ax.barh(pos, hi, left=y0, color=plots.COLORS[1], label="High setting")
            ax.axvline(y0, color="k", lw=1)
            ax.set_yticks(pos, [p.factors[r["i"]].name for r in rows])
            ax.set_xlabel(p.response_label(j))
            ax.set_title(p.response_label(j))
            ax.grid(alpha=0.3, axis="x")
            ax.legend(fontsize=7)
        fig.suptitle(f"Local sensitivity: each factor changed by ±{self.sp_delta.value()}% of its range "
                     "(categoric factors: lowest and highest level effect)", fontsize=10)
        self.can_tor.draw()
        h = [self._header("Local Sensitivity (Tornado)", z0)]
        for j in js:
            y0 = res["y0"][j]
            h.append(f"<h4>{html.escape(p.response_label(j))} (reference value {fmt(y0, 5)})</h4><table><tr>"
                     "<th class='l'>Factor</th><th>Low setting</th><th>High setting</th><th>Response at low</th>"
                     "<th>Response at high</th><th>Swing</th><th>Normalized sensitivity</th></tr>")
            rows = sorted(res["results"][j], key=lambda r: -np.nan_to_num(abs(r["y_hi"] - r["y_lo"])))
            for r in rows:
                i = r["i"]
                swing = r["y_hi"] - r["y_lo"]
                if r["categoric"] or r["x_hi"] == r["x_lo"]:
                    ns = ""
                else:
                    span = float(np.ptp(p.factor_to_actual(i, np.array(sens.factor_range(p, i)))))
                    dx = abs(float(p.factor_to_actual(i, r["x_hi"]) - p.factor_to_actual(i, r["x_lo"])))
                    ns = fmt(swing / dx * span / abs(y0), 4) if y0 != 0 and dx > 0 else ""
                h.append(f"<tr><td class='l'>{html.escape(p.factor_label(i))}</td>"
                         f"<td>{self._label_value(i, r['x_lo'])}</td><td>{self._label_value(i, r['x_hi'])}</td>"
                         f"<td>{fmt(r['y_lo'], 5)}</td><td>{fmt(r['y_hi'], 5)}</td><td>{fmt(swing, 4)}</td>"
                         f"<td>{ns}</td></tr>")
            h.append("</table>")
        h.append("<p class='note'><b>How to read:</b> the longest bars belong to the factors that change the response "
                 "most near the reference point. Normalized sensitivity = (change of the response / change of the "
                 "factor) × factor range / reference response: the relative change of the response for a change of "
                 "the factor over its whole range, at the local slope. Unlike Sobol indices, this result is valid "
                 "only near the reference point.</p>")
        self.txt_tor.setHtml("".join(h))

    # ------------------------------------------------------------------ Pareto front
    def run_front(self):
        p = self.project
        if p is None:
            return
        r = p.nsga_result
        if not r:
            self.can_front.message("Run NSGA-II / NSGA-III on the multi-objective optimization page first.")
            self.txt_front.setHtml("")
            return
        res = sens.front_analysis(p, r)
        objs = res["objectives"]
        onames = [p.responses[j].name for j in objs]
        fnames = [p.factors[i].name for i in res["factors"]]
        X = np.asarray(r["X"], float)
        Y0 = np.asarray(r["Yobj"][0], float)
        order = np.argsort(Y0)
        fig = self.can_front.figure
        fig.clear()
        n_f = len(res["factors"])
        two = len(objs) == 2
        n_pan = 1 + n_f + (1 if two else 0)
        axs = _grid(fig, n_pan)
        ax = axs[0]
        if n_f:
            im = ax.imshow(res["rho"], cmap="bwr", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(objs)), onames, rotation=30 if len(objs) > 3 else 0)
            ax.set_yticks(range(n_f), fnames)
            for a in range(n_f):
                for b in range(len(objs)):
                    v = res["rho"][a, b]
                    if np.isfinite(v):
                        ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=8,
                                color="white" if abs(v) > 0.6 else "black")
            fig.colorbar(im, ax=ax, label="Spearman ρ")
        ax.set_title("Factor vs objective on the front")
        for ax, a in zip(axs[1:1 + n_f], range(n_f)):
            i = res["factors"][a]
            xa = X[order, i] if p.factors[i].categoric else p.factor_to_actual(i, X[order, i])
            ax.plot(Y0[order], xa, "o-", ms=3, color=plots.COLORS[a % len(plots.COLORS)])
            ax.set_xlabel(p.response_label(objs[0]))
            ax.set_ylabel(p.factor_label(i))
            ax.set_title(f"{p.factors[i].name} along the front")
            ax.grid(alpha=0.3)
        if two:
            ax = axs[-1]
            mid, rate = sens.tradeoff_rates(r["Yobj"][0], r["Yobj"][1])
            ax.plot(mid, rate, "o-", ms=3, color=plots.COLORS[3])
            ax.axhline(0, color="#888", lw=0.8)
            ax.set_xlabel(p.response_label(objs[0]))
            ax.set_ylabel(f"Δ{onames[1]} / Δ{onames[0]}")
            ax.set_title("Trade-off rate along the front")
            ax.grid(alpha=0.3)
        fig.suptitle(f"Pareto front trade-off ({res['n']} solutions, {r.get('algorithm', 'NSGA')})", fontsize=10)
        self.can_front.draw()

        h = [REPORT_CSS, "<h3>Pareto Front Trade-off</h3><h4>Spearman correlation of each factor with each objective "
                         "along the front</h4><table><tr><th class='l'>Factor</th>"]
        h += [f"<th>{html.escape(n)} ρ</th><th>p</th>" for n in onames] + ["</tr>"]
        for a, i in enumerate(res["factors"]):
            h.append(f"<tr><td class='l'>{html.escape(p.factor_label(i))}</td>")
            for b in range(len(objs)):
                h.append(f"<td>{res['rho'][a, b]:.3f}</td><td>{fmt_p(res['p'][a, b])}</td>")
            h.append("</tr>")
        h.append("</table><h4>Correlation between the objectives on the front</h4><table><tr><th class='l'></th>")
        h += [f"<th>{html.escape(n)}</th>" for n in onames] + ["</tr>"]
        for a, n in enumerate(onames):
            h.append(f"<tr><td class='l'>{html.escape(n)}</td>"
                     + "".join(f"<td>{res['rho_obj'][a, b]:.3f}</td>" for b in range(len(objs))) + "</tr>")
        h.append("</table><p class='note'><b>How to read:</b> a factor with |ρ| close to 1 changes steadily along the "
                 "front and drives the trade-off between the objectives; a factor with ρ close to 0 stays nearly "
                 "constant on the front, so it can be set once. A negative correlation between two objectives means "
                 "they conflict: one improves only when the other gets worse. The trade-off rate (two objectives) is "
                 "the change of the second objective per unit change of the first between neighbouring solutions. "
                 "The p-values describe the ordering of the solutions on this front, not the uncertainty of the "
                 "models.</p>")
        self.txt_front.setHtml("".join(h))
