"""Design evaluation tab: degrees of freedom, aliases, standard error, VIF, power, FDS, SE contour.
No response data needed."""
import numpy as np
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout,
                               QWidget)
from PySide6.QtCore import Qt

from doe import evaluation, models

from . import plots
from .copying import ReportBrowser
from .common import REPORT_CSS, MplCanvas, fmt

HIGHER = {"mean": "linear", "linear": "2fi", "2fi": "3fi", "3fi": "3fi", "quadratic": "cubic", "cubic": "cubic",
          "special_cubic": "cubic"}


class EvalTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._loading = False
        self.ev = None

        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Model to evaluate:"))
        self.cb_order = QComboBox()
        bar.addWidget(self.cb_order)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Signal Δ:"))
        self.sp_delta = QDoubleSpinBox()
        self.sp_delta.setRange(0.01, 1e6)
        self.sp_delta.setValue(2.0)
        self.sp_delta.setToolTip("Response difference to detect between the low and high levels")
        bar.addWidget(self.sp_delta)
        bar.addWidget(QLabel("Noise σ:"))
        self.sp_sigma = QDoubleSpinBox()
        self.sp_sigma.setRange(0.0001, 1e6)
        self.sp_sigma.setValue(1.0)
        bar.addWidget(self.sp_sigma)
        bar.addWidget(QLabel("α:"))
        self.cb_alpha = QComboBox()
        for a in (0.01, 0.05, 0.10):
            self.cb_alpha.addItem(f"{a:g}", a)
        self.cb_alpha.setCurrentIndex(1)
        bar.addWidget(self.cb_alpha)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Graph:"))
        self.cb_graph = QComboBox()
        self.cb_graph.addItems(["Fraction of Design Space (FDS)", "Prediction Std. Error Contour", "Leverage per Run",
                                "Coefficient Correlation Matrix", "Pearson r Between Terms"])
        bar.addWidget(self.cb_graph)
        bar.addStretch()
        lay.addLayout(bar)

        split = QSplitter(Qt.Horizontal)
        self.txt = ReportBrowser()
        self.canvas = MplCanvas(figsize=(6, 5))
        split.addWidget(self.txt)
        split.addWidget(self.canvas)
        split.setSizes([620, 560])
        lay.addWidget(split)

        for w in (self.cb_order, self.cb_alpha, self.cb_graph):
            w.currentIndexChanged.connect(self.refresh)
        for w in (self.sp_delta, self.sp_sigma):
            w.valueChanged.connect(self.refresh)

    def set_project(self, project):
        self.project = project
        self._loading = True
        self.cb_order.clear()
        for o in project.available_orders():
            self.cb_order.addItem(models.order_label(o), o)
        default = project.options.get("model_order") or models.default_order(project.design_type)
        self.cb_order.setCurrentIndex(max(self.cb_order.findData(default), 0))
        self._loading = False
        self.refresh()

    def refresh(self, *_):
        p = self.project
        if p is None or self._loading:
            return
        order = self.cb_order.currentData()
        terms = p.terms_for_order(order)
        higher = HIGHER.get(order.split("|")[0], order) if "|" not in order else order
        try:
            potential = p.terms_for_order(higher) if "|" not in order else None
        except ValueError:
            potential = None
        coded = p.coded[np.all(np.isfinite(p.coded), axis=1)]
        if len(coded) < 2:
            self.txt.setHtml(REPORT_CSS + "<p>No complete runs to evaluate yet.</p>")
            self.canvas.message("")
            return
        try:
            ev = evaluation.evaluate(coded, terms, p.is_mixture, p.blocks[np.all(np.isfinite(p.coded), axis=1)]
                                     if p.n_blocks > 1 else None, p.space, self.sp_delta.value(),
                                     self.sp_sigma.value(), self.cb_alpha.currentData(), potential)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.txt.setHtml(REPORT_CSS + f"<p class='ns'>{exc}</p>")
            return
        self.ev = ev
        self.corr = evaluation.correlations(ev)
        self.measures = evaluation.matrix_measures(ev, p.region_samples(3000))
        self.txt.setHtml(REPORT_CSS + self.html(ev, order))
        self.draw()

    def html(self, ev, order):
        p = self.project
        h = [f"<h2>Design Evaluation - {models.order_label(order)} model</h2>",
             f"<p class='note'>{p.design_description()}</p>",
             "<h3>Degrees of Freedom</h3><table>"]
        rows = [("Number of runs", ev["n"]), ("Model coefficients (including intercept/blocks)", ev["p"]),
                ("Residual", ev["df_resid"]), ("Lack of fit", ev["df_lof"]), ("Pure error", ev["df_pe"])]
        for lab, v in rows:
            h.append(f"<tr><td class='l'>{lab}</td><td>{v}</td></tr>")
        h.append("</table>")
        notes = []
        if ev["df_resid"] <= 0:
            notes.append("<span class='ns'>No residual df - the model cannot be tested. Add runs.</span>")
        if ev["df_pe"] < 3:
            notes.append("Pure error &lt; 3 df: add replicates so lack of fit can be tested properly.")
        if ev["df_lof"] < 3 and ev["df_resid"] > 0:
            notes.append("Lack of fit &lt; 3 df: add unique points so the model fit can be checked.")
        if notes:
            h.append("<p class='note'>" + "<br>".join(notes) + "</p>")
        h.append("<div class='box'><p class='note' style='margin: 2px'>At least <b>3 lack of fit df</b> and "
                 "<b>4 pure error df</b> are recommended for a valid lack of fit test, plus ≥ 5 residual df for a "
                 "reliable error estimate.</p></div>")
        if ev["aliased"]:
            h.append("<p class='ns'>Terms that cannot be estimated (aliased): "
                     + ", ".join(models.term_name(t) for t in ev["aliased"]) + "</p>")
        h.append(f"<h3>Coefficients (Δ = {fmt(self.sp_delta.value())}, σ = {fmt(self.sp_sigma.value())}, "
                 f"α = {self.cb_alpha.currentData():g})</h3>")
        h.append("<table><tr><th class='l'>Term</th><th>Std. Error (×σ)</th><th>VIF</th><th>Ri²</th>"
                 "<th>Power</th></tr>")
        for lab, pos in zip(ev["labels"], ev["dc"].term_idx):
            v = ev["vif"][pos]
            pw = ev["power"][pos]
            cls = "" if np.isnan(pw) else ("sig" if pw >= 0.8 else "ns")
            h.append(f"<tr><td class='l'>{lab}</td><td>{fmt(ev['se'][pos], 4)}</td>"
                     f"<td>{'' if np.isnan(v) else fmt(v, 4)}</td>"
                     f"<td>{'' if np.isnan(v) else fmt(max(0.0, round(1 - 1 / v, 10)), 4)}</td>"
                     f"<td class='{cls}'>{'' if np.isnan(pw) else f'{100 * pw:.1f} %'}</td></tr>")
        h.append("</table><p class='note'>Power = probability of detecting an effect of size Δ (response difference "
                 "between the low and high levels) given noise σ. The usual target is ≥ 80%. Ideal VIF = 1 "
                 "(orthogonal); VIF &gt; 10 indicates high correlation between terms.</p>")
        if ev["alias"]:
            h.append("<h3>Alias Structure (model term ← higher-order term)</h3><table>"
                     "<tr><th class='l'>Term</th><th class='l'>Alias</th></tr>")
            for t, links in ev["alias"]:
                h.append(f"<tr><td class='l'>[{t}]</td><td class='l'>"
                         + " ".join(f"{c:+.3g}·{name}" for name, c in links) + "</td></tr>")
            h.append("</table><p class='note'>The estimate of the term on the left is biased by the terms on the right "
                     "if those terms actually have an effect.</p>")
        m = self.measures
        rows = [("Condition number (X'X)", fmt(m["cond"], 4), "1 = orthogonal; > 100 multicollinear, > 1000 severe"),
                ("Determinant (X'X)⁻¹", fmt(m["det_inv"], 4), "smaller is better (D criterion)"),
                ("Trace (X'X)⁻¹", fmt(m["trace_inv"], 4), "smaller is better (A criterion)"),
                ("D-efficiency (%)", fmt(m["d_eff"], 4), "relative to an ideal ±1 orthogonal design"),
                ("A-efficiency (%)", fmt(m["a_eff"], 4), "mean coefficient variance"),
                ("G-efficiency (%)", fmt(m.get("g_eff", np.nan), 4), "worst prediction variance in the design region"),
                ("Mean prediction variance (I, ×σ²)", fmt(m.get("pv_mean", np.nan), 4), "I criterion - smaller is better"),
                ("Max / min prediction variance (×σ²)", f"{fmt(m.get('pv_max', np.nan), 4)} / {fmt(m.get('pv_min', np.nan), 4)}",
                 "in the design region"),
                ("Max / mean leverage", f"{fmt(m['lev_max'], 4)} / {fmt(m['lev_mean'], 4)}", "mean = p/n")]
        h.append("<h3>Matrix Measures</h3><table><tr><th class='l'>Measure</th><th>Value</th><th class='l'>Meaning</th></tr>"
                 + "".join(f"<tr><td class='l'>{a}</td><td>{b}</td><td class='l'>{c}</td></tr>" for a, b, c in rows)
                 + "</table><p class='note'>Efficiencies are computed in coded units. Use them to compare designs "
                   "with different numbers of runs, or to assess historical data.</p>")
        labs, coef, labs2, R = self.corr
        if len(labs2) > 1:
            off = np.abs(R - np.eye(len(R)))
            pairs = sorted(((off[a, b], labs2[a], labs2[b], R[a, b]) for a in range(len(R)) for b in range(a + 1, len(R))),
                           reverse=True)
            top = [q for q in pairs if q[0] >= 0.3][:10]
            h.append("<h3>Correlation Between Terms (Pearson r)</h3>")
            if top:
                h.append("<table><tr><th class='l'>Term</th><th class='l'>Term</th><th>r</th></tr>" + "".join(
                    f"<tr><td class='l'>{a}</td><td class='l'>{b}</td><td class='{'ns' if abs(r) >= 0.7 else ''}'>"
                    f"{r:+.3f}</td></tr>" for _, a, b, r in top) + "</table>")
            else:
                h.append("<p class='sig'>No term pair has |r| ≥ 0.3 - the terms are nearly independent.</p>")
            h.append(f"<p class='note'>Maximum correlation |r| = {max((q[0] for q in pairs), default=0):.3f}. "
                     "See the full heatmaps in the 'Pearson r Between Terms' and 'Coefficient Correlation Matrix' graphs.</p>")
        lev = ev["leverage"]
        h.append(f"<h3>Leverage</h3><p>Mean = {fmt(lev.mean(), 4)}, maximum = {fmt(lev.max(), 4)}. "
                 "A leverage close to 1 means that run strongly determines the model (if it fails, the model is hard "
                 "to estimate).</p>")
        return "".join(h)

    def draw(self):
        p, ev = self.project, self.ev
        fig = self.canvas.figure
        fig.clear()
        kind = self.cb_graph.currentIndex()
        if kind == 0:
            pts = p.region_samples(4000)
            if not len(pts):
                self.canvas.message("The design region is empty.")
                return
            frac, se = evaluation.fds_curve(ev, pts)
            ax = fig.add_subplot(111)
            ax.plot(frac, se * self.sp_sigma.value(), color=plots.BLUE, lw=2)
            for q in (0.5, 0.8):
                v = np.interp(q, frac, se) * self.sp_sigma.value()
                ax.axvline(q, color="#999", ls=":", lw=1)
                ax.annotate(f"{fmt(v, 3)}", (q, v), xytext=(4, 6), textcoords="offset points", fontsize=9)
            ax.set_xlabel("Fraction of design space")
            ax.set_ylabel("Std. error of mean prediction")
            ax.set_title("Fraction of Design Space (FDS)")
            ax.grid(alpha=0.3)
        elif kind == 1:
            num = p.comp_idx[:3] if p.is_mixture else p.num_idx[:2]
            need = 3 if p.is_mixture else 2
            if len(num) < need:
                self.canvas.message("At least 2 numeric factors (or 3 components) are required.")
                return
            base = np.nanmean(p.coded, axis=0) if p.is_mixture else np.zeros(p.k)
            for i in p.cat_idx:
                base[i] = 0
            grid = plots.Grid(p, num, base)
            if grid.empty:
                self.canvas.message("The slice is empty.")
                return
            Z = evaluation.std_error_prediction(ev, grid.pts) * self.sp_sigma.value()
            ax = fig.add_subplot(111)
            grid.decorate(ax)
            cf = grid.fill(ax, Z, 12, cmap="viridis_r")
            cs = grid.lines(ax, Z, 12, colors="k", linewidths=0.5)
            grid.clabel(ax, cs, inline=True, fontsize=8, fmt=lambda v: fmt(v, 3))
            fig.colorbar(cf, ax=ax, label="Prediction std. error")
            grid.design_points(ax)
            ax.set_title("Prediction Std. Error Contour (other factors at center)")
        elif kind in (3, 4):
            labs, coef, labs2, R = self.corr
            M, L = (coef, labs) if kind == 3 else (R, labs2)
            ax = fig.add_subplot(111)
            im = ax.imshow(M, cmap="bwr", vmin=-1, vmax=1)
            ax.set_xticks(range(len(L)), L, rotation=90, fontsize=8)
            ax.set_yticks(range(len(L)), L, fontsize=8)
            if len(L) <= 16:
                for a in range(len(L)):
                    for b in range(len(L)):
                        if a != b and abs(M[a, b]) >= 0.05:
                            ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=6.5,
                                    color="white" if abs(M[a, b]) > 0.6 else "#1e293b")
            fig.colorbar(im, ax=ax, label="Correlation", fraction=0.046)
            ax.set_title("Correlation between coefficients, from (X'X)⁻¹" if kind == 3 else "Pearson r between term columns")
            ax.grid(False)
        else:
            ax = fig.add_subplot(111)
            run = p.run_order[np.all(np.isfinite(p.coded), axis=1)]
            o = np.argsort(run)
            plots._stems(ax, run[o], ev["leverage"][o], plots.BLUE)
            ax.axhline(2 * ev["p"] / ev["n"], color=plots.RED, ls="--", lw=1, label="2p/n")
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Run number")
            ax.set_ylabel("Leverage")
            ax.set_title("Leverage per Run")
            ax.legend()
            ax.grid(alpha=0.3)
        self.canvas.draw()
