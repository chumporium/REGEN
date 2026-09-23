"""Analysis tab: Fit Summary, term selection, transformation, ANOVA, equation, diagnostics, Box-Cox."""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from doe import models

from . import plots, report
from .copying import ReportBrowser
from .common import REPORT_CSS, MplCanvas
from .diagnostics import DiagnosticsPanel


class AnalysisTab(QWidget):
    model_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._loading = False
        self._suggested = None
        self._bc = None

        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Response:"))
        self.cb_resp = QComboBox()
        self.cb_resp.setMinimumWidth(190)
        bar.addWidget(self.cb_resp)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Model:"))
        self.cb_order = QComboBox()
        bar.addWidget(self.cb_order)
        self.btn_suggest = QPushButton("Use Suggested Model")
        bar.addWidget(self.btn_suggest)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Alpha out:"))
        self.sp_alpha = QDoubleSpinBox()
        self.sp_alpha.setRange(0.01, 0.5)
        self.sp_alpha.setSingleStep(0.01)
        self.sp_alpha.setValue(0.10)
        bar.addWidget(self.sp_alpha)
        self.btn_backward = QPushButton("Backward Elimination")
        self.btn_backward.setToolTip("Remove non-significant terms one at a time (keeps the model hierarchical)")
        bar.addWidget(self.btn_backward)
        bar.addStretch()
        lay.addLayout(bar)

        bar2 = QHBoxLayout()
        bar2.addWidget(QLabel("Response transformation:"))
        self.cb_tr = QComboBox()
        for key, label in models.TRANSFORMS.items():
            self.cb_tr.addItem(label, key)
        bar2.addWidget(self.cb_tr)
        bar2.addWidget(QLabel("λ:"))
        self.sp_lam = QDoubleSpinBox()
        self.sp_lam.setRange(-3, 3)
        self.sp_lam.setSingleStep(0.1)
        self.sp_lam.setDecimals(2)
        self.sp_lam.setValue(1.0)
        bar2.addWidget(self.sp_lam)
        bar2.addWidget(QLabel("Constant k (y + k):"))
        self.sp_shift = QDoubleSpinBox()
        self.sp_shift.setRange(-1e9, 1e9)
        self.sp_shift.setDecimals(4)
        self.sp_shift.setToolTip("Add a constant if any response value is ≤ 0 (for log/inverse)")
        bar2.addWidget(self.sp_shift)
        self.lbl_err = QLabel()
        self.lbl_err.setStyleSheet("color: #b42318; font-weight: bold;")
        bar2.addWidget(self.lbl_err, 1)
        lay.addLayout(bar2)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Model terms (checked = in model):"))
        self.lst_terms = QListWidget()
        ll.addWidget(self.lst_terms)
        split.addWidget(left)

        self.tabs = QTabWidget()
        self.txt_summary = ReportBrowser()
        self.txt_anova = ReportBrowser()
        self.txt_equation = ReportBrowser()
        self.diag = DiagnosticsPanel()
        self.diag.excluded_changed.connect(self.on_excluded)
        bc_w = QWidget()
        bcl = QVBoxLayout(bc_w)
        bcl.setContentsMargins(0, 0, 0, 0)
        self.boxcox = MplCanvas(figsize=(7, 5))
        bcl.addWidget(self.boxcox)
        bcb = QHBoxLayout()
        self.lbl_bc = QLabel()
        self.lbl_bc.setWordWrap(True)
        self.btn_bc = QPushButton("Apply Recommendation")
        bcb.addWidget(self.lbl_bc, 1)
        bcb.addWidget(self.btn_bc)
        bcl.addLayout(bcb)
        eff_w = QWidget()
        effl = QVBoxLayout(eff_w)
        effl.setContentsMargins(0, 0, 0, 0)
        effb = QHBoxLayout()
        effb.addWidget(QLabel("Graph:"))
        self.cb_eff = QComboBox()
        self.cb_eff.addItems(["Half-Normal", "Normal", "Pareto"])
        effb.addWidget(self.cb_eff)
        self.btn_eff_auto = QPushButton("Select Automatically (Lenth ME Limit)")
        self.btn_eff_clear = QPushButton("Clear Selection")
        effb.addWidget(self.btn_eff_auto)
        effb.addWidget(self.btn_eff_clear)
        effb.addStretch()
        effl.addLayout(effb)
        self.effplot = MplCanvas(figsize=(7, 5))
        effl.addWidget(self.effplot)
        self.lbl_eff = QLabel("Click a point on the plot to add the term to or remove it from the model.")
        self.lbl_eff.setWordWrap(True)
        self.lbl_eff.setStyleSheet("color: #555;")
        effl.addWidget(self.lbl_eff)
        self.eff_tab = eff_w
        self._eff = None
        self._eff_pts = None
        self.effplot.canvas.mpl_connect("button_press_event", self.on_effect_click)
        self.cb_eff.currentIndexChanged.connect(self.draw_effects)
        self.btn_eff_auto.clicked.connect(self.auto_effects)
        self.btn_eff_clear.clicked.connect(self.clear_effects)

        self.tabs.addTab(self.txt_summary, "Fit Summary")
        self.tabs.addTab(self.txt_anova, "ANOVA")
        self.tabs.addTab(self.txt_equation, "Model Equation")
        self.tabs.addTab(self.diag, "Diagnostics")
        self.tabs.addTab(bc_w, "Box-Cox")
        self.tabs.addTab(eff_w, "Effects (Factorial)")
        split.addWidget(self.tabs)
        split.setSizes([200, 950])
        lay.addWidget(split)

        self.cb_resp.currentIndexChanged.connect(self.on_response_changed)
        self.cb_order.currentIndexChanged.connect(self.on_order_changed)
        self.lst_terms.itemChanged.connect(self.on_terms_changed)
        self.btn_backward.clicked.connect(self.on_backward)
        self.btn_suggest.clicked.connect(self.on_suggest)
        self.cb_tr.currentIndexChanged.connect(self.on_transform_changed)
        self.sp_lam.valueChanged.connect(self.on_transform_changed)
        self.sp_shift.valueChanged.connect(self.on_transform_changed)
        self.btn_bc.clicked.connect(self.apply_box_cox)

    # ---------------------------------------------------------------
    def set_project(self, project):
        self.project = project
        self.tabs.setTabVisible(self.tabs.indexOf(self.eff_tab), project.is_two_level_factorial)
        self._loading = True
        self.cb_order.clear()
        for o in project.available_orders():
            self.cb_order.addItem(models.order_label(o), o)
        self._loading = False
        self.refresh_responses()

    def refresh_responses(self):
        p = self.project
        self._loading = True
        cur = max(self.cb_resp.currentIndex(), 0)
        self.cb_resp.clear()
        for j in range(len(p.responses)):
            self.cb_resp.addItem(p.response_label(j))
        self.cb_resp.setCurrentIndex(min(cur, len(p.responses) - 1))
        self._loading = False
        self.on_response_changed()

    def refresh(self):
        if self.project is not None:
            self.on_response_changed()

    def resp(self):
        return self.cb_resp.currentIndex()

    def on_response_changed(self):
        if self._loading or self.project is None or self.resp() < 0:
            return
        spec = self.project.model_spec(self.resp())
        tr = spec.get("transform") or {"kind": "none"}
        self._loading = True
        self.cb_order.setCurrentIndex(max(self.cb_order.findData(spec["order"]), 0))
        self.cb_tr.setCurrentIndex(max(self.cb_tr.findData(tr.get("kind", "none")), 0))
        self.sp_lam.setValue(float(tr.get("lam", 1.0)))
        self.sp_shift.setValue(float(tr.get("shift", 0.0)))
        self._loading = False
        self._sync_transform_widgets()
        self.fill_terms()
        self.update_all()

    def _sync_transform_widgets(self):
        kind = self.cb_tr.currentData()
        glm = self.project is not None and self.resp() >= 0 and self.project.analysis_kind(self.resp()) == "glm"
        self.cb_tr.setEnabled(not glm)
        self.sp_lam.setEnabled(kind == "power" and not glm)
        self.sp_shift.setEnabled(kind != "none" and not glm)
        if self.project is not None and self.resp() >= 0:
            ak = self.project.analysis_kind(self.resp())
            self.tabs.setTabText(1, {"glm": "GLM (Logistic/Poisson)", "reml": "Split-Plot (REML)"}.get(ak, "ANOVA"))
            self.btn_suggest.setVisible(ak != "reml")

    def _changed(self):
        self.project.dirty = True
        self.update_all()
        self.model_changed.emit()

    def on_order_changed(self):
        if self._loading or self.project is None:
            return
        spec = self.project.model_spec(self.resp())
        spec["order"] = self.cb_order.currentData()
        spec["terms"] = None
        self.fill_terms()
        self._changed()

    def on_transform_changed(self):
        self._sync_transform_widgets()
        if self._loading or self.project is None:
            return
        kind = self.cb_tr.currentData()
        tr = {"kind": kind}
        if kind == "power":
            tr["lam"] = round(self.sp_lam.value(), 4)
        if kind != "none" and self.sp_shift.value():
            tr["shift"] = self.sp_shift.value()
        self.project.model_spec(self.resp())["transform"] = tr
        self._changed()

    def fill_terms(self):
        p = self.project
        j = self.resp()
        spec = p.model_spec(j)
        selected = set(p.model_terms(j))
        listed = list(p.terms_for_order(spec["order"]))
        listed += sorted((t for t in selected if t not in listed), key=models._term_sort_key)
        self._loading = True
        self.lst_terms.clear()
        for t in listed:
            if models.is_intercept(t):
                continue
            it = QListWidgetItem(models.term_name(t))
            it.setData(Qt.UserRole, list(t))
            flags = it.flags() | Qt.ItemIsUserCheckable
            if p.is_mixture and models.degree(t) == 1:
                flags &= ~Qt.ItemIsEnabled  # linear mixture terms are mandatory
            it.setFlags(flags)
            it.setCheckState(Qt.Checked if t in selected else Qt.Unchecked)
            self.lst_terms.addItem(it)
        self._loading = False

    def on_terms_changed(self):
        if self._loading:
            return
        terms = [] if self.project.is_mixture else [tuple([0] * self.project.k)]
        for i in range(self.lst_terms.count()):
            it = self.lst_terms.item(i)
            if it.checkState() == Qt.Checked:
                terms.append(models.normalize_term(it.data(Qt.UserRole)))
        self.project.model_spec(self.resp())["terms"] = terms
        self._changed()

    def on_backward(self):
        p = self.project
        j = self.resp()
        try:
            terms = p.backward_eliminate(j, self.sp_alpha.value())
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.lbl_err.setText(str(exc))
            return
        p.model_spec(j)["terms"] = [models.normalize_term(t) for t in terms]
        self.fill_terms()
        self._changed()

    def on_suggest(self):
        if not self._suggested or self._suggested == "mean":
            return
        i = self.cb_order.findData(self._suggested)
        if i >= 0:
            self.cb_order.setCurrentIndex(i)

    def apply_box_cox(self):
        rec = models.recommend_transform(self._bc)
        if rec is None:
            return
        self._loading = True
        self.cb_tr.setCurrentIndex(max(self.cb_tr.findData(rec["kind"]), 0))
        if rec["kind"] == "power":
            self.sp_lam.setValue(rec["lam"])
        self._loading = False
        self.on_transform_changed()

    # ---------------------------------------------------------------
    def update_all(self):
        p = self.project
        j = self.resp()
        fit, err = p.fit_with_error(j)
        self.lbl_err.setText("" if fit is not None or err is None or "at least 3" in err else err)
        if fit is None:
            msg = REPORT_CSS + f"<h2>The model cannot be computed yet</h2><p>{err}</p>"
            for w in (self.txt_summary, self.txt_anova, self.txt_equation):
                w.setHtml(msg)
            self.diag.set_fit(p, None, j, err or "Not enough response data yet")
            self.boxcox.message(err or "")
            self.btn_suggest.setEnabled(False)
            return
        rows, suggested = p.fit_summary(j)
        self._suggested = suggested
        self.btn_suggest.setEnabled(suggested != "mean")
        self.txt_summary.setHtml(REPORT_CSS + report.summary_html(p, j, rows, suggested))
        self.txt_anova.setHtml(REPORT_CSS + report.anova_html(p, j, fit))
        self.txt_equation.setHtml(REPORT_CSS + report.equation_html(p, j, fit))

        self.diag.set_fit(p, fit, j)

        self._bc = p.box_cox(j)
        tr = p.model_spec(j).get("transform") or {}
        cur = {"none": 1.0, "sqrt": 0.5, "ln": 0.0, "log10": 0.0, "inverse": -1.0}.get(
            tr.get("kind", "none"), tr.get("lam", 1.0))
        self.draw_effects()
        self.boxcox.figure.clear()
        plots.draw_box_cox(self.boxcox.figure, self._bc, cur)
        self.boxcox.draw()
        rec = models.recommend_transform(self._bc)
        self.btn_bc.setEnabled(rec is not None)
        if rec is None:
            self.lbl_bc.setText("Box-Cox is not available for this response.")
        else:
            self.lbl_bc.setText(f"Recommendation: <b>{models.transform_label(rec)}</b>. "
                                "If λ = 1 lies inside the CI, no transformation is needed.")

    def on_excluded(self):
        """Runs ignored/restored: all models are recomputed without those runs."""
        self.project.dirty = True
        self.update_all()
        self.model_changed.emit()

    # --------------------------------------------------------------- factorial effects
    def _selected_terms(self):
        return {t for t in self.project.model_terms(self.resp()) if not models.is_intercept(t)}

    def _set_terms(self, terms):
        p = self.project
        k = p.k
        spec = p.model_spec(self.resp())
        spec["terms"] = [tuple([0] * k)] + sorted(set(terms), key=models._term_sort_key)
        self.fill_terms()
        self._changed()

    def draw_effects(self):
        p = self.project
        if p is None or not p.is_two_level_factorial:
            return
        fig = self.effplot.figure
        fig.clear()
        try:
            self._eff = p.effects(self.resp())
        except (ValueError, np.linalg.LinAlgError) as exc:
            self._eff = None
            self.effplot.message(str(exc))
            return
        sel = self._selected_terms()
        kind = self.cb_eff.currentIndex()
        self._eff_pts = None
        if kind == 2:
            plots.draw_pareto(fig, self._eff, sel)
        else:
            _, xs, q, order = plots.draw_half_normal(fig, self._eff, sel, normal=kind == 1)
            self._eff_pts = (xs, q, order)
        self.effplot.draw()
        e = self._eff
        self.lbl_eff.setText(f"PSE Lenth = {e['pse']:.4g}, ME = {e['me']:.4g}, SME = {e['sme']:.4g}. "
                             "Click a point to add the term to or remove it from the model."
                             + (" Aliased terms: " + ", ".join(models.term_name(t) for t in e["aliased"])
                                if e["aliased"] else ""))

    def on_effect_click(self, event):
        if self._eff_pts is None or event.inaxes is None or event.xdata is None:
            return
        xs, q, order = self._eff_pts
        ax = event.inaxes
        # distance in screen pixels
        disp = ax.transData.transform(np.column_stack([xs, q]))
        d = np.hypot(disp[:, 0] - event.x, disp[:, 1] - event.y)
        i = int(np.argmin(d))
        if d[i] > 12:
            return
        t = self._eff["terms"][order[i]]
        sel = self._selected_terms()
        sel.symmetric_difference_update({t})
        self._set_terms(sel)

    def auto_effects(self):
        if self._eff is None:
            return
        e = self._eff
        k = self.project.k
        sel = {t for t, v in zip(e["terms"], e["effects"]) if abs(v) >= e["me"]}
        for t in list(sel):  # keep hierarchy: add parent terms
            for o in e["terms"]:
                if models.contains(o, t, k):
                    sel.add(o)
        if not sel:
            self.lbl_eff.setText("No effect exceeds the ME limit.")
            return
        self._set_terms(sel)

    def clear_effects(self):
        self._set_terms(set())
