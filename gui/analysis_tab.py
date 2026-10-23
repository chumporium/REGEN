"""Analysis tab: Fit Summary, term selection, transformation, ANOVA, equation, diagnostics, Box-Cox."""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QListWidget,
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
        self._order_alias = {}

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
        self.chk_hier = QCheckBox("Maintain hierarchy")
        self.chk_hier.setChecked(True)
        self.chk_hier.setToolTip("Selecting a term also selects its parent terms (A and B for AB); removing a term "
                                 "also removes the terms that contain it. Applies to the term list and to clicks "
                                 "on the effects plot.")
        ll.addWidget(self.chk_hier)
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
        lay.addWidget(split, 1)

        self.cb_resp.currentIndexChanged.connect(self.on_response_changed)
        self.cb_order.currentIndexChanged.connect(self.on_order_changed)
        self.lst_terms.itemChanged.connect(self.on_terms_changed)
        self.btn_backward.clicked.connect(self.on_backward)
        self.btn_suggest.clicked.connect(self.on_suggest)
        self.chk_hier.toggled.connect(self.on_hierarchy_toggled)
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
        # orders with the same terms as an earlier order are not listed (for example 2FI = Linear with
        # one process factor); a saved order of that kind is shown as its equivalent
        self._order_alias = {}
        seen = {}
        for o in project.available_orders():
            key = frozenset(t for t in project.terms_for_order(o) if not models.is_intercept(t))
            if key in seen:
                self._order_alias[o] = seen[key]
                continue
            seen[key] = o
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
        order = self._order_alias.get(spec["order"], spec["order"])
        self.cb_order.setCurrentIndex(max(self.cb_order.findData(order), 0))
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

    def _hierarchy(self, sel, t, add, universe):
        """Add the parent terms of t (add=True) or remove the terms that contain t (add=False)."""
        k = self.project.k
        sel = set(sel)
        if add:
            sel.add(t)
            sel |= {o for o in universe if models.contains(o, t, k)}
        else:
            sel.discard(t)
            sel -= {o for o in sel if models.contains(t, o, k)}
        return sel

    def on_terms_changed(self, item=None):
        if self._loading:
            return
        items = [self.lst_terms.item(i) for i in range(self.lst_terms.count())]
        listed = [models.normalize_term(it.data(Qt.UserRole)) for it in items]
        checked = {t for t, it in zip(listed, items) if it.checkState() == Qt.Checked}
        refill = False
        if item is not None and self.chk_hier.isChecked():
            t = models.normalize_term(item.data(Qt.UserRole))
            checked = self._hierarchy(checked, t, item.checkState() == Qt.Checked, listed)
            self._loading = True
            for tt, it in zip(listed, items):
                if it.flags() & Qt.ItemIsEnabled:
                    it.setCheckState(Qt.Checked if tt in checked else Qt.Unchecked)
            self._loading = False
            refill = bool(checked - set(listed))
        terms = [] if self.project.is_mixture else [tuple([0] * self.project.k)]
        terms += [t for t in listed if t in checked]
        terms += sorted(checked - set(listed), key=models._term_sort_key)
        self.project.model_spec(self.resp())["terms"] = terms
        if refill:
            self.fill_terms()
        self._changed()

    def on_hierarchy_toggled(self, on):
        """Turning hierarchy on adds the missing parent terms of the current model."""
        if not on or self.project is None or self.resp() < 0:
            return
        sel = self._selected_terms()
        full = set(sel)
        universe = set(self.project.terms_for_order(self.project.model_spec(self.resp())["order"])) | sel
        for t in sel:
            full = self._hierarchy(full, t, True, universe)
        if full != sel:
            self._set_terms(full)

    def on_backward(self):
        p = self.project
        j = self.resp()
        try:
            terms = p.backward_eliminate(j, self.sp_alpha.value(), keep_hierarchy=self.chk_hier.isChecked())
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
        html = report.anova_html(p, j, fit)
        warnings = getattr(fit, "warnings", None)
        if warnings:
            self.lbl_err.setText("The GLM fit has warnings - see the GLM (Logistic/Poisson) tab.")
            if warnings[0] not in html:
                block = "".join(f"<p class='ns'><b>Warning:</b> {w}</p>" for w in warnings)
                i = html.find("</h2>")
                html = html[:i + 5] + block + html[i + 5:] if i >= 0 else block + html
        self.txt_anova.setHtml(REPORT_CSS + html)
        self.txt_equation.setHtml(REPORT_CSS + report.equation_html(p, j, fit))

        self.diag.set_fit(p, fit, j)

        self._bc = self._box_cox(j)
        tr = p.model_spec(j).get("transform") or {}
        cur = {"none": 1.0, "sqrt": 0.5, "ln": 0.0, "log10": 0.0, "inverse": -1.0}.get(
            tr.get("kind", "none"), tr.get("lam", 1.0))
        self.draw_effects()
        self.boxcox.figure.clear()
        plots.draw_box_cox(self.boxcox.figure, self._bc, cur)
        self.boxcox.draw()
        rec = models.recommend_transform(self._bc)
        self.btn_bc.setEnabled(rec is not None)
        bc = self._bc
        if bc is None:
            self.lbl_bc.setText("Box-Cox is not available for this response.")
            return
        lam = np.asarray(bc["lambdas"])
        rng = f"{lam.min():g} to {lam.max():g}"
        k = bc.get("shift", 0.0)
        tail = f" Computed for y + k with k = {k:g}." if k else ""
        if rec is None:
            self.lbl_bc.setText(f"No recommendation: the smallest residual SS lies at the edge of the λ range "
                                f"({rng}), so the 95% CI is open on that side and no reliable λ can be given. "
                                "Check the model (lack of fit, outliers) before choosing a transformation." + tail)
            return
        edge = " or ".join(s for s, e in zip(("lower", "upper"), bc.get("edge", (False, False))) if e)
        note = (f" The CI reaches the {edge} end of the λ range ({rng}), so it continues beyond the plotted range."
                if edge else "")
        self.lbl_bc.setText(f"Recommendation: <b>{models.transform_label(rec)}</b>. "
                            "If λ = 1 lies inside the CI, no transformation is needed." + note + tail)

    def _box_cox(self, j):
        """Box-Cox for y + k (k = constant of the current transformation). Ordinary least squares only."""
        p = self.project
        if p.analysis_kind(j) != "ols":
            return None
        tr = p.model_spec(j).get("transform") or {}
        k = float(tr.get("shift", 0.0) or 0.0) if tr.get("kind", "none") != "none" else 0.0
        return models.box_cox(p.coded, p.model_data[:, j], p.model_terms(j), mixture=p.is_mixture,
                              blocks=p.blocks if p.n_blocks > 1 else None, space=p.space, shift=k)

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
            # Plackett-Burman: two-factor interactions are partially aliased with main effects, so only main
            # effects are estimated; the unassigned columns form the error estimate
            pb = p.design_type == "plackett_burman"
            self._eff = models.factorial_effects(p.coded, p.model_data[:, self.resp()], p.space,
                                                 max_order=1 if pb else None)
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
        self.lbl_eff.setText(f"PSE Lenth = {e['pse']:.4g}, ME = {e['me']:.4g}, SME = {e['sme']:.4g} (Lenth 1989). "
                             "Effect = 2 × coded coefficient. "
                             + ("Plackett-Burman design: only main effects are shown, because two-factor "
                                "interactions are partially aliased with them. " if pb else "")
                             + "Click a point to add the term to or remove it from the model."
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
        if self.chk_hier.isChecked():
            sel = self._hierarchy(sel, t, t not in sel, set(self._eff["terms"]) | sel)
        else:
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
