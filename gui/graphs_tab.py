"""Model graphs tab: contour/3D/effects/interaction/cube/perturbation/overlay/desirability; ternary & trace."""
import numpy as np
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QSpinBox, QVBoxLayout, QWidget)

from . import plots
from .common import MplCanvas

PROCESS_PLOTS = [("contour", "2D Contour"), ("surface", "3D Surface"), ("oneway", "One Factor Effect"),
                 ("allfactors", "All Factors"), ("interaction", "Interaction"), ("cube", "Cube Plot"), ("perturbation", "Perturbation"),
                 ("overlay", "Overlay (Graphical Optimization)"), ("desirability", "Desirability Contour")]
MIXTURE_PLOTS = [("contour", "Ternary Contour"), ("surface", "3D Ternary Surface"), ("trace", "Trace (Piepel)"),
                 ("overlay", "Overlay (Graphical Optimization)"), ("desirability", "Desirability Contour")]

# axes needed per graph type: (number of axes, numeric only?)
AXES_NEED = {"poe": (2, True), "contour": (2, True), "surface": (2, True), "overlay": (2, True), "desirability": (2, True),
             "oneway": (1, False), "interaction": (2, False), "cube": (3, True), "perturbation": (0, True), "allfactors": (0, False),
             "trace": (0, True)}
AXIS_LABELS = {"interaction": ("X axis:", "Lines (second factor):", ""),
               "cube": ("X axis:", "Y axis:", "Depth axis:"),
               "mixture": ("Top vertex:", "Left vertex:", "Right vertex:")}


class GraphsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self._loading = False
        self.slice_widgets = []

        lay = QHBoxLayout(self)
        side = QVBoxLayout()
        lay.addLayout(side, 0)

        g = QGroupBox("Graph Settings")
        self.f_main = QFormLayout(g)
        self.cb_resp = QComboBox()
        self.cb_type = QComboBox()
        self.cb_axes = [QComboBox(), QComboBox(), QComboBox()]
        self.sp_levels = QSpinBox()
        self.sp_levels.setRange(4, 40)
        self.sp_levels.setValue(12)
        self.f_main.addRow("Response:", self.cb_resp)
        self.f_main.addRow("Type:", self.cb_type)
        for cb in self.cb_axes:
            self.f_main.addRow("Axis:", cb)
        self.f_main.addRow("Contour lines:", self.sp_levels)
        side.addWidget(g)

        self.g_slice = QGroupBox("Other Factor Values")
        self.f_slice = QFormLayout(self.g_slice)
        side.addWidget(self.g_slice)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #555;")
        side.addWidget(self.note)
        side.addStretch()

        self.canvas = MplCanvas(figsize=(7, 6))
        lay.addWidget(self.canvas, 1)

        for w in [self.cb_resp, self.cb_type] + self.cb_axes:
            w.currentIndexChanged.connect(self.on_controls_changed)
        self.sp_levels.valueChanged.connect(self.draw)

    # ---------------------------------------------------------------
    def set_project(self, project):
        self.project = project
        self._loading = True
        self.cb_type.clear()
        for key, label in (MIXTURE_PLOTS if project.is_mixture else PROCESS_PLOTS):
            self.cb_type.addItem(label, key)
        if project.has_poe:
            self.cb_type.addItem("Propagation of Error (POE)", "poe")
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
        self.on_controls_changed(rebuild_axes=True)

    def refresh(self):
        if self.project is not None:
            self.draw()

    def kind(self):
        return self.cb_type.currentData()

    def _fill_axis_combos(self):
        p = self.project
        kind = self.kind()
        need, numeric_only = AXES_NEED[kind]
        if p.is_mixture:
            need = 3 if kind != "trace" else 0
        candidates = [i for i in range(p.k) if not (numeric_only and p.factors[i].categoric)]
        if p.is_mixture:
            candidates = [i for i in p.comp_idx] if kind != "trace" else []
        prev = [cb.currentData() for cb in self.cb_axes]
        labels = AXIS_LABELS["mixture"] if p.is_mixture else AXIS_LABELS.get(kind, ("X axis:", "Y axis:", ""))
        for n, cb in enumerate(self.cb_axes):
            cb.blockSignals(True)
            cb.clear()
            for i in candidates:
                cb.addItem(p.factor_label(i), i)
            want = prev[n] if prev[n] in candidates else (candidates[n] if n < len(candidates) else None)
            if want is not None:
                cb.setCurrentIndex(cb.findData(want))
            cb.blockSignals(False)
            visible = n < need
            self.f_main.setRowVisible(cb, visible)
            if visible:
                self.f_main.labelForField(cb).setText(labels[n] or "Axis:")
        self.f_main.setRowVisible(self.sp_levels, kind in ("contour", "desirability", "poe"))
        self._dedupe(need)

    def _dedupe(self, need):
        used = []
        for cb in self.cb_axes[:need]:
            if cb.currentData() in used:
                for n in range(cb.count()):
                    if cb.itemData(n) not in used:
                        cb.blockSignals(True)
                        cb.setCurrentIndex(n)
                        cb.blockSignals(False)
                        break
            used.append(cb.currentData())

    def axes(self):
        p = self.project
        kind = self.kind()
        need = AXES_NEED[kind][0]
        if p.is_mixture:
            need = 3 if kind != "trace" else 0
        if kind == "perturbation":
            return list(p.num_idx)
        if kind == "allfactors":
            return list(range(p.k))
        if kind == "trace":
            return list(p.comp_idx)
        return [cb.currentData() for cb in self.cb_axes[:need]]

    def on_controls_changed(self, *_, rebuild_axes=False):
        if self._loading or self.project is None:
            return
        self._loading = True
        self._fill_axis_combos()
        self._loading = False
        p = self.project
        need = 3 if p.is_mixture and self.kind() != "trace" else AXES_NEED[self.kind()][0]
        if len([i for i in range(p.k) if not (AXES_NEED[self.kind()][1] and p.factors[i].categoric)]) < need:
            self.canvas.message("This graph needs more numeric factors.")
            return
        self.rebuild_slices()
        self.update_note()
        self.draw()

    def update_note(self):
        kind = self.kind()
        notes = {
            "overlay": "Each response's limits are taken from the Optimization page (Lower & Upper Limit for "
                       "responses with a goal selected). Yellow area = all responses meet their limits.",
            "desirability": "Combined desirability from the criteria on the Optimization page. Star = latest best "
                            "solution (if it lies in this slice).",
            "cube": "Corner numbers = predicted response at each low/high level combination.",
            "perturbation": "All numeric factors are moved away from the reference point (the values below are "
                            "the reference for categorical factors).",
            "interaction": "Non-parallel lines = interaction present. Error bars = 95% CI.",
        }
        base = "Red points = design runs in this slice (number = replicate count). "
        if self.project.is_mixture:
            base += "Mixture graphs are drawn in pseudo-component space. "
        elif kind in ("contour", "surface", "overlay", "desirability"):
            base += "Graph range = levels -1 to +1. Gray area = outside the constraints. "
        self.note.setText(notes.get(kind, "") + " " + base)

    def rebuild_slices(self):
        old = {i: w for i, w in self.slice_widgets}
        old_vals = {i: (w.currentIndex() if isinstance(w, QComboBox) else w.value()) for i, w in old.items()}
        while self.f_slice.rowCount():
            self.f_slice.removeRow(0)
        self.slice_widgets = []
        p = self.project
        used = self.axes()
        center = np.nanmean(p.coded, axis=0) if p.is_mixture else None
        for i in range(p.k):
            if i in used:
                continue
            f = p.factors[i]
            if f.categoric:
                w = QComboBox()
                w.addItems(f.levels)
                w.setCurrentIndex(old_vals.get(i, 0) if isinstance(old.get(i), QComboBox) else 0)
                w.currentIndexChanged.connect(self.draw)
            else:
                w = QDoubleSpinBox()
                if p.is_mixture and f.kind != "process":
                    lo, hi = f.low, f.high
                    default = float(p.factor_to_actual(i, center[i]))
                else:
                    act = p.actual[:, i]
                    act = act[np.isfinite(act)]
                    lo, hi = (min(act.min(), f.low), max(act.max(), f.high)) if len(act) else (f.low, f.high)
                    default = (f.low + f.high) / 2
                span = hi - lo
                w.setDecimals(4 if span < 10 else 2)
                w.setRange(lo, hi)
                w.setSingleStep(span / 20 if span > 0 else 0.1)
                w.setValue(old_vals.get(i, default) if isinstance(old.get(i), QDoubleSpinBox) else default)
                w.valueChanged.connect(self.draw)
            self.f_slice.addRow(p.factor_label(i) + ":", w)
            self.slice_widgets.append((i, w))
        self.g_slice.setVisible(bool(self.slice_widgets))
        self.g_slice.setTitle("Other Components (Actual Units)" if p.is_mixture else "Other Factor Values")

    def base(self):
        p = self.project
        b = np.zeros(p.k)
        for i, w in self.slice_widgets:
            b[i] = w.currentIndex() if isinstance(w, QComboBox) else float(p.factor_to_coded(i, w.value()))
        return b

    def _criteria(self):
        p = self.project
        out = []
        for j in range(len(p.responses)):
            c = p.opt_criteria.get(j)
            if not c or c.get("goal", "none") == "none":
                continue
            fit = p.model(j)
            if fit is not None:
                out.append({**c, "fit": fit, "index": j})
        return out

    def draw(self):
        p = self.project
        if p is None or self._loading:
            return
        j = self.cb_resp.currentIndex()
        kind = self.kind()
        axes = self.axes()
        base = self.base()
        fig = self.canvas.figure
        fig.clear()
        mark = p.solutions[0][1] if p.solutions else None
        if kind in ("overlay", "desirability"):
            crit = self._criteria()
            if not crit:
                self.canvas.message("No criteria yet. Choose response goals and limits on the Optimization page.")
                return
            if kind == "overlay":
                items = [(p.responses[c["index"]].name, c["fit"], c["low"], c["high"]) for c in crit]
                plots.draw_overlay(fig, p, items, axes, base, mark)
            else:
                plots.draw_desirability(fig, p, crit, axes, base, self.sp_levels.value(), mark)
            self.canvas.draw()
            return
        fit, err = p.model_with_error(j) if j >= 0 else (None, None)
        if fit is None:
            self.canvas.message((err or "Model not available yet.") + "\nEnter response data, then check Analysis.")
            return
        label = p.response_label(j) + (" [ANN]" if p.source(j) == "ann" else "")
        if kind in ("contour", "surface"):
            plots.draw_response_2d(fig, p, fit, axes, base, label, kind, self.sp_levels.value(), mark)
        elif kind == "oneway":
            plots.draw_oneway(fig, p, fit, axes[0], base, label)
        elif kind == "interaction":
            plots.draw_interaction(fig, p, fit, axes[0], axes[1], base, label)
        elif kind == "cube":
            plots.draw_cube(fig, p, fit, axes, base, label)
        elif kind == "perturbation":
            plots.draw_perturbation(fig, p, fit, base, label)
        elif kind == "allfactors":
            plots.draw_all_factors(fig, p, fit, base, label)
        elif kind == "trace":
            plots.draw_trace(fig, p, fit, label)
        elif kind == "poe":
            if not p.has_poe:
                self.canvas.message("Enter the factor standard deviations first (Data page → Factors & Responses).")
                return
            plots.draw_poe(fig, p, fit, axes, base, label, self.sp_levels.value(), mark)
        self.canvas.draw()
