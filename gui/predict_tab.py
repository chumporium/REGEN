"""Point prediction + confirmation tab (CI for the mean, PI for n confirmation runs)."""
import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from .copying import install_table_copy
from .common import fmt, parse_float

COLS = ["Response", "Prediction", "Std. Error of mean", "CI low", "CI high", "PI low", "PI high",
        "Confirmation mean", "Status"]


class PredictTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.inputs = []
        self._busy = False

        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        lay.addLayout(left, 0)
        self.g_in = QGroupBox("Prediction Point (Actual Units)")
        self.f_in = QFormLayout(self.g_in)
        left.addWidget(self.g_in)
        self.lbl_sum = QLabel()
        left.addWidget(self.lbl_sum)

        g_opt = QGroupBox("Options")
        fo = QFormLayout(g_opt)
        self.cb_level = QComboBox()
        for v in (90, 95, 99):
            self.cb_level.addItem(f"{v}%", v / 100)
        self.cb_level.setCurrentIndex(1)
        self.sp_n = QSpinBox()
        self.sp_n.setRange(1, 100)
        self.sp_n.setValue(3)
        fo.addRow("Confidence level:", self.cb_level)
        fo.addRow("Number of confirmation runs (n):", self.sp_n)
        left.addWidget(g_opt)
        self.btn_center = QPushButton("Reset to Center Point")
        left.addWidget(self.btn_center)
        left.addStretch()

        right = QVBoxLayout()
        lay.addLayout(right, 1)
        self.lbl_warn = QLabel()
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setStyleSheet("color: #b42318; font-weight: bold;")
        right.addWidget(self.lbl_warn)
        self.tbl = QTableWidget(0, len(COLS))
        install_table_copy(self.tbl)
        self.tbl.setHorizontalHeaderLabels(COLS)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl.verticalHeader().setVisible(False)
        right.addWidget(self.tbl)
        info = QLabel(
            "<b>CI</b> = confidence interval for the <i>mean</i> response at this point. "
            "<b>PI</b> = prediction interval for the <i>mean of the n confirmation runs</i> you will perform. "
            "Perform n runs at this point and type the mean result in the <b>Confirmation mean</b> column: "
            "if it falls inside the PI, the model is <b>confirmed</b>. If the response is transformed, the "
            "intervals are computed on the transformed scale and then converted back to the original scale.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #444;")
        right.addWidget(info)

        self.cb_level.currentIndexChanged.connect(self.compute)
        self.sp_n.valueChanged.connect(self.compute)
        self.btn_center.clicked.connect(self.reset_center)
        self.tbl.itemChanged.connect(self.on_conf_changed)

    # ---------------------------------------------------------------
    def set_project(self, project):
        self.project = project
        while self.f_in.rowCount():
            self.f_in.removeRow(0)
        self.inputs = []
        self._confirm = {}
        for i, f in enumerate(project.factors):
            if f.categoric:
                w = QComboBox()
                w.addItems(f.levels)
                w.currentIndexChanged.connect(self.compute)
            else:
                w = QDoubleSpinBox()
                act = project.actual[:, i]
                act = act[np.isfinite(act)]
                lo = min(f.low, act.min()) if len(act) else f.low
                hi = max(f.high, act.max()) if len(act) else f.high
                span = hi - lo
                w.setDecimals(4 if span < 10 else 3)
                w.setRange(lo - span, hi + span)
                w.setSingleStep(span / 20 if span else 0.1)
                w.valueChanged.connect(self.compute)
            self.f_in.addRow(project.factor_label(i) + ":", w)
            self.inputs.append(w)
        self.lbl_sum.setVisible(project.is_mixture)
        self.reset_center()

    def refresh(self):
        if self.project is not None:
            self.compute()

    def reset_center(self):
        p = self.project
        center = np.zeros(p.k)
        if p.is_mixture:
            center[p.comp_idx] = np.nanmean(p.coded, axis=0)[p.comp_idx]
        self.set_point(center)

    def set_point(self, coded):
        p = self.project
        self._busy = True
        for i, w in enumerate(self.inputs):
            if isinstance(w, QComboBox):
                w.setCurrentIndex(int(round(coded[i])) if np.isfinite(coded[i]) else 0)
            else:
                w.setValue(float(p.factor_to_actual(i, coded[i])))
        self._busy = False
        self.compute()

    def point(self):
        p = self.project
        x = np.zeros(p.k)
        for i, w in enumerate(self.inputs):
            x[i] = w.currentIndex() if isinstance(w, QComboBox) else float(p.factor_to_coded(i, w.value()))
        return x

    def compute(self):
        p = self.project
        if p is None or self._busy:
            return
        x = self.point()
        warn = []
        if p.is_mixture:
            total = sum(self.inputs[i].value() for i in p.comp_idx)
            ok_sum = abs(total - p.mixture_total) < 1e-6 * max(1, p.mixture_total)
            self.lbl_sum.setText(f"Component sum = {total:g} (must be {p.mixture_total:g})")
            self.lbl_sum.setStyleSheet("color: #1a7f37;" if ok_sum else "color: #b42318; font-weight: bold;")
            if not ok_sum:
                warn.append("The component sum does not equal the mixture total.")
        if not p.feasible(x)[0]:
            warn.append("This point violates the component bounds / constraints.")
        num = p.num_idx
        if not p.is_mixture and np.any(np.abs(x[num]) > 1 + 1e-9):
            ext = np.abs(p.coded[:, num][np.all(np.isfinite(p.coded), axis=1)]).max(initial=1)
            if np.any(np.abs(x[num]) > ext + 1e-9):
                warn.append("The point is outside the design region - the prediction is an extrapolation.")
        self.lbl_warn.setText(" ".join(warn))

        level = self.cb_level.currentData()
        n = self.sp_n.value()
        self._busy = True
        self.tbl.setRowCount(len(p.responses))
        for j in range(len(p.responses)):
            fit, err = p.model_with_error(j)
            cells = [p.response_label(j) + (" [ANN]" if p.source(j) == "ann" else "")]
            if fit is None:
                cells += ["-"] * 6
            else:
                iv = fit.intervals(x[None, :], n_obs=n, level=level)
                se = iv["se_mean"][0]
                cells += [fmt(iv["pred"][0], 6), fmt(se, 4), fmt(iv["ci"][0][0], 6), fmt(iv["ci"][1][0], 6),
                          fmt(iv["pi"][0][0], 6), fmt(iv["pi"][1][0], 6)]
                self._confirm.setdefault(j, "")
            for c, v in enumerate(cells):
                it = QTableWidgetItem(v)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                it.setTextAlignment(Qt.AlignCenter)
                if c == 1:
                    it.setBackground(QBrush(QColor("#e6f4ea")))
                self.tbl.setItem(j, c, it)
            conf = QTableWidgetItem(self._confirm.get(j, ""))
            conf.setTextAlignment(Qt.AlignCenter)
            if fit is None:
                conf.setFlags(conf.flags() & ~Qt.ItemIsEditable)
            self.tbl.setItem(j, 7, conf)
            self._status(j, fit, x, n, level)
        self._busy = False

    def _status(self, j, fit, x, n, level):
        text, color = "", "white"
        raw = self._confirm.get(j, "")
        if fit is not None and raw.strip():
            try:
                v = parse_float(raw)
            except ValueError:
                v = math.nan
            if math.isnan(v):
                text, color = "Not a number", "#fde2e1"
            else:
                iv = fit.intervals(x[None, :], n_obs=n, level=level)
                lo, hi = iv["pi"] if np.isfinite(iv["pi"][0][0]) else iv["ci"]
                name = "PI" if np.isfinite(iv["pi"][0][0]) else "CI"
                if not np.isfinite(lo[0]):
                    # model without intervals (ANN): use the relative error, typically < 5 %
                    pred = float(iv["pred"][0])
                    err = abs(v - pred) / abs(pred) * 100 if pred else math.inf
                    ok = err < 5
                    text = f"{'✓' if ok else '✗'} Relative error {err:.2f}% ({'< 5%' if ok else '≥ 5%'})"
                    color = "#d3f0dc" if ok else "#fde2e1"
                elif lo[0] <= v <= hi[0]:
                    text, color = f"✓ Confirmed (inside {name})", "#d3f0dc"
                else:
                    text, color = f"✗ Outside {name}", "#fde2e1"
        it = QTableWidgetItem(text)
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        it.setBackground(QBrush(QColor(color)))
        self.tbl.setItem(j, 8, it)

    def on_conf_changed(self, item):
        if self._busy or item.column() != 7:
            return
        self._confirm[item.row()] = item.text()
        self.compute()
