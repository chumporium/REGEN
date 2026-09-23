"""Numerical optimization tab using desirability."""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                               QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from doe import models
from doe.optimize import GOALS, desirability

from .copying import install_table_copy
from .common import MplCanvas, fmt, model_source_combo, parse_float

IMPORTANCE = ["+", "++", "+++", "++++", "+++++"]
COLS = ["Name", "Goal", "Lower Limit", "Upper Limit", "Target", "Importance", "Model"]


class OptimTab(QWidget):
    send_to_prediction = Signal(object)
    model_changed = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.solutions = []
        self.criteria_used = []

        lay = QVBoxLayout(self)
        g = QGroupBox("Optimization Criteria")
        gl = QVBoxLayout(g)
        self.tbl = QTableWidget(0, len(COLS))
        self.tbl.setHorizontalHeaderLabels(COLS)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.verticalHeader().setVisible(False)
        gl.addWidget(self.tbl)
        bar = QHBoxLayout()
        self.btn_reset = QPushButton("Reset Limits to Data")
        self.btn_run = QPushButton("Run Optimization")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 6px 18px;")
        info = QLabel("Factor limits are in actual units. A response is included only if a goal is selected "
                      "and its model is available.")
        info.setStyleSheet("color: #555;")
        bar.addWidget(info, 1)
        bar.addWidget(self.btn_reset)
        bar.addWidget(self.btn_run)
        gl.addLayout(bar)
        lay.addWidget(g, 2)

        split = QSplitter(Qt.Horizontal)
        g2 = QGroupBox("Solutions (Sorted by Desirability)")
        g2l = QVBoxLayout(g2)
        self.tbl_sol = QTableWidget()
        install_table_copy(self.tbl_sol)
        self.tbl_sol.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_sol.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_sol.verticalHeader().setVisible(False)
        g2l.addWidget(self.tbl_sol)
        split.addWidget(g2)
        self.canvas = MplCanvas(toolbar=False, figsize=(4, 3))
        split.addWidget(self.canvas)
        split.setSizes([700, 350])
        lay.addWidget(split, 3)

        self.btn_send = QPushButton("Send Selected Solution to Prediction")
        g2l.addWidget(self.btn_send)
        self.btn_send.clicked.connect(self.send_selected)
        self.btn_run.clicked.connect(self.run)
        self.btn_reset.clicked.connect(lambda: self.fill_table(reset=True))
        self.tbl_sol.itemSelectionChanged.connect(self.show_selected)

    def set_project(self, project):
        self.project = project
        self.solutions = list(project.solutions)
        self.criteria_used = list(project.criteria_used)
        self.tbl_sol.clear()
        self.tbl_sol.setRowCount(0)
        self.tbl_sol.setColumnCount(0)
        self.canvas.clear()
        self.canvas.draw()
        self.fill_table()
        if self.solutions:
            self.show_solutions()

    def refresh(self):
        if self.project is not None:
            self.save_criteria()
            self.fill_table()

    # ---------------------------------------------------------------
    def _item(self, text, editable=True):
        it = QTableWidgetItem(text)
        if not editable:
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            it.setBackground(QBrush(QColor("#eef1f5")))
        it.setTextAlignment(Qt.AlignCenter)
        return it

    def fill_table(self, reset=False):
        p = self.project
        crit = p.opt_criteria
        self._rows = [("resp", j) for j in range(len(p.responses))]
        if p.has_poe:
            self._rows += [("poe", j) for j in range(len(p.responses)) if p.analysis_kind(j) != "glm"]
        self.tbl.setRowCount(p.k + len(self._rows))
        for i, f in enumerate(p.factors):
            c = {} if reset else crit.get(f"f{i}", {})
            self.tbl.setItem(i, 0, self._item(p.factor_label(i), False))
            if f.categoric:
                self.tbl.setItem(i, 1, self._item("All levels", False))
                self.tbl.setItem(i, 2, self._item(", ".join(f.levels), False))
                self.tbl.setItem(i, 3, self._item("", False))
            else:
                self.tbl.setItem(i, 1, self._item("In range", False))
                self.tbl.setItem(i, 2, self._item(fmt(c.get("low", f.low), 8)))
                self.tbl.setItem(i, 3, self._item(fmt(c.get("high", f.high), 8)))
            self.tbl.setItem(i, 4, self._item("", False))
            self.tbl.setItem(i, 5, self._item("", False))
            self.tbl.setItem(i, 6, self._item("", False))
            self.tbl.removeCellWidget(i, 1)
            self.tbl.removeCellWidget(i, 5)
            self.tbl.removeCellWidget(i, 6)
        for n, (kind, j) in enumerate(self._rows):
            row = p.k + n
            y = p.data[:, j]
            if p.response_family(j) == "binomial":
                y = y / (p.responses[j].trials or 1)
            has = np.any(~np.isnan(y))
            ymin = float(np.nanmin(y)) if has else 0.0
            ymax = float(np.nanmax(y)) if has else 1.0
            key = j if kind == "resp" else f"poe{j}"
            label = p.response_label(j)
            if kind == "poe":
                label = "POE " + label
                fit = p.fit(j)
                ymin = 0.0
                ymax = float(np.nanmax(models.poe(fit, p.coded[np.all(np.isfinite(p.coded), axis=1)],
                                                  p.sd_coded()))) if fit is not None else 1.0
            c = {} if reset else crit.get(key, {})
            self.tbl.setItem(row, 0, self._item(label, False))
            cb = QComboBox()
            for key, label in GOALS.items():
                cb.addItem(label, key)
            cb.setCurrentIndex(max(cb.findData(c.get("goal", "none")), 0))
            self.tbl.setCellWidget(row, 1, cb)
            self.tbl.setItem(row, 2, self._item(fmt(c.get("low", ymin), 8)))
            self.tbl.setItem(row, 3, self._item(fmt(c.get("high", ymax), 8)))
            self.tbl.setItem(row, 4, self._item(fmt(c.get("target", (ymin + ymax) / 2), 8)))
            cbi = QComboBox()
            cbi.addItems(IMPORTANCE)
            cbi.setCurrentIndex(int(c.get("importance", 3)) - 1)
            self.tbl.setCellWidget(row, 5, cbi)
            if kind == "resp":
                self.tbl.setCellWidget(row, 6, model_source_combo(p, j, self.model_changed.emit))
            else:
                self.tbl.removeCellWidget(row, 6)
                self.tbl.setItem(row, 6, self._item("follows response", False))
        self.tbl.resizeRowsToContents()

    def _num(self, row, col):
        return parse_float(self.tbl.item(row, col).text())

    def save_criteria(self):
        """Read the table into project.opt_criteria. Returns an error message or None."""
        p = self.project
        if not hasattr(self, "_rows") or self.tbl.rowCount() != p.k + len(self._rows):
            return None
        crit = {}
        try:
            for i in p.num_idx:
                crit[f"f{i}"] = {"low": self._num(i, 2), "high": self._num(i, 3)}
            for n, (kind, j) in enumerate(self._rows):
                row = p.k + n
                crit[j if kind == "resp" else f"poe{j}"] = {"goal": self.tbl.cellWidget(row, 1).currentData(),
                           "low": self._num(row, 2), "high": self._num(row, 3),
                           "target": self._num(row, 4),
                           "importance": self.tbl.cellWidget(row, 5).currentIndex() + 1}
        except (ValueError, AttributeError):
            return "Some limits are not numbers."
        if crit != p.opt_criteria:
            p.opt_criteria = crit
            p.dirty = True
        return None

    # ---------------------------------------------------------------
    def run(self):
        p = self.project
        err = self.save_criteria()
        if err:
            QMessageBox.warning(self, "Optimization", err)
            return
        bounds = {}
        for i in p.num_idx:
            c = p.opt_criteria[f"f{i}"]
            if c["low"] >= c["high"]:
                QMessageBox.warning(self, "Optimization", f"Invalid limits for factor {p.factors[i].name}.")
                return
            bounds[i] = (c["low"], c["high"])
        criteria, skipped = [], []
        for kind, j in self._rows:
            r = p.responses[j]
            c = p.opt_criteria[j if kind == "resp" else f"poe{j}"]
            if c["goal"] == "none":
                continue
            fit = p.model(j)
            if fit is None:
                skipped.append(r.name)
                continue
            if kind == "poe":
                criteria.append({**c, "fit": models.PoeModel(fit, p.sd_coded()), "index": j,
                                 "label": f"POE {r.name}"})
                continue
            if c["low"] >= c["high"]:
                QMessageBox.warning(self, "Optimization", f"Limits for response {r.name}: lower must be < upper.")
                return
            if c["goal"] == "target" and not (c["low"] <= c["target"] <= c["high"]):
                QMessageBox.warning(self, "Optimization", f"The target for {r.name} must be between the limits.")
                return
            criteria.append({**c, "fit": fit, "index": j})
        if not criteria:
            QMessageBox.information(self, "Optimization",
                                    "Choose a goal (Maximize/Minimize/Target/In range) for at least one "
                                    "response whose model is available.")
            return
        if skipped:
            QMessageBox.information(self, "Optimization", "Skipped responses without a model: " + ", ".join(skipped))
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            sols = p.optimize(criteria, bounds)
        except ValueError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Optimization", str(exc))
            return
        QApplication.restoreOverrideCursor()
        self.criteria_used = criteria
        self.solutions = sols[:20]
        p.solutions, p.criteria_used = self.solutions, criteria
        self.show_solutions()

    def show_solutions(self):
        p = self.project
        fits = {j: p.model(j) for j in range(len(p.responses))}
        headers = ["No"] + [f.name for f in p.factors] + \
                  [r.name for r in p.responses] + ["Desirability"]
        self.tbl_sol.clear()
        self.tbl_sol.setColumnCount(len(headers))
        self.tbl_sol.setHorizontalHeaderLabels(headers)
        self.tbl_sol.setRowCount(len(self.solutions))
        for r, (D, x) in enumerate(self.solutions):
            actual = p.to_actual(x)
            vals = [str(r + 1)] + [p.format_value(i, v, 5) for i, v in enumerate(actual)]
            for j in range(len(p.responses)):
                vals.append(fmt(float(fits[j].predict(x[None, :])[0]), 5) if fits[j] else "-")
            vals.append(f"{D:.3f}")
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignCenter)
                if r == 0:
                    it.setBackground(QBrush(QColor("#e6f4ea")))
                self.tbl_sol.setItem(r, c, it)
        self.tbl_sol.resizeColumnsToContents()
        if self.solutions:
            self.tbl_sol.selectRow(0)
        else:
            self.canvas.message("No solutions.")

    def show_selected(self):
        rows = self.tbl_sol.selectionModel().selectedRows()
        if not rows or not self.solutions:
            return
        D, x = self.solutions[rows[0].row()]
        p = self.project
        names, ds = [], []
        for c in self.criteria_used:
            y = c["fit"].predict(x[None, :])
            ds.append(float(desirability(y, c["goal"], c["low"], c["high"], c.get("target"))[0]))
            names.append(c.get("label", p.responses[c["index"]].name))
        names.append("Combined")
        ds.append(D)
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        colors = ["#1f5f99"] * (len(ds) - 1) + ["#1a7f37"]
        ax.barh(names, ds, color=colors)
        for i, v in enumerate(ds):
            inside = v > 0.25
            ax.text(v - 0.02 if inside else v + 0.02, i, f"{v:.3f}", va="center",
                    ha="right" if inside else "left", color="white" if inside else "black",
                    fontweight="bold")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_title(f"Desirability of solution {rows[0].row() + 1}")
        ax.grid(axis="x", alpha=0.3)
        self.canvas.draw()

    def send_selected(self):
        rows = self.tbl_sol.selectionModel().selectedRows() if self.tbl_sol.selectionModel() else []
        if not rows or not self.solutions:
            QMessageBox.information(self, "Prediction", "Run the optimization, then select a solution.")
            return
        self.send_to_prediction.emit(self.solutions[rows[0].row()][1])
