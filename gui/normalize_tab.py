"""Data Normalization page: puts factors and responses with different ranges/units on a common scale."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QSplitter, QTableView, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                               QWidget)

from doe import ann

from . import theme
from .common import ArrayModel, MplCanvas, fmt
from .copying import install_table_copy

FORMULAS = {"mapminmax": "x' = 2(x − min)/(max − min) − 1   → range [−1, 1]",
            "minmax01": "x' = (x − min)/(max − min)   → range [0, 1]",
            "zscore": "x' = (x − mean)/SD   → mean 0, SD 1",
            "none": "x' = x   (original data)"}


def columns(project, use_factors=True, use_responses=True):
    """-> (column names, matrix of actual values) in run order. Categorical factors use the level index."""
    p = project
    order = p.display_order()
    names, cols = [], []
    if use_factors:
        act = p.actual
        for i in range(p.k):
            names.append(p.factor_label(i))
            cols.append(act[order, i])
    if use_responses:
        for j in range(len(p.responses)):
            names.append(p.response_label(j))
            cols.append(p.data[order, j])
    return names, (np.column_stack(cols) if cols else np.zeros((len(order), 0))), p.run_order[order]


class NormalizeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.cb_method = QComboBox()
        for k, lab in ann.NORMS.items():
            self.cb_method.addItem(lab, k)
        self.chk_f = QCheckBox("Factors")
        self.chk_r = QCheckBox("Responses")
        self.chk_f.setChecked(True)
        self.chk_r.setChecked(True)
        self.btn_excel = QPushButton("Export to Excel...")
        self.btn_excel.setIcon(theme.glyph_icon("excel", theme.ACCENT))
        top.addWidget(QLabel("Method:"))
        top.addWidget(self.cb_method)
        top.addSpacing(12)
        top.addWidget(self.chk_f)
        top.addWidget(self.chk_r)
        top.addStretch()
        top.addWidget(self.btn_excel)
        lay.addLayout(top)
        self.lbl_formula = QLabel()
        self.lbl_formula.setStyleSheet(f"color: {theme.ACCENT}; font-family: Consolas; font-size: 10pt;")
        lay.addWidget(self.lbl_formula)
        note = QLabel("Normalization puts variables with different units and ranges (e.g. temperature 60–80 °C "
                      "and concentration 1–3%) on a common scale so that no variable dominates just because its "
                      "numbers are large. RSM automatically uses coded factors (−1…+1); ANN automatically "
                      "normalizes inputs & outputs with the method chosen on the ANN page. Use this page to view, "
                      "compare and export normalized data.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.MUTED};")
        lay.addWidget(note)

        split = QSplitter(Qt.Horizontal)
        lay.addWidget(split, 1)
        self.tabs_t = QTabWidget()
        self.tbl = QTableView()
        self.tbl_model = ArrayModel()
        self.tbl.setModel(self.tbl_model)
        self.tbl.horizontalHeader().setResizeContentsPrecision(60)
        self.tbl_stats = QTableWidget()
        for t in (self.tbl, self.tbl_stats):
            t.setEditTriggers(QTableWidget.NoEditTriggers)
            t.setAlternatingRowColors(True)
            install_table_copy(t)
        self.tabs_t.addTab(self.tbl, "Normalized Data")
        self.tabs_t.addTab(self.tbl_stats, "Statistics && Parameters")
        split.addWidget(self.tabs_t)
        self.canvas = MplCanvas(figsize=(7, 5))
        split.addWidget(self.canvas)
        split.setSizes([620, 560])

        self.cb_method.currentIndexChanged.connect(self.refresh)
        self.chk_f.toggled.connect(self.refresh)
        self.chk_r.toggled.connect(self.refresh)
        self.btn_excel.clicked.connect(self.export)

    def set_project(self, project):
        self.project = project
        self.refresh()

    def compute(self):
        names, A, runs = columns(self.project, self.chk_f.isChecked(), self.chk_r.isChecked())
        method = self.cb_method.currentData()
        Z, off, scale, base = ann.normalize(A, method)
        return names, A, Z, off, scale, base, runs, method

    def refresh(self):
        if self.project is None:
            return
        names, A, Z, off, scale, base, runs, method = self.compute()
        self.lbl_formula.setText(FORMULAS[method])
        t = self.tbl
        self.tbl_model.set_data(["Run"] + names, [runs] + [Z[:, c] for c in range(len(names))],
                                lambda c, v: str(v) if c == 0 else ("" if np.isnan(v) else f"{v:.5f}"))
        t.verticalHeader().setVisible(False)
        t.resizeColumnsToContents()

        heads = ["Variable", "Min", "Max", "Mean", "SD", "Range", "Offset", "Scale", "Base",
                 "New min", "New max"]
        s = self.tbl_stats
        s.clear()
        s.setRowCount(len(names))
        s.setColumnCount(len(heads))
        s.setHorizontalHeaderLabels(heads)
        for c, name in enumerate(names):
            col, zc = A[:, c], Z[:, c]
            ok = np.isfinite(col)
            vals = [name]
            if ok.sum():
                vals += [fmt(col[ok].min()), fmt(col[ok].max()), fmt(col[ok].mean()),
                         fmt(col[ok].std(ddof=1)) if ok.sum() > 1 else "-", fmt(np.ptp(col[ok])), fmt(off[c]),
                         fmt(scale[c]), fmt(base), fmt(np.nanmin(zc)), fmt(np.nanmax(zc))]
            else:
                vals += ["-"] * (len(heads) - 1)
            for k, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if k:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                s.setItem(c, k, it)
        s.verticalHeader().setVisible(False)
        s.resizeColumnsToContents()
        self.draw(names, A, Z, method)

    def draw(self, names, A, Z, method):
        if len(A) > 5000:                               # a random sample is enough for the boxplot
            pick = np.random.default_rng(0).choice(len(A), 5000, replace=False)
            A, Z = A[pick], Z[pick]
        fig = self.canvas.figure
        fig.clear()
        if not names:
            self.canvas.message("Select factors and/or responses")
            return
        ax1, ax2 = fig.subplots(2, 1)
        short = [n.split(":")[0] for n in names]
        for ax, M, title in ((ax1, A, "Before normalization (original units)"),
                             (ax2, Z, f"After normalization - {ann.NORMS[method].split(' - ')[0]}")):
            data = [M[np.isfinite(M[:, c]), c] for c in range(M.shape[1])]
            bp = ax.boxplot(data, patch_artist=True, widths=0.55)
            ax.set_xticks(range(1, len(short) + 1), short)
            for n, box in enumerate(bp["boxes"]):
                box.set_facecolor(theme.PLOT_COLORS[0] if n < self.project.k and self.chk_f.isChecked()
                                  else theme.PLOT_COLORS[1])
                box.set_alpha(0.55)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.7)
        ax1.set_ylabel("Original value")
        ax2.set_ylabel("Normalized value")

    def export(self):
        if self.project is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Normalized Data", "normalized_data.xlsx",
                                              "Excel (*.xlsx)")
        if not path:
            return
        from openpyxl import Workbook
        names, A, Z, off, scale, base, runs, method = self.compute()
        wb = Workbook()
        ws = wb.active
        ws.title = "Normalized"
        ws.append(["Method", ann.NORMS[method], FORMULAS[method]])
        ws.append(["Run"] + names)
        for r in range(len(Z)):
            ws.append([int(runs[r])] + [None if np.isnan(v) else float(v) for v in Z[r]])
        ws2 = wb.create_sheet("Original Data")
        ws2.append(["Run"] + names)
        for r in range(len(A)):
            ws2.append([int(runs[r])] + [None if np.isnan(v) else float(v) for v in A[r]])
        ws3 = wb.create_sheet("Parameters")
        ws3.append(["Variable", "Offset", "Scale", "Base", "Formula: x' = base + (x - offset)/scale"])
        for c, n in enumerate(names):
            ws3.append([n, float(off[c]), float(scale[c]), float(base)])
        try:
            wb.save(path)
        except OSError as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
