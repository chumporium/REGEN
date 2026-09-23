"""Shared components: matplotlib canvas, number formatting, HTML report style."""
import math

import matplotlib
import numpy as np

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QThread, Signal  # noqa: E402
from PySide6.QtGui import QBrush, QColor  # noqa: E402
from PySide6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402

from . import theme  # noqa: E402

theme.apply_matplotlib()

ACCENT = theme.ACCENT
SIGNIFICANT = "#107c10"
NOT_SIGNIFICANT = "#c42b1c"

REPORT_CSS = f"""
<style>
body {{ font-family: 'Segoe UI', sans-serif; font-size: 9pt; color: #1f1f1f; }}
h2 {{ font-size: 12pt; font-weight: 600; margin: 10px 0 4px 0; }}
h3 {{ font-size: 10pt; font-weight: 600; margin: 10px 0 3px 0; }}
h4 {{ font-size: 9pt; font-weight: 600; margin: 8px 0 3px 0; }}
table {{ border-collapse: collapse; margin: 3px 0 8px 0; }}
th {{ background: #f0f0f0; border: 1px solid #c8c8c8; padding: 2px 7px; text-align: right; font-weight: 600; }}
th.l, td.l {{ text-align: left; }}
td {{ border: 1px solid #d8d8d8; padding: 2px 7px; text-align: right; }}
tr.model td {{ font-weight: 600; background: #f7f7f7; }}
.ok {{ color: #107c10; font-weight: 600; }}
.warn {{ color: #b45309; font-weight: 600; }}
.bad {{ color: #c42b1c; font-weight: 600; }}
.info {{ color: #5f6368; font-weight: 600; }}
.sig {{ color: {SIGNIFICANT}; font-weight: 600; }}
.ns {{ color: {NOT_SIGNIFICANT}; }}
.note {{ color: #5f6368; font-size: 8.5pt; }}
.box {{ background: #fafafa; border: 1px solid #d8d8d8; padding: 4px; }}
pre {{ font-family: Consolas, monospace; font-size: 9pt; }}
</style>
"""


def parse_float(text):
    """Accept numbers with a decimal point or decimal comma. Empty -> NaN."""
    t = str(text).strip().replace(" ", "")
    if t == "":
        return math.nan
    if "," in t and "." in t:
        # format 1.234,5 or 1,234.5
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    else:
        t = t.replace(",", ".")
    return float(t)


def fmt(v, digits=5):
    if v is None:
        return ""
    try:
        if math.isnan(v):
            return "-"
        if math.isinf(v):
            return "∞"
    except TypeError:
        return str(v)
    if v != 0 and (abs(v) >= 1e6 or abs(v) < 1e-4):
        return f"{v:.{digits - 1}e}"
    return f"{v:.{digits}g}"


def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    return "&lt; 0.0001" if p < 1e-4 else f"{p:.4f}"


def p_class(p, alpha=0.05):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    return "sig" if p < alpha else "ns"


class MplCanvas(QWidget):
    """Matplotlib canvas widget with a toolbar (zoom, pan, save image)."""

    def __init__(self, parent=None, toolbar=True, figsize=(6, 4)):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.toolbar = NavigationToolbar2QT(self.canvas, self) if toolbar else None
        if toolbar:
            lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

    def clear(self):
        self.figure.clear()

    def draw(self):
        self.canvas.draw_idle()

    def message(self, text):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=11, color="#555", wrap=True)
        self.draw()


def model_source_combo(project, j, on_change=None):
    """'RSM' / 'ANN' combo to choose the model for response j used by graphs, optimization and prediction."""
    from PySide6.QtWidgets import QComboBox
    cb = QComboBox()
    cb.addItem("RSM", "rsm")
    cb.addItem("ANN", "ann")
    m = project.ann.get(j)
    if m is None:
        cb.model().item(1).setEnabled(False)
        cb.setToolTip("No ANN model for this response yet (train one on the ANN page first).")
    else:
        te = m.metrics.get("test", {}) if isinstance(m.metrics, dict) else {}
        cb.setToolTip(f"ANN {m.arch}" + (f", test R2 {te['r2']:.4f}" if te.get("r2") is not None else ""))
    fit = project.fit(j)
    if fit is not None:
        pr2 = fit.stats.get("pred_r2")
        cb.setItemData(0, f"RSM, Pred R2 {pr2:.4f}" if pr2 is not None and np.isfinite(pr2) else "RSM",
                       Qt.ToolTipRole)
    cb.setCurrentIndex(1 if project.source(j) == "ann" else 0)

    def changed():
        project.model_source[j] = cb.currentData()
        project.dirty = True
        if on_change:
            on_change()
    cb.currentIndexChanged.connect(changed)
    return cb


class ArrayModel(QAbstractTableModel):
    """Read-only table model for large data: cells are formatted only when shown (QTableView creates no items)."""

    def __init__(self, headers=(), columns=(), fmt_fn=None, parent=None):
        super().__init__(parent)
        self.set_data(headers, columns, fmt_fn)

    def set_data(self, headers, columns, fmt_fn=None, bg_fn=None):
        """bg_fn(row, column) -> background color (str) or None."""
        self.beginResetModel()
        self.headers = list(headers)
        self.columns = [np.asarray(c) for c in columns]
        self.fmt_fn = fmt_fn
        self.bg_fn = bg_fn
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() or not self.columns else len(self.columns[0])

    def columnCount(self, parent=QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() else len(self.columns)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            v = self.columns[index.column()][index.row()]
            if self.fmt_fn is not None:
                return self.fmt_fn(index.column(), v)
            if isinstance(v, (float, np.floating)):
                return "" if np.isnan(v) else f"{v:.5g}"
            return str(v)
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.BackgroundRole and self.bg_fn is not None:
            c = self.bg_fn(index.row(), index.column())
            return QBrush(QColor(c)) if c else None
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role == Qt.DisplayRole and orientation == Qt.Horizontal and section < len(self.headers):
            return self.headers[section]
        return None


class Task(QThread):
    """Run a heavy function in a separate thread so the window stays responsive.

    fn is called as fn(report, cancelled): report(text, i, n) sends progress, cancelled() -> bool.
    """
    progressed = Signal(str, int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            res = self.fn(lambda text, i=0, n=0: self.progressed.emit(text, int(i), int(n)), lambda: self._cancel)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.done.emit(res)
