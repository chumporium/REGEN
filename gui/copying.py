"""Copy tables to the clipboard (TSV → pastes straight into Excel columns) with the Windows decimal separator."""
import html as html_mod
import re
from html.parser import HTMLParser

from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import QMenu, QTextBrowser

_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?%?$")


def system_decimal():
    try:
        sep = QLocale.system().decimalPoint()
        return sep if isinstance(sep, str) and sep else "."
    except Exception:  # noqa: BLE001
        return "."


def localize(text, sep):
    """Convert the decimal point to a comma when needed (only for cells that are purely numeric)."""
    t = text.strip()
    if sep != ",":
        return t
    if _NUM.match(t.replace(" ", "")):
        return t.replace(".", ",")
    m = re.match(r"^([<>≤≥]=?\s*)(.+)$", t)          # e.g. "< 0.0001"
    if m and _NUM.match(m.group(2).replace(" ", "")):
        return m.group(1) + m.group(2).replace(".", ",")
    return t


def _clean(text):
    return " ".join(str(text).replace("\t", " ").split())


def to_clipboard(rows, sep):
    tsv = "\n".join("\t".join(localize(_clean(c), sep) for c in r) for r in rows)
    QGuiApplication.clipboard().setText(tsv)
    return tsv


# ------------------------------------------------------------------ QTableWidget
def table_rows(table, selection_only=True, headers=True):
    """Table contents (QTableWidget or QTableView) as a list of rows (cell text or combo box text)."""
    model = table.model()
    widget = hasattr(table, "cellWidget")

    def cell(r, c):
        if widget:
            w = table.cellWidget(r, c)
            if w is not None and hasattr(w, "currentText"):
                return w.currentText()
        v = model.data(model.index(r, c), Qt.DisplayRole)
        return "" if v is None else str(v)

    sel = table.selectionModel().selectedIndexes() if table.selectionModel() else []
    if selection_only and sel:
        rows_idx = sorted({i.row() for i in sel})
        cols_idx = sorted({i.column() for i in sel})
    else:
        rows_idx, cols_idx = list(range(model.rowCount())), list(range(model.columnCount()))
    rows_idx = [r for r in rows_idx if not table.isRowHidden(r)]
    cols_idx = [c for c in cols_idx if not table.isColumnHidden(c)]
    out = []
    if headers:
        out.append([str(model.headerData(c, Qt.Horizontal, Qt.DisplayRole) or "") for c in cols_idx])
    for r in rows_idx:
        out.append([cell(r, c) for c in cols_idx])
    return out


def install_table_copy(table):
    """Add Ctrl+C and a right-click 'Copy ...' menu to a QTableWidget."""
    sc = QShortcut(QKeySequence.Copy, table)
    sc.setContext(Qt.WidgetWithChildrenShortcut)
    sc.activated.connect(lambda: to_clipboard(table_rows(table, True, False), system_decimal()))
    table.setContextMenuPolicy(Qt.CustomContextMenu)

    def menu(pos):
        m = QMenu(table)
        sep = system_decimal()
        other = "." if sep == "," else ","
        acts = [("Copy selected cells", True, False, sep), ("Copy selected cells with column headers", True, True, sep),
                ("Copy entire table (for Excel)", False, True, sep), None,
                (f"Copy entire table - decimal '{other}'", False, True, other)]
        for a in acts:
            if a is None:
                m.addSeparator()
                continue
            label, sel, head, s = a
            act = QAction(label, m)
            act.triggered.connect(lambda _=False, sel=sel, head=head, s=s: to_clipboard(table_rows(table, sel, head), s))
            m.addAction(act)
        m.exec(table.viewport().mapToGlobal(pos))

    table.customContextMenuRequested.connect(menu)
    return table


# ------------------------------------------------------------------ HTML reports
class _TableParser(HTMLParser):
    """Collect every <table> together with the last heading (h1–h3) before it."""

    def __init__(self):
        super().__init__()
        self.tables, self.title = [], ""
        self._in_head, self._head = False, []
        self._table, self._row, self._cell, self._in_cell = None, None, [], False
        self._span = 1

    def handle_starttag(self, tag, attrs):
        if tag in ("h1", "h2", "h3"):
            self._in_head, self._head = True, []
        elif tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._in_cell, self._cell = True, []
            self._span = int(dict(attrs).get("colspan", 1) or 1)
        elif tag == "br" and self._in_cell:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("h1", "h2", "h3") and self._in_head:
            self._in_head = False
            self.title = _clean("".join(self._head))
        elif tag in ("td", "th") and self._in_cell:
            self._row.append(_clean("".join(self._cell)))
            self._row.extend([""] * (self._span - 1))
            self._in_cell = False
        elif tag == "tr" and self._row is not None:
            if self._table is not None:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append((self.title, self._table))
            self._table = None

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)
        elif self._in_head:
            self._head.append(data)


def html_tables(html_text):
    p = _TableParser()
    p.feed(html_text)
    return p.tables


class ReportBrowser(QTextBrowser):
    """QTextBrowser that can copy all of its tables to Excel (right-click)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._html = ""
        self.setOpenLinks(False)

    def setHtml(self, text):  # noqa: N802 (Qt API name)
        self._html = text
        super().setHtml(text)

    def table_rows(self):
        rows = []
        for title, table in html_tables(self._html):
            if title:
                rows.append([html_mod.unescape(title)])
            rows.extend([[html_mod.unescape(c) for c in r] for r in table])
            rows.append([])
        return rows

    def copy_tables(self, sep=None):
        return to_clipboard(self.table_rows(), sep or system_decimal())

    def contextMenuEvent(self, event):  # noqa: N802
        m = self.createStandardContextMenu()
        m.addSeparator()
        sep = system_decimal()
        other = "." if sep == "," else ","
        a1 = QAction("Copy all tables (for Excel)", m)
        a1.triggered.connect(lambda: self.copy_tables(sep))
        a2 = QAction(f"Copy all tables - decimal '{other}'", m)
        a2.triggered.connect(lambda: self.copy_tables(other))
        m.addAction(a1)
        m.addAction(a2)
        m.exec(event.globalPos())
