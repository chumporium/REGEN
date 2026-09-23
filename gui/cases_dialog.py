"""Case study gallery: one menu, then choose the case to open."""
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QVBoxLayout, QWidget)

from doe.project import CASES

from . import theme

CATEGORY_GLYPH = {"Response Surface": "graph", "Mixture": "analysis", "Screening": "eval", "GLM": "analysis",
                  "Split-Plot": "data", "Mixture + Process": "optim", "Machine Learning": "ann"}


class CasesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Case Studies")
        self.resize(820, 470)
        self.factory = None
        lay = QVBoxLayout(self)
        head = QLabel("Choose a case study to open. All sample data can be edited and saved as your own project.")
        head.setStyleSheet(f"color: {theme.MUTED};")
        lay.addWidget(head)
        row = QHBoxLayout()
        lay.addLayout(row, 1)
        self.list = QListWidget()
        self.list.setMinimumWidth(330)
        self.list.setIconSize(QSize(20, 20))
        for key, title, cat, fn, desc in CASES:
            it = QListWidgetItem(theme.glyph_icon(CATEGORY_GLYPH.get(cat, "cases"), theme.ACCENT), f"{title}\n{cat}")
            it.setData(Qt.UserRole, key)
            it.setSizeHint(QSize(300, 52))
            self.list.addItem(it)
        row.addWidget(self.list)
        side = QWidget()
        side.setAutoFillBackground(True)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(18, 16, 18, 16)
        self.lbl_cat = QLabel()
        self.lbl_cat.setStyleSheet(f"color: {theme.MUTED};")
        self.lbl_title = QLabel()
        self.lbl_title.setWordWrap(True)
        self.lbl_title.setStyleSheet("font-size: 12pt; font-weight: 600;")
        self.lbl_desc = QLabel()
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_desc.setStyleSheet("font-size: 10pt; line-height: 140%;")
        for w in (self.lbl_cat, self.lbl_title, self.lbl_desc):
            sl.addWidget(w)
        sl.addStretch()
        row.addWidget(side, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Open).setText("Open Case Study")
        bb.button(QDialogButtonBox.Open).setObjectName("Primary")
        bb.button(QDialogButtonBox.Cancel).setText("Cancel")
        bb.accepted.connect(self.accept_case)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.list.currentRowChanged.connect(self.show_case)
        self.list.itemDoubleClicked.connect(lambda _: self.accept_case())
        self.list.setCurrentRow(0)

    def show_case(self, row):
        if row < 0:
            return
        key, title, cat, fn, desc = CASES[row]
        self.lbl_cat.setText(cat)
        self.lbl_title.setText(title)
        self.lbl_desc.setText(desc)

    def accept_case(self):
        row = self.list.currentRow()
        if row >= 0:
            self.factory = CASES[row][3]
            self.accept()
