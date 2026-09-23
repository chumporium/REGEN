"""Main window: menus, toolbar, navigation tree on the left and the work pages on the right."""
import os

import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QApplication, QCommandLinkButton, QFileDialog, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QSizePolicy, QSplitter, QStackedWidget, QToolBar,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from doe.project import Project

from . import theme
from .analysis_tab import AnalysisTab
from .ann_tab import AnnTab
from .cases_dialog import CasesDialog
from .data_tab import DataTab
from .eval_tab import EvalTab
from .graphs_tab import GraphsTab
from .import_dialog import ImportDialog
from .info_tabs import CoefTableTab, GraphColumnsTab, NotesTab, SummaryTab
from .new_design_dialog import NewDesignDialog
from .normalize_tab import NormalizeTab
from .nsga_tab import NsgaTab
from .optim_tab import OptimTab
from .predict_tab import PredictTab
from .readiness_tab import ReadinessTab
from .report import write_excel, write_pdf

APP_NAME = "REGEN"
APP_FULL = "Regression, Experimental design, GEnetic optimisation and Neural networks"
SETTINGS_KEY = "DOE Studio"      # old name: keeps the recent-projects list readable
APP_VERSION = "7.0.0"
FILE_FILTER = "REGEN Project (*.doe)"
MAX_RECENT = 8


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = None
        self.settings = QSettings(SETTINGS_KEY, SETTINGS_KEY)
        self.resize(1360, 860)
        self.setMinimumSize(960, 600)

        self.data_tab = DataTab()
        self.eval_tab = EvalTab()
        self.analysis_tab = AnalysisTab()
        self.graphs_tab = GraphsTab()
        self.optim_tab = OptimTab()
        self.predict_tab = PredictTab()
        self.ann_tab = AnnTab()
        self.nsga_tab = NsgaTab()
        self.norm_tab = NormalizeTab()
        self.ready_tab = ReadinessTab()
        self.summary_tab = SummaryTab()
        self.notes_tab = NotesTab()
        self.gcols_tab = GraphColumnsTab()
        self.coef_tab = CoefTableTab()
        self.home = self._start_page()

        # (key, icon, navigation label, title, description, widget, needs project, group)
        self.pages = [
            ("home", "home", "Start Page", "Start Page", "", self.home, False, None),
            ("summary", "summary", "Summary", "Project Summary", "", self.summary_tab, True, "Project"),
            ("notes", "notes", "Notes", "Notes", "", self.notes_tab, True, "Project"),
            ("gcols", "scatter", "Column Graphs", "Column Graphs", "", self.gcols_tab, True, "Project"),
            ("data", "data", "Data", "Data & Design", "", self.data_tab, True, "Design"),
            ("ready", "check", "Data Readiness", "Data Readiness", "", self.ready_tab, True, "Design"),
            ("eval", "eval", "Design Evaluation", "Design Evaluation", "", self.eval_tab, True, "Design"),
            ("analysis", "analysis", "RSM Analysis", "RSM Analysis", "", self.analysis_tab, True, "Analysis"),
            ("coef", "coef", "Coefficient Table", "Coefficient Table", "", self.coef_tab, True, "Analysis"),
            ("ann", "ann", "ANN", "Artificial Neural Network", "", self.ann_tab, True, "Analysis"),
            ("graphs", "graph", "Model Graphs", "Model Graphs", "", self.graphs_tab, True, "Analysis"),
            ("optim", "optim", "Desirability", "Desirability Optimization", "", self.optim_tab, True, "Optimization"),
            ("nsga", "nsga", "NSGA-II / III", "Multi-objective Optimization (NSGA-II / NSGA-III)", "", self.nsga_tab,
             True, "Optimization"),
            ("predict", "predict", "Prediction & Confirmation", "Prediction & Confirmation", "", self.predict_tab,
             True, "Optimization"),
            ("norm", "norm", "Data Normalization", "Data Normalization", "", self.norm_tab, True, "Tools"),
        ]

        self.stack = QStackedWidget()
        for page in self.pages:
            self.stack.addWidget(page[5])
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._navigator())
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(8, 6, 8, 4)
        bl.addWidget(self.stack)
        split.addWidget(body)
        split.setStretchFactor(1, 1)
        split.setSizes([215, 1145])
        split.setCollapsible(0, False)
        self.setCentralWidget(split)

        self.lbl_status = QLabel()
        self.statusBar().addPermanentWidget(self.lbl_status)
        self.statusBar().showMessage("Ready")

        self.data_tab.data_changed.connect(self.update_title)
        self.data_tab.responses_changed.connect(self.on_responses_changed)
        self.analysis_tab.model_changed.connect(self.update_title)
        self.data_tab.design_changed.connect(self.on_design_changed)
        self.optim_tab.send_to_prediction.connect(self.on_send_to_prediction)
        self.nsga_tab.send_to_prediction.connect(self.on_send_to_prediction)
        self.ann_tab.model_changed.connect(self.update_title)
        self.ready_tab.project_changed.connect(self.on_design_changed)
        self.ready_tab.model_changed.connect(self.update_title)
        self.notes_tab.changed.connect(self.update_title)
        self.nsga_tab.model_changed.connect(self.update_title)
        self.optim_tab.model_changed.connect(self.update_title)

        self._build_actions()
        self.go("home")
        self.update_title()

    # ---------------------------------------------------------------- layout
    def _navigator(self):
        self.tree = QTreeWidget()
        self.tree.setObjectName("Nav")
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.setIconSize(QSize(16, 16))
        self.tree.setMinimumWidth(190)
        self.nav, groups = {}, {}
        for key, glyph, label, *_rest in self.pages:
            group = _rest[-1]
            if group is None:
                parent = self.tree.invisibleRootItem()
            else:
                if group not in groups:
                    g = QTreeWidgetItem(self.tree, [group])
                    f = g.font(0)
                    f.setBold(True)
                    g.setFont(0, f)
                    g.setFlags(Qt.ItemIsEnabled)
                    g.setExpanded(True)
                    groups[group] = g
                parent = groups[group]
            it = QTreeWidgetItem(parent, [label])
            it.setIcon(0, theme.glyph_icon(glyph, "#404040", 16))
            it.setData(0, Qt.UserRole, key)
            self.nav[key] = it
        for g in groups.values():
            g.setExpanded(True)
        self.tree.currentItemChanged.connect(self._on_nav)
        return self.tree

    def _on_nav(self, cur, _prev):
        if cur is None or self._syncing:
            return
        key = cur.data(0, Qt.UserRole)
        if key:
            self.go(key)

    _syncing = False

    def _start_page(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(24, 18, 24, 18)
        lay.setSpacing(30)
        left = QVBoxLayout()
        t = QLabel(APP_NAME)
        t.setObjectName("StartTitle")
        v = QLabel(f"{APP_FULL}<br>Version {APP_VERSION}")
        v.setStyleSheet(f"color: {theme.MUTED};")
        left.addWidget(t)
        left.addWidget(v)
        left.addSpacing(18)
        left.addWidget(QLabel("<b>Start</b>"))
        left.setSpacing(4)
        for text, desc, slot in (
                ("New design", "Factorial, CCD, Box-Behnken, mixture, optimal, screening, space-filling",
                 self.new_design),
                ("Design guide", "Design and number of runs by goal: screening, RSM optimization, ANN",
                 self.design_guide),
                ("Open project", "A saved .doe file", self.open_project),
                ("Import data", "Paste from Excel or open an Excel/CSV file", self.import_data),
                ("Case studies", "Example projects for practice", self.open_cases)):
            b = QCommandLinkButton(text, desc)
            b.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            b.setMaximumWidth(520)
            b.clicked.connect(slot)
            left.addWidget(b)
        left.addStretch()
        lay.addLayout(left, 1)
        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Recent projects</b>"))
        self.lst_recent = QListWidget()
        self.lst_recent.itemActivated.connect(lambda it: self.open_recent(it.data(Qt.UserRole)))
        right.addWidget(self.lst_recent, 1)
        hint = QLabel("Double-click to open.")
        hint.setStyleSheet(f"color: {theme.MUTED};")
        right.addWidget(hint)
        lay.addLayout(right, 1)
        self._fill_recent()
        return w

    def _fill_recent(self):
        if not hasattr(self, "lst_recent"):
            return
        self.lst_recent.clear()
        for f in (f for f in self.recent_files() if os.path.exists(f)):
            folder = os.path.basename(os.path.dirname(f)) or os.path.dirname(f)
            it = QListWidgetItem(theme.glyph_icon("open", "#404040", 16), f"{os.path.basename(f)}   -   {folder}")
            it.setData(Qt.UserRole, f)
            it.setToolTip(f)
            self.lst_recent.addItem(it)
        if not self.lst_recent.count():
            it = QListWidgetItem("(none yet)")
            it.setFlags(Qt.NoItemFlags)
            self.lst_recent.addItem(it)
        if hasattr(self, "menu_recent"):
            self.menu_recent.clear()
            for f in (f for f in self.recent_files() if os.path.exists(f)):
                self.menu_recent.addAction(os.path.basename(f), lambda path=f: self.open_recent(path))
            self.menu_recent.setEnabled(bool(self.menu_recent.actions()))

    def recent_files(self):
        v = self.settings.value("recent", [])
        if isinstance(v, str):
            v = [v]
        return list(v or [])

    def add_recent(self, path):
        path = os.path.abspath(path)
        files = [path] + [f for f in self.recent_files() if os.path.normcase(f) != os.path.normcase(path)]
        self.settings.setValue("recent", files[:MAX_RECENT])
        self._fill_recent()

    # ---------------------------------------------------------------- menu & toolbar
    def _action(self, menu, text, slot, shortcut=None, glyph=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(shortcut)
        if glyph:
            a.setIcon(theme.glyph_icon(glyph, "#303030", 16))
        a.triggered.connect(slot)
        if menu is not None:
            menu.addAction(a)
        return a

    def _build_actions(self):
        mb = self.menuBar()
        m = mb.addMenu("&File")
        self.act_new = self._action(m, "&New Design...", self.new_design, QKeySequence.New, "new")
        self.act_guide = self._action(m, "Design Guide...", self.design_guide, None, "help")
        self.act_open = self._action(m, "&Open...", self.open_project, QKeySequence.Open, "open")
        self.menu_recent = m.addMenu("Open &Recent")
        m.addSeparator()
        self.act_import = self._action(m, "&Import Data...", self.import_data, None, "import")
        self.act_cases = self._action(m, "&Case Studies...", self.open_cases, None, "cases")
        m.addSeparator()
        self.act_save = self._action(m, "&Save", self.save_project, QKeySequence.Save, "save")
        self.act_save_as = self._action(m, "Save &As...", self.save_project_as, QKeySequence.SaveAs)
        m.addSeparator()
        self.act_export = self._action(m, "Export to &Excel...", self.export_excel, None, "excel")
        self.act_pdf = self._action(m, "&PDF Report...", self.export_pdf, None, "pdf")
        m.addSeparator()
        self._action(m, "E&xit", self.close, QKeySequence.Quit)

        view = mb.addMenu("&View")
        group = None
        self.view_actions = {}
        for key, _g, label, *_rest in self.pages:
            if _rest[-1] != group and group is not None:
                view.addSeparator()
            group = _rest[-1]
            self.view_actions[key] = self._action(view, label, lambda _=False, k=key: self.go(k))

        tools = mb.addMenu("&Tools")
        self._action(tools, "Data Normalization", lambda: self.go("norm"))

        h = mb.addMenu("&Help")
        self._action(h, "Getting Started", self.show_guide, QKeySequence.HelpContents)
        self._action(h, "Design Guide (RSM / ANN Run Counts)...", self.design_guide)
        h.addSeparator()
        self._action(h, f"About {APP_NAME}", self.show_about)

        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for a in (self.act_new, self.act_guide, self.act_open, self.act_save):
            tb.addAction(a)
        tb.addSeparator()
        for a in (self.act_import, self.act_cases):
            tb.addAction(a)
        tb.addSeparator()
        for a in (self.act_export, self.act_pdf):
            tb.addAction(a)
        self.addToolBar(tb)
        self._fill_recent()

    def go(self, key):
        for k, glyph, label, title, sub, widget, needs, _ in self.pages:
            if k != key:
                continue
            if needs and self.project is None:
                return
            self.stack.setCurrentWidget(widget)
            self._syncing = True
            self.tree.setCurrentItem(self.nav[k])
            self._syncing = False
            if self.project is not None and hasattr(widget, "refresh"):
                widget.refresh()
            self.update_title()
            return

    def current_key(self):
        w = self.stack.currentWidget()
        return next(page[0] for page in self.pages if page[5] is w)

    def update_title(self):
        has = self.project is not None
        for a in (self.act_save, self.act_save_as, self.act_export, self.act_pdf):
            a.setEnabled(has)
        for page in self.pages:
            ok = has or not page[6]
            it = self.nav[page[0]]
            it.setFlags((Qt.ItemIsEnabled | Qt.ItemIsSelectable) if ok else Qt.NoItemFlags)
            self.view_actions[page[0]].setEnabled(ok)
        if not has:
            self.setWindowTitle(APP_NAME)
            self.lbl_status.setText("")
            return
        p = self.project
        name = os.path.basename(p.path) if p.path else "Untitled"
        star = "*" if p.dirty else ""
        self.setWindowTitle(f"{name}{star} - {APP_NAME}")
        filled = int(np.all(np.isfinite(p.data), axis=1).sum()) if p.data.size else 0
        self.lbl_status.setText(f"{p.design_description()}   |   {p.n} runs, {filled} complete"
                                + (f", {len(p.excluded)} excluded" if p.excluded else "") + "  ")

    # ---------------------------------------------------------------- project
    def set_project(self, project, page="data"):
        self.project = project
        for w in (self.data_tab, self.eval_tab, self.analysis_tab, self.graphs_tab, self.optim_tab,
                  self.predict_tab, self.ann_tab, self.nsga_tab, self.norm_tab, self.ready_tab,
                  self.summary_tab, self.notes_tab, self.gcols_tab, self.coef_tab):
            w.set_project(project)
        self.update_title()
        self.go(page)

    def on_responses_changed(self):
        self.analysis_tab.refresh_responses()
        self.graphs_tab.refresh_responses()
        self.optim_tab.fill_table()
        self.ann_tab.refresh_responses()
        self.update_title()

    def confirm_discard(self):
        if self.project is None or not self.project.dirty:
            return True
        self.optim_tab.save_criteria()
        r = QMessageBox.question(self, APP_NAME, "The current project has unsaved changes. Save it first?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save_project()
        return r == QMessageBox.Discard

    def new_design(self):
        if not self.confirm_discard():
            return
        dlg = NewDesignDialog(self)
        if dlg.exec() and dlg.project is not None:
            self.set_project(dlg.project)

    def design_guide(self):
        from .design_guide import DesignGuideDialog
        k = self.project.k if self.project is not None else 3
        dlg = DesignGuideDialog(self, k)
        if not (dlg.exec() and dlg.choice):
            return
        if not self.confirm_discard():
            return
        nd = NewDesignDialog(self)
        nd.preset(*dlg.choice)
        if nd.exec() and nd.project is not None:
            self.set_project(nd.project)

    def import_data(self):
        if not self.confirm_discard():
            return
        dlg = ImportDialog(self)
        if dlg.exec() and dlg.project is not None:
            self.set_project(dlg.project, "analysis")

    def open_cases(self):
        if not self.confirm_discard():
            return
        dlg = CasesDialog(self)
        if dlg.exec() and dlg.factory is not None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.set_project(dlg.factory())
            finally:
                QApplication.restoreOverrideCursor()

    def on_design_changed(self):
        self.set_project(self.project, self.current_key())

    def on_send_to_prediction(self, coded):
        self.predict_tab.set_point(coded)
        self.go("predict")

    def open_project(self):
        if not self.confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", FILE_FILTER)
        if path:
            self.load_path(path)

    def open_recent(self, path):
        if self.confirm_discard():
            self.load_path(path)

    def load_path(self, path):
        try:
            project = Project.load(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Open Failed", f"The file could not be opened:\n{exc}")
            return
        self.set_project(project)
        self.add_recent(path)

    def save_project(self):
        if self.project is None:
            return False
        if not self.project.path:
            return self.save_project_as()
        self.optim_tab.save_criteria()
        try:
            self.project.save(self.project.path)
        except OSError as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            return False
        self.add_recent(self.project.path)
        self.update_title()
        self.statusBar().showMessage(f"Saved: {self.project.path}", 4000)
        return True

    def save_project_as(self):
        if self.project is None:
            return False
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", self.project.path or "experiment.doe",
                                              FILE_FILTER)
        if not path:
            return False
        if not path.lower().endswith(".doe"):
            path += ".doe"
        self.project.path = path
        return self.save_project()

    def export_excel(self):
        p = self.project
        if p is None:
            return
        default = os.path.splitext(p.path)[0] + ".xlsx" if p.path else "doe_results.xlsx"
        path, _ = QFileDialog.getSaveFileName(self, "Export to Excel", default, "Excel (*.xlsx)")
        if not path:
            return
        try:
            write_excel(p, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export Failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported to {path}", 5000)

    def export_pdf(self):
        p = self.project
        if p is None:
            return
        self.optim_tab.save_criteria()
        default = os.path.splitext(p.path)[0] + ".pdf" if p.path else "doe_report.pdf"
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF Report", default, "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            write_pdf(p, path, self.optim_tab.criteria_used, self.optim_tab.solutions, APP_NAME)
        except Exception as exc:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "PDF Export Failed", str(exc))
            return
        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"PDF report saved: {path}", 6000)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def show_guide(self):
        QMessageBox.information(self, "Getting Started", (
            "Choose a work page in the left panel or from the View menu.\n\n"
            "1. Create a design (File > New Design; File > Design Guide to choose a design and number of runs by "
            "goal), import existing data (File > Import Data), or open a case study.\n"
            "2. Data: run the experiments in Run order, then enter the responses. You can paste from Excel "
            "(Ctrl+V).\n"
            "3. Data Readiness and Design Evaluation: check whether the data are sufficient and add the suggested "
            "runs.\n"
            "4. RSM Analysis: choose a model, check the ANOVA and diagnostics. If the pattern is strongly "
            "nonlinear, try ANN.\n"
            "5. Model Graphs: contour, 3D surface, factor effects.\n"
            "6. Optimization (desirability or NSGA-II / NSGA-III), then Prediction & Confirmation.\n\n"
            "Save, Excel export and PDF report are in the File menu and on the toolbar."))

    def show_about(self):
        QMessageBox.about(self, f"About {APP_NAME}",
                          f"<b>{APP_NAME}</b> version {APP_VERSION}<br><i>{APP_FULL}</i><br><br>"
                          "Software for design of experiments, response surface methodology, artificial neural "
                          "networks and multi-objective optimization.<br><br>"
                          "© 2026 Erdiyanto Munandar and Nasruddin<br>"
                          "Department of Mechanical Engineering, Universitas Indonesia<br>"
                          "Released under the MIT License.")

    def closeEvent(self, event):  # noqa: N802
        task = self.ann_tab.task
        if task is not None and task.isRunning():
            self.ann_tab.cancel_search()
            task.wait(20000)
        ntask = self.nsga_tab.task
        if ntask is not None and ntask.isRunning():
            ntask.cancel()
            ntask.wait(20000)
        if self.confirm_discard():
            from doe import parallel
            parallel.shutdown()
            event.accept()
        else:
            event.ignore()
