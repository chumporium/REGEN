"""Data Readiness page: readiness scores for modeling & optimization, check results, recommendations that can
be applied directly (add suggested runs, ignore outliers, transform, change model), and a suggested runs table."""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSplitter, QTableView, QVBoxLayout, QWidget)

from doe import models, readiness

from . import theme
from .common import REPORT_CSS, ArrayModel, fmt
from .copying import ReportBrowser, install_table_copy

STATUS_LABEL = {"ok": "Good", "warn": "Needs attention", "bad": "Problem", "info": "Info"}
COLOR = {"ok": theme.GOOD, "warn": theme.WARN, "bad": theme.BAD, "info": theme.MUTED}


def score_text(v):
    return "-" if not np.isfinite(v) else f"{v:.0f}"


def readiness_html(p, j, a, overview=None):
    h = []
    if overview:
        h.append("<h2>Readiness Summary for All Responses</h2><table><tr><th class='l'>Response</th><th>Modeling score</th>"
                 "<th>Optimization score</th><th class='l'>Conclusion</th><th>Recommendations</th></tr>")
        for jj, aa in overview:
            v, cls = readiness.verdict(aa["score_opt"])
            h.append(f"<tr><td class='l'>{p.response_label(jj)}</td><td>{score_text(aa['score_model'])}</td>"
                     f"<td>{score_text(aa['score_opt'])}</td><td class='l'><span class='{cls}'>{v}</span></td>"
                     f"<td>{len(aa['recs'])}</td></tr>")
        h.append("</table>")
    vm, cm_ = readiness.verdict(a["score_model"])
    vo, co = readiness.verdict(a["score_opt"])
    h.append(f"<h2>Data Readiness - {p.response_label(j)}</h2>"
             f"<div class='box'><p style='margin: 2px'>For <b>modeling</b>: <span class='{cm_}'>{vm} "
             f"(score {score_text(a['score_model'])})</span> &nbsp;·&nbsp; for <b>optimization</b>: "
             f"<span class='{co}'>{vo} (score {score_text(a['score_opt'])})</span></p>"
             f"<p class='note' style='margin: 2px'>{a['n']} complete runs, reference model "
             f"{models.order_label(a['order'])} ({a['n_par']} parameters).</p></div>")
    for cat, title in readiness.CATS.items():
        cs = [c for c in a["checks"] if c["cat"] == cat]
        if not cs:
            continue
        h.append(f"<h3>{title} - score {score_text(a['scores'][cat])}</h3><table><tr><th class='l'>Check</th>"
                 "<th class='l'>Value</th><th class='l'>Status</th><th class='l'>Explanation</th></tr>")
        for c in cs:
            h.append(f"<tr><td class='l'>{c['name']}</td><td class='l'>{c['value']}</td><td class='l'><span "
                     f"class='{c['status']}'>{STATUS_LABEL[c['status']]}</span></td><td class='l'>{c['detail']}</td></tr>")
        h.append("</table>")
    if a["recs"]:
        h.append("<h3>Recommendations</h3><table><tr><th>No</th><th class='l'>Type</th><th class='l'>Recommendation</th>"
                 "<th class='l'>From check</th></tr>")
        for n, r in enumerate(a["recs"], 1):
            kind = "Complete data" if r["kind"] == "complete" else "Treatment"
            h.append(f"<tr><td>{n}</td><td class='l'>{kind}</td><td class='l'>{r['text']}</td>"
                     f"<td class='l'><span class='{r['status']}'>{r['check']}</span></td></tr>")
        h.append("</table>")
    else:
        h.append("<p class='ok'>No recommendations - the data are already good for modeling and optimization.</p>")
    h.append("<p class='note'>Score = mean of the check statuses (Good = 100, Needs attention = 50, Problem = 0). "
             "The modeling score uses the data, structure and response quality groups; the optimization score adds "
             "optimization readiness. ≥ 85 ready, 60–85 adequate, < 60 not ready yet.</p>")
    return "".join(h)


class ScoreCard(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(0)
        self.t = QLabel(title)
        self.t.setStyleSheet(f"color: {theme.MUTED}; font-size: 8.5pt;")
        self.v = QLabel("-")
        self.v.setStyleSheet("font-size: 15pt;")
        self.s = QLabel("")
        self.s.setStyleSheet("font-size: 8.5pt;")
        for w in (self.t, self.v, self.s):
            lay.addWidget(w)

    def set(self, score, sub=None):
        text, cls = readiness.verdict(score)
        self.v.setText(score_text(score))
        self.v.setStyleSheet(f"font-size: 15pt; color: {COLOR[cls]};")
        self.s.setText(sub if sub is not None else text)
        self.s.setStyleSheet(f"font-size: 8.5pt; color: {COLOR[cls]};")


class ReadinessTab(QWidget):
    project_changed = Signal()          # runs added / ignored -> all pages need reloading
    model_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.a = None
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Response:"))
        self.cb_resp = QComboBox()
        self.cb_resp.setMinimumWidth(200)
        bar.addWidget(self.cb_resp)
        bar.addSpacing(10)
        bar.addWidget(QLabel("Reference model:"))
        self.cb_order = QComboBox()
        bar.addWidget(self.cb_order)
        self.btn_eval = QPushButton("Reassess")
        self.btn_eval.setIcon(theme.glyph_icon("eval", theme.ACCENT))
        bar.addWidget(self.btn_eval)
        bar.addStretch()
        lay.addLayout(bar)

        cards = QHBoxLayout()
        self.cards = {"model": ScoreCard("Modeling"), "opt": ScoreCard("Optimization")}
        for cat, title in readiness.CATS.items():
            self.cards[f"c_{cat}"] = ScoreCard(title)
        for key in ("model", "opt", "c_data", "c_model", "c_resp", "c_opt"):
            cards.addWidget(self.cards[key])
        lay.addLayout(cards)

        split = QSplitter(Qt.Horizontal)
        self.txt = ReportBrowser()
        split.addWidget(self.txt)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)
        rl.addWidget(QLabel("<b>Recommendations</b> (select one, then click Apply):"))
        self.lst = QListWidget()
        self.lst.setWordWrap(True)
        rl.addWidget(self.lst, 2)
        self.btn_apply = QPushButton("Apply Selected Recommendation")
        self.btn_apply.setObjectName("Primary")
        self.btn_apply.setIcon(theme.glyph_icon("check", "white"))
        rl.addWidget(self.btn_apply)
        row = QHBoxLayout()
        row.addWidget(QLabel("<b>Suggested runs:</b>"))
        self.cb_sugg = QComboBox()
        row.addWidget(self.cb_sugg, 1)
        rl.addLayout(row)
        self.tbl = QTableView()
        self.tbl_model = ArrayModel()
        self.tbl.setModel(self.tbl_model)
        self.tbl.verticalHeader().setVisible(False)
        install_table_copy(self.tbl)
        rl.addWidget(self.tbl, 2)
        self.btn_add = QPushButton("Add Suggested Runs to Data Sheet")
        self.btn_add.setIcon(theme.glyph_icon("add", theme.ACCENT))
        self.btn_add.setToolTip("New runs are added at the end of the run order with empty responses; perform the "
                                "experiments, then enter the responses on the Data page.")
        rl.addWidget(self.btn_add)
        note = QLabel("Suggested runs are generated automatically: D-optimal (adds the most model information), "
                      "factor space gap filling, replicates, or confirmation around the optimum.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.MUTED};")
        rl.addWidget(note)
        split.addWidget(right)
        split.setSizes([760, 480])
        lay.addWidget(split, 1)

        self.btn_eval.clicked.connect(self.evaluate)
        self.cb_resp.currentIndexChanged.connect(self.evaluate)
        self.cb_order.currentIndexChanged.connect(self.evaluate)
        self.cb_sugg.currentIndexChanged.connect(self.show_suggest)
        self.btn_apply.clicked.connect(self.apply_selected)
        self.btn_add.clicked.connect(self.add_suggest)
        self.lst.currentRowChanged.connect(self.on_rec_selected)
        self._loading = False

    def set_project(self, project):
        self.project = project
        self._loading = True
        self.cb_resp.clear()
        for j in range(len(project.responses)):
            self.cb_resp.addItem(project.response_label(j))
        self.cb_order.clear()
        for o in project.available_orders():
            self.cb_order.addItem(models.order_label(o), o)
        self._loading = False
        self._sync_order()
        self.evaluate()

    def refresh(self):
        if self.project is not None:
            self._sync_order()
            self.evaluate()

    def _sync_order(self):
        p = self.project
        j = max(self.cb_resp.currentIndex(), 0)
        if not p.responses:
            return
        self._loading = True
        i = self.cb_order.findData(p.model_spec(j)["order"])
        self.cb_order.setCurrentIndex(max(i, 0))
        self._loading = False

    def evaluate(self):
        p = self.project
        if p is None or self._loading or self.cb_resp.currentIndex() < 0:
            return
        j = self.cb_resp.currentIndex()
        order = self.cb_order.currentData()
        try:
            self.a = readiness.assess(p, j, order)
            overview = [(jj, self.a if jj == j else readiness.assess(p, jj)) for jj in range(len(p.responses))] \
                if len(p.responses) > 1 else None
        except Exception as exc:  # noqa: BLE001
            self.txt.setHtml(REPORT_CSS + f"<p class='bad'>Assessment failed: {exc}</p>")
            return
        a = self.a
        self.cards["model"].set(a["score_model"])
        self.cards["opt"].set(a["score_opt"])
        for cat in readiness.CATS:
            n = sum(1 for c in a["checks"] if c["cat"] == cat and c["status"] in ("warn", "bad"))
            self.cards[f"c_{cat}"].set(a["scores"][cat], f"{n} notes" if n else "good")
        self.txt.setHtml(REPORT_CSS + readiness_html(p, j, a, overview))
        self.lst.clear()
        for r in a["recs"]:
            icon = theme.glyph_icon("add" if r["kind"] == "complete" else "unc", COLOR.get(r["status"], theme.INK), 16)
            tag = "[Complete]" if r["kind"] == "complete" else "[Treatment]"
            it = QListWidgetItem(icon, f"{tag} {r['text']}" + ("" if r["action"] else "  (manual)"))
            it.setData(Qt.UserRole, r)
            self.lst.addItem(it)
        if not a["recs"]:
            self.lst.addItem("No recommendations - the data are already good.")
        self.cb_sugg.blockSignals(True)
        self.cb_sugg.clear()
        for key, arr in a["suggest"].items():
            if len(arr):
                self.cb_sugg.addItem(f"{readiness.SUGGEST_LABEL.get(key, key)} - {len(arr)} run", key)
        self.cb_sugg.blockSignals(False)
        self.show_suggest()
        self.btn_apply.setEnabled(bool(a["recs"]))

    def show_suggest(self):
        p, a = self.project, self.a
        key = self.cb_sugg.currentData()
        if a is None or key is None:
            self.tbl_model.set_data([], [])
            self.btn_add.setEnabled(False)
            return
        Z = a["suggest"][key]
        act = p.to_actual(Z)
        heads = ["No"] + [p.factor_label(i) for i in range(p.k)]
        cols = [np.arange(1, len(Z) + 1)] + [act[:, i] for i in range(p.k)]
        if key in ("near", "dopt", "space", "dopt_quad") and p.fit(a["j"]) is not None:
            m = p.model(a["j"])
            heads.append(f"Predicted {p.responses[a['j']].name}")
            cols.append(m.predict(Z))

        def f(c, v):
            if c == 0:
                return str(int(v))
            if 1 <= c <= p.k:
                return p.format_value(c - 1, v, 5)
            return fmt(float(v), 5)

        self.tbl_model.set_data(heads, cols, f)
        self.tbl.resizeColumnsToContents()
        self.btn_add.setEnabled(True)

    def on_rec_selected(self, row):
        if row < 0 or self.a is None:
            return
        r = self.lst.item(row).data(Qt.UserRole)
        if r and r.get("action") and r["action"][0] == "add_runs":
            i = self.cb_sugg.findData(r["action"][1])
            if i >= 0:
                self.cb_sugg.setCurrentIndex(i)

    def apply_selected(self):
        it = self.lst.currentItem()
        r = it.data(Qt.UserRole) if it is not None else None
        if not r:
            QMessageBox.information(self, "Recommendations", "Select a recommendation in the list.")
            return
        act = r.get("action")
        if not act:
            QMessageBox.information(self, "Recommendations",
                                    "This recommendation must be carried out manually:\n\n" + r["text"])
            return
        p, j = self.project, self.a["j"]
        kind, val = act
        if kind == "add_runs":
            i = self.cb_sugg.findData(val)
            if i >= 0:
                self.cb_sugg.setCurrentIndex(i)
            self.add_suggest()
            return
        if kind == "exclude":
            runs = ", ".join(str(p.run_order[r_]) for r_ in val[:15])
            if QMessageBox.question(self, "Ignore Runs", f"Ignore runs {runs} in the analysis? The data are not deleted "
                                    "and can be restored on the Data or Diagnostics page.") != QMessageBox.Yes:
                return
            p.set_excluded(val, True)
            self.project_changed.emit()
            return
        spec = p.model_spec(j)
        if kind == "transform":
            spec["transform"] = dict(val)
        elif kind == "order":
            spec["order"], spec["terms"] = val, None
        elif kind == "backward":
            try:
                spec["terms"] = [models.normalize_term(t) for t in p.backward_eliminate(j, 0.10)]
            except (ValueError, np.linalg.LinAlgError) as exc:
                QMessageBox.warning(self, "Recommendations", str(exc))
                return
        p.dirty = True
        self.model_changed.emit()
        self._sync_order()
        self.evaluate()

    def add_suggest(self):
        p, a = self.project, self.a
        key = self.cb_sugg.currentData()
        if a is None or key is None:
            return
        Z = a["suggest"][key]
        label = {"dopt": "D-optimal", "dopt_quad": "Mid-level", "rep": "Replicate", "space": "Space-filling",
                 "near": "Confirmation"}.get(key, "Suggested")
        if QMessageBox.question(self, "Add Runs", f"Add {len(Z)} runs ({label}) to the data sheet? Their responses "
                                "are empty and must be entered after the experiments are run.") != QMessageBox.Yes:
            return
        try:
            p.add_runs(Z, label)
        except ValueError as exc:
            QMessageBox.warning(self, "Add Runs", str(exc))
            return
        self.project_changed.emit()
