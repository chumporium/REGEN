"""Design Guide: design recommendations by goal, a two-level factorial table (color-coded resolution), and a
run count calculator for RSM and ANN."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)

from doe import designs, planning

from .common import REPORT_CSS
from .copying import ReportBrowser, install_table_copy

RES_COLORS = {"full": "#ffffff", 5: "#cfe8cc", 4: "#fbefb4", 3: "#f3c9c4", None: "#d4d4d4"}
ORDER_LABEL = {"linear": "Linear", "2fi": "2FI (2-factor interactions)", "quadratic": "Quadratic", "cubic": "Cubic"}

GOAL_HELP = {
    "screening": "Many factors (usually 6 or more) and it is not yet known which ones matter. Main effects are "
                 "enough; interactions and curvature are not needed yet. Once the important factors are found, "
                 "continue to characterization or optimization with fewer factors.",
    "characterize": "Few factors (2 to 8). You want clean main effects and two-factor interactions: this needs "
                    "resolution V or a full factorial. Center points are used to detect curvature.",
    "optimize": "Looking for the best combination. The response usually curves (has a peak or valley), so a "
                "quadratic model is needed: at least 3 levels per factor. Minimum runs = number of quadratic "
                "coefficients + about 5 for the lack of fit test + 3 to 5 replicates for pure error.",
    "ann": "An ANN learns from examples, so it needs more data than RSM and points spread across the whole space "
           "(many levels per factor). Latin Hypercube gives each factor as many levels as there are runs. "
           "Replicated center points give a noise estimate for detecting overfitting.",
    "rsm_ann": "One data set for two models: a classic RSM design (CCD / Box-Behnken) plus space-filling points. "
               "RSM keeps the structure it needs, the ANN gets extra points in between, and both can be compared "
               "fairly on the same data.",
    "robust": "To show that small variations in the factors do NOT change the response (for example, method "
              "validation). Fewest runs: resolution III or Plackett-Burman.",
}


SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def power_text(k, p):
    """2^(k-p) with superscript digits, e.g. 2 to the power 14-10."""
    return "2" + (f"{k}" if p == 0 else f"{k}-{p}").translate(SUP)


def _res_key(res):
    if res == 0:
        return "full"
    return 5 if res >= 5 else res


class DesignGuideDialog(QDialog):
    """self.choice = (design_type, k, options) when the user chooses 'Create This Design'."""

    def __init__(self, parent=None, k=3):
        super().__init__(parent)
        self.setWindowTitle("Design Guide")
        self.resize(1180, 760)
        self.choice = None
        self._recs = []
        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self.tabs.addTab(self._build_recommend(k), "Recommendations by Goal")
        self.tabs.addTab(self._build_table(), "Two-Level Factorial Table")
        self.tabs.addTab(self._build_calc(k), "Run Count Calculator")
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Close)
        self.btn_make = bb.button(QDialogButtonBox.Ok)
        self.btn_make.setText("Create This Design...")
        bb.button(QDialogButtonBox.Close).setText("Close")
        bb.accepted.connect(self.make)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)
        self.tabs.currentChanged.connect(self._sync_make)
        self.update_recs()
        self.select_cell(3, 3)
        self.update_calc()

    # ================================================================ recommendations
    def _build_recommend(self, k):
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QHBoxLayout()
        self.cb_goal = QComboBox()
        for key, label in planning.GOALS.items():
            self.cb_goal.addItem(label, key)
        self.cb_goal.setCurrentIndex(self.cb_goal.findData("optimize"))
        self.sp_k = QSpinBox()
        self.sp_k.setRange(1, 21)
        self.sp_k.setValue(max(1, min(21, k)))
        self.sp_budget = QSpinBox()
        self.sp_budget.setRange(0, 5000)
        self.sp_budget.setSpecialValueText("no limit")
        self.cb_nonlin = QComboBox()
        for key, label in planning.NONLINEAR.items():
            self.cb_nonlin.addItem(label, key)
        self.cb_nonlin.setCurrentIndex(1)
        self.lbl_nonlin = QLabel("Nonlinearity:")
        for lab, wd in (("Goal:", self.cb_goal), ("Numeric factors:", self.sp_k), ("Max run budget:",
                                                                                   self.sp_budget),
                        (self.lbl_nonlin, self.cb_nonlin)):
            form.addWidget(lab if isinstance(lab, QLabel) else QLabel(lab))
            form.addWidget(wd)
        form.addStretch()
        lay.addLayout(form)
        self.lbl_goal = QLabel()
        self.lbl_goal.setWordWrap(True)
        self.lbl_goal.setStyleSheet("color: #444; padding: 4px 0;")
        lay.addWidget(self.lbl_goal)
        split = QSplitter(Qt.Vertical)
        self.tbl_rec = QTableWidget(0, 6)
        self.tbl_rec.setHorizontalHeaderLabels(["Design", "Runs", "Levels per factor", "Model", "Power (2σ signal)",
                                                "Notes"])
        self.tbl_rec.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_rec.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.tbl_rec.verticalHeader().setVisible(False)
        self.tbl_rec.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_rec.setSelectionMode(QTableWidget.SingleSelection)
        self.tbl_rec.setEditTriggers(QTableWidget.NoEditTriggers)
        install_table_copy(self.tbl_rec)
        split.addWidget(self.tbl_rec)
        self.txt_rec = ReportBrowser()
        split.addWidget(self.txt_rec)
        split.setSizes([300, 300])
        lay.addWidget(split, 1)
        self.cb_goal.currentIndexChanged.connect(self.update_recs)
        self.sp_k.valueChanged.connect(self.update_recs)
        self.sp_budget.valueChanged.connect(self.update_recs)
        self.cb_nonlin.currentIndexChanged.connect(self.update_recs)
        self.tbl_rec.itemSelectionChanged.connect(self.show_rec)
        self.tbl_rec.itemDoubleClicked.connect(lambda *_: self.make())
        return w

    def update_recs(self):
        goal = self.cb_goal.currentData()
        ann_goal = goal in ("ann", "rsm_ann")
        self.cb_nonlin.setVisible(ann_goal)
        self.lbl_nonlin.setVisible(ann_goal)
        self.lbl_goal.setText(GOAL_HELP[goal])
        k = self.sp_k.value()
        self._recs = planning.recommend(goal, k, self.sp_budget.value(), self.cb_nonlin.currentData())
        self.tbl_rec.setRowCount(len(self._recs))
        for r, o in enumerate(self._recs):
            pw = "" if not np.isfinite(o.get("power", np.nan)) else f"{100 * o['power']:.0f}%"
            note = ("over budget" if o.get("over") else "") or o.get("note", "")
            vals = [o["title"], str(o["runs"]), o.get("levels", ""), ORDER_LABEL.get(o.get("order"), "-"), pw, note]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c in (1, 4):
                    it.setTextAlignment(Qt.AlignCenter)
                if o.get("over") or o["type"] is None:
                    it.setForeground(QBrush(QColor("#8a8a8a")))
                self.tbl_rec.setItem(r, c, it)
        self.tbl_rec.resizeColumnsToContents()
        self.tbl_rec.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        if self._recs:
            first = next((i for i, o in enumerate(self._recs) if not o.get("over") and o["type"]), 0)
            self.tbl_rec.selectRow(first)
            self.show_rec()
        else:
            self.txt_rec.setHtml(REPORT_CSS + "<p>No standard design exists for this combination.</p>")
        self._sync_make()

    def _rec(self):
        rows = self.tbl_rec.selectionModel().selectedRows()
        return self._recs[rows[0].row()] if rows and rows[0].row() < len(self._recs) else None

    def show_rec(self):
        o = self._rec()
        if o is None:
            return
        k = self.sp_k.value()
        h = [f"<h2>{o['title']}</h2>",
             f"<p><b>{o['runs']} runs</b>" + (f", {o['levels']}" if o.get("levels") else "") + ".</p>"]
        if o.get("why"):
            h.append(f"<p><b>Advantages:</b> {o['why']}</p>")
        if o.get("cons"):
            h.append(f"<p><b>Disadvantages:</b> {o['cons']}</p>")
        if o.get("note"):
            h.append(f"<p class='note'>{o['note']}</p>")
        if np.isfinite(o.get("power", np.nan)):
            h.append(f"<p>Lowest power among the {ORDER_LABEL.get(o['order'], '')} model coefficients: "
                     f"<b>{100 * o['power']:.0f}%</b> to detect an effect of 2 sigma (low-to-high response difference "
                     f"= 2 times the noise standard deviation). The usual target is 80%. Residual degrees of freedom: "
                     f"{o['df_resid']}.</p>")
        if o.get("order") and o["type"]:
            rr = planning.rsm_runs(k, o["order"])
            h.append(f"<p class='note'>A {ORDER_LABEL[o['order']]} model with {k} factors has {rr['p']} "
                     "coefficients.</p>")
        tag, txt = planning.ann_adequacy_text(k, o["runs"])
        h.append(f"<p class='note'>For ANN: {txt}</p>")
        self.txt_rec.setHtml(REPORT_CSS + "".join(h))

    # ================================================================ factorial table
    def _build_table(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        top = QHBoxLayout()
        self.sp_rep = QSpinBox()
        self.sp_rep.setRange(1, 10)
        self.sp_cp = QSpinBox()
        self.sp_cp.setRange(0, 50)
        self.sp_cp.setValue(0)
        self.chk_gen = QCheckBox("Show Generators")
        for lab, wd in (("Replicates:", self.sp_rep), ("Center points:", self.sp_cp)):
            top.addWidget(QLabel(lab))
            top.addWidget(wd)
        top.addWidget(self.chk_gen)
        top.addSpacing(20)
        legend = QLabel(" ".join(f"<span style='background:{c}; border:1px solid #999;'>&nbsp;&nbsp;&nbsp;&nbsp;</span>"
                                 f" {t}&nbsp;&nbsp;" for c, t in (
                                     (RES_COLORS[5], "Res V and higher: clean main effects & 2-factor interactions "
                                                     "(characterization)"),
                                     (RES_COLORS[4], "Res IV: clean main effects (screening)"),
                                     (RES_COLORS[3], "Res III: main effects aliased with 2-factor interactions "
                                                     "(ruggedness testing)"))))
        legend.setWordWrap(True)
        top.addWidget(legend, 1)
        lay.addLayout(top)
        cap = QLabel("Number of factors (columns) and number of runs (rows). Click a cell to see its generators and "
                     "aliases; double-click to create the design.")
        cap.setStyleSheet("color: #555;")
        lay.addWidget(cap)
        self.cells = designs.factorial_table()
        ks = list(range(2, 22))
        runs = list(designs.FACTORIAL_RUNS)
        self.grid = QTableWidget(len(runs), len(ks))
        self.grid.setHorizontalHeaderLabels([str(k) for k in ks])
        self.grid.setVerticalHeaderLabels([str(r) for r in runs])
        self.grid.setEditTriggers(QTableWidget.NoEditTriggers)
        self.grid.setSelectionMode(QTableWidget.SingleSelection)
        self.grid.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.grid.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        bold = QFont()
        bold.setBold(True)
        for r, n in enumerate(runs):
            for c, k in enumerate(ks):
                p = self.cells.get((n, k))
                it = QTableWidgetItem()
                it.setTextAlignment(Qt.AlignCenter)
                if p is None:
                    it.setBackground(QBrush(QColor(RES_COLORS[None])))
                    it.setFlags(Qt.NoItemFlags)
                else:
                    res = designs.resolution(k, p)
                    it.setText(f"{power_text(k, 0)}\nfull" if p == 0 else f"{power_text(k, p)}\n{designs.roman(res)}")
                    it.setBackground(QBrush(QColor(RES_COLORS[_res_key(res)])))
                    it.setForeground(QBrush(QColor("#1f1f1f")))
                    if p == 0:
                        it.setFont(bold)
                    it.setToolTip(self._cell_tip(k, p))
                self.grid.setItem(r, c, it)
        lay.addWidget(self.grid, 3)
        self.txt_cell = ReportBrowser()
        lay.addWidget(self.txt_cell, 2)
        self.grid.currentCellChanged.connect(lambda r, c, *_: self.select_cell(r, c))
        self.grid.cellDoubleClicked.connect(lambda *_: self.make())
        for wd in (self.sp_rep, self.sp_cp):
            wd.valueChanged.connect(lambda *_: self.select_cell(self.grid.currentRow(), self.grid.currentColumn()))
        self.chk_gen.toggled.connect(self._toggle_gen)
        return w

    def _cell_tip(self, k, p):
        if p == 0:
            return f"Full factorial, {k} factors, {2 ** k} runs"
        return f"{2 ** (k - p)} runs, resolution {designs.roman(designs.resolution(k, p))}\n" + \
            "\n".join(designs.generator_text(k, p))

    def _toggle_gen(self, on):
        for r, n in enumerate(designs.FACTORIAL_RUNS):
            for c, k in enumerate(range(2, 22)):
                p = self.cells.get((n, k))
                if p is None or p == 0:
                    continue
                it = self.grid.item(r, c)
                base = f"{power_text(k, p)}\n{designs.roman(designs.resolution(k, p))}"
                it.setText(base + ("\n" + ", ".join(g.replace(" ", "") for g in designs.generator_text(k, p))
                                   if on and p <= 3 else ""))
        self.grid.resizeRowsToContents() if on else None
        self.select_cell(self.grid.currentRow(), self.grid.currentColumn())

    def _cell(self):
        r, c = self.grid.currentRow(), self.grid.currentColumn()
        if r < 0 or c < 0:
            return None
        n, k = designs.FACTORIAL_RUNS[r], c + 2
        p = self.cells.get((n, k))
        return None if p is None else (k, p)

    def select_cell(self, r, c):
        if r is None or r < 0 or c < 0:
            return
        if self.grid.currentRow() != r or self.grid.currentColumn() != c:
            self.grid.setCurrentCell(r, c)
            return
        cell = self._cell()
        if cell is None:
            self.txt_cell.setHtml(REPORT_CSS + "<p>No two-level design exists for this combination.</p>")
            self._sync_make()
            return
        k, p = cell
        n = 2 ** (k - p) * self.sp_rep.value() + self.sp_cp.value()
        h = []
        if p == 0:
            h.append(f"<h2>Full factorial {power_text(k, 0)}</h2><p><b>{n} runs</b>. All main effects and all "
                     "interactions can be estimated without aliasing.</p>")
        else:
            res = designs.resolution(k, p)
            meaning = {3: "main effects are aliased with two-factor interactions",
                       4: "main effects are clean, two-factor interactions are aliased with each other",
                       5: "main effects and two-factor interactions are clean (aliased with three-factor and higher "
                          "interactions)"}
            h.append(f"<h2>Fractional factorial {power_text(k, p)}, resolution {designs.roman(res)}</h2>"
                     f"<p><b>{n} runs</b> ({2 ** (k - p)} factorial points"
                     + (f" x {self.sp_rep.value()} replicates" if self.sp_rep.value() > 1 else "")
                     + (f" + {self.sp_cp.value()} center points" if self.sp_cp.value() else "")
                     + f"). Resolution {designs.roman(res)}: {meaning.get(min(res, 5))}.</p>")
            h.append("<p><b>Generators:</b> " + ", ".join(designs.generator_text(k, p)) + "</p>")
            rel = designs.defining_relation(k, p)
            wlp = designs.word_length_pattern(k, p)
            h.append(f"<p><b>Defining relation</b> ({len(rel)} words): I = " + " = ".join(rel[:12])
                     + (" = ..." if len(rel) > 12 else "") + "</p>")
            h.append("<p class='note'>Word length pattern: "
                     + ", ".join(f"{i + 1} letters: {c}" for i, c in enumerate(wlp) if c)
                     + ". The fewer short words, the lower the aberration (the better the design).</p>")
            al = designs.alias_summary(k, p, 3)
            if al:
                h.append("<h3>Aliases of main effects and 2-factor interactions (up to 3 letters)</h3><table><tr>"
                         "<th class='l'>Effect</th><th class='l'>Aliased with</th></tr>")
                for e, lst in al[:60]:
                    h.append(f"<tr><td class='l'>{e}</td><td class='l'>{', '.join(lst[:10])}"
                             + (" ..." if len(lst) > 10 else "") + "</td></tr>")
                h.append("</table>" + (f"<p class='note'>{len(al) - 60} more effects not shown.</p>"
                                       if len(al) > 60 else ""))
            else:
                h.append("<p>Main effects and two-factor interactions are not aliased with any effect of up to "
                         "3 letters.</p>")
        tag, txt = planning.ann_adequacy_text(k, n)
        h.append(f"<p class='note'>Two-level designs have only 2 levels (no curvature), so they are not suitable for "
                 f"ANN or quadratic models; use them for screening/characterization. {txt}</p>")
        self.txt_cell.setHtml(REPORT_CSS + "".join(h))
        self._sync_make()

    # ================================================================ calculator
    def _build_calc(self, k):
        w = QWidget()
        lay = QHBoxLayout(w)
        left = QVBoxLayout()
        g = QGroupBox("RSM")
        f = QFormLayout(g)
        self.sp_ck = QSpinBox()
        self.sp_ck.setRange(1, 21)
        self.sp_ck.setValue(max(1, min(21, k)))
        self.cb_order = QComboBox()
        for key in ("linear", "2fi", "quadratic", "cubic"):
            self.cb_order.addItem(ORDER_LABEL[key], key)
        self.cb_order.setCurrentIndex(2)
        self.sp_lof = QSpinBox()
        self.sp_lof.setRange(0, 50)
        self.sp_lof.setValue(5)
        self.sp_nrep = QSpinBox()
        self.sp_nrep.setRange(0, 50)
        self.sp_nrep.setValue(5)
        f.addRow("Numeric factors:", self.sp_ck)
        f.addRow("Model:", self.cb_order)
        f.addRow("Lack of fit points:", self.sp_lof)
        f.addRow("Replicate points:", self.sp_nrep)
        left.addWidget(g)
        g2 = QGroupBox("ANN")
        f2 = QFormLayout(g2)
        self.ed_hidden = QLineEdit("")
        self.ed_hidden.setPlaceholderText("automatic (2 x number of factors)")
        self.sp_out = QSpinBox()
        self.sp_out.setRange(1, 20)
        self.cb_calg = QComboBox()
        self.cb_calg.addItem("Bayesian Regularization (trainbr)", "trainbr")
        self.cb_calg.addItem("Levenberg-Marquardt / Adam", "trainlm")
        self.cb_scheme = QComboBox()
        self.cb_scheme.addItem("k-fold cross-validation (data < 400)", "kfold")
        self.cb_scheme.addItem("Hold-out 70/15/15 (large data)", "holdout")
        self.sp_have = QSpinBox()
        self.sp_have.setRange(0, 100000)
        self.sp_have.setSpecialValueText("-")
        f2.addRow("Neurons per hidden layer:", self.ed_hidden)
        f2.addRow("Number of outputs:", self.sp_out)
        f2.addRow("Algorithm:", self.cb_calg)
        f2.addRow("Validation:", self.cb_scheme)
        f2.addRow("Data you have:", self.sp_have)
        left.addWidget(g2)
        left.addStretch()
        lay.addLayout(left, 1)
        self.txt_calc = ReportBrowser()
        lay.addWidget(self.txt_calc, 2)
        for wd in (self.sp_ck, self.sp_lof, self.sp_nrep, self.sp_out, self.sp_have):
            wd.valueChanged.connect(self.update_calc)
        for wd in (self.cb_order, self.cb_calg, self.cb_scheme):
            wd.currentIndexChanged.connect(self.update_calc)
        self.ed_hidden.textChanged.connect(self.update_calc)
        return w

    def update_calc(self):
        k = self.sp_ck.value()
        order = self.cb_order.currentData()
        rr = planning.rsm_runs(k, order, self.sp_lof.value(), self.sp_nrep.value())
        h = [f"<h2>RSM: {ORDER_LABEL[order]} model, {k} factors</h2>",
             f"<table><tr><td class='l'>Number of coefficients (including intercept)</td><td><b>{rr['p']}</b></td></tr>"
             f"<tr><td class='l'>Absolute minimum (coefficients + 1 to estimate error)</td><td>{rr['n_abs']}</td></tr>"
             f"<tr><td class='l'>Recommended (coefficients + lack of fit + replicates)</td><td><b>{rr['n_min']}</b>"
             f"</td></tr></table>"]
        std = []
        builds = []
        if order in ("linear", "2fi") and k <= 9:
            builds.append((f"Full factorial 2^{k} + 4 center", "full_factorial", {"center_points": 4}))
            r5 = designs.min_runs_for_resolution(k, 5 if order == "2fi" else 4)
            if r5 and r5[1]:
                builds.append((f"Fractional 2^({k}-{r5[1]}) + 4 center", "fractional_factorial",
                               {"fraction_p": r5[1], "center_points": 4}))
        if order in ("quadratic", "cubic") or order == "2fi":
            if 2 <= k <= 6:
                builds.append(("CCD rotatable", "ccd", {"alpha_type": "rotatable",
                                                         "center_points": designs.default_center_points("ccd", k)}))
                builds.append(("CCD face-centered", "ccd", {"alpha_type": "face",
                                                             "center_points": designs.default_center_points("ccd", k)}))
            if 3 <= k <= 7:
                builds.append(("Box-Behnken", "bbd", {"center_points": designs.default_center_points("bbd", k)}))
        for name, t, opts in builds:
            try:
                coded, _, _ = designs.build_design(t, k, **opts)
            except (ValueError, KeyError):
                continue
            pw, df = planning._power(coded, order)
            std.append((name, len(coded), pw, df))
        if std:
            h.append(f"<h3>Standard designs for {k} factors</h3><table><tr><th class='l'>Design</th><th>Runs</th>"
                     "<th>Residual df</th><th>Lowest power (2σ signal)</th></tr>")
            for name, n, pw, df in std:
                pws = "not enough df" if not np.isfinite(pw) else f"{100 * pw:.0f}%"
                h.append(f"<tr><td class='l'>{name}</td><td>{n}</td><td>{df}</td><td>{pws}</td></tr>")
            h.append("</table><p class='note'>Power = probability of detecting a model coefficient of 2 sigma. "
                     "Below 80%: add replicates/center points, or accept that small effects may go undetected.</p>")
        # ANN
        txt = self.ed_hidden.text().strip()
        try:
            hidden = [int(float(t)) for t in txt.replace(";", ",").split(",") if t.strip()] if txt else [2 * k]
            if not hidden or min(hidden) < 1:
                raise ValueError
        except ValueError:
            hidden = [2 * k]
        ar = planning.ann_runs(k, hidden, self.sp_out.value(), self.cb_calg.currentData(), self.cb_scheme.currentData())
        lo, (r1, r2), hi = planning.ann_table(k)
        h.append(f"<h2>ANN: {k} input, hidden {'/'.join(map(str, hidden))}, {self.sp_out.value()} output</h2>"
                 f"<table><tr><td class='l'>Number of weights & biases</td><td><b>{ar['weights']}</b></td></tr>"
                 f"<tr><td class='l'>Minimum training data ({ar['ratio']:g} x weights)</td>"
                 f"<td>{int(np.ceil(ar['ratio'] * ar['weights']))}</td></tr>"
                 f"<tr><td class='l'>Total runs so the training share ({100 * ar['train_frac']:.0f}%) is enough</td>"
                 f"<td><b>{ar['n_need']}</b></td></tr></table>"
                 "<p class='note'>trainbr uses a ratio of 1 because Bayesian regularization reduces the effective "
                 "parameters; trainlm and Adam use a ratio of 2.</p>")
        h.append(f"<h3>Rules of thumb for {k} factors</h3><table><tr><th class='l'></th><th>Runs</th></tr>"
                 f"<tr><td class='l'>Minimum (small ANN, trainbr, cross-validation)</td><td>{lo}</td></tr>"
                 f"<tr><td class='l'>Recommended</td><td>{r1} to {r2}</td></tr>"
                 f"<tr><td class='l'>Highly nonlinear pattern</td><td>{hi}+</td></tr></table>"
                 "<p class='note'>The spread of points matters more than their number: use at least 5 levels per "
                 "factor (Latin Hypercube or hybrid). Once the data is collected, the ANN &gt; Learning Curve page "
                 "shows objectively whether more data would still help.</p>")
        if self.sp_have.value():
            tag, t = planning.ann_adequacy_text(k, self.sp_have.value())
            rsm_ok = self.sp_have.value() >= rr["n_min"]
            h.append(f"<h3>Your data: {self.sp_have.value()} runs</h3><ul><li>RSM {ORDER_LABEL[order]}: "
                     + ("enough." if rsm_ok else f"not enough, needs about {rr['n_min']} runs.") + f"</li><li>ANN: {t}"
                     + (f" This architecture needs about {ar['n_need']} runs." if self.sp_have.value() < ar["n_need"]
                        else "") + "</li></ul>")
        self.txt_calc.setHtml(REPORT_CSS + "".join(h))

    # ================================================================ choice
    def _sync_make(self):
        i = self.tabs.currentIndex()
        if i == 0:
            o = self._rec() if hasattr(self, "tbl_rec") else None
            ok = o is not None and o["type"] is not None
        elif i == 1:
            ok = hasattr(self, "grid") and self._cell() is not None
        else:
            ok = False
        self.btn_make.setEnabled(ok)

    def make(self):
        i = self.tabs.currentIndex()
        if i == 0:
            o = self._rec()
            if o is None or not o["type"]:
                return
            self.choice = (o["type"], self.sp_k.value(), dict(o["options"]))
        elif i == 1:
            cell = self._cell()
            if cell is None:
                return
            k, p = cell
            opts = {"center_points": self.sp_cp.value(), "replicates": self.sp_rep.value()}
            if p:
                opts["fraction_p"] = p
            self.choice = ("fractional_factorial" if p else "full_factorial", k, opts)
        else:
            return
        self.accept()
