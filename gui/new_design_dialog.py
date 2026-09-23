"""Dialog for creating a new design (process, categorical, mixture, optimal, historical)."""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout)

from doe import constraints as cons_mod
from doe import designs, models, planning
from doe.optimal import CRITERIA
from doe.project import Factor, Project, Response

from .common import parse_float

OPTIMAL = ("optimal_rsm", "optimal_mixture", "combined_optimal")
SPLIT_OK = ("full_factorial", "general_factorial", "ccd", "bbd", "optimal_rsm")
RESP_KINDS = [("normal", "Normal (continuous)"), ("binomial", "Binomial (successes out of n)"),
              ("poisson", "Poisson (counts)")]


class NewDesignDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Design")
        self.resize(860, 760)
        self.project = None

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        root.addLayout(top)

        g_type = QGroupBox("Design Type")
        f1 = QFormLayout(g_type)
        self.cb_type = QComboBox()
        for key, label in designs.DESIGN_TYPES.items():
            if key != "historical_mixture":   # created via File > Import Data
                self.cb_type.addItem(label, key)
        self.cb_type.setCurrentIndex(self.cb_type.findData("ccd"))
        self.sp_k = QSpinBox()
        self.sp_cat = QSpinBox()
        self.sp_cat.setRange(0, 4)
        self.sp_resp = QSpinBox()
        self.sp_resp.setRange(1, 20)
        self.sp_resp.setValue(1)
        self.lbl_k = QLabel("Numeric factors:")
        f1.addRow("Type:", self.cb_type)
        f1.addRow(self.lbl_k, self.sp_k)
        f1.addRow("Categorical factors:", self.sp_cat)
        f1.addRow("Responses:", self.sp_resp)
        self.f1 = f1
        top.addWidget(g_type, 3)

        g_opt = QGroupBox("Options")
        self.f_opt = QFormLayout(g_opt)
        self.sp_center = QSpinBox()
        self.sp_center.setRange(0, 50)
        self.sp_rep = QSpinBox()
        self.sp_rep.setRange(1, 10)
        self.cb_blocks = QComboBox()
        self.cb_fraction = QComboBox()
        self.cb_alpha = QComboBox()
        for key, label in designs.CCD_ALPHA_TYPES.items():
            self.cb_alpha.addItem(label, key)
        self.sp_alpha = QDoubleSpinBox()
        self.sp_alpha.setRange(0.1, 5)
        self.sp_alpha.setDecimals(4)
        self.sp_alpha.setValue(1.5)
        self.sp_degree = QSpinBox()
        self.sp_degree.setRange(1, 4)
        self.sp_degree.setValue(2)
        self.chk_augment = QCheckBox("Add axial points + centroid")
        self.chk_augment.setChecked(True)
        self.chk_reps = QCheckBox("Replicate vertices + centroid")
        self.chk_reps.setChecked(True)
        self.sp_total = QDoubleSpinBox()
        self.sp_total.setRange(0.0001, 1e9)
        self.sp_total.setDecimals(4)
        self.sp_total.setValue(100)
        self.cb_crit = QComboBox()
        for key, label in CRITERIA.items():
            self.cb_crit.addItem(label, key)
        self.cb_model = QComboBox()
        self.sp_extra = QSpinBox()
        self.sp_extra.setRange(0, 50)
        self.sp_lof = QSpinBox()
        self.sp_lof.setRange(0, 30)
        self.sp_lof.setValue(5)
        self.sp_nrep = QSpinBox()
        self.sp_nrep.setRange(0, 30)
        self.sp_nrep.setValue(5)
        self.sp_runs = QSpinBox()
        self.sp_runs.setRange(1, 5000)
        self.sp_runs.setValue(20)
        self.cb_pb = QComboBox()
        for r in (12, 20, 24):
            self.cb_pb.addItem(f"{r} runs", r)
        self.cb_taguchi = QComboBox()
        for key, (label, _, lv) in designs.TAGUCHI.items():
            self.cb_taguchi.addItem(label, key)
        self.cb_taguchi.setCurrentIndex(1)
        self.sp_proc = QSpinBox()
        self.sp_proc.setRange(1, 4)
        self.sp_proc.setValue(1)
        self.chk_split = QCheckBox("Split-plot (hard-to-change factors)")
        self.txt_htc = QLineEdit()
        self.txt_htc.setPlaceholderText("factor letters, e.g. A or A, C")
        self.sp_wp = QSpinBox()
        self.sp_wp.setRange(0, 100)
        self.sp_wp.setSpecialValueText("automatic")
        self.sp_lhs = QSpinBox()
        self.sp_lhs.setRange(4, 5000)
        self.sp_lhs.setValue(30)
        self.sp_lhs.setToolTip("Number of Latin Hypercube points. Each factor has this many levels.")
        self.cb_hybrid = QComboBox()
        for key, label in designs.HYBRID_BASES.items():
            self.cb_hybrid.addItem(label, key)
        self.sp_extra_pts = QSpinBox()
        self.sp_extra_pts.setRange(0, 5000)
        self.sp_extra_pts.setValue(12)
        self.sp_extra_pts.setToolTip("Additional points that fill the gaps of the base design (maximin), for ANN.")
        self._seed = int(np.random.default_rng().integers(1, 1_000_000))
        rows = [("Center points:", self.sp_center), ("Replicates:", self.sp_rep), ("Blocks:", self.cb_blocks),
                ("Fraction:", self.cb_fraction), ("Alpha value:", self.cb_alpha), ("Custom alpha:", self.sp_alpha),
                ("Lattice degree (m):", self.sp_degree), ("Mixture total:", self.sp_total),
                ("", self.chk_augment), ("", self.chk_reps), ("Criterion:", self.cb_crit),
                ("Planned model:", self.cb_model), ("Additional model points:", self.sp_extra),
                ("Lack of fit points:", self.sp_lof), ("Replicate points:", self.sp_nrep),
                ("Data rows:", self.sp_runs), ("Plackett-Burman size:", self.cb_pb),
                ("Orthogonal array:", self.cb_taguchi), ("Process factors:", self.sp_proc),
                ("", self.chk_split), ("Hard-to-change factors:", self.txt_htc), ("Runs per whole plot:", self.sp_wp),
                ("Latin Hypercube points:", self.sp_lhs), ("Base design:", self.cb_hybrid),
                ("Space-filling points:", self.sp_extra_pts)]
        for lab, w in rows:
            self.f_opt.addRow(lab, w)
        top.addWidget(g_opt, 3)

        self.g_fac = QGroupBox()
        lf = QVBoxLayout(self.g_fac)
        self.tbl_fac = QTableWidget(0, 4)
        self.tbl_fac.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lf.addWidget(self.tbl_fac)
        root.addWidget(self.g_fac, 3)

        self.g_proc = QGroupBox("Process Factors (Low = -1, High = +1)")
        lp = QVBoxLayout(self.g_proc)
        self.tbl_proc = QTableWidget(0, 4)
        self.tbl_proc.setHorizontalHeaderLabels(["Name", "Unit", "Low (-1)", "High (+1)"])
        self.tbl_proc.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lp.addWidget(self.tbl_proc)
        root.addWidget(self.g_proc, 1)

        self.g_cat = QGroupBox("Categorical Factors (separate level names with semicolons)")
        lc = QVBoxLayout(self.g_cat)
        self.tbl_cat = QTableWidget(0, 2)
        self.tbl_cat.setHorizontalHeaderLabels(["Name", "Levels"])
        self.tbl_cat.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lc.addWidget(self.tbl_cat)
        root.addWidget(self.g_cat, 1)

        self.g_cons = QGroupBox("Linear Constraints (one per line, actual units, letters = factors/components)")
        lk = QVBoxLayout(self.g_cons)
        self.txt_cons = QPlainTextEdit()
        self.txt_cons.setPlaceholderText("example:\nA + B <= 60\n2A - C >= 0")
        self.txt_cons.setMaximumHeight(70)
        lk.addWidget(self.txt_cons)
        root.addWidget(self.g_cons, 1)

        g_resp = QGroupBox("Responses")
        lr = QVBoxLayout(g_resp)
        self.tbl_resp = QTableWidget(0, 4)
        self.tbl_resp.setHorizontalHeaderLabels(["Name", "Unit", "Response type", "Trials per run (binomial)"])
        self.tbl_resp.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lr.addWidget(self.tbl_resp)
        root.addWidget(g_resp, 1)

        self.lbl_info = QLabel()
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("font-weight: bold; color: #1f5f99; padding: 4px;")
        root.addWidget(self.lbl_info)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_guide = buttons.addButton("Design Guide...", QDialogButtonBox.HelpRole)
        self.btn_guide.setToolTip("Choose a design and number of runs by goal (screening, RSM optimization, ANN, ...)")
        self.btn_guide.clicked.connect(self.open_guide)
        buttons.button(QDialogButtonBox.Ok).setText("Create Design")
        buttons.button(QDialogButtonBox.Cancel).setText("Cancel")
        buttons.accepted.connect(self.accept_design)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.cb_type.currentIndexChanged.connect(self.on_type_changed)
        self.sp_k.valueChanged.connect(self.on_k_changed)
        self.sp_cat.valueChanged.connect(self.sync_cat_rows)
        self.sp_resp.valueChanged.connect(self.sync_resp_rows)
        for w in (self.sp_center, self.sp_rep, self.sp_alpha, self.sp_degree, self.sp_total, self.sp_extra,
                  self.sp_lof, self.sp_nrep, self.sp_runs, self.sp_lhs, self.sp_extra_pts):
            w.valueChanged.connect(self.update_info)
        for w in (self.cb_fraction, self.cb_alpha, self.cb_blocks, self.cb_model, self.cb_crit, self.cb_pb,
                  self.cb_taguchi, self.cb_hybrid):
            w.currentIndexChanged.connect(self.update_info)
        self.sp_proc.valueChanged.connect(self.sync_proc_rows)
        self.chk_split.toggled.connect(self.sync_split)
        self.sp_wp.valueChanged.connect(self.update_info)
        for w in (self.chk_augment, self.chk_reps):
            w.toggled.connect(self.update_info)
        self.tbl_fac.itemChanged.connect(self.update_info)
        self.tbl_cat.itemChanged.connect(self.update_info)

        self.sync_resp_rows()
        self.on_type_changed()

    # ---------------------------------------------------------------
    def open_guide(self):
        from .design_guide import DesignGuideDialog
        dlg = DesignGuideDialog(self, self.sp_k.value())
        if dlg.exec() and dlg.choice:
            self.preset(*dlg.choice)

    def preset(self, dtype, k, options):
        """Fill the dialog from a Design Guide choice."""
        i = self.cb_type.findData(dtype)
        if i < 0:
            return
        self.cb_type.setCurrentIndex(i)
        self.on_type_changed()
        self.sp_k.setValue(int(k))
        o = dict(options)
        if "fraction_p" in o:
            j = self.cb_fraction.findData(int(o["fraction_p"]))
            if j >= 0:
                self.cb_fraction.setCurrentIndex(j)
        if "alpha_type" in o:
            self.cb_alpha.setCurrentIndex(max(self.cb_alpha.findData(o["alpha_type"]), 0))
        if "model_order" in o:
            self.cb_model.setCurrentIndex(max(self.cb_model.findData(o["model_order"]), 0))
        if "pb_runs" in o:
            self.cb_pb.setCurrentIndex(max(self.cb_pb.findData(int(o["pb_runs"])), 0))
        if "hybrid_base" in o:
            self.cb_hybrid.setCurrentIndex(max(self.cb_hybrid.findData(o["hybrid_base"]), 0))
        for key, w in (("lhs_runs", self.sp_lhs), ("extra_points", self.sp_extra_pts),
                       ("center_points", self.sp_center), ("replicates", self.sp_rep)):
            if key in o:
                w.setValue(int(o[key]))
        self.update_info()

    def dtype(self):
        return self.cb_type.currentData()

    def is_mix(self):
        return designs.is_mixture(self.dtype())

    def _show(self, widget, visible):
        self.f_opt.setRowVisible(widget, visible)

    def on_type_changed(self):
        t = self.dtype()
        mix = self.is_mix()
        lo, hi = designs.FACTOR_LIMITS[t]
        general = t == "general_factorial"
        self.sp_k.blockSignals(True)
        self.sp_k.setRange(lo, hi)
        if self.tbl_fac.rowCount() == 0 and not general:
            self.sp_k.setValue(max(lo, min(3, hi)))
        self.sp_k.blockSignals(False)
        self.f1.setRowVisible(self.sp_k, not general)
        cat_ok = t in designs.CATEGORIC_OK
        self.f1.setRowVisible(self.sp_cat, cat_ok)
        self.sp_cat.blockSignals(True)
        self.sp_cat.setRange(1 if general else 0, 4 if cat_ok else 0)
        if general and getattr(self, "_prev_type", None) != "general_factorial":
            self.sp_cat.setValue(2)
        elif not general and getattr(self, "_prev_type", None) == "general_factorial":
            self.sp_cat.setValue(0)
        self._prev_type = t
        self.sp_cat.blockSignals(False)
        self.lbl_k.setText("Components:" if mix else "Numeric factors:")
        if t == "taguchi":
            self.lbl_k.setText("Factors:")
        self.g_fac.setTitle("Mixture Components (actual units)" if mix
                            else "Numeric Factors (Low = level -1, High = level +1)")
        self.g_fac.setVisible(not general)
        optimal = t in OPTIMAL
        self._show(self.sp_center, t in ("full_factorial", "fractional_factorial", "ccd", "bbd", "plackett_burman",
                                         "lhs", "hybrid"))
        self._show(self.sp_lhs, t == "lhs")
        self._show(self.cb_hybrid, t == "hybrid")
        self._show(self.sp_extra_pts, t == "hybrid")
        self._show(self.sp_rep, t in ("full_factorial", "fractional_factorial", "ccd", "bbd", "general_factorial"))
        self._show(self.cb_fraction, t == "fractional_factorial")
        self._show(self.cb_alpha, t == "ccd")
        self._show(self.sp_degree, t == "simplex_lattice")
        self._show(self.sp_total, mix)
        self._show(self.chk_augment, mix and not optimal)
        self._show(self.chk_reps, mix and not optimal)
        for w in (self.cb_crit, self.cb_model, self.sp_extra, self.sp_lof, self.sp_nrep):
            self._show(w, optimal)
        self._show(self.sp_runs, t == "historical")
        self._show(self.cb_pb, t == "plackett_burman")
        self._show(self.cb_taguchi, t == "taguchi")
        self._show(self.sp_proc, t == "combined_optimal")
        self._show(self.chk_split, t in SPLIT_OK)
        if t not in SPLIT_OK:
            self.chk_split.setChecked(False)
        self.g_proc.setVisible(t == "combined_optimal")
        self.sync_proc_rows()
        self.sync_split()
        self.g_cons.setVisible(optimal)
        self.cb_model.blockSignals(True)
        self.cb_model.clear()
        for o in models.orders_for_design(t):
            self.cb_model.addItem(models.order_label(o), o)
        self.cb_model.setCurrentIndex(max(self.cb_model.findData(models.default_order(t)), 0))
        self.cb_model.blockSignals(False)
        self.tbl_fac.clear()
        self.tbl_fac.setRowCount(0)
        self.sync_cat_rows()
        self.on_k_changed()

    def on_k_changed(self):
        t = self.dtype()
        k = self.sp_k.value()
        self.sp_center.blockSignals(True)
        self.sp_center.setValue(designs.default_center_points(t, k) if t not in ("lhs", "hybrid") else 4)
        self.sp_center.blockSignals(False)
        if t == "lhs":
            self.sp_lhs.blockSignals(True)
            self.sp_lhs.setValue(planning.ann_target(k))
            self.sp_lhs.blockSignals(False)
        if t == "fractional_factorial":
            self.cb_fraction.blockSignals(True)
            self.cb_fraction.clear()
            for p, res in designs.fraction_options(k):
                self.cb_fraction.addItem(f"2^({k}-{p}) = {2 ** (k - p)} runs, Resolution {designs.roman(res)}", p)
            self.cb_fraction.blockSignals(False)
        self.cb_blocks.blockSignals(True)
        cur = self.cb_blocks.currentData() or 1
        self.cb_blocks.clear()
        for b in designs.blocking_options(t, k):
            self.cb_blocks.addItem("No blocks" if b == 1 else f"{b} blocks", b)
        self.cb_blocks.setCurrentIndex(max(self.cb_blocks.findData(cur), 0))
        self.cb_blocks.blockSignals(False)
        self._show(self.cb_blocks, self.cb_blocks.count() > 1)
        self.sync_factor_rows()
        self.update_info()

    def sync_factor_rows(self):
        mix = self.is_mix()
        opt_mix = self.dtype() in ("optimal_mixture", "combined_optimal")
        k = 0 if self.dtype() == "general_factorial" else self.sp_k.value()
        self.tbl_fac.blockSignals(True)
        if mix:
            cols = ["Name", "Unit", "Lower limit"] + (["Upper limit"] if opt_mix else [])
        else:
            cols = ["Name", "Unit", "Low (-1)", "High (+1)"]
        self.tbl_fac.setColumnCount(len(cols))
        self.tbl_fac.setHorizontalHeaderLabels(cols)
        old = self.tbl_fac.rowCount()
        self.tbl_fac.setRowCount(k)
        for i in range(old, k):
            name = f"{'Component' if mix else 'Factor'} {designs.LETTERS[i]}"
            vals = ([name, "%", "0"] + (["100"] if opt_mix else [])) if mix else [name, "", "-1", "1"]
            for c, v in enumerate(vals):
                self.tbl_fac.setItem(i, c, QTableWidgetItem(v))
        self.tbl_fac.setVerticalHeaderLabels([designs.LETTERS[i] for i in range(k)])
        self.tbl_fac.blockSignals(False)
        self.sync_cat_rows()

    def sync_cat_rows(self):
        if self.dtype() not in designs.CATEGORIC_OK:
            n = 0
        else:
            n = self.sp_cat.value()
        self.g_cat.setVisible(n > 0)
        k0 = 0 if self.dtype() == "general_factorial" else self.sp_k.value()
        old = self.tbl_cat.rowCount()
        self.tbl_cat.blockSignals(True)
        self.tbl_cat.setRowCount(n)
        for i in range(old, n):
            self.tbl_cat.setItem(i, 0, QTableWidgetItem(f"Category {i + 1}"))
            self.tbl_cat.setItem(i, 1, QTableWidgetItem("Level 1; Level 2"))
        self.tbl_cat.setVerticalHeaderLabels([designs.LETTERS[k0 + i] for i in range(n)])
        self.tbl_cat.blockSignals(False)
        self.update_info()

    def sync_resp_rows(self):
        r = self.sp_resp.value()
        old = self.tbl_resp.rowCount()
        self.tbl_resp.setRowCount(r)
        for i in range(old, r):
            self.tbl_resp.setItem(i, 0, QTableWidgetItem(f"Response {i + 1}"))
            self.tbl_resp.setItem(i, 1, QTableWidgetItem(""))
            cb = QComboBox()
            for key, label in RESP_KINDS:
                cb.addItem(label, key)
            self.tbl_resp.setCellWidget(i, 2, cb)
            self.tbl_resp.setItem(i, 3, QTableWidgetItem(""))
        self.tbl_resp.setVerticalHeaderLabels([f"R{i + 1}" for i in range(r)])

    def options(self):
        t = self.dtype()
        opts = self._options(t)
        if self.chk_split.isChecked() and t in SPLIT_OK:
            htc = self._htc()
            if htc:
                opts["split_plot"] = {"htc": htc, "wp_size": self.sp_wp.value() or None}
        return opts

    def _htc(self):
        out = []
        for part in self.txt_htc.text().replace(";", ",").split(","):
            part = part.strip().upper()
            if part and part in designs.LETTERS:
                out.append(designs.LETTERS.index(part))
        return sorted(set(out))

    def _options(self, t):
        if t == "plackett_burman":
            return {"pb_runs": self.cb_pb.currentData(), "center_points": self.sp_center.value()}
        if t == "dsd":
            return {}
        if t == "lhs":
            return {"lhs_runs": self.sp_lhs.value(), "center_points": self.sp_center.value(), "design_seed": self._seed}
        if t == "hybrid":
            return {"hybrid_base": self.cb_hybrid.currentData(), "extra_points": self.sp_extra_pts.value(),
                    "center_points": self.sp_center.value(), "design_seed": self._seed}
        if t == "taguchi":
            return {"taguchi_array": self.cb_taguchi.currentData()}
        if t in OPTIMAL:
            return {"criterion": self.cb_crit.currentData(), "model_order": self.cb_model.currentData(),
                    "extra_model": self.sp_extra.value(), "n_lof": self.sp_lof.value(), "n_rep": self.sp_nrep.value()}
        if self.is_mix():
            opts = {"augment": self.chk_augment.isChecked(), "replicate_points": self.chk_reps.isChecked()}
            if t == "simplex_lattice":
                opts["lattice_degree"] = self.sp_degree.value()
            return opts
        if t in ("general_factorial", "historical"):
            return {"replicates": self.sp_rep.value()} if t == "general_factorial" else {}
        opts = {"center_points": self.sp_center.value(), "replicates": self.sp_rep.value(),
                "blocks": self.cb_blocks.currentData() or 1}
        if t == "fractional_factorial":
            opts["fraction_p"] = self.cb_fraction.currentData() or 1
        if t == "ccd":
            opts["alpha_type"] = self.cb_alpha.currentData()
            opts["alpha_custom"] = self.sp_alpha.value()
        return opts

    def _cell(self, tbl, r, c):
        it = tbl.item(r, c)
        return it.text().strip() if it else ""

    def _cat_levels(self):
        out = []
        for i in range(self.tbl_cat.rowCount() if self.dtype() in designs.CATEGORIC_OK else 0):
            levels = [s.strip() for s in self._cell(self.tbl_cat, i, 1).replace(",", ";").split(";") if s.strip()]
            out.append((self._cell(self.tbl_cat, i, 0) or f"Category {i + 1}", levels))
        return out

    def sync_proc_rows(self):
        n = self.sp_proc.value() if self.dtype() == "combined_optimal" else 0
        k0 = self.sp_k.value()
        old = self.tbl_proc.rowCount()
        self.tbl_proc.blockSignals(True)
        self.tbl_proc.setRowCount(n)
        for i in range(old, n):
            for c, v in enumerate([f"Process {i + 1}", "", "-1", "1"]):
                self.tbl_proc.setItem(i, c, QTableWidgetItem(v))
        self.tbl_proc.setVerticalHeaderLabels([designs.LETTERS[k0 + i] for i in range(n)])
        self.tbl_proc.blockSignals(False)
        self.update_info()

    def sync_split(self):
        on = self.chk_split.isChecked() and self.dtype() in SPLIT_OK
        self._show(self.txt_htc, on)
        self._show(self.sp_wp, on)
        if on and not self.txt_htc.text():
            self.txt_htc.setText("A")
        self.update_info()

    def update_info(self):
        t = self.dtype()
        self._show(self.sp_alpha, t == "ccd" and self.cb_alpha.currentData() == "custom")
        if t == "combined_optimal":
            k = self.sp_k.value()
            npr = self.sp_proc.value()
            order = self.cb_model.currentData() or "quadratic|linear"
            if "|" not in order:
                order = "quadratic|linear"
            p = len(models.combined_terms(list(range(k)), list(range(k, k + npr)), k + npr, order))
            n = p + self.sp_extra.value() + self.sp_lof.value() + self.sp_nrep.value()
            self.lbl_info.setText(f"Model {models.order_label(order)}: {p} coefficients → {n} runs in total")
            return
        if t == "taguchi":
            _, _, lv = designs.TAGUCHI[self.cb_taguchi.currentData()]
            k = self.sp_k.value()
            if k > len(lv):
                self.lbl_info.setText(f"This array allows at most {len(lv)} factors.")
                return
            levels = ", ".join(f"{designs.LETTERS[i]}={lv[i]} levels" for i in range(k))
            n = designs.TAGUCHI[self.cb_taguchi.currentData()][1]().shape[0]
            self.lbl_info.setText(f"Runs: {n}    |    {levels} (3 levels = low/middle/high)")
            return
        combos = 1
        for _, lv in self._cat_levels():
            combos *= max(len(lv), 1)
        if t in OPTIMAL:
            k = self.sp_k.value()
            order = self.cb_model.currentData() or "quadratic"
            if self.is_mix():
                p = len(models.mixture_terms(k, order))
            else:
                space = models.Space([0] * k + [len(lv) for _, lv in self._cat_levels()],
                                     [None] * (k + len(self._cat_levels())))
                terms = models.model_terms(space.k, order, space)
                p = models.expand([[0] * space.k], terms, space)[0].shape[1]
            n = p + self.sp_extra.value() + self.sp_lof.value() + self.sp_nrep.value()
            self.lbl_info.setText(f"Model {models.order_label(order)}: {p} coefficients → {n} runs in total "
                                  f"({p + self.sp_extra.value()} model + {self.sp_lof.value()} lack of fit + "
                                  f"{self.sp_nrep.value()} replicates)")
            return
        if t == "historical":
            self.lbl_info.setText(f"{self.sp_runs.value()} empty rows - enter the factor & response values "
                                  "from your data.")
            return
        if t == "general_factorial":
            self.lbl_info.setText(f"Runs: {combos * self.sp_rep.value()}")
            return
        if t in ("lhs", "hybrid"):
            k = self.sp_k.value()
            if t == "lhs":
                n = self.sp_lhs.value() + self.sp_center.value()
                desc = f"{self.sp_lhs.value()} levels per factor"
            else:
                base = self.cb_hybrid.currentData()
                if base == "bbd" and k < 3:
                    self.lbl_info.setText("Box-Behnken requires at least 3 factors.")
                    return
                sub = {"ccd_face": ("ccd", {"alpha_type": "face"}), "ccd": ("ccd", {}), "bbd": ("bbd", {}),
                       "full_factorial": ("full_factorial", {})}[base]
                nb = designs.build_design(sub[0], k, 0, 1, **sub[1])[0].shape[0]
                n = nb + self.sp_extra_pts.value() + self.sp_center.value()
                desc = f"{nb} base design points + {self.sp_extra_pts.value()} space-filling + " \
                       f"{self.sp_center.value()} center"
            n *= combos
            self.lbl_info.setText(f"Runs: {n} ({desc})    |    ANN: {planning.ann_adequacy_text(k, n)[1]}")
            return
        try:
            coded, _, blocks = designs.build_design(t, self.sp_k.value(), **self._options(t))
        except Exception as exc:  # noqa: BLE001
            self.lbl_info.setText(f"Invalid: {exc}")
            return
        info = f"Runs: {coded.shape[0] * combos}"
        if t == "fractional_factorial":
            p = self.cb_fraction.currentData() or 1
            gens = designs.generator_text(self.sp_k.value(), p)
            info += "    |    generator: " + ", ".join(gens[:6]) + (" ..." if len(gens) > 6 else "")
        if self.chk_split.isChecked() and t in SPLIT_OK:
            htc = ", ".join(designs.LETTERS[i] for i in self._htc()) or "-"
            info += f"    |    split-plot, hard-to-change factors: {htc}"
        if combos > 1:
            info += f" ({coded.shape[0]} × {combos} categorical level combinations)"
        nb = len(set(blocks.tolist()))
        if nb > 1:
            info += f"    |    {nb} blocks"
        if t == "ccd":
            info += f"    |    alpha = {abs(coded).max():.4f}"
        if self.is_mix():
            try:
                s = sum(parse_float(self._cell(self.tbl_fac, i, 2) or "0") for i in range(self.tbl_fac.rowCount()))
            except ValueError:
                s = None
            total = self.sp_total.value()
            if s is None:
                info += "    |    Lower limits must be numbers"
            elif s >= total:
                info += f"    |    Σ lower limits ({s:g}) must be < total ({total:g})"
            else:
                info += f"    |    Each component can rise up to its lower limit + {total - s:g}"
        self.lbl_info.setText(info)

    def accept_design(self):
        t = self.dtype()
        mix = self.is_mix()
        factors = []
        n_num = 0 if t == "general_factorial" else self.tbl_fac.rowCount()
        for i in range(n_num):
            name = self._cell(self.tbl_fac, i, 0) or f"Factor {designs.LETTERS[i]}"
            try:
                lo = parse_float(self._cell(self.tbl_fac, i, 2) or "0")
                if mix:
                    hi = lo
                    upper = parse_float(self._cell(self.tbl_fac, i, 3)) if t in ("optimal_mixture", "combined_optimal") else None
                else:
                    hi = parse_float(self._cell(self.tbl_fac, i, 3))
                    upper = None
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", f"The limits of {name} are not numbers.")
                return
            if not mix and not (lo < hi):
                QMessageBox.warning(self, "Invalid Input", f"Factor {name}: Low must be less than High.")
                return
            if mix and lo < 0:
                QMessageBox.warning(self, "Invalid Input", f"The lower limit of {name} cannot be negative.")
                return
            if upper is not None and upper <= lo:
                QMessageBox.warning(self, "Invalid Input",
                                    f"The upper limit of {name} must be greater than the lower limit.")
                return
            factors.append(Factor(name, self._cell(self.tbl_fac, i, 1), lo, hi, upper=upper))
        if t == "combined_optimal":
            for i in range(self.tbl_proc.rowCount()):
                name = self._cell(self.tbl_proc, i, 0) or f"Process {i + 1}"
                try:
                    lo = parse_float(self._cell(self.tbl_proc, i, 2))
                    hi = parse_float(self._cell(self.tbl_proc, i, 3))
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", f"The limits of {name} are not numbers.")
                    return
                if not lo < hi:
                    QMessageBox.warning(self, "Invalid Input", f"{name}: Low must be less than High.")
                    return
                factors.append(Factor(name, self._cell(self.tbl_proc, i, 1), lo, hi, kind="process"))
        if t in designs.CATEGORIC_OK:
            for name, levels in self._cat_levels():
                if len(levels) < 2 or len(set(levels)) != len(levels):
                    QMessageBox.warning(self, "Invalid Input",
                                        f"Categorical factor {name} needs at least 2 distinct levels.")
                    return
                factors.append(Factor(name, "", 0, 0, kind="categoric", levels=levels))
        if not factors:
            QMessageBox.warning(self, "Invalid Input", "There are no factors.")
            return
        extra = {}
        if mix:
            total = self.sp_total.value()
            if sum(f.low for f in factors if f.kind == "numeric") >= total:
                QMessageBox.warning(self, "Invalid Input",
                                    "The sum of the component lower limits must be less than the total.")
                return
            extra["mixture_total"] = total
        if t in OPTIMAL:
            try:
                extra["constraints"] = cons_mod.parse_many(self.txt_cons.toPlainText(), len(factors))
            except ValueError as exc:
                QMessageBox.warning(self, "Constraint", str(exc))
                return
        if t == "historical":
            extra["n_runs"] = self.sp_runs.value()
        responses = []
        for i in range(self.tbl_resp.rowCount()):
            kind = self.tbl_resp.cellWidget(i, 2).currentData()
            trials = None
            if kind == "binomial":
                try:
                    trials = parse_float(self._cell(self.tbl_resp, i, 3))
                except ValueError:
                    trials = float("nan")
                if not trials == trials or trials < 1:
                    QMessageBox.warning(self, "Invalid Input",
                                        f"Response {i + 1}: enter the number of trials per run (≥ 1).")
                    return
            responses.append(Response(self._cell(self.tbl_resp, i, 0) or f"Response {i + 1}",
                                      self._cell(self.tbl_resp, i, 1), kind=kind, trials=trials))
        if self.chk_split.isChecked() and t in SPLIT_OK and not self._htc():
            QMessageBox.warning(self, "Split-plot", "Enter the letters of the hard-to-change factors (e.g. A).")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.project = Project.create(t, factors, responses, **extra, **self.options())
        except (ValueError, np.linalg.LinAlgError) as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Design Could Not Be Created", str(exc))
            return
        QApplication.restoreOverrideCursor()
        self.project.dirty = True
        self.accept()
