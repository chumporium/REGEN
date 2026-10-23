"""Make the screenshots for the REGEN user manual.

Run from the project folder with the project's virtual environment:

    .venv\\Scripts\\python.exe docs\\manual\\make_screenshots.py

The main window is opened off screen (it never appears on the desktop), the case studies are loaded, the
computations are run with fixed seeds on one CPU core, and every page or dialog is saved as a PNG file in
docs/manual/figures/. The user's own settings (recent projects, window state) are not touched: a temporary
settings file is used instead. Message boxes are answered automatically and their texts are printed.

Options:
    --only name1,name2   make only the listed sections (see SECTIONS at the end of this file)
    --list               list the sections and exit
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys
import tempfile
import time
import traceback
import warnings

# canvases on hidden tabs have no size yet; matplotlib warns about it, which is harmless here
warnings.filterwarnings("ignore", message="constrained_layout not applied")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "figures")

WIN_W, WIN_H = 1500, 900          # main window size (logical pixels)
SEED = 12345                      # design seed in the New Design dialog


class Shooter:
    """Holds the application, the main window and helpers to wait, grab and save."""

    def __init__(self, app, window, tmp):
        self.app, self.w, self.tmp = app, window, tmp
        self.saved, self.failed, self.messages, self.notes = [], [], [], []

    # ------------------------------------------------------------------ waiting
    def pump(self, n=6):
        for _ in range(n):
            self.app.processEvents()
            time.sleep(0.01)

    def wait(self, tab, timeout=900):
        """Wait until a background computation (QThread) of a page has finished."""
        t0 = time.time()
        while tab.task is not None:
            self.app.processEvents()
            time.sleep(0.02)
            if time.time() - t0 > timeout:
                raise TimeoutError(f"{type(tab).__name__}: computation did not finish within {timeout} s")
        self.pump()
        return time.time() - t0

    # ------------------------------------------------------------------ saving
    def save(self, widget_or_pixmap, name, note=""):
        from PySide6.QtGui import QPixmap
        self.pump()
        if widget_or_pixmap is self.w:
            # a small resize forces every layout to be recomputed (word-wrapped labels on pages that were hidden
            # while their text changed can otherwise leave stretched gaps)
            self.w.resize(WIN_W, WIN_H + 1)
            self.pump(3)
            self.w.resize(WIN_W, WIN_H)
            self.pump()
        pm = widget_or_pixmap if isinstance(widget_or_pixmap, QPixmap) else widget_or_pixmap.grab()
        path = os.path.join(OUT, name)
        if pm.isNull() or not pm.save(path, "PNG"):
            raise RuntimeError(f"could not save {name}")
        self.saved.append(name)
        self.notes.append((name, note))
        print(f"  saved {name} ({pm.width()} x {pm.height()})", flush=True)

    def page(self, key, name, note=""):
        self.w.go(key)
        self.pump()
        self.save(self.w, name, note() if callable(note) else note)

    def dialog(self, dlg, size=None):
        from PySide6.QtCore import Qt
        dlg.setAttribute(Qt.WA_DontShowOnScreen)
        if size:
            dlg.resize(*size)      # otherwise: the dialog's own size (or its size hint)
        dlg.show()
        self.pump()
        return dlg

    def with_menu(self, trigger, base_widget, name, note=""):
        """Run trigger(), which opens a context menu (QMenu.exec). The menu is kept off screen, grabbed and
        closed, then drawn over a grab of base_widget at the position where it opened."""
        from PySide6.QtCore import QEvent, QObject, Qt, QTimer
        from PySide6.QtGui import QPainter
        from PySide6.QtWidgets import QMenu
        got = {}

        def capture(menu):
            got["pm"], got["pos"] = menu.grab(), menu.pos()
            menu.close()

        class Filter(QObject):
            def eventFilter(self, obj, event):  # noqa: N802
                if isinstance(obj, QMenu):
                    if event.type() == QEvent.Polish:          # before the native window exists
                        obj.setAttribute(Qt.WA_DontShowOnScreen)
                    elif event.type() == QEvent.Show and "pm" not in got:
                        QTimer.singleShot(100, lambda m=obj: capture(m))
                return False

        flt = Filter()
        self.app.installEventFilter(flt)
        try:
            trigger()
        finally:
            self.app.removeEventFilter(flt)
        if "pm" not in got:
            raise RuntimeError("no context menu was opened")
        self.pump()
        base = base_widget.grab()
        from PySide6.QtCore import QRect
        from PySide6.QtGui import QColor
        at = base_widget.mapFromGlobal(got["pos"])
        size = got["pm"].deviceIndependentSize().toSize()
        painter = QPainter(base)
        painter.fillRect(QRect(at.x() + 3, at.y() + 3, size.width(), size.height()), QColor(0, 0, 0, 40))  # shadow
        painter.drawPixmap(at, got["pm"])
        painter.setPen(QColor("#a0a0a0"))
        painter.drawRect(QRect(at.x(), at.y(), size.width() - 1, size.height() - 1))
        painter.end()
        self.save(base, name, note)


def text_of(widget, limit=700):
    """Plain text of a report browser (for the notes printed next to each screenshot)."""
    t = " ".join(widget.toPlainText().split())
    return t[:limit]


def combo_select(cb, data=None, text=None):
    i = cb.findData(data) if data is not None else cb.findText(text)
    if i < 0:
        raise ValueError(f"combo item not found: {data or text}")
    cb.setCurrentIndex(i)


# ====================================================================== sections
def sec_start(s):
    """Start page with a few recent projects, and the dialogs that open from it."""
    from doe.project import sample_ann_project, sample_mixture_project, sample_project
    w = s.w
    folder = os.path.join(s.tmp, "Projects")
    os.makedirs(folder, exist_ok=True)
    for fn, factory in (("dye_adsorption.doe", sample_ann_project), ("edible_film.doe", sample_mixture_project),
                        ("extraction_bbd.doe", sample_project)):
        path = os.path.join(folder, fn)
        factory().save(path)
        w.add_recent(path)
    w.go("home")
    s.save(w, "start_page.png", "Start page, no project open; three recent projects in the list.")


def sec_new_design(s):
    from PySide6.QtWidgets import QTableWidgetItem
    from gui.new_design_dialog import NewDesignDialog
    w = s.w

    def fill(tbl, rows):
        for r, vals in enumerate(rows):
            for c, v in enumerate(vals):
                tbl.setItem(r, c, QTableWidgetItem(v))

    # CCD, 3 factors, 2 responses
    d = NewDesignDialog(w)
    d.sp_seed.setValue(SEED)
    combo_select(d.cb_type, "ccd")
    d.sp_k.setValue(3)
    fill(d.tbl_fac, [["Temperature", "°C", "60", "80"], ["Time", "min", "30", "90"], ["Concentration", "%", "1", "3"]])
    d.sp_resp.setValue(2)
    fill(d.tbl_resp, [["Yield", "%"], ["Purity", "%"]])
    s.dialog(d)
    s.save(d, "new_design_ccd.png", d.lbl_info.text())
    d.close()

    # mixture: simplex lattice
    d = NewDesignDialog(w)
    d.sp_seed.setValue(SEED)
    combo_select(d.cb_type, "simplex_lattice")
    d.sp_k.setValue(3)
    fill(d.tbl_fac, [["Starch", "%", "50"], ["Glycerol", "%", "10"], ["Chitosan", "%", "10"]])
    d.sp_resp.setValue(2)
    fill(d.tbl_resp, [["Tensile Strength", "MPa"], ["Elongation", "%"]])
    d.update_info()
    s.dialog(d)
    s.save(d, "new_design_mixture.png", d.lbl_info.text())
    d.close()

    # optimal (custom) process design with a categorical factor and a linear constraint
    d = NewDesignDialog(w)
    d.sp_seed.setValue(SEED)
    combo_select(d.cb_type, "optimal_rsm")
    d.sp_k.setValue(3)
    fill(d.tbl_fac, [["Temperature", "°C", "60", "80"], ["Time", "min", "30", "90"], ["Concentration", "%", "1", "3"]])
    d.sp_cat.setValue(1)
    fill(d.tbl_cat, [["Solvent", "Ethanol; Methanol"]])
    d.txt_cons.setPlainText("A + B <= 160")
    fill(d.tbl_resp, [["Yield", "%"]])
    d.update_info()
    s.dialog(d, (860, 900))                    # taller, so that all three factor rows are visible
    s.save(d, "new_design_optimal.png", d.lbl_info.text())
    d.close()


def sec_guide(s):
    from gui.design_guide import DesignGuideDialog
    g = DesignGuideDialog(s.w, 3)
    s.dialog(g, (1180, 760))
    g.tabs.setCurrentIndex(0)
    s.save(g, "design_guide_recommend.png", "Goal: optimize, 3 numeric factors, no run budget. " + text_of(g.txt_rec, 400))
    g.tabs.setCurrentIndex(1)
    g.select_cell(2, 4)                       # 16 runs, 6 factors: 2^(6-2), resolution IV
    s.save(g, "design_guide_table.png", "Selected cell: 16 runs x 6 factors. " + text_of(g.txt_cell, 400))
    g.tabs.setCurrentIndex(2)
    g.sp_have.setValue(20)
    s.save(g, "design_guide_calculator.png", "3 factors, quadratic, 5 LOF + 5 replicates, data you have = 20. "
           + text_of(g.txt_calc, 600))
    g.close()


def sec_cases(s):
    from gui.cases_dialog import CasesDialog
    d = CasesDialog(s.w)
    s.dialog(d, (820, 470))
    d.list.setCurrentRow(0)
    s.save(d, "case_studies.png", "First case (Extraction Optimization, Box-Behnken) selected.")
    d.close()


def sec_import(s):
    from doe.project import sample_project
    from gui.import_dialog import ImportDialog
    p = sample_project()
    head = ["Std", "Run", "A: Temperature (°C)", "B: Time (min)", "C: Concentration (%)", "R1: Yield (%)",
            "R2: Purity (%)"]
    rows = [head]
    for i in range(p.n):
        rows.append([str(i + 1), str(p.run_order[i])] + [f"{v:g}" for v in p.actual[i]]
                    + [f"{v:g}" for v in p.data[i]])
    d = ImportDialog(s.w)
    d.load_rows(rows)
    s.dialog(d, (1080, 700))
    d.select_column(5)
    d.tbl.selectColumn(5)
    s.save(d, "import_dialog.png", "17-run Box-Behnken table pasted with 'Std', 'Run' and 'R1:' style headers; "
           f"column 'Yield' selected. Summary: {d.lbl_sum.text()}")
    d.close()


def sec_bbd(s):
    """Box-Behnken extraction case: data, readiness, evaluation, analysis, graphs, desirability, NSGA,
    sensitivity, prediction."""
    from doe.project import sample_project
    from gui.data_tab import InfoDialog, SimulateDialog
    w = s.w
    p = sample_project()
    p.notes = ("Goal: maximize extraction yield and purity.\n"
               "Material: dried leaves, batch 2026-03, particle size < 0.5 mm.\n"
               "Equipment: water bath with shaker, HPLC for purity.\n"
               "[12-03-2026 09:10] Runs 1-9 done. Run 6 restarted after a power cut.\n"
               "[13-03-2026 14:30] Runs 10-17 done.\n"
               "Analysis: quadratic model for both responses; no transformation needed (Box-Cox).")
    s.bbd = p
    w.set_project(p)
    s.page("data", "data_page.png", "Box-Behnken 3 factors, 17 runs (5 center points), responses Yield and Purity.")

    d = InfoDialog(p, w)
    for i, sd in enumerate(("1.0", "2.0", "0.05")):
        d.tf.item(i, 2).setText(sd)
    s.dialog(d, (640, 460))
    s.save(d, "factor_response_info.png", "Standard deviations typed in: Temperature 1.0 °C, Time 2.0 min, "
           "Concentration 0.05 % (not saved here; they are applied later for the POE graph).")
    d.reject()

    d = SimulateDialog(p, w)
    s.dialog(d)
    s.save(d, "simulate_dialog.png", "Default equation 50 + 3*A - 2*B - 4*A**2, coded factors, noise 1, seed 1.")
    d.reject()

    s.page("ready", "readiness.png", lambda: text_of(w.ready_tab.txt, 600))
    s.page("eval", "design_evaluation.png", lambda: text_of(w.eval_tab.txt, 600))
    s.page("notes", "notes.png", "Example notes typed into the Notes page.")

    g = w.gcols_tab
    w.go("gcols")
    g.chk_trend.setChecked(True)
    s.save(w, "column_graphs.png", f"Scatterplot Temperature vs Yield, colored by Run, trend line on. {g.lbl_corr.text()}")
    s.page("coef", "coefficient_table.png", lambda: text_of(w.coef_tab.txt, 600))

    a = w.analysis_tab
    w.go("analysis")
    a.cb_resp.setCurrentIndex(0)
    for i, (name, browser) in enumerate((("rsm_fit_summary.png", a.txt_summary), ("rsm_anova.png", a.txt_anova),
                                         ("rsm_equation.png", a.txt_equation))):
        a.tabs.setCurrentIndex(i)
        s.save(w, name, "Response Yield. " + text_of(browser, 900))
    a.tabs.setCurrentIndex(4)
    s.save(w, "rsm_boxcox.png", "Response Yield. " + a.lbl_bc.text())
    a.tabs.setCurrentWidget(a.diag)
    s.save(w, "rsm_diagnostics.png", "Response Yield, 4 default plots (normal plot, residuals vs predicted, "
           "predicted vs actual, Cook's distance).")
    a.tabs.setCurrentIndex(0)

    # model graphs (Yield)
    gt = w.graphs_tab
    w.go("graphs")
    gt.cb_resp.setCurrentIndex(0)
    for key, name, note in (
            ("contour", "graph_contour.png", "Yield, Temperature (X) x Time (Y), Concentration held at 2 %."),
            ("surface", "graph_surface3d.png", "Yield, Temperature x Time, Concentration 2 %."),
            ("oneway", "graph_oneway.png", "Yield vs Temperature, other factors at the center, 95% CI band."),
            ("allfactors", "graph_all_factors.png", "Yield, one panel per factor, others at the center."),
            ("interaction", "graph_interaction.png", "Yield, Temperature (X) with lines for Time (low/high)."),
            ("cube", "graph_cube.png", "Yield predicted at the 8 low/high corners of Temperature, Time and "
                                       "Concentration."),
            ("perturbation", "graph_perturbation.png", "Yield, deviation from the reference point (center).")):
        combo_select(gt.cb_type, key)
        s.save(w, name, note + " " + gt.note.text())
    # the plot toolbar alone, with the Copy data / Export data buttons
    combo_select(gt.cb_type, "contour")
    s.save(gt.canvas.toolbar, "plot_toolbar.png", "Matplotlib toolbar of a plot: Home, Back, Forward, Pan, Zoom, "
           "Subplots, Customize, Save, then Copy data and Export data...")
    s.with_menu(lambda: gt.canvas.canvas.customContextMenuRequested.emit(gt.canvas.canvas.rect().center()),
                w, "plot_context_menu.png", "Right-click on a plot: Copy plot data / Export plot data...")

    # desirability optimization: maximize both responses
    o = w.optim_tab
    w.go("optim")
    for n, (j, lo) in enumerate(((0, "75"), (1, "89"))):
        row = p.k + n
        combo_select(o.tbl.cellWidget(row, 1), "maximize")
        o.tbl.item(row, 2).setText(lo)
    o.run()
    s.pump()
    best = o.tbl_sol.item(0, o.tbl_sol.columnCount() - 1).text() if o.tbl_sol.rowCount() else "-"
    s.save(w, "desirability.png", f"Yield maximize (lower limit 75), Purity maximize (lower limit 89), importance 3, "
           f"shape 1. {o.tbl_sol.rowCount()} solutions; best desirability {best}: "
           + ", ".join(o.tbl_sol.item(0, c).text() for c in range(o.tbl_sol.columnCount())))
    s.with_menu(lambda: o.tbl_sol.customContextMenuRequested.emit(o.tbl_sol.visualItemRect(o.tbl_sol.item(1, 2)).center()),
                w, "table_context_menu.png", "Right-click on the solutions table of the Desirability page.")
    w.go("analysis")
    a.tabs.setCurrentIndex(1)
    s.with_menu(lambda: _report_menu(a.txt_anova), w, "report_context_menu.png",
                "Right-click in a report (RSM Analysis > ANOVA): standard text menu plus Copy all tables / Export "
                "all tables...")

    w.go("graphs")
    for key, name in (("overlay", "graph_overlay.png"), ("desirability", "graph_desirability.png")):
        combo_select(gt.cb_type, key)
        s.save(w, name, "Criteria from the Desirability page (Yield >= 75, Purity >= 89). " + gt.note.text())

    # NSGA-II: maximize both responses
    n = w.nsga_tab
    w.go("nsga")
    for r in range(2):
        combo_select(n.tbl.cellWidget(r, 2), "maximize")
    n.run()
    t = s.wait(n)
    print(f"  NSGA-II run: {t:.1f} s", flush=True)
    for key, name in (("front", "nsga_front.png"), ("front2d", "nsga_front2d.png"), ("matrix", "nsga_matrix.png"),
                      ("parallel", "nsga_parallel.png"), ("convergence", "nsga_convergence.png")):
        combo_select(n.cb_plot, key)
        n.rtabs.setCurrentIndex(0)
        s.save(w, name, "Yield & Purity maximized, population 100, max 300 generations, seed 1, automatic stop. "
               + n.lbl_best.text())
    n.run_adequacy()
    t = s.wait(n)
    print(f"  NSGA population check: {t:.1f} s", flush=True)
    s.save(w, "nsga_population_check.png", text_of(n.txt_conv, 900))
    combo_select(n.cb_plot, "front")
    n.rtabs.setCurrentIndex(0)

    # ANN for both responses (needed for the ANN vs RSM comparison)
    at = w.ann_tab
    w.go("ann")
    at.sp_cpu.setValue(1)
    at.chk_use_after.setChecked(False)
    at.search(True)
    t = s.wait(at)
    print(f"  ANN search, BBD case (2 responses): {t:.1f} s  {at.lbl_status.text()}", flush=True)
    s.bbd_ann_status = at.lbl_status.text()

    w.go("nsga")
    n.run_compare()
    t = s.wait(n)
    print(f"  NSGA ANN vs RSM: {t:.1f} s", flush=True)
    for key, name in (("cmp_front", "nsga_compare_front.png"), ("cmp_cross", "nsga_compare_cross.png"),
                      ("cmp_factors", "nsga_compare_factors.png")):
        n.rtabs.setCurrentIndex(0)
        combo_select(n.cb_plot, key)
        s.save(w, name, s.bbd_ann_status)
    n.rtabs.setCurrentWidget(n.txt_cmp)
    s.save(w, "nsga_compare_report.png", text_of(n.txt_cmp, 900))
    n.rtabs.setCurrentIndex(0)
    combo_select(n.cb_plot, "front")

    # sensitivity & parametric study from the best desirability solution
    st = w.sens_tab
    w.go("sens")
    combo_select(st.cb_ref, "desirability")
    st.tabs.setCurrentIndex(0)
    combo_select(st.cb_layout, "all")
    st.run_parametric()
    s.save(w, "sens_parametric_all.png", "Reference: best desirability solution; model: selected model (RSM). "
           + text_of(st.txt_par, 500))
    combo_select(st.cb_layout, "one")
    combo_select(st.cb_px, st.cb_px.itemData(0))
    combo_select(st.cb_pz, st.cb_px.itemData(1))
    st.sp_nz.setValue(3)
    st.run_parametric()
    s.save(w, "sens_parametric_one.png", "Vary Temperature 60-80 (11 points), curves for Time (3 values). "
           + text_of(st.txt_par, 500))
    st.tabs.setCurrentIndex(1)
    combo_select(st.cb_gmethod, "src")
    st.run_global()
    s.save(w, "sens_src.png", "Standardized coefficients, Latin hypercube 120 points. " + text_of(st.txt_sob, 700))
    combo_select(st.cb_gmethod, "sobol")
    st.run_global()
    t = s.wait(st)
    print(f"  Sobol: {t:.1f} s", flush=True)
    s.save(w, "sens_sobol.png", f"Sobol indices, N = {st.sp_n.value()}. " + text_of(st.txt_sob, 700))
    st.tabs.setCurrentIndex(2)
    st.run_tornado()
    s.save(w, "sens_tornado.png", "Each factor changed by +/-10% of its range. " + text_of(st.txt_tor, 700))
    st.tabs.setCurrentIndex(3)
    st.run_front()
    s.save(w, "sens_front.png", "Last NSGA-II result (Yield & Purity maximized). " + text_of(st.txt_front, 700))

    # project summary (after the optimizations)
    s.page("summary", "summary.png", lambda: text_of(w.summary_tab.txt, 900))

    # prediction & confirmation of the best desirability solution
    w.on_send_to_prediction(p.solutions[0][1])
    pt = w.predict_tab
    for j, delta in ((0, 0.4), (1, -0.3)):
        pred = float(pt.tbl.item(j, 1).text())
        pt.tbl.item(j, 7).setText(f"{pred + delta:.2f}")
    s.pump()
    s.save(w, "prediction.png", "Best desirability solution sent to Prediction; 95% level, n = 3; confirmation "
           "means typed in: " + "; ".join(" | ".join(pt.tbl.item(r, c).text() for c in range(pt.tbl.columnCount()))
                                          for r in range(pt.tbl.rowCount())))

    # propagation of error: enter the factor standard deviations, then the POE graph
    for f, sd in zip(p.factors, (1.0, 2.0, 0.05)):
        f.sd = sd
    w.set_project(p, "graphs")
    combo_select(gt.cb_resp, text=p.response_label(0))
    combo_select(gt.cb_type, "poe")
    s.save(w, "graph_poe.png", "Yield, sigma Temperature 1.0 °C, Time 2.0 min, Concentration 0.05 %. "
           + gt.note.text())


def _report_menu(browser):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QContextMenuEvent
    pos = QPoint(260, 160)
    ev = QContextMenuEvent(QContextMenuEvent.Mouse, pos, browser.viewport().mapToGlobal(pos))
    browser.contextMenuEvent(ev)


def sec_factorial(s):
    from doe.project import sample_factorial_project
    from gui.data_tab import AugmentDialog
    w = s.w
    p = sample_factorial_project()
    w.set_project(p, "analysis")
    a = w.analysis_tab
    a.tabs.setCurrentWidget(a.eff_tab)
    combo_select(a.cb_eff, text="Half-Normal")
    a.auto_effects()
    s.save(w, "effects_halfnormal.png", "2^4 factorial, Biomass; terms selected with 'Select Automatically (Lenth "
           "ME Limit)'. " + a.lbl_eff.text())
    combo_select(a.cb_eff, text="Pareto")
    s.save(w, "effects_pareto.png", "Same selection as the half-normal plot. " + a.lbl_eff.text())
    a.tabs.setCurrentIndex(1)
    s.save(w, "effects_anova.png", "ANOVA of the selected factorial model. " + text_of(a.txt_anova, 700))
    combo_select(a.cb_eff, text="Half-Normal")

    d = AugmentDialog(p, w)
    combo_select(d.cb_kind, "ccd")
    s.dialog(d)
    s.save(d, "augment_dialog.png", "2^4 factorial case; Type 'Factorial -> CCD (add axial + center points)', "
           "3 additional center points, rotatable alpha, new block.")
    d.reject()


def sec_logistic(s):
    from doe.project import sample_logistic_project
    w = s.w
    w.set_project(sample_logistic_project(), "analysis")
    a = w.analysis_tab
    a.tabs.setCurrentIndex(1)
    s.save(w, "glm_logistic.png", "Germinated out of 50 seeds (binomial). " + text_of(a.txt_anova, 900))


def sec_splitplot(s):
    from doe.project import sample_splitplot_project
    w = s.w
    w.set_project(sample_splitplot_project(), "data")
    s.save(w, "data_splitplot.png", "Split-plot: WP (whole plot) column; Oven Temperature is hard to change.")
    w.go("analysis")
    a = w.analysis_tab
    a.tabs.setCurrentIndex(1)
    s.save(w, "reml_splitplot.png", "Hardness. " + text_of(a.txt_anova, 900))


def sec_mixture(s):
    from doe.project import sample_mixture_project
    w = s.w
    w.set_project(sample_mixture_project(), "graphs")
    gt = w.graphs_tab
    gt.cb_resp.setCurrentIndex(0)
    combo_select(gt.cb_type, "contour")
    s.save(w, "graph_ternary.png", "Tensile Strength, ternary contour in pseudo-components. " + gt.note.text())
    combo_select(gt.cb_type, "trace")
    s.save(w, "graph_trace.png", "Tensile Strength, Piepel trace from the overall centroid. " + gt.note.text())
    w.go("data")
    s.save(w, "data_mixture.png", "Simplex lattice {3,2} + axial points + replicated vertices and centroid.")


def sec_ann(s):
    from doe.project import sample_ann_project
    w = s.w
    p = sample_ann_project()
    w.set_project(p, "ready")
    s.save(w, "readiness_dye.png", "Dye adsorption, 40 historical runs. " + text_of(w.ready_tab.txt, 600))
    s.page("norm", "normalization.png", "Dye adsorption case, default method; factors and responses.")
    at = w.ann_tab
    w.go("ann")
    at.sp_cpu.setValue(1)
    at.ed_hidden.setText("5")              # default entry (an earlier case may have left its own architecture)
    at.tabs.setCurrentIndex(0)
    s.save(w, "ann_adequacy.png", at.lbl_verdict.text() + " " + text_of(at.adq_txt, 600))
    at.learning()
    t = s.wait(at)
    print(f"  ANN learning curve: {t:.1f} s", flush=True)
    at.tabs.setCurrentWidget(at.tab_learn)
    s.save(w, "ann_learning_curve.png", f"Hidden layer '5' (default), trainbr. {at.lbl_status.text()} "
           + text_of(at.lc_txt, 700))
    at.search(False)
    t = s.wait(at)
    print(f"  ANN architecture search (dye): {t:.1f} s  {at.lbl_status.text()}", flush=True)
    s.ann_search_time = t
    at.tabs.setCurrentWidget(at.tab_search)
    s.save(w, "ann_search.png", "Default search settings (no layer/neuron limit, 5-fold CV, patience 2, no weight "
           f"limit), 1 CPU core, seed 1. {at.lbl_status.text()} " + text_of(at.srch_txt, 700))
    for widget, name in ((at.txt, "ann_summary.png"), (at.reg, "ann_regression.png"),
                         (at.perf, "ann_performance.png"), (at.imp, "ann_importance.png"),
                         (at.cmp, "ann_rsm_vs_ann.png"), (at.net, "ann_diagram.png"), (at.wts, "ann_weights.png")):
        at.tabs.setCurrentWidget(widget)
        note = text_of(widget, 700) if hasattr(widget, "toPlainText") else at.lbl_status.text()
        s.save(w, name, note)


def sec_diag_table(s):
    """Diagnostics Report (Table) sub-tab. Made last: once this sub-tab has been shown, the bars above the tabs of
    the RSM Analysis page are drawn with extra vertical space in the screenshots that follow."""
    from doe.project import sample_project
    w = s.w
    w.set_project(sample_project(), "analysis")
    a = w.analysis_tab
    a.tabs.setCurrentWidget(a.diag)
    a.diag.tabs.setCurrentIndex(1)
    s.save(w, "rsm_diagnostics_table.png", "Response Yield, Diagnostics Report (Table) sub-tab.")
    a.diag.tabs.setCurrentIndex(0)


SECTIONS = [("start", sec_start), ("new_design", sec_new_design), ("guide", sec_guide), ("cases", sec_cases),
            ("import", sec_import), ("bbd", sec_bbd), ("factorial", sec_factorial), ("logistic", sec_logistic),
            ("splitplot", sec_splitplot), ("mixture", sec_mixture), ("ann", sec_ann),
            ("diag_table", sec_diag_table)]


def main():
    sys.path.insert(0, ROOT)
    if "--list" in sys.argv:
        print("\n".join(name for name, _ in SECTIONS))
        return
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="regen_shots_")

    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication(sys.argv)
    from gui import theme
    theme.apply(app)

    messages = []

    def record(kind, answer):
        def fn(*args, **kwargs):
            title = args[1] if len(args) > 1 else ""
            text = args[2] if len(args) > 2 else ""
            messages.append(f"{kind}: {title}: {text}")
            print(f"  [message box] {kind}: {title}: {' '.join(str(text).split())[:200]}", flush=True)
            return answer
        return staticmethod(fn)

    QMessageBox.question = record("question", QMessageBox.Discard)
    QMessageBox.information = record("information", QMessageBox.Ok)
    QMessageBox.warning = record("warning", QMessageBox.Ok)
    QMessageBox.critical = record("critical", QMessageBox.Ok)

    import gui.main_window as mw
    ini = os.path.join(tmp, "settings.ini")
    mw.QSettings = lambda *a: QSettings(ini, QSettings.IniFormat)

    w = mw.MainWindow()
    w.setAttribute(Qt.WA_DontShowOnScreen)
    w.resize(WIN_W, WIN_H)
    w.show()
    s = Shooter(app, w, tmp)
    s.messages = messages
    t_all = time.time()
    for name, fn in SECTIONS:
        if only and name not in only:
            continue
        print(f"[{name}]", flush=True)
        t0 = time.time()
        try:
            fn(s)
        except Exception:  # noqa: BLE001 - keep going with the other sections
            traceback.print_exc()
            s.failed.append(name)
        print(f"  ({time.time() - t0:.1f} s)", flush=True)

    notes_path = os.environ.get("REGEN_SHOT_NOTES")
    if notes_path:
        with open(notes_path, "w", encoding="utf-8") as fh:
            for name, note in s.notes:
                fh.write(f"## {name}\n{note}\n\n")
            fh.write("## message boxes\n" + "\n".join(messages) + "\n")
    print(f"\n{len(s.saved)} screenshots saved to {OUT} in {time.time() - t_all:.0f} s")
    if s.failed:
        print("FAILED sections: " + ", ".join(s.failed))
    w.project = None                     # nothing to save on close
    from doe import parallel
    parallel.shutdown()


if __name__ == "__main__":
    main()
