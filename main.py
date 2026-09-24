"""REGEN (Response Engineering with GEnetic algorithms and Neural networks) - application entry point."""
import os
import sys

# DOE matrices are small (< 200 x 200): multi-threaded BLAS is actually slower due to overhead & CPU contention.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


def resource_path(name):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def self_test(out_dir):
    """Self-test for the installed version: `"REGEN.exe" --self-test <folder>` -> selftest.log."""
    import traceback

    os.makedirs(out_dir, exist_ok=True)
    log = os.path.join(out_dir, "selftest.log")
    try:
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from PySide6.QtWidgets import QApplication

        from doe import constraints, evaluation, models
        from doe.project import (Factor, Project, Response, sample_categoric_project, sample_combined_project,
                                 sample_factorial_project, sample_logistic_project, sample_mixture_project,
                                 sample_project, sample_splitplot_project)
        from gui import plots
        from gui.report import write_excel, write_pdf

        QApplication(sys.argv)
        opt = Project.create("optimal_mixture", [Factor("A", "%", 40, 0, upper=70), Factor("B", "%", 5, 0, upper=30),
                                                 Factor("C", "%", 2, 0, upper=10)], [Response("Y")], seed=1,
                             mixture_total=100, constraints=constraints.parse_many("B - 2C >= 0", 3))
        opt.data[:, 0] = 10 * opt.coded[:, 0] + 30 * opt.coded[:, 1] + 5 * opt.coded[:, 2] \
            + np.random.default_rng(0).normal(0, 0.3, opt.n)
        lines = []
        projects = (("rsm", sample_project()), ("mixture", sample_mixture_project()),
                    ("categoric", sample_categoric_project()), ("factorial", sample_factorial_project()),
                    ("optimal", opt), ("logistic", sample_logistic_project()),
                    ("splitplot", sample_splitplot_project()), ("combined", sample_combined_project()))
        for name, p in projects:
            fit = p.fit(0)
            ev = evaluation.evaluate(p.coded, p.model_terms(0), p.is_mixture, None, p.space)
            evaluation.fds_curve(ev, p.region_samples(500))
            if p.has_poe:
                models.poe(fit, p.coded, p.sd_coded())
            yv = p.data[:, 0] / ((p.responses[0].trials or 1) if p.response_family(0) == "binomial" else 1)
            crit = [{"fit": fit, "goal": "maximize", "low": float(np.nanmin(yv)), "high": float(np.nanmax(yv)),
                     "index": 0}]
            sols = p.optimize(crit, {})
            p.solutions, p.criteria_used = sols, crit
            axes = [0, 1, 2] if p.is_mixture else p.num_idx[:2]
            base = np.nanmean(p.coded, axis=0) if p.is_mixture else np.zeros(p.k)
            for kind in ("contour", "surface", "overlay", "desirability"):
                fig = Figure()
                FigureCanvasAgg(fig)
                if kind == "overlay":
                    plots.draw_overlay(fig, p, [("R1", fit, crit[0]["low"], crit[0]["high"])], axes, base, sols[0][1])
                elif kind == "desirability":
                    plots.draw_desirability(fig, p, crit, axes, base, 10, sols[0][1])
                else:
                    plots.draw_response_2d(fig, p, fit, axes, base, "R1", kind)
                fig.savefig(os.path.join(out_dir, f"{name}_{kind}.png"))
            if p.is_two_level_factorial:
                fig = Figure()
                FigureCanvasAgg(fig)
                plots.draw_half_normal(fig, p.effects(0), set(fit.terms))
                fig.savefig(os.path.join(out_dir, f"{name}_halfnormal.png"))
            iv = fit.intervals(sols[0][1][None, :], n_obs=3)
            write_pdf(p, os.path.join(out_dir, f"{name}.pdf"), crit, sols)
            write_excel(p, os.path.join(out_dir, f"{name}.xlsx"))
            r2 = fit.stats.get("r2", fit.stats.get("mcfadden", fit.stats.get("r2_cond", float("nan"))))
            lines.append(f"{name} [{p.analysis_kind(0)}]: R2={r2:.4f} D={sols[0][0]:.3f} "
                         f"PI=[{iv['pi'][0][0]:.3g}, {iv['pi'][1][0]:.3g}] OK")
        # ANN + NSGA-II (RSM, 2 responses)
        p = sample_project()
        m = p.train_ann(0, hidden=[6, 3], algorithm="trainbr", split=(0.85, 0, 0.15), restarts=5, seed=1)
        p.model_source[0] = "ann"
        res = p.run_nsga([{"index": 0, "goal": "maximize"}, {"index": 1, "goal": "maximize"}], {}, pop=60,
                         gens=60, seed=1)
        from gui.nsga_tab import draw_pareto, write_pareto_excel
        fig = Figure()
        FigureCanvasAgg(fig)
        draw_pareto(fig, p, res)
        fig.savefig(os.path.join(out_dir, "nsga_pareto.png"))
        write_pareto_excel(p, res, os.path.join(out_dir, "nsga_pareto.xlsx"))
        write_pdf(p, os.path.join(out_dir, "ann_nsga.pdf"))
        lines.append(f"ann+nsga2: R2 ANN={m.metrics['all']['r2']:.4f} Pareto={len(res['X'])} solutions OK")
        # normalization, data adequacy, automatic architecture, case studies
        from doe import adequacy, ann
        from doe.project import CASES
        for norm in ann.NORMS:
            mn = p.train_ann(1, hidden=[4, 2], algorithm="trainbr", split=(0.85, 0, 0.15), restarts=2, seed=2,
                             norm=norm)
            assert np.isfinite(mn.metrics["all"]["r2"]), norm
        srch = ann.architecture_search(p.coded, p.data[:, 0], p.space, max_layers=2, max_neurons=4, folds=4,
                                       restarts=1, max_epochs=60)
        lines.append(f"architecture: {len(srch['table'])} candidates, selected {srch['best']} OK")
        for key, _t, _c, fn, _d in CASES:
            a = adequacy.assess(fn(), 0)
            lines.append(f"case {key}: {a['level']} ({len(a['checks'])} checks) OK")
        from gui.ann_tab import adequacy_html, ann_summary_html, search_html, weights_html
        adequacy_html(p, 0, adequacy.assess(p, 0))
        srch["resp"] = [0]
        srch["rsm_cv"] = p.rsm_cv([0], srch["rows"], srch["splits"])
        search_html(p, srch)
        mc = p.train_ann([0, 1], hidden=[5, 3], algorithm="trainbr", restarts=2, seed=1)
        ann_summary_html(p, 1, p.ann[1])
        weights_html(p, p.ann[1])
        lines.append(f"combined ANN: {mc.arch}, R2 R1={p.ann[0].metrics['all']['r2']:.3f} "
                     f"R2 R2={p.ann[1].metrics['all']['r2']:.3f}, diagnosis {mc.diagnosis['status']} OK")
        p.model_source[0] = "rsm"
        write_pdf(p, os.path.join(out_dir, "v61.pdf"))
        # data readiness & recommendations, interactive diagnostics, ignored runs, all factors graph
        from doe import readiness
        from gui.diagnostics import DiagnosticsPanel, dfbetas
        from gui.readiness_tab import readiness_html
        for key, _t, _c, fn, _d in CASES:
            q = fn()
            a = readiness.assess(q, 0)
            readiness_html(q, 0, a)
            for arr in a["suggest"].values():
                q.add_runs(arr[:1], "Suggested")
            lines.append(f"readiness {key}: modeling {a['score_model']:.0f}, optimization {a['score_opt']:.0f}, "
                         f"{len(a['recs'])} recommendations OK")
        q = sample_project()
        f0 = q.fit(0)
        dfb = dfbetas(q, f0)
        panel = DiagnosticsPanel()
        panel.set_fit(q, f0, 0)
        for pn in panel.panels:
            for k in range(pn.cb.count()):
                pn.cb.setCurrentIndex(k)
        q.set_excluded([0, 1])
        lines.append(f"diagnostics: DFBETAS {dfb.shape}, runs ignored -> n = {q.fit(0).n} OK")
        fig = Figure()
        FigureCanvasAgg(fig)
        plots.draw_all_factors(fig, q, q.fit(0), np.zeros(q.k), "R1")
        fig.savefig(os.path.join(out_dir, "all_factors.png"))
        # matrix measures & correlation, summary, coefficient table, column graphs, notes
        from gui.info_tabs import GraphColumnsTab, coef_table_html, summary_html
        q = sample_project()
        q.notes = "test notes"
        ev = evaluation.evaluate(q.coded, q.model_terms(0), False, None, q.space)
        labs, coef, labs2, R = evaluation.correlations(ev)
        mm = evaluation.matrix_measures(ev, q.region_samples(1000))
        summary_html(q)
        coef_table_html(q)
        gc = GraphColumnsTab()
        gc.set_project(q)
        for k in range(gc.cb_kind.count()):
            gc.cb_kind.setCurrentIndex(k)
        gc.canvas.figure.savefig(os.path.join(out_dir, "column_graphs.png"))
        write_pdf(q, os.path.join(out_dir, "v63.pdf"))
        lines.append(f"evaluation: D-eff {mm['d_eff']:.1f}%, cond {mm['cond']:.2f}, correlation {coef.shape} OK")
        # parallel multi-core ANN, NSGA-III, automatic stop, adequacy check, ANN vs RSM comparison
        from doe import nsga2, parallel
        from gui.nsga_tab import KINDS, CMP_KINDS, compare_fronts, compare_html, convergence_html, draw_compare
        q = sample_project()
        s1 = ann.architecture_search(q.coded, q.data[:, 0], q.space, max_layers=2, max_neurons=4, folds=4,
                                     restarts=2, max_epochs=60, n_jobs=1)
        s2 = ann.architecture_search(q.coded, q.data[:, 0], q.space, max_layers=2, max_neurons=4, folds=4,
                                     restarts=2, max_epochs=60, n_jobs=2)
        assert s1["best"] == s2["best"] and s2["workers"] == 2, (s1["best"], s2["best"])
        mp1 = q.train_ann(0, hidden=[4], restarts=4, n_jobs=2, algorithm="trainbr", seed=1)
        q.train_ann(1, hidden=[3], restarts=2, algorithm="trainbr", seed=1)
        parallel.shutdown()
        objs = [{"index": 0, "goal": "maximize"}, {"index": 1, "goal": "maximize"}]
        r3 = q.run_nsga(objs, {}, pop=40, gens=80, seed=1, algorithm="nsga3")
        rr = q.run_nsga(objs, {}, pop=40, gens=80, seed=1, source="rsm", store=False)
        ra = q.run_nsga(objs, {}, pop=40, gens=80, seed=1, source="ann", store=False)
        cmp = compare_fronts(q, rr, ra)
        q.nsga_compare = cmp
        compare_html(q, cmp)
        adq = q.nsga_adequacy(r3, runs=[("other seed", 40, 9)])
        convergence_html(q, r3, adq)
        for key, _l in KINDS:
            fig = Figure()
            FigureCanvasAgg(fig)
            draw_pareto(fig, q, r3, 1, key, {"x": 0, "y": 1, "color": 1, "size": 0})
        for key, _l in CMP_KINDS:
            fig = Figure()
            FigureCanvasAgg(fig)
            draw_compare(fig, q, cmp, key, {"x": 0, "y": 1})
            fig.savefig(os.path.join(out_dir, f"{key}.png"))
        hv = nsga2.hypervolume(np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.5]]), np.zeros(3), np.ones(3))
        write_pdf(q, os.path.join(out_dir, "v67.pdf"))
        # factorial table up to 21 factors, Latin Hypercube, hybrid, design guide, learning curve
        from doe import designs, planning
        from doe.project import sample_ann_project
        from gui.design_guide import DesignGuideDialog
        cells = designs.factorial_table()
        Xf = designs.fractional_factorial(21, 12)
        assert np.allclose(Xf.T @ Xf, 512 * np.eye(21)) and designs.resolution(21, 12) == 5
        assert designs.resolution(17, 9) == 5 and designs.resolution(21, 15) == 4
        lh = Project.create("lhs", [Factor("A", "", 0, 1), Factor("B", "", 0, 1), Factor("C", "", 0, 1)],
                            [Response("Y")], seed=1, lhs_runs=20, center_points=3, design_seed=5)
        hy = Project.create("hybrid", [Factor("A", "", 0, 1), Factor("B", "", 0, 1), Factor("C", "", 0, 1)],
                            [Response("Y")], seed=1, hybrid_base="bbd", extra_points=10, center_points=3)
        recs = {g: len(planning.recommend(g, 4, 40)) for g in planning.GOALS}
        gd = DesignGuideDialog(None, 4)
        for gi in range(gd.cb_goal.count()):
            gd.cb_goal.setCurrentIndex(gi)
        gd.select_cell(5, 10)
        gd.update_calc()
        qa = sample_ann_project()
        lc = ann.learning_curve(qa.coded, qa.data[:, 0], qa.space, [4], rsm_eval=qa.rsm_eval(0), n_jobs=2,
                                repeats=3, max_epochs=80)
        parallel.shutdown()
        qa.learning = {0: lc}
        from gui.ann_tab import draw_learning, learning_html
        learning_html(qa, 0, lc)
        fig = Figure()
        FigureCanvasAgg(fig)
        draw_learning(fig, qa, 0, lc)
        fig.savefig(os.path.join(out_dir, "learning_curve.png"))
        lines.append(f"design: {len(cells)} factorial cells, 2^(21-12) Res V orthogonal, LHS {lh.n} runs "
                     f"({len(np.unique(lh.coded[:, 0]))} levels), hybrid {hy.n} runs, recommendations {recs}; learning "
                     f"curve {lc['status']} OK")
        lines.append(f"parallel: architecture {s2['best']} ({s2['workers']} cores) = 1 core, parallel restarts "
                     f"{mp1.workers} cores; NSGA-III {r3['gens_run']} generations ({r3['stop_reason'] or 'max'}); "
                     f"test HV {hv:.3f}; ANN vs RSM difference {cmp['diff'][0]['mean']:.1f}% OK")
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\nSELFTEST OK\n")
        return 0
    except Exception:  # noqa: BLE001
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        return 1


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--self-test":
        sys.exit(self_test(sys.argv[2]))

    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from gui import theme
    from gui.main_window import APP_NAME, MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    theme.apply(app)
    icon = resource_path(os.path.join("assets", "icon.ico"))
    if os.path.exists(icon):
        app.setWindowIcon(QIcon(icon))

    win = MainWindow()
    win.show()
    # open a .doe file double-clicked in Explorer
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".doe") and os.path.exists(sys.argv[1]):
        win.load_path(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()        # parallel worker processes (multi-core ANN) in the installed version
    main()
