# REGEN user manual - screenshot list

All files in this folder are made by `docs/manual/make_screenshots.py`:

    .venv\Scripts\python.exe docs\manual\make_screenshots.py            (all, about 2.5 minutes)
    .venv\Scripts\python.exe docs\manual\make_screenshots.py --only bbd,ann
    .venv\Scripts\python.exe docs\manual\make_screenshots.py --list

General notes for the manual writers

- The main window is 1500 x 900 (logical pixels). The script runs at the screen scaling of the computer it runs on;
  at 150 % the PNG files are 2250 x 1350. Dialogs keep their own size. Window grabs include the menu bar, toolbar,
  navigation tree and status bar, but not the Windows title bar; dialog grabs do not show the dialog title bar
  either (the dialog name is given below).
- Everything is reproducible: fixed case-study data, fixed seeds (design seed 12345 in New Design, ANN seed 1,
  NSGA seed 1) and 1 CPU core. Re-running the script gives the same numbers.
- The case studies come from File > Case Studies (`doe/project.py`, `CASES`): BBD extraction (rsm), edible film
  mixture, 2^4 fermentation screening, seed germination (logistic), oven hardness (split-plot), dye adsorption (ANN).
- Context menus cannot be shown off screen in the normal way; the script captures the real menu and draws it over
  the window at the place where it opens (with a thin border and a small shadow). The menu contents are genuine.
- Qt drops the trailing "..." from toolbar button texts, so the plot toolbar shows "Export data" although the
  action is called "Export data..." (the plot right-click menu shows "Export plot data...").
- Case "BBD extraction" is used for the whole RSM-to-optimization workflow, in this order: data, readiness,
  evaluation, analysis (Yield), model graphs, desirability, NSGA-II, ANN for both responses, ANN vs RSM, sensitivity,
  summary, prediction, POE. Numbers below therefore refer to that state of the project.

---

## 1. Start, new design and design guide

| File | Page / dialog | Case & settings | What is visible / worth explaining |
|---|---|---|---|
| `start_page.png` | Start Page, no project open | Three projects saved by the script into a folder "Projects" and added to the recent list | Left: REGEN name, "Response Engineering with GEnetic algorithms and Neural networks", Version 7.0.0, five start buttons (New design, Design guide, Open project, Import data, Case studies). Right: Recent projects (extraction_bbd.doe, edible_film.doe, dye_adsorption.doe), "Double-click to open". Save / Export / PDF are greyed out because no project is open; the navigation tree shows all pages grouped Project / Design / Analysis / Optimization / Tools. |
| `new_design_ccd.png` | New Design dialog | Type Central Composite Design (CCD), 3 numeric factors Temperature 60-80 °C, Time 30-90 min, Concentration 1-3 %, 2 responses Yield (%), Purity (%), seed 12345 | Options: 6 center points, 1 replicate, No blocks, Alpha "Rotatable", split-plot check box, random seed. Blue info line at the bottom: "Runs: 20 \| alpha = 1.6818". Buttons Create Design / Cancel / Design Guide... |
| `new_design_mixture.png` | New Design dialog, mixture page | Mixture: Simplex Lattice, 3 components Starch (lower limit 50 %), Glycerol (10 %), Chitosan (10 %), 2 responses Tensile Strength (MPa), Elongation (%) | The dialog switches to "Components", "Mixture Components (actual units)" with a Lower limit column, and options Lattice degree (m) = 2, Mixture total = 100, "Add axial points + centroid", "Replicate vertices + centroid". Info line: "Runs: 14 \| Each component can rise up to its lower limit + 30". |
| `new_design_optimal.png` | New Design dialog, optimal page | Optimal (Custom) - Process, same 3 factors, 1 categorical factor Solvent (Ethanol; Methanol), linear constraint `A + B <= 160`, 1 response Yield | Options change to Criterion (I-optimal), Planned model (Quadratic), Additional model points 0, Lack of fit points 5, Replicate points 5. Extra boxes for Categorical Factors and Linear Constraints. Info line: "Model Quadratic: 14 coefficients -> 24 runs in total (14 model + 5 lack of fit + 5 replicates)". |
| `design_guide_recommend.png` | Design Guide dialog, tab "Recommendations by Goal" | Goal "optimize", 3 numeric factors, no run budget | Table of candidate designs (runs, levels, model, power for a 2-sigma signal, notes); the first suitable row, Central Composite (rotatable), 20 runs, 5 levels, is selected and explained below (advantages, disadvantages, lowest power 72 % for the quadratic coefficients, ANN adequacy note). Button "Create This Design..." passes the choice to New Design. |
| `design_guide_table.png` | Design Guide, tab "Two-Level Factorial Table" | Cell 16 runs x 6 factors selected | Colour-coded grid of 2^(k-p) designs (white = full, green = Res V+, yellow = Res IV, red = Res III). Below: "Fractional factorial 2^(6-2), resolution IV", 16 runs, generators E = ABC, F = BCD, defining relation I = ABCE = ADEF = BCDF, word length pattern and the alias table. |
| `design_guide_calculator.png` | Design Guide, tab "Run Count Calculator" | 3 factors, Quadratic, 5 lack-of-fit + 5 replicate points, ANN with automatic hidden size, "Data you have" = 20 | RSM: 10 coefficients, absolute minimum 11 runs, recommended 20; standard designs for 3 factors (CCD rotatable 20 runs, 72 % power; CCD face-centered 20 runs, 32 %; Box-Behnken 17 runs, 41 %). ANN: 3 inputs, hidden 6, 31 weights and biases with the runs needed; rules of thumb; "Your data: 20 runs" verdict for RSM and ANN. |
| `case_studies.png` | Case Studies dialog | First case selected | List of the 8 case studies with category, and on the right the title "Extraction Optimization (Box-Behnken)", category "Response Surface" and the description. Button "Open Case Study". |
| `import_dialog.png` | Import Data Set dialog | The 17-run BBD table pasted with headers "Std", "Run", "A: Temperature (°C)" ... "R1: Yield (%)", "R2: Purity (%)"; column Yield selected | Shows automatic role detection: Std and Run -> Ignore (grey), A-C -> Numeric factor (blue), R1/R2 -> Response (yellow); names and units are taken from the headers. Right panel "Selected Column": Name Yield, Units %, Role Response, "17 values, min 64.13, max 78.29, 17 unique values"; summary "3 factors, 2 responses, 17 rows". |

## 2. Data and design (BBD extraction unless noted)

| File | Page / dialog | Case & settings | What is visible / worth explaining |
|---|---|---|---|
| `data_page.png` | Data | BBD extraction | "Box-Behnken Design (BBD) - 3 factors, 17 runs", sort by Run, columns Std, Run, Type (Edge / Center, center rows shaded), factor columns (grey = read-only) and white response columns Yield and Purity. Buttons: Add/Delete/Rename Response, Exclude/Restore Runs, Augment Design..., Factor & Response Info..., Simulate...; hint about Ctrl+V from Excel. Status bar: "17 runs, 17 complete". |
| `factor_response_info.png` | Factor & Response Info dialog (from Data) | BBD; standard deviations typed in: Temperature 1.0, Time 2.0, Concentration 0.05 | Factor table with Range and "Standard deviation (sigma)" used for Propagation of Error; response table with Name, Unit, Type (Normal / Binomial / Poisson) and Trials per run. The same sigma values are used later for `graph_poe.png`. |
| `simulate_dialog.png` | Simulate Response dialog | BBD, default entries | Response R1: Yield, equation `50 + 3*A - 2*B - 4*A**2`, factor units Coded, noise sigma 1, seed 1, "Fill empty cells only"; the long note lists the allowed operators and functions. |
| `augment_dialog.png` | Augment Design dialog | **2^4 factorial screening case** (the Factorial -> CCD option only exists for two-level factorials) | Type "Factorial -> CCD (add axial + center points)", Additional center points 3, Alpha Rotatable, "Make it a new block" checked, explanatory note. |
| `data_splitplot.png` | Data | Oven hardness split-plot | Extra column WP (whole plot) next to Std/Run; Oven Temperature (hard to change) is constant within each whole plot. Status bar "2-Level Full Factorial, split-plot (HTC: A; 4 whole plots) - 3 factors, 16 runs". |
| `data_mixture.png` | Data | Edible film mixture | "Mixture: Simplex Lattice {3, 2} - 3 components, total = 100, 14 runs"; point types Vertex, Edge, Axial and Centroid are colour-coded (vertices and centroid replicated); component columns in actual % (each row sums to 100); "Show pseudo-components" check box instead of "Show coded values". |
| `readiness.png` | Data Readiness | BBD, response Yield, reference model Quadratic | Score cards: Modeling 94 Ready, Optimization 96 Ready, Data Completeness 100, Data Structure 100, Response & Model Quality 83 (2 notes), Optimization Readiness 100. Summary table for both responses (Yield 94/96, Purity 100/97, both Ready). Recommendations: check runs 1 and 7 (Cook's distance > 1) and "Significant lack of fit: try a higher-order model (Cubic), a transform, or an ANN". Suggested runs: "Confirmation around the optimum - 4 run" with predicted Yield about 80.2-80.3. |
| `readiness_dye.png` | Data Readiness | Dye adsorption (40 historical runs), Removal Efficiency | Same page for nonlinear historical data: Modeling score 82, Optimization 77, both "Fair - some things need fixing"; reference model Quadratic (15 parameters). Good contrast with `readiness.png`. |
| `design_evaluation.png` | Design Evaluation | BBD, model Quadratic, signal Delta 2, noise sigma 1, alpha 0.05, graph FDS | Degrees of freedom (17 runs, 10 coefficients, residual 7, lack of fit 3, pure error 4); coefficient table with standard error, VIF (1 / 1.006) and power (68.1 % for main effects, 40.8 % for interactions, 42.5 % for squares - red because < 80 %); alias structure. Right: Fraction of Design Space curve with the standard error at 50 % (0.548) and 80 % (0.668) of the space. |
| `normalization.png` | Data Normalization (Tools) | Dye adsorption, method "Min-max to [-1, 1] (mapminmax)", factors and responses | Formula x' = 2(x - min)/(max - min) - 1, normalized data table, and box plots before (factors on very different scales: pH 2-10 vs concentration 20-200) and after normalization (all in -1..1). |

## 3. Project pages (BBD extraction)

| File | Page | Settings | What is visible / worth explaining |
|---|---|---|---|
| `summary.png` | Summary | After desirability, NSGA-II and ANN | Factor table (range, data min/max, mean, SD, unique levels, POE sigma), response table (Yield: Quadratic, R² 0.9919, Adj R² 0.9815, Pred R² 0.8857; Purity: R² 0.9886, Adj 0.974, Pred 0.9302; both "Used RSM"), "Other project contents: 2 ANN models, desirability optimization results (5 solutions), NSGA-II results (99 Pareto solutions)", then the notes. |
| `notes.png` | Notes | Example notes (goal, material, equipment, dated events, analysis decision) | Free-text page; the "Insert Date" button adds a [dd-mm-yyyy hh:mm] stamp. Notes are saved in the .doe file and printed in the PDF report. |
| `column_graphs.png` | Column Graphs | Scatterplot, X = Temperature, Y = Yield, colour by Run, trend line on | Scatter with linear trend y = 0.3934x + 45.55; "Correlation (Pearson r) = 0.53 · Spearman rho = 0.4551 · n = 17"; correlation heatmap of all columns (Std, Run, A, B, C, R1, R2, residuals) - clicking a cell plots that pair. |
| `coefficient_table.png` | Coefficient Table | p-values shown | Coded coefficients of both responses side by side with colour by p-value, e.g. Yield intercept 77.772, A 3.9337, A² -5.4298; Purity A -2.3538. AC and BC are not significant for both responses (p 0.34-0.41). |

## 4. RSM analysis

| File | Page / tab | Case & settings | What is visible / worth explaining |
|---|---|---|---|
| `rsm_fit_summary.png` | RSM Analysis > Fit Summary | BBD, Yield | Sequential model table: Linear p 0.0171, 2FI p 0.7955, Quadratic p < 0.0001 (LOF p 0.0302, Adj R² 0.9815, Pred R² 0.8857, remarks "Suggested" and "Significant lack of fit"), Cubic "Aliased"; how-to-read note, then the orange warning "Warning: the suggested model (Quadratic) has significant lack of fit (p = 0.0302). It does not describe the data within the pure error: check for outliers, try a response transform or a higher-order model, or consider an ANN." Left panel: model term list with check boxes and "Maintain hierarchy". |
| `rsm_anova.png` | RSM Analysis > ANOVA | BBD, Yield, Quadratic | Type III ANOVA: Model F = 95.37, p < 0.0001; AC (p 0.3373) and BC (p 0.4066) not significant (red); Lack of Fit F = 8.935, p 0.0302 (significant - worth explaining: pure error is very small, 4 df from 5 center points). Fit statistics: Std. Dev. 0.71361, Mean 73.089, C.V. 0.976 %, R² 0.99191, Adj R² 0.98151, Pred R² 0.88574, Adeq Precision 26.79. |
| `report_context_menu.png` | RSM Analysis > ANOVA, right-click in the report | as above | Standard text menu (Copy, Select All) plus "Copy all tables (for Excel)", "Copy all tables - decimal ','" and "Export all tables...". |
| `rsm_equation.png` | RSM Analysis > Model Equation | BBD, Yield | Coded equation Yield = +77.772 +3.9337 A +3.3975 B +1.4887 C +2.14 AB -0.3675 AC -0.315 BC -5.4298 A² -2.7423 B² -1.7798 C², and the actual-units equation (Yield = -220.13 + 7.6405 Temperature + ...), with the warning not to judge effects from the actual equation. |
| `rsm_diagnostics.png` | RSM Analysis > Diagnostics > Graphs | BBD, Yield, 4 plots (2 x 2), no colouring | Normal plot of externally studentized residuals, residuals vs predicted with Bonferroni limits ±4.820, predicted vs actual, Cook's distance with the 1.0 and 4/n lines (runs 1 and 7 above 1.0). Buttons Exclude Selected Runs / Restore Excluded Runs; each plot has its own selector and toolbar. |
| `rsm_diagnostics_table.png` | RSM Analysis > Diagnostics > Diagnostics Report (Table) | BBD, Yield (made at the very end of the script) | Per-run table (actual, predicted, residual, leverage 0.75 / 0.2, studentized residuals, Cook's, DFFITS, DFBETAS) with red cells beyond the limits; the header line gives the limits and counts (Cook's > 1: 2 runs, DFFITS > 1.534: 8 runs). The page layout is compact (no extra space above the tab bar), so all 17 runs are visible; no transformation, so the columns read "Predicted" and "Residual" (with a transform they read "Predicted (transformed scale)" and "Residual (transformed scale)"). |
| `rsm_boxcox.png` | RSM Analysis > Box-Cox | BBD, Yield | ln(SS residual) vs lambda from -3 to 3 with the 95 % CI limit; title "95% CI for lambda: -2.95 to > 3.00"; recommendation "None" (lambda = 1 lies inside the CI), plus the note that the CI reaches the upper end of the lambda range. Button "Apply Recommendation". |
| `effects_halfnormal.png` | RSM Analysis > Effects (Factorial), Half-Normal | **2^4 factorial screening**, Biomass, after "Select Automatically (Lenth ME Limit)" | Selected terms (orange): A, D, AD, AB, B, BD; error effects (blue) on the line; ME (Lenth) = 0.198 and SME = 0.401 lines. Text below: "PSE Lenth = 0.07688, ME = 0.1976, SME = 0.4012". The term list on the left shows the automatic selection (hierarchy keeps B for AB/BD). The Effects tab appears only for two-level factorials. |
| `effects_pareto.png` | Effects (Factorial), Pareto | same | Pareto chart of absolute effects with the ME/SME limits, same selection. |
| `effects_anova.png` | ANOVA of the selected factorial model | same | Model F = 1385, p < 0.0001 with A, B, D, AB, AD, BD all significant; R² 0.99892, Adj R² 0.9982, Pred R² 0.99658, 9 residual df. |
| `glm_logistic.png` | RSM Analysis > GLM (Logistic/Poisson) | **Seed germination**, Germinated out of 50 seeds, Quadratic | Tab name changes to "GLM (Logistic/Poisson)" and the transformation box is disabled. Likelihood-ratio chi² table: Model 114.1 (5 df, p < 0.0001), AB p 0.0622 and B² p 0.5312 not significant; deviance 11.27, McFadden pseudo R² 0.640, dispersion 1.61; goodness of fit Deviance/Pearson p 0.127, Hosmer-Lemeshow p 0.819; coefficients with odds ratios. |
| `reml_splitplot.png` | RSM Analysis > Split-Plot (REML) | **Oven hardness split-plot**, Hardness, 2FI | Tab name "Split-Plot (REML)"; the "Use Suggested Model" button is hidden. Wald F tests with Type (whole plot / subplot): A (whole plot) F 190.8 with only 2 denominator df, p 0.0052; subplot terms with 7 df. Variance components: whole plot 0.2006 (41.8 %), subplot 0.2790 (58.2 %); 4 whole plots; conditional R² 0.99455. |

## 5. Model graphs

All BBD extraction, response Yield, X = Temperature, Y = Time, Concentration held at 2 % unless noted.

| File | Graph type | What is visible / worth explaining |
|---|---|---|
| `graph_contour.png` | 2D Contour | Contour of Yield with 12 contour lines; red points = design runs in this slice (the centre point shows "5" = five replicates). Side note explains the -1..+1 graph range and grey area outside constraints. |
| `graph_surface3d.png` | 3D Surface | Response surface of Yield over Temperature x Time with the curved maximum near the middle-to-high Time. |
| `graph_oneway.png` | One Factor Effect | Yield vs Temperature with 95 % CI band, Time 60 min and Concentration 2 % ("Other Factor Values"); curved effect with its maximum of about 78.5 % near 73.5 °C; the five center runs are shown as red points at 70 °C. |
| `graph_all_factors.png` | All Factors | One panel per factor (Temperature, Time, Concentration), others held at the center. |
| `graph_interaction.png` | Interaction | Yield vs Temperature with lines for Time low/high and 95 % CI error bars; non-parallel lines show the AB interaction. |
| `graph_cube.png` | Cube Plot | Predicted Yield at the 8 corners: 60.46 (all low) to 78.1 (high Temperature, high Time, high Concentration); depth axis Concentration 1 -> 3. |
| `graph_perturbation.png` | Perturbation | Group "Reference Point" with a control for every factor (A: Temperature 70.00, B: Time 60.00, C: Concentration 2.0000, the center). Each factor is moved over its range (-1..+1 coded) while the others stay at the reference point; all traces pass through the reference point (Yield 77.77 at deviation 0); x axis "Deviation from reference point (coded units)". Note: "Each numeric factor is moved over its range (-1 to +1) while the other factors stay at the reference point set below; the x axis is the deviation from the reference value in coded units. Red points = ...". Temperature (A) has the steepest and most curved trace, Concentration (C) the flattest. |
| `plot_toolbar.png` | Plot toolbar (crop) | The toolbar above every plot: Home, Back, Forward, Pan, Zoom, Subplot settings, Edit axes, Save image, then the REGEN additions "Copy data" and "Export data(...)". |
| `plot_context_menu.png` | Right-click on a plot (contour) | Menu "Copy plot data" / "Export plot data..." (exports the data behind the plot to Excel or CSV). |
| `graph_overlay.png` | Overlay (Graphical Optimization) | Uses the limits from the Desirability page: Yield 75-78.29, Purity 89-93.6. Yellow = all limits met. Worth explaining: with goal "maximize" the upper limit is the data maximum (78.29), so the overlay also excludes the area *above* 78.29 (the hole inside the yellow band). |
| `graph_desirability.png` | Desirability Contour | Combined desirability (0-1) of the Desirability-page criteria; best region at high Time and Temperature about 69-72 °C (0.83 contour). The optimum star is not drawn because the best solution has Concentration 1.7975, not 2 (it is drawn only when it lies in the slice). |
| `graph_poe.png` | Propagation of Error (POE) | After entering sigma Temperature 1.0, Time 2.0, Concentration 0.05 (Factor & Response Info). POE of Yield ranges about 0.7-1.9 %; lowest (most robust) region at Temperature about 72-77 °C and Time above 55 min. The POE type appears in the list only after the sigmas are entered. |
| `graph_ternary.png` | Ternary Contour | **Edible film mixture**, Tensile Strength, in pseudo-components; vertices Starch 80 / Glycerol 40 / Chitosan 40 %; highest strength (about 15.2 MPa) toward the Starch-Chitosan edge; red points = design runs (2 = replicated). |
| `graph_trace.png` | Trace (Piepel) | **Edible film mixture**, Tensile Strength, from the design centroid: increasing Starch raises strength up to about +0.45, Glycerol lowers it steadily, Chitosan has little effect at first. |

## 6. Desirability optimization and prediction (BBD extraction)

| File | Page | Settings | What is visible / worth explaining |
|---|---|---|---|
| `desirability.png` | Desirability | Yield maximize, limits 75-78.29; Purity maximize, limits 89-93.6; shape 1, importance +++ (3), model RSM | Criteria table (factor limits "In range", response goal, limits, target, shape, importance, model). 5 solutions; best: Temperature 70.265, Time 89.781, Concentration 1.7975 -> Yield 78.29, Purity 92.416, D = 0.862. Bar chart: Yield d = 1.000, Purity d = 0.743, combined 0.862. Note that Yield sits exactly at its upper limit (d = 1 is reached, so the search does not push Yield higher). The empty rows below the two response rows of the criteria table hold no widgets (re-shot after the fix that removes widgets of a previously opened project). |
| `table_context_menu.png` | Desirability, right-click on the solutions table | row 1 selected | Table menu: Copy selected cells, ... with column headers, Copy entire table (for Excel), Copy entire table - decimal ',', Export table... (available on every result table). |
| `prediction.png` | Prediction & Confirmation | Best desirability solution sent from the Desirability page; 95 %, n = 3 confirmation runs; confirmation means typed in: Yield 78.69, Purity 92.12 | Yield prediction 78.29 (SE 0.430, CI 77.27-79.31, PI 76.88-79.70); Purity 92.42 (SE 0.224, CI 91.89-92.95, PI 91.68-93.15). Both confirmation means fall inside the PI: "✓ Confirmed (inside PI)" in green. Explanatory text on CI vs PI, the transformed scale, the link scale (logit or log) of GLM responses with no PI, "Typed confirmation means are cleared when a different point is sent to this page.", and ANN confirmation (±5 % of data range). |

## 7. NSGA-II / NSGA-III (BBD extraction, Yield and Purity maximized)

Settings for all: population 100, max generations 300, P crossover 0.9, eta crossover 15, eta mutation 20, seed 1,
algorithm Automatic, automatic stop on (0.1 % within 20 generations), model RSM for both objectives. The note under
the buttons, visible in every NSGA shot, reads "Factor limits are taken from the Desirability page. Response lower/upper
limits (optional) become constraints. ...". Result line:
"NSGA-II: 99 Pareto solutions, automatic stop at generation 78. Best compromise (TOPSIS): Temperature = 71.527,
Time = 88.239, Concentration = 1.8783 -> Yield = 79.157, Purity = 91.802".

| File | Graph / tab | What is visible / worth explaining |
|---|---|---|
| `nsga_front.png` | Graph "Pareto front (2D / 3D / matrix)" | 2D front Purity vs Yield coloured by TOPSIS score, TOPSIS star and knee diamond; clear trade-off (Purity 95 at Yield 63 down to Purity 89.5 at Yield 80.4). Solutions table on the left with Mark column. |
| `nsga_front2d.png` | "Pareto 2D (choose axes)" | Same front with the X / Y / Color / Size selectors visible. |
| `nsga_matrix.png` | "Objective matrix (each pair)" | Pair-plot of the objectives (useful mainly for 3+ objectives). |
| `nsga_parallel.png` | "Parallel coordinates" | Each Pareto solution as a line across factors and objectives; TOPSIS/knee highlighted. |
| `nsga_convergence.png` | "Convergence (hypervolume)" | Hypervolume (% of max) rising and flattening near 84 %; lower panel: mean hypervolume gain and ideal-point shift against the tolerance; green dotted line = criterion first met (generation 58), red line = stop (78). |
| `nsga_population_check.png` | Tab "Convergence & Adequacy" after "Check Population Size" | Text: 78 generations run, converged (last change -0.029 %, limit 0.10 %), criterion first met at generation 58; Quick Assessment "Enough generations" and "All 100 individuals are on the first front (normal for 2 objectives)"; "How to Read" explains the generation and population criteria. The result of the population check itself (2 other seeds and a 2x population, hypervolume per run) follows further down in the same report (scroll down in the app; not visible in this crop of the page). |
| `nsga_compare_front.png` | "ANN vs RSM: 2D front" (after "Compare ANN vs RSM") | RSM front (blue, 99 solutions) and ANN front (orange, 100 solutions) with their TOPSIS choices; hollow stars = each TOPSIS choice predicted by the other model. The ANN front reaches Yield about 84, beyond the RSM front. |
| `nsga_compare_cross.png` | "ANN vs RSM: cross-prediction" | Each Pareto solution predicted by both models; Yield differs by 22.4 % on average, Purity by 4.3 %; dashed line = agreement. |
| `nsga_compare_factors.png` | "ANN vs RSM: factors" | Factor settings of both fronts as parallel coordinates (Temperature, Time, Concentration); the ANN front prefers lower Temperature and Time at its upper limit. |
| `nsga_compare_report.png` | Tab "ANN vs RSM" | Model quality per response: RSM Pred R² 0.8857 / 0.9302; ANN test R² -0.7466 / -9.71; ANN cross-validation Q² 0.6244 / 0.7742. Front size, generations (78 vs 86), hypervolume by own model (71.5 % vs 82.4 %) and by the other model (69.0 % vs 57.6 %). **Worth explaining:** with only 17 runs the ANN single test split has 2-3 runs, so its test R² is meaningless (negative); the cross-validated Q² is the number to use, and the RSM is the more trustworthy model for this case. The ANNs were found with "Optimize All Responses" (1 hidden layer: 4 neurons for Yield, 5 for Purity, both "Good fit"). Further down (not in this crop), the conclusion bullet reads "Prediction accuracy on new data: RSM mean Pred R² 0.908, ANN mean cross-validation Q² 0.699" (the cross-validated Q² is used because both ANNs come from a search). |

## 8. Sensitivity & parametric study (BBD extraction)

Reference point for all: "Best desirability solution" (Temperature 70.265, Time 89.7807, Concentration 1.7975),
model "Selected model of each response" (RSM).

| File | Tab | Settings | What is visible / worth explaining |
|---|---|---|---|
| `sens_parametric_all.png` | Parametric Study | Layout "All factors, one panel each", 11 points | One panel per factor with Yield (black, left axis) and Purity (blue, right axis); dotted line = reference value. Temperature drives the conflict: Yield rises from 66.6 to 79.9, Purity falls from 94.1 to 87.3 between 60 and 80 °C. Table of the values below. |
| `sens_parametric_one.png` | Parametric Study | Layout "One factor, one panel per response", vary Temperature 60-80 (11 points), curves for Time (3 values: 30, 60, 90) | One panel per response with one curve per Time value. |
| `sens_src.png` | Global Sensitivity | Method "Standardized coefficients (Latin hypercube)", 120 points | Standardized beta with p-values: Yield: Temperature +0.579, Time +0.555, Concentration +0.168; Purity: Temperature -0.734, Time +0.355, Concentration -0.367; R² of the linear fit 0.701 / 0.860 (a low R² warns that the response is nonlinear). |
| `sens_sobol.png` | Global Sensitivity | Method "Sobol indices (variance-based)", N = 8192 | First-order S1 and total ST with 95 % bootstrap bars. Yield: Temperature S1 0.562 / ST 0.599, Time 0.326 / 0.364, Concentration 0.074 / 0.076; sum of S1 = 0.961 (almost additive). Purity: Temperature S1 0.676 / ST 0.714. |
| `sens_tornado.png` | Local Sensitivity (Tornado) | ±10 % of each range | Yield (reference 78.29): Temperature swing 2.338 (76.90 -> 79.24), Concentration 0.755, Time -0.290; Purity (reference 92.416): Temperature swing -1.398. Blue = low setting, orange = high setting. |
| `sens_front.png` | Pareto Front Trade-off | Last NSGA-II result | Spearman correlation of each factor with each objective along the front: Temperature +0.998 with Yield / -0.998 with Purity, Concentration +0.980 / -0.980, Time -0.783 / +0.783; Yield and Purity perfectly conflicting (rho = -1); plots of each factor and the trade-off rate dPurity/dYield along the front. Title "Pareto front trade-off (99 solutions, NSGA-II)". The trade-off rate is computed between the means of 12 groups of neighbouring solutions (11 points): -0.03 at Yield 65.7, -0.25 at 74.4, -0.48 at 77.5, -0.92 at 79.3, -3.36 at 80.2. |

## 9. ANN (dye adsorption, 40 historical runs, response Removal Efficiency)

Network settings visible on the left in all ANN shots: Bayesian Regularization (trainbr), mapminmax normalization,
tansig, early stopping with 6 validation failures, 70/15/15 % split, 10 restarts, 500 epochs, seed 1, 1 CPU core
("1 of 28" on the machine used). The left panel is cut at the right edge at this window width (combo texts end in
"- re..."); that is the app layout.

| File | Tab | Settings | What is visible / worth explaining |
|---|---|---|---|
| `ann_adequacy.png` | Data Adequacy | default (hidden "5") | Verdict "Adequate with caveats" (orange, also in the left panel). Checks: 40 runs Good; training rows / weights ratio 2.62 Caution (4-2-1 network, 13 weights); factor levels 40/39/40/40 Good; signal/noise "cannot be calculated" (no replicates); response skewness 0.85, max/min 21.4 Caution, with the explanation "Highly skewed (|skewness| > 1) or max/min > 10: errors at large values will dominate training. A response transform (e.g. ln) applies only to the RSM model (RSM Analysis page); the ANN is trained on the response in its original units."; factor correlation max 0.25 Good. Response histogram and check summary (5 Good, 2 Caution, 0 Not adequate, 3 Info). |
| `ann_learning_curve.png` | Learning Curve | Architecture from the entry "5" (1 hidden layer, 5 neurons), fixed 2 restarts / 300 epochs | ANN test R² rises from about 0 at 10 training runs to 0.876 at 32; RSM test R² only 0.429. Orange text: "More data would still help ... about 100 runs in total (range 95 to 111; add about 60 runs)", followed by the warning that this run estimate is unreliable (fitted exponent at the edge of its range). |
| `ann_search.png` | Architecture Search | Default search settings: no layer/neuron/weight limit, 5-fold CV, patience 2 | 15 architectures evaluated; best per depth: 1 layer, 3 neurons (4-3-1, 19 weights, training R² 0.9872, test R² 0.9381, Good fit) and 2 layers 3/3 (31 weights, test R² 0.9277); RSM test R² 0.3755. "Selected: 1 hidden layers, neurons 3 ..."; 9 overfitting and 3 underfitting candidates excluded. Scatter of training vs test R² against network size, bar chart per depth. Search time about 5 s on 1 core. |
| `ann_summary.png` | Summary & Diagnosis | Selected 4-3-1 network | Architecture, 19 weights, data split 28/6/6 (seed 1), 21 epochs with early stopping, "effective parameters γ not estimated yet (best epoch 7 lies within the warm-up; γ is first estimated after epoch 15)"; fit diagnosis "Good fit" from 5-fold CV (training R² 0.9872, validation 0.9313, test 0.9381). |
| `ann_regression.png` | Regression Plots | 4-3-1 | Output vs target for training (R = 0.9870), validation (R = 0.9979), test (R = 0.9505) and all data (R = 0.9809). |
| `ann_performance.png` | Training Performance | 4-3-1 | MSE (normalized, log scale) per epoch for training, validation and test; weights used at epoch 7 (dotted line), pink band = validation not improving until the stop. |
| `ann_importance.png` | Factor Importance (Garson) | 4-3-1 | pH 68.2 %, Initial Concentration 13.3 %, Contact Time 11.1 %, Adsorbent Dose 7.4 % - matches the pH optimum built into this case. |
| `ann_rsm_vs_ann.png` | RSM vs ANN | 4-3-1 vs the RSM quadratic | Predicted vs actual (RSM R² 0.8769, ANN R² 0.9614) and residuals per run; RSM has large residuals (up to +16) and even a negative prediction, the ANN stays within about ±6 except one run. |
| `ann_diagram.png` | Network Diagram | 4-3-1 | Inputs A-D, 3 tansig hidden neurons, 1 linear output; blue = positive, red = negative weight, thickness proportional to the weight. |
| `ann_weights.png` | Weights & Biases | 4-3-1 | Input-to-hidden weights and biases (IW, b1), hidden-to-output weights (LW, b output), and the normalization parameters (min/max of training + validation data) needed to reproduce the network outside REGEN. |

## Not captured / limitations

- Windows title bars of the main window and dialogs (grabs are of the client area only).
- Message boxes (e.g. "Simulation: 17 response values filled") are answered automatically and not captured; none
  were raised during the run.
- Combo-box drop-down lists and tooltips are not captured (they are popups like menus; the context menus above are
  the only popups composited into the screenshots).
- The ANN "Combined" mode, the combined mixture+process case (cake dough) and the categoric extraction case are
  not shown.
