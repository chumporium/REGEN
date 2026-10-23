# Changelog

Versions 1 to 6 were developed under the name DOE Studio.

## Unreleased
Sensitivity & Parametric Study (new page under Optimization)
- Reference point: design center, best desirability solution, NSGA TOPSIS choice or knee point, or custom
  values; model: the selected model of each response, RSM or ANN.
- Parametric study: every factor varied over its range in its own panel, all responses on separate vertical axes;
  or one factor with curves for a second factor, one panel per response.
- Global sensitivity: standardized regression coefficients β with p-values from a Latin hypercube sample (F-test
  for categoric factors with more than two levels, R² of the linear regression), or Sobol indices S1 and ST
  (Saltelli et al. 2010 and Jansen 1999 estimators) with bootstrap intervals.
- Local sensitivity: tornado chart and normalized sensitivity for ±δ of each factor's range.
- Pareto front trade-off: Spearman correlation of each factor with each objective along the front, correlation
  between the objectives, factor values along the front and the trade-off rate for two objectives.

Data export
- Every plot has "Copy data" and "Export data..." (toolbar and right-click): the data behind each panel are
  written to the clipboard, an Excel workbook (one sheet per series) or a CSV file, so the plot can be redrawn
  elsewhere. Contour, surface, overlay, desirability and POE plots export the full grid in actual units with the
  plotted values; confidence bands and box plots export their underlying values.
- Every result table has "Export table..." (Excel or CSV), and every report panel has "Export all tables..." (one
  sheet per table).

Design and data
- New Design: random seed option; the same seed gives the same run order and the same optimal, Latin
  hypercube and hybrid designs. The seed is saved with the project and shown on the Summary page.
- Excluded runs are left out of Design Evaluation and Data Normalization; Excel export and the PDF report mark
  them in an "Excluded" column.
- Mixture center points: the centroid of the (constrained) mixture region is added instead of the mean of the
  existing runs.
- Factorial to CCD augmentation states the fractional factorial core; rotatable alpha counts all factorial runs
  of the core, replicates included.
- Data Readiness: zero pure-error df is rated "bad"; the score footnote matches the verdicts.
- Summary: correct number of whole plots for split-plot designs; "POE σ" column header.
- Design Evaluation shows the scaled average prediction variance (n·I) next to I.
- Simulation: the dialog lists all accepted functions, "." is the decimal separator, min(a, b) and max(a, b)
  work, and noise σ is disabled for binomial and Poisson responses.

Analysis
- Fit Summary for combined mixture-process designs uses a nested sequence of models.
- Lack of fit is judged at p = 0.05 everywhere.
- Box-Cox uses the constant k and the standard confidence interval ln(1 + t²/ν); an interval that reaches the end
  of the λ range is reported as such.
- Effects plots are labelled Effect and |Effect| and show Lenth's ME and SME lines; Plackett-Burman designs show
  main effects only.
- "Maintain hierarchy" option for the term list and the effects plot.
- GLM: predicted vs actual on the response scale (proportion or count), convergence and separation warnings.
- Split-plot (REML): coefficient tests use the same denominator df as the Wald F tests; Cook's distance and DFFITS
  are not shown because they are not defined for REML fits.
- All Factors graph: reference level for each categoric factor.

ANN and optimization
- The noise level from replicated runs is used in the ANN diagnosis.
- After an architecture search, the status and suggestions follow the cross-validated diagnosis; the single
  train/test split of the final model is shown separately.
- Learning curve: warning when the run estimate is unreliable; the message states the test used.
- Desirability: shape exponent for each response (default 1, linear).
- Prediction: standard error labelled with its scale when a transformation is used; ANN confirmation compares
  the error with 5% of the response range.
- NSGA-II with four or more objectives: a note explains that the automatic stop is disabled.
- Search results, learning curves, NSGA results, front comparisons and desirability solutions are saved in the
  project file.

## 7.0.0
- The application is renamed REGEN (Response Engineering with GEnetic algorithms and Neural networks).
- The user interface, reports and source code are in English.
- `.doe` project files and the recent-files list of earlier versions are still read; the installer replaces
  DOE Studio.
- ANN vs RSM front comparison: the conclusion names the criterion that was not met (mean prediction difference
  or distance between the compromise choices) and states the limits used.

## 6.9
- Measurement uncertainty features removed; the noise level for ANN diagnosis is taken from pure error.

## 6.8
- Design Guide: recommended designs and run counts by goal, two-level factorial table (2 to 21 factors),
  run-size calculator for RSM and ANN.
- Maximin Latin hypercube and hybrid designs (CCD/BBD plus space-filling points).
- ANN learning curve for data sufficiency.

## 6.7
- Parallel ANN training and architecture search on several CPU cores (results identical to one core).
- NSGA-III for three or more objectives, automatic stop based on the hypervolume, population-size check.
- 2D Pareto plots with color and marker size for additional objectives; comparison of RSM and ANN fronts.

## 6.6
- Unlimited number of hidden layers and neurons; Adam for large networks.

## 6.5
- RSM or ANN model per response in Optimization and NSGA-II; objective matrix; parallel coordinates.

## 6.4
- Native Windows look: menu bar, toolbar and navigation tree.

## 6.3
- Design evaluation: coefficient correlation and matrix measures.
- Project summary, project notes, column graphs and coefficient table for all responses.

## 6.2
- Interactive diagnostics, ignoring runs without deleting data, graphs for all factors.
- Data readiness assessment with suggested extra runs.

## 6.1
- Large data sets; multi-output ANN; early stopping for weights, number of neurons and number of layers.

## 6.0
- Navigation by work stage, case study gallery, ANN data adequacy, automatic ANN architecture search,
  data normalization.

## 5.x
- Artificial neural networks (Levenberg-Marquardt and Bayesian regularization) and NSGA-II with TOPSIS.
- Excel/CSV import; copying tables to Excel/Word.

## 4.0
- Screening designs (Plackett-Burman, Definitive Screening, Taguchi), design evaluation, propagation of error,
  response simulation, logistic and Poisson regression, combined mixture-process designs, split-plot REML.

## 3.0
- D/I-optimal designs with constraints, categorical factors, historical data and augmentation, factorial effect
  plots, additional graphs, prediction and confirmation.

## 2.0
- Mixture designs, blocking, response transformations and Box-Cox, PDF report.

## 1.0
- Full and fractional factorials, CCD, Box-Behnken, ANOVA, contour and 3D plots, desirability optimization.
