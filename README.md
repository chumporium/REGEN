# REGEN

**R**esponse **E**ngineering with **GE**netic algorithms and **N**eural networks

A Windows desktop application for design of experiments (DOE): experimental design, response surface methodology
(RSM), artificial neural networks (ANN) and multi-objective optimization in one project.

## Workflow

1. **Design**: choose a design and the number of runs with the Design Guide.
2. **Data**: enter or import data, check data readiness, and add suggested runs.
3. **Modeling**: fit RSM and ANN to the same data; the ANN architecture is selected automatically by
   cross-validation.
4. **Data sufficiency**: a learning curve shows whether the ANN would still improve with more runs.
5. **Optimization**: desirability, or NSGA-II / NSGA-III with an automatic stop based on the hypervolume.
6. **Checking the result**: compare the Pareto fronts of the RSM and ANN models before running confirmation
   experiments.

## Features

**Experimental design**
- Two-level full and fractional factorials (2 to 21 factors, 4 to 512 runs, maximum-resolution and
  minimum-aberration generators, alias structure), Plackett-Burman, Definitive Screening Design, Taguchi arrays.
- Central composite (rotatable, face-centered, orthogonal, custom alpha), Box-Behnken, maximin Latin hypercube,
  and hybrid designs (CCD/BBD plus space-filling points).
- Mixture designs (simplex lattice, simplex centroid, component bounds), D/I-optimal designs with linear
  constraints, combined mixture-process designs, categorical factors, blocking, split-plot.
- Design Guide: recommended designs and run counts by goal (screening, characterization, RSM optimization,
  ANN, RSM + ANN, robustness testing) with power, and a run-size calculator.
- Design evaluation before any experiment: degrees of freedom, power, aliasing, VIF, coefficient correlation,
  D/A/G efficiency, FDS curves.

**Data**
- Spreadsheet-style data sheet (paste from Excel, decimal comma or point), Excel/CSV import with column roles,
  historical data, augmentation (center points, replicates, factorial to CCD, foldover).
- Data readiness assessment for modeling and optimization, with suggested extra runs that can be added directly.
- Large data sets (tested with 15,000 runs).

**Response surface methodology**
- Fit summary, manual term selection or backward elimination, Type III ANOVA, fit statistics (R², adjusted R²,
  predicted R², PRESS, adequate precision), coefficients with 95% CI and VIF, coded and actual equations.
- Scheffé mixture models, logistic and Poisson regression, split-plot REML, response transformations and Box-Cox.
- Interactive diagnostics (residuals, Cook's distance, leverage, DFFITS, DFBETAS), click a point to identify the
  run, ignore runs without deleting data.
- Contour and 3D surface plots, factor effects, interaction, cube, perturbation, ternary contour and surface plots.

**Artificial neural networks**
- Any number of hidden layers and neurons; Levenberg-Marquardt, Bayesian regularization, or Adam for large
  networks; normalization from the training data only.
- Automatic architecture search (cross-validation, early stopping on the number of neurons and layers,
  one-standard-error rule, underfitting/overfitting diagnosis).
- Learning curve for data sufficiency, separate or multi-output models, parallel training on several CPU cores.

**Optimization**
- Multi-response desirability (maximize, minimize, target, in range, importance), graphical optimization.
- NSGA-II and NSGA-III with response constraints, automatic stop based on the hypervolume, population-size check,
  TOPSIS or knee selection.
- 2D/3D Pareto fronts (color and marker size for the third and fourth objectives), objective matrix, parallel
  coordinates, and comparison of RSM and ANN fronts (hypervolume, cross-prediction, compromise solutions).
- RSM or ANN can be chosen per response for graphs, optimization and prediction.

**Reporting**
- Prediction with CI/PI and confirmation runs, PDF report, Excel export, copy tables directly to Excel/Word.

## Download

A Windows installer (Windows 10/11, 64-bit, no Python required) is available on the **Releases** page of this
repository.

## Running from source

Tested with Python 3.13 on Windows 11.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Self-test (the result is written to `selftest.log` in the output folder):

```powershell
.\.venv\Scripts\python.exe main.py --self-test <output-folder>
```

## Building the installer

1. Install [Inno Setup 6](https://jrsoftware.org/isdl.php).
2. Run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File build.ps1
   ```

3. The installer is written to `installer_output\`.

## Code structure

```
main.py                application entry point and self-test
doe/designs.py         design matrices
doe/fraction_table.py  fractional factorial generators
doe/planning.py        Design Guide and run-size calculator
doe/optimal.py         D/I-optimal designs
doe/constraints.py     linear constraints and design region
doe/evaluation.py      design evaluation (power, aliasing, FDS)
doe/models.py          regression, ANOVA, mixture models, transformations, Box-Cox
doe/glm.py             logistic and Poisson regression
doe/mixed.py           split-plot REML
doe/simulate.py        response simulation
doe/ann.py             neural networks, architecture search, learning curve
doe/parallel.py        multi-core training
doe/adequacy.py        data adequacy for ANN
doe/readiness.py       data readiness for modeling and optimization
doe/optimize.py        desirability and numerical optimization
doe/nsga2.py           NSGA-II, NSGA-III, hypervolume, TOPSIS
doe/project.py         project data, saving and loading .doe files
gui/                   user interface (PySide6)
build.ps1              builds the .exe and the installer
installer.iss          Inno Setup script
```

## Version history

See [CHANGELOG.md](CHANGELOG.md). Versions 1 to 6 were developed under the name DOE Studio; `.doe` project files
from those versions can still be opened.

## Authors

Erdiyanto Munandar and Nasruddin (corresponding author), Department of Mechanical Engineering, Faculty of
Engineering, Universitas Indonesia, Depok, Indonesia.

## License

REGEN is released under the MIT License (see `LICENSE`). Third-party libraries keep their own licenses; the user
interface uses PySide6 (Qt for Python), which is licensed under the LGPL-3.0.
