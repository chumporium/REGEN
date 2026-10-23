"""Simulate responses from an equation (for practice, planning, and trying out analyses)."""
import ast

import numpy as np

from .designs import LETTERS

_FUNCS = {"exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10, "sqrt": np.sqrt, "sin": np.sin,
          "cos": np.cos, "tan": np.tan, "abs": np.abs, "min": np.minimum, "max": np.maximum, "pi": np.pi,
          "e": np.e}
_TWO_ARGS = ("min", "max")
DECIMAL_NOTE = ("Use a point as the decimal separator (for example 2.5); commas only separate the arguments "
                "of min and max.")
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call, ast.Add,
            ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.Compare, ast.Eq, ast.NotEq,
            ast.Lt, ast.Gt, ast.LtE, ast.GtE)


def evaluate_equation(text, variables, hint="Use the factor letters (A, B, ...)."):
    """Evaluate an equation safely (arithmetic, math functions, and factor variables only).

    The decimal separator is the point; a comma separates function arguments (min(A, B), max(A, B)).
    """
    expr = text.replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid equation: {exc.msg}" + (f". {DECIMAL_NOTE}" if "," in expr else "")) from None
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple):
            raise ValueError(f"Invalid equation: unexpected comma. {DECIMAL_NOTE}")
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"'{type(node).__name__}' is not allowed in an equation.")
        if isinstance(node, ast.Name) and node.id not in variables and node.id not in _FUNCS:
            raise ValueError(f"Unknown name '{node.id}'. {hint}")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS) or node.keywords:
                raise ValueError("Only basic math functions are allowed.")
            want = 2 if node.func.id in _TWO_ARGS else 1
            if len(node.args) != want:
                raise ValueError(f"{node.func.id}() takes {want} argument{'s' if want > 1 else ''}. "
                                 + DECIMAL_NOTE)
    return eval(compile(tree, "<equation>", "eval"), {"__builtins__": {}}, {**_FUNCS, **variables})  # noqa: S307


def noise_note(project, j, noise_sd):
    """Note shown when the noise σ is not used: binomial and Poisson responses get their own random variation."""
    kind = getattr(project.responses[j], "kind", "normal")
    if not noise_sd or noise_sd <= 0 or kind not in ("binomial", "poisson"):
        return None
    if kind == "binomial":
        return ("Noise (σ) was not used: a binomial response is drawn from the binomial distribution with the "
                "probability given by the equation.")
    return ("Noise (σ) was not used: a Poisson response is drawn from the Poisson distribution with the mean "
            "given by the equation.")


def simulate(project, j, equation, units="coded", noise_sd=0.0, seed=None, only_empty=False, notes=None):
    """Fill response j from an equation. units: 'coded' or 'actual'. Categoric = level index (0, 1, ...).

    Returns the number of values filled. Noise σ applies to normal responses only; if `notes` (list) is given,
    a note is appended when σ is ignored (see noise_note)."""
    rng = np.random.default_rng(seed)
    X = project.coded if units == "coded" else project.actual
    variables = {LETTERS[i]: X[:, i] for i in range(project.k)}
    mean = np.asarray(evaluate_equation(equation, variables), float) * np.ones(project.n)
    resp = project.responses[j]
    kind = getattr(resp, "kind", "normal")
    if kind == "binomial":
        prob = np.clip(mean, 0, 1)
        y = rng.binomial(int(resp.trials or 1), prob).astype(float)
    elif kind == "poisson":
        y = rng.poisson(np.maximum(mean, 0)).astype(float)
    else:
        y = mean + (rng.normal(0, noise_sd, project.n) if noise_sd > 0 else 0)
    target = project.data[:, j]
    fill = np.isnan(target) if only_empty else np.ones(project.n, bool)
    fill &= np.all(np.isfinite(project.coded), axis=1)
    target[fill] = y[fill]
    project.dirty = True
    note = noise_note(project, j, noise_sd)
    if note and notes is not None:
        notes.append(note)
    return int(fill.sum())
