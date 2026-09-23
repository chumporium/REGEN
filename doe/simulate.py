"""Simulate responses from an equation (for practice, planning, and trying out analyses)."""
import ast

import numpy as np

from .designs import LETTERS

_FUNCS = {"exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10, "sqrt": np.sqrt, "sin": np.sin,
          "cos": np.cos, "tan": np.tan, "abs": np.abs, "min": np.minimum, "max": np.maximum, "pi": np.pi,
          "e": np.e}
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call, ast.Add,
            ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.Compare, ast.Eq, ast.NotEq,
            ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.IfExp)


def evaluate_equation(text, variables, hint="Use the factor letters (A, B, ...)."):
    """Evaluate an equation safely (arithmetic, math functions, and factor variables only)."""
    expr = text.replace("^", "**").replace(",", ".")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid equation: {exc.msg}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"'{type(node).__name__}' is not allowed in an equation.")
        if isinstance(node, ast.Name) and node.id not in variables and node.id not in _FUNCS:
            raise ValueError(f"Unknown name '{node.id}'. {hint}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
            raise ValueError("Only basic math functions are allowed.")
    return eval(compile(tree, "<equation>", "eval"), {"__builtins__": {}}, {**_FUNCS, **variables})  # noqa: S307


def simulate(project, j, equation, units="coded", noise_sd=0.0, seed=None, only_empty=False):
    """Fill response j from an equation. units: 'coded' or 'actual'. Categoric = level index (0, 1, ...)."""
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
    return int(fill.sum())
