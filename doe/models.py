"""Polynomial & Scheffé models, least-squares regression, ANOVA, transforms, Box-Cox, effects, intervals.

Term representation:
- monomial: tuple of exponents per factor, e.g. (1, 1, 0) = AB, (2, 0, 0) = A²
- full cubic mixture term: ("D", i, j) = x_i * x_j * (x_i - x_j)

Categorical factors are stored as level indices (0..L-1) in the coded matrix and expanded into
L-1 columns with effect coding (last level = -1 in all columns).
"""
import itertools
from dataclasses import dataclass, field
from math import comb

import numpy as np
from scipy import stats

from .designs import LETTERS

MODEL_ORDERS = {
    "mean": "Mean",
    "linear": "Linear",
    "2fi": "2FI (2-Factor Interaction)",
    "3fi": "3FI (3-Factor Interaction)",
    "quadratic": "Quadratic",
    "cubic": "Cubic",
    "special_cubic": "Special Cubic",
}

TRANSFORMS = {
    "none": "None",
    "sqrt": "Square root",
    "ln": "Natural log (ln)",
    "log10": "Base 10 log",
    "inverse": "Inverse (1/y)",
    "power": "Power (y^λ)",
}

SUPERSCRIPT = {2: "²", 3: "³"}
PROCESS_FACTORIAL = ("full_factorial", "fractional_factorial", "general_factorial")
MIXTURE = ("simplex_lattice", "simplex_centroid", "optimal_mixture", "combined_optimal", "historical_mixture")
SCREENING = ("plackett_burman", "taguchi")


# ------------------------------------------------------------------ factor space
@dataclass
class Space:
    """Type of each factor: cat[i] = 0 for numeric, L for categorical with L levels."""
    cat: list
    level_names: list

    @classmethod
    def numeric(cls, k):
        return cls([0] * k, [None] * k)

    @property
    def k(self):
        return len(self.cat)

    @property
    def has_categoric(self):
        return any(self.cat)


def _space(space, k):
    return space if space is not None else Space.numeric(k)


def effect_code(idx, L):
    """Effect coding: n x (L-1)."""
    idx = np.rint(np.asarray(idx, float)).astype(int)
    D = np.zeros((len(idx), L - 1))
    for j in range(L - 1):
        D[idx == j, j] = 1.0
    D[idx == L - 1, :] = -1.0
    return D


# ------------------------------------------------------------------ term
def normalize_term(t):
    t = tuple(t)
    if len(t) == 3 and t[0] == "D":
        return ("D", int(t[1]), int(t[2]))
    return tuple(int(e) for e in t)


def is_delta(t):
    return len(t) == 3 and t[0] == "D"


def term_exps(t, k):
    """'Equivalent' exponents (for ordering & hierarchy). A delta term counts as x_i² x_j."""
    if is_delta(t):
        e = [0] * k
        e[t[1]], e[t[2]] = 2, 1
        return tuple(e)
    return t


def is_intercept(t):
    return not is_delta(t) and sum(t) == 0


def degree(t):
    return 3 if is_delta(t) else sum(t)


def _term_sort_key(t):
    if is_delta(t):
        return (3, 9, (t[1], t[2]))
    return (sum(t), max(t) if t else 0, tuple(-e for e in t))


def order_label(order):
    if "|" in order:
        a, b = order.split("|")
        return f"{MODEL_ORDERS[a]} × {MODEL_ORDERS[b]}"
    return MODEL_ORDERS.get(order, order)


def combined_terms(comp_idx, proc_idx, k, order):
    """Combined mixture x process model: each Scheffé term is multiplied by each process term."""
    mix_order, proc_order = order.split("|")
    q = len(comp_idx)
    mix = [t for t in mixture_terms(q, mix_order) if not is_delta(t)]
    proc = model_terms(len(proc_idx), proc_order)
    out = []
    for pt in proc:
        for mt in mix:
            e = [0] * k
            for i, v in zip(comp_idx, mt):
                e[i] = v
            for i, v in zip(proc_idx, pt):
                e[i] = v
            out.append(tuple(e))
    return out


COMBINED_ORDERS = ["linear|mean", "linear|linear", "quadratic|linear", "linear|2fi", "quadratic|2fi",
                   "quadratic|quadratic", "special_cubic|linear", "special_cubic|2fi"]


def orders_for_design(design_type):
    if design_type == "combined_optimal":
        return COMBINED_ORDERS
    if design_type in MIXTURE:
        return ["linear", "quadratic", "special_cubic", "cubic"]
    if design_type in PROCESS_FACTORIAL or design_type == "plackett_burman":
        return ["linear", "2fi", "3fi"]
    return ["linear", "2fi", "quadratic", "cubic"]


def default_order(design_type):
    if design_type == "combined_optimal":
        return "quadratic|linear"
    if design_type in SCREENING:
        return "linear"
    return "2fi" if design_type in PROCESS_FACTORIAL else "quadratic"


def model_terms(k, order, space=None):
    """Process model terms (intercept first). Categorical factors have at most power 1."""
    if order == "mean":
        return [tuple([0] * k)]
    if "|" in order:
        raise ValueError("Combined models are only for combined mixture-process designs.")
    max_deg = {"linear": 1, "2fi": 2, "3fi": 3, "quadratic": 2, "cubic": 3}[order]
    max_exp = 1 if order in ("linear", "2fi", "3fi") else max_deg
    cat = _space(space, k).cat
    terms = []
    for e in itertools.product(range(max_exp + 1), repeat=k):
        if sum(e) <= max_deg and all(not cat[i] or e[i] <= 1 for i in range(k)):
            terms.append(tuple(e))
    return sorted(terms, key=_term_sort_key)


def mixture_terms(q, order):
    """Scheffé model terms (no intercept)."""
    if order == "mean":
        return [tuple([0] * q)]

    def mono(idx):
        e = [0] * q
        for i in idx:
            e[i] += 1
        return tuple(e)

    terms = [mono([i]) for i in range(q)]
    if order in ("quadratic", "special_cubic", "cubic"):
        terms += [mono(c) for c in itertools.combinations(range(q), 2)]
    if order in ("special_cubic", "cubic"):
        terms += [mono(c) for c in itertools.combinations(range(q), 3)]
    if order == "cubic":
        terms += [("D", i, j) for i, j in itertools.combinations(range(q), 2)]
    return terms


def term_name(t, names=None, levels=None):
    """Term name. `levels`: {factor index: level name} for a specific categorical column."""
    lab = (lambda i: LETTERS[i]) if names is None else (lambda i: names[i])
    if is_delta(t):
        i, j = t[1], t[2]
        if names is None:
            return f"{lab(i)}{lab(j)}({lab(i)}-{lab(j)})"
        return f"{lab(i)}*{lab(j)}*({lab(i)}-{lab(j)})"
    if sum(t) == 0:
        return "Intercept"
    parts = []
    for i, e in enumerate(t):
        if e:
            s = lab(i) + (SUPERSCRIPT.get(e, f"^{e}") if e > 1 else "")
            if levels and i in levels:
                s += f"[{levels[i]}]"
            parts.append(s)
    return ("*" if names is not None else "").join(parts)


def expand(coded, terms, space=None):
    """Model matrix. Returns (M, col_term, col_levels) - col_levels: {factor: level index}."""
    coded = np.atleast_2d(np.asarray(coded, float))
    n, k = coded.shape
    sp = _space(space, k)
    cols, col_term, col_levels = [], [], []
    for ti, t in enumerate(terms):
        if is_delta(t):
            xi, xj = coded[:, t[1]], coded[:, t[2]]
            cols.append(xi * xj * (xi - xj))
            col_term.append(ti)
            col_levels.append({})
            continue
        num = np.ones(n)
        cats = []
        for i, e in enumerate(t):
            if not e:
                continue
            if sp.cat[i]:
                cats.append(i)
            else:
                num = num * coded[:, i] ** e
        if not cats:
            cols.append(num)
            col_term.append(ti)
            col_levels.append({})
            continue
        dummies = [effect_code(coded[:, i], sp.cat[i]) for i in cats]
        for combo in itertools.product(*[range(sp.cat[i] - 1) for i in cats]):
            col = num.copy()
            for D, j in zip(dummies, combo):
                col = col * D[:, j]
            cols.append(col)
            col_term.append(ti)
            col_levels.append(dict(zip(cats, combo)))
    M = np.column_stack(cols) if cols else np.zeros((n, 0))
    return M, col_term, col_levels


def term_matrix(coded, terms, space=None):
    return expand(coded, terms, space)[0]


def contains(parent, child, k):
    """True if term `parent` is contained in `child` (A is contained in AB, A², ABC, ...)."""
    if parent == child or is_intercept(parent):
        return False
    pe, ce = term_exps(parent, k), term_exps(child, k)
    return all(p <= c for p, c in zip(pe, ce))


def remove_aliased(X, tol=1e-8):
    """Indices of columns that are not linearly dependent on the preceding columns.

    The rank check is done on R from the QR decomposition (X = QR): the singular values of X[:, S] equal those of
    R[:, S], so the result is identical without repeated SVDs of an n-row matrix (important for large data).
    """
    kept = []
    tol_abs = tol * (max(1.0, float(np.abs(X).max())) if X.size else 1.0) * max(X.shape)
    R = np.linalg.qr(X, mode="r") if X.shape[0] > X.shape[1] else X
    basis = np.zeros((R.shape[0], 0))
    for j in range(R.shape[1]):
        cand = np.column_stack([basis, R[:, j]])
        if np.linalg.matrix_rank(cand, tol=tol_abs) > basis.shape[1]:
            basis = cand
            kept.append(j)
    return kept


# ------------------------------------------------------------------ transforms
def transform_label(tr):
    if not tr or tr.get("kind", "none") == "none":
        return "None"
    if tr["kind"] == "power":
        return f"Power, λ = {tr.get('lam', 1):g}"
    return TRANSFORMS[tr["kind"]]


def apply_transform(y, tr):
    kind = (tr or {}).get("kind", "none")
    if kind == "none":
        return np.asarray(y, float)
    k = float((tr or {}).get("shift", 0.0))
    v = np.asarray(y, float) + k
    ok = v[~np.isnan(v)]
    if kind == "sqrt":
        if np.any(ok < 0):
            raise ValueError("The square root transform requires all response values ≥ 0.")
        return np.sqrt(v)
    if np.any(ok <= 0):
        raise ValueError(f"The {TRANSFORMS[kind]} transform requires all response values > 0.")
    if kind == "ln":
        return np.log(v)
    if kind == "log10":
        return np.log10(v)
    if kind == "inverse":
        return 1.0 / v
    if kind == "power":
        lam = float(tr.get("lam", 1.0))
        return np.log(v) if abs(lam) < 1e-12 else v ** lam
    raise ValueError(f"Unknown transform: {kind}")


def inverse_transform(z, tr):
    kind = (tr or {}).get("kind", "none")
    z = np.asarray(z, float)
    if kind == "none":
        return z
    k = float((tr or {}).get("shift", 0.0))
    with np.errstate(all="ignore"):
        if kind == "sqrt":
            v = np.where(z < 0, 0, z) ** 2
        elif kind == "ln":
            v = np.exp(z)
        elif kind == "log10":
            v = 10 ** z
        elif kind == "inverse":
            v = 1.0 / z
        else:
            lam = float(tr.get("lam", 1.0))
            v = np.exp(z) if abs(lam) < 1e-12 else np.sign(z) * np.abs(z) ** (1 / lam)
    return v - k


# ------------------------------------------------------------------ fit results
@dataclass
class AnovaRow:
    source: str
    ss: float
    df: int
    ms: float
    f: float = np.nan
    p: float = np.nan


@dataclass
class FitResult:
    terms: list             # terms in the model (including the intercept, if any)
    aliased: list
    beta: np.ndarray        # coefficients of all X columns (including blocks)
    term_idx: list          # positions of the term columns in X
    block_idx: list         # positions of the block columns in X
    exp_terms: list         # expanded terms (no intercept)
    exp_idx: list           # for each term column: expanded column index (-1 = intercept)
    col_terms: list         # for each term column: its term
    col_levels: list        # for each term column: {categorical factor: level index}
    space: Space
    se: np.ndarray
    vif: np.ndarray
    n: int
    p: int
    df_resid: int
    sse: float
    sst: float
    mse: float
    y: np.ndarray           # response (transformed scale)
    y_orig: np.ndarray      # response on the original scale
    yhat: np.ndarray
    resid: np.ndarray
    leverage: np.ndarray
    stud_int: np.ndarray
    stud_ext: np.ndarray
    cooks: np.ndarray
    dffits: np.ndarray
    xtx_inv: np.ndarray
    mixture: bool = False
    transform: dict = None
    ss_block: float = 0.0
    anova: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    rows_used: np.ndarray = None

    @property
    def coef(self):
        """Term column coefficients (without block effects)."""
        return self.beta[self.term_idx]

    @property
    def coef_se(self):
        return self.se[self.term_idx]

    @property
    def coef_vif(self):
        return self.vif[self.term_idx]

    def coef_labels(self, names=None):
        out = []
        for t, lv in zip(self.col_terms, self.col_levels):
            levels = {i: (self.space.level_names[i][j] if self.space.level_names[i] else j + 1)
                      for i, j in lv.items()}
            out.append(term_name(t, names, levels))
        return out

    def _rows(self, coded):
        coded = np.atleast_2d(np.asarray(coded, float))
        E = expand(coded, self.exp_terms, self.space)[0]
        full = np.zeros((coded.shape[0], self.p))
        for pos, ei in zip(self.term_idx, self.exp_idx):
            full[:, pos] = 1.0 if ei < 0 else E[:, ei]
        return full  # block columns = 0 -> average over blocks

    def predict_t(self, coded):
        return self._rows(coded) @ self.beta

    def predict(self, coded, original=True):
        z = self.predict_t(coded)
        return inverse_transform(z, self.transform) if original else z

    def leverage_at(self, coded):
        R = self._rows(coded)
        return np.einsum("ij,jk,ik->i", R, self.xtx_inv, R)

    def predict_se(self, coded):
        return np.sqrt(np.maximum(self.leverage_at(coded) * self.mse, 0))

    def t_crit(self, level=0.95):
        if self.df_resid <= 0:
            return np.nan
        return stats.t.ppf(0.5 + level / 2, self.df_resid)

    def _back(self, lo, hi):
        a = inverse_transform(lo, self.transform)
        b = inverse_transform(hi, self.transform)
        return np.minimum(a, b), np.maximum(a, b)

    def confidence_band(self, coded, level=0.95):
        """Confidence interval of the mean, original scale."""
        z = self.predict_t(coded)
        w = self.t_crit(level) * self.predict_se(coded)
        return self._back(z - w, z + w)

    def intervals(self, coded, n_obs=1, level=0.95):
        """Prediction + CI of the mean + PI for the mean of n_obs new observations (original scale)."""
        z = self.predict_t(coded)
        h = self.leverage_at(coded)
        t = self.t_crit(level)
        se_mean = np.sqrt(np.maximum(h * self.mse, 0))
        se_pred = np.sqrt(np.maximum(self.mse * (1.0 / max(n_obs, 1) + h), 0))
        return {"pred": inverse_transform(z, self.transform), "pred_t": z,
                "se_mean": se_mean, "se_pred": se_pred, "t": t,
                "ci": self._back(z - t * se_mean, z + t * se_mean),
                "pi": self._back(z - t * se_pred, z + t * se_pred)}


def _sse(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r)


BIG_N = 400          # above this, partial SS are computed from the Gram matrix (p x p), not an n-row lstsq


class _Gram:
    """SSE of sub-models (column subsets) from X'X and X'y - O(p³) per model, independent of the number of rows."""

    def __init__(self, X, y):
        yc = y - y.mean()
        self.G, self.Xy, self.yy = X.T @ X, X.T @ yc, float(yc @ yc)

    def sse(self, cols):
        cols = np.asarray(cols, int)
        G, Xy = self.G[np.ix_(cols, cols)], self.Xy[cols]
        b = np.linalg.pinv(G) @ Xy
        return max(self.yy - float(b @ Xy), 0.0)


def _sse_sub(X, y, cols, gram=None):
    if gram is None:
        return _sse(X[:, cols], y)
    return gram.sse(cols)       # centered y: correct because the intercept column is always in `cols`


def replicate_groups(coded, blocks=None, decimals=6):
    """Group index of identical points (same factors, and same block if any) for each row."""
    key = np.round(np.asarray(coded, float), decimals)
    if blocks is not None:
        key = np.column_stack([np.asarray(blocks, float), key])
    if not len(key):
        return np.zeros(0, int)
    return np.unique(key, axis=0, return_inverse=True)[1].ravel()


def pure_error(coded, y, blocks=None, decimals=6):
    y = np.asarray(y, float)
    g = replicate_groups(coded, blocks, decimals)
    if not len(g):
        return 0.0, 0
    cnt = np.bincount(g)
    mean = np.bincount(g, weights=y) / cnt
    rep = cnt[g] > 1
    ss = float(((y[rep] - mean[g[rep]]) ** 2).sum())
    return ss, int((cnt[cnt > 1] - 1).sum())


def block_columns(blocks):
    """Sum-to-zero coding for blocks: b blocks -> b-1 columns."""
    if blocks is None:
        return np.zeros((0, 0)), []
    levels = sorted(set(blocks.tolist()))
    if len(levels) < 2:
        return np.zeros((len(blocks), 0)), levels
    Z = np.zeros((len(blocks), len(levels) - 1))
    for j, lev in enumerate(levels[:-1]):
        Z[blocks == lev, j] = 1.0
        Z[blocks == levels[-1], j] = -1.0
    return Z, levels


class DesignColumns:
    """Full model matrix (intercept, blocks, terms) + column -> term mapping."""
    pass


def design_columns(coded, terms, mixture=False, blocks=None, space=None):
    coded = np.asarray(coded, float)
    n, k = coded.shape
    sp = _space(space, k)
    terms = [normalize_term(t) for t in terms]
    intercept = tuple([0] * k)
    body = [t for t in terms if not is_intercept(t)]
    use_intercept = not mixture or not body
    Z, _ = block_columns(None if blocks is None else np.asarray(blocks))
    E, e_term, e_levels = expand(coded, body, sp)
    labels, cols = [], []
    if use_intercept:
        labels.append(("int", -1))
        cols.append(np.ones((n, 1)))
    for j in range(Z.shape[1]):
        labels.append(("block", j))
        cols.append(Z[:, [j]])
    for j in range(E.shape[1]):
        labels.append(("term", j))
    if E.shape[1]:
        cols.append(E)
    X_all = np.hstack(cols) if cols else np.zeros((n, 0))
    kept = remove_aliased(X_all)
    labels = [labels[i] for i in kept]
    dc = DesignColumns()
    dc.X = X_all[:, kept]
    dc.term_idx, dc.exp_idx, dc.col_terms, dc.col_levels, dc.block_idx = [], [], [], [], []
    for pos, (kind, j) in enumerate(labels):
        if kind == "int":
            dc.term_idx.append(pos)
            dc.exp_idx.append(-1)
            dc.col_terms.append(intercept)
            dc.col_levels.append({})
        elif kind == "block":
            dc.block_idx.append(pos)
        else:
            dc.term_idx.append(pos)
            dc.exp_idx.append(j)
            dc.col_terms.append(body[e_term[j]])
            dc.col_levels.append(e_levels[j])
    kept_terms = set(dc.col_terms)
    dc.fterms = ([intercept] if use_intercept else []) + [t for t in body if t in kept_terms]
    dc.aliased = [t for t in body if t not in kept_terms]
    dc.body, dc.use_intercept, dc.intercept, dc.space = body, use_intercept, intercept, sp
    dc.term_positions = {}
    for pos, t in zip(dc.term_idx, dc.col_terms):
        if not is_intercept(t):
            dc.term_positions.setdefault(t, []).append(pos)
    return dc


def fit_model(coded, y, terms, mixture=False, blocks=None, transform=None, space=None):
    """Least-squares regression + Type III ANOVA (partial SS per term)."""
    coded = np.asarray(coded, float)
    y_orig = np.asarray(y, float)
    mask = ~np.isnan(y_orig) & np.all(np.isfinite(coded), axis=1)
    coded, y_orig = coded[mask], y_orig[mask]
    y = apply_transform(y_orig, transform)
    blk = None if blocks is None else np.asarray(blocks)[mask]
    n, k = len(y), coded.shape[1]
    sp = _space(space, k)
    if n < 2:
        raise ValueError("Not enough response data (at least 2 values).")

    dc = design_columns(coded, terms, mixture, blk, sp)
    X, p = dc.X, dc.X.shape[1]
    term_idx, exp_idx, col_terms, col_levels = dc.term_idx, dc.exp_idx, dc.col_terms, dc.col_levels
    block_idx, fterms, aliased, body = dc.block_idx, dc.fterms, dc.aliased, dc.body
    use_intercept, intercept = dc.use_intercept, dc.intercept

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    yhat = X @ beta
    resid = y - yhat
    sse = float(resid @ resid)
    ybar = y.mean()
    sst = float(((y - ybar) ** 2).sum())
    df_resid = n - p
    mse = sse / df_resid if df_resid > 0 else np.nan

    se = np.sqrt(np.maximum(np.diag(xtx_inv) * mse, 0)) if df_resid > 0 else np.full(p, np.nan)
    vif = np.full(p, np.nan)
    if not (mixture and not use_intercept):
        for pos, ei in zip(term_idx, exp_idx):
            if ei < 0:
                continue
            cj = X[:, pos] - X[:, pos].mean()
            vif[pos] = xtx_inv[pos, pos] * float(cj @ cj)

    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    with np.errstate(divide="ignore", invalid="ignore"):
        stud_int = resid / np.sqrt(mse * (1 - leverage))
        stud_int[leverage > 1 - 1e-10] = np.nan
        stud_ext = stud_int * np.sqrt((df_resid - 1) / np.maximum(df_resid - stud_int ** 2, 1e-12)) \
            if df_resid > 1 else np.full(n, np.nan)
        cooks = stud_int ** 2 * leverage / (p * (1 - leverage))
        dffits = stud_ext * np.sqrt(leverage / (1 - leverage))

    gram = None
    if n > BIG_N and not (mixture and not use_intercept):
        gram = _Gram(X, y)          # the model has an intercept -> SSE from centered y is still correct
    ss_block = 0.0
    if block_idx:
        ss_block = sst - _sse(np.column_stack([np.ones(n), X[:, block_idx]]), y)
    df_block = len(block_idx)

    # columns per term
    term_positions = {}
    for pos, t in zip(term_idx, col_terms):
        if not is_intercept(t):
            term_positions.setdefault(t, []).append(pos)

    anova = []
    is_mix = mixture and not use_intercept
    df_model = (p - 1) if is_mix else sum(len(v) for v in term_positions.values())
    ssm = sst - ss_block - sse
    if df_resid > 0 and df_model > 0:
        if block_idx:
            anova.append(AnovaRow("Block", ss_block, df_block, ss_block / df_block))
        msm = ssm / df_model
        anova.append(AnovaRow("Model", ssm, df_model, msm, msm / mse, stats.f.sf(msm / mse, df_model, df_resid)))
        if is_mix:
            lin = [pos for t, v in term_positions.items() if degree(t) == 1 for pos in v]
            other = [j for j in range(p) if j not in lin]
            X_red = np.column_stack([np.ones(n)] + [X[:, [j]] for j in other])
            ss_lin = max(_sse(X_red, y) - sse, 0.0)
            df_lin = len(lin) - 1
            if df_lin > 0:
                f_lin = (ss_lin / df_lin) / mse
                anova.append(AnovaRow("Linear Mixture", ss_lin, df_lin, ss_lin / df_lin, f_lin,
                                      stats.f.sf(f_lin, df_lin, df_resid)))
        for t in fterms:
            if is_intercept(t) or (is_mix and degree(t) == 1):
                continue
            pos = term_positions[t]
            keep = [c for c in range(p) if c not in pos]
            ss_t = max(_sse_sub(X, y, keep, gram) - sse, 0.0)
            df_t = len(pos)
            f_t = (ss_t / df_t) / mse
            anova.append(AnovaRow(term_name(t), ss_t, df_t, ss_t / df_t, f_t, stats.f.sf(f_t, df_t, df_resid)))
        anova.append(AnovaRow("Residual", sse, df_resid, mse))
        sspe, dfpe = pure_error(coded, y, blk)
        df_lof = df_resid - dfpe
        if dfpe > 0 and df_lof > 0:
            sslof = max(sse - sspe, 0.0)
            ms_lof, ms_pe = sslof / df_lof, sspe / dfpe
            f_lof = ms_lof / ms_pe if ms_pe > 0 else np.inf
            anova.append(AnovaRow("Lack of Fit", sslof, df_lof, ms_lof, f_lof,
                                  stats.f.sf(f_lof, df_lof, dfpe) if np.isfinite(f_lof) else 0.0))
            anova.append(AnovaRow("Pure Error", sspe, dfpe, ms_pe))
        anova.append(AnovaRow("Corrected Total", sst, n - 1, np.nan))

    st = {"n": n, "p": p, "mean": ybar}
    denom = sst - ss_block
    if df_resid > 0:
        sd = np.sqrt(mse)
        st["std_dev"] = sd
        st["cv"] = 100 * sd / abs(ybar) if ybar != 0 else np.nan
        st["r2"] = 1 - sse / denom if denom > 0 else np.nan
        st["adj_r2"] = 1 - mse / (denom / (df_model + df_resid)) if denom > 0 else np.nan
        if np.any(leverage > 1 - 1e-10):
            st["pred_r2"] = st["press"] = np.nan
        else:
            press = float(((resid / (1 - leverage)) ** 2).sum())
            st["press"] = press
            st["pred_r2"] = 1 - press / denom if denom > 0 else np.nan
        mean_var = p * mse / n
        st["adeq_precision"] = (yhat.max() - yhat.min()) / np.sqrt(mean_var) if mean_var > 0 else np.nan
        if 3 <= n <= 5000:
            st["shapiro_p"] = float(stats.shapiro(resid).pvalue) if np.ptp(resid) > 0 else np.nan

    return FitResult(terms=fterms, aliased=aliased, beta=beta, term_idx=term_idx, block_idx=block_idx,
                     exp_terms=body, exp_idx=exp_idx, col_terms=col_terms, col_levels=col_levels, space=sp,
                     se=se, vif=vif, n=n, p=p, df_resid=df_resid, sse=sse, sst=sst, mse=mse, y=y,
                     y_orig=y_orig, yhat=yhat, resid=resid, leverage=leverage, stud_int=stud_int,
                     stud_ext=stud_ext, cooks=cooks, dffits=dffits, xtx_inv=xtx_inv, mixture=is_mix,
                     transform=transform, ss_block=ss_block, anova=anova, stats=st,
                     rows_used=np.where(mask)[0])


# ------------------------------------------------------------------ model selection
def sequential_summary(order_fits):
    """order_fits: list of (order, FitResult) in order starting from 'mean'. Returns (rows, suggested)."""
    rows, prev = [], None
    for order, fit in order_fits:
        row = {"order": order, "fit": fit, "aliased": bool(fit.aliased), "seq_p": np.nan, "lof_p": np.nan}
        if prev is not None and fit.df_resid > 0:
            df_add = fit.p - prev.p
            if df_add > 0:
                f = ((prev.sse - fit.sse) / df_add) / fit.mse if fit.mse > 0 else np.inf
                row["seq_p"] = stats.f.sf(f, df_add, fit.df_resid) if np.isfinite(f) else 0.0
        for a in fit.anova:
            if a.source == "Lack of Fit":
                row["lof_p"] = a.p
        rows.append(row)
        prev = fit
    suggested = "mean"
    for r in rows[1:]:
        if r["aliased"]:
            break
        if not np.isnan(r["seq_p"]) and r["seq_p"] < 0.05:
            suggested = r["order"]
    return rows, suggested


def backward_elimination(fit_fn, terms, k, alpha_out=0.10, keep_hierarchy=True, mixture=False):
    """fit_fn(terms) -> FitResult. Remove the term with the largest p > alpha_out, one at a time."""
    terms = [normalize_term(t) for t in terms if not is_intercept(normalize_term(t))]
    while True:
        fit = fit_fn(terms)
        if fit.df_resid <= 0:
            return fit.terms
        terms = [t for t in fit.terms if not is_intercept(t)]
        pvals = {a.source: a.p for a in fit.anova}
        candidates = []
        for t in terms:
            if mixture and degree(t) == 1:
                continue
            pv = pvals.get(term_name(t), np.nan)
            if np.isnan(pv) or pv <= alpha_out:
                continue
            if keep_hierarchy and any(contains(t, o, k) for o in terms):
                continue
            candidates.append((pv, t))
        if not candidates:
            return fit.terms
        candidates.sort(key=lambda c: -c[0])
        terms.remove(candidates[0][1])


def box_cox(coded, y, terms, mixture=False, blocks=None, lambdas=None, space=None):
    """Box-Cox plot: ln(residual SS) versus λ. Requires all y > 0."""
    y = np.asarray(y, float)
    yy = y[~np.isnan(y)]
    if len(yy) < 3 or np.any(yy <= 0):
        return None
    lambdas = np.linspace(-3, 3, 121) if lambdas is None else np.asarray(lambdas)
    gm = float(np.exp(np.mean(np.log(yy))))
    # model matrix built once; residual SS for each λ = ||w||² − ||Q'w||² (projection onto the column space of X)
    coded = np.asarray(coded, float)
    mask = ~np.isnan(y) & np.all(np.isfinite(coded), axis=1)
    blk = None if blocks is None else np.asarray(blocks)[mask]
    dc = design_columns(coded[mask], terms, mixture, blk, _space(space, coded.shape[1]))
    Q = np.linalg.qr(dc.X)[0]
    df = len(yy) - dc.X.shape[1]
    ym = y[mask]
    lnss = []
    for lam in lambdas:
        w = gm * np.log(ym) if abs(lam) < 1e-9 else (ym ** lam - 1) / (lam * gm ** (lam - 1))
        r = w - Q @ (Q.T @ w)
        lnss.append(np.log(max(float(r @ r), 1e-300)))
    lnss = np.array(lnss)
    i = int(np.argmin(lnss))
    res = {"lambdas": lambdas, "lnss": lnss, "best": float(lambdas[i]), "min": float(lnss[i]),
           "ci": (np.nan, np.nan), "threshold": np.nan}
    if df and df > 0:
        thr = lnss[i] + stats.t.ppf(0.975, df) ** 2 / df
        inside = lambdas[lnss <= thr]
        res["threshold"] = float(thr)
        res["ci"] = (float(inside.min()), float(inside.max()))
    return res


def recommend_transform(bc):
    if bc is None:
        return None
    lo, hi = bc["ci"]
    if not np.isnan(lo) and lo <= 1 <= hi:
        return {"kind": "none"}
    for lam, kind in ((0.5, "sqrt"), (0.0, "ln"), (-1.0, "inverse")):
        if not np.isnan(lo) and lo <= lam <= hi:
            return {"kind": kind}
    return {"kind": "power", "lam": round(bc["best"], 2)}


# ------------------------------------------------------------------ factorial effects
def factorial_effects(coded, y, space=None, max_order=None):
    """Effects (= 2 x coefficient) of all interactions of 2-level numeric factors, plus Lenth limits.

    Returns a dict: terms, effects, pse, me, sme, df_resid, t (effect/SE or effect/PSE).
    """
    coded = np.asarray(coded, float)
    y = np.asarray(y, float)
    mask = ~np.isnan(y) & np.all(np.isfinite(coded), axis=1)
    coded, y = coded[mask], y[mask]
    k = coded.shape[1]
    sp = _space(space, k)
    if sp.has_categoric:
        raise ValueError("Effect plots are only for 2-level numeric factors.")
    corner = np.abs(coded) > 1e-9
    if np.any(np.abs(np.abs(coded[corner]) - 1) > 1e-6):
        raise ValueError("Effect plots are only for 2-level factorial designs.")
    max_order = max_order or k
    terms = sorted((tuple(e) for e in itertools.product((0, 1), repeat=k) if sum(e) <= max_order),
                   key=_term_sort_key)
    fit = fit_model(coded, y, terms)
    body = [t for t in fit.terms if not is_intercept(t)]
    pos = {t: p for p, t in zip(fit.term_idx, fit.col_terms)}
    coef = np.array([fit.beta[pos[t]] for t in body])
    eff = 2 * coef
    a = np.abs(eff)
    m = len(a)
    s0 = 1.5 * np.median(a) if m else np.nan
    small = a[a < 2.5 * s0]
    pse = 1.5 * np.median(small) if len(small) else s0
    d = max(m / 3.0, 1.0)
    me = stats.t.ppf(0.975, d) * pse
    gamma = (1 + 0.95 ** (1.0 / max(m, 1))) / 2
    sme = stats.t.ppf(gamma, d) * pse
    if fit.df_resid > 0:
        se_eff = 2 * np.array([fit.se[pos[t]] for t in body])
        tval = a / se_eff
        t_lim = stats.t.ppf(0.975, fit.df_resid)
        bonf = stats.t.ppf(1 - 0.05 / (2 * max(m, 1)), fit.df_resid)
    else:
        tval = a / pse if pse > 0 else np.full(m, np.nan)
        t_lim = stats.t.ppf(0.975, d)
        bonf = stats.t.ppf(gamma, d)
    return {"terms": body, "effects": eff, "pse": pse, "me": me, "sme": sme,
            "df_resid": fit.df_resid, "t": tval, "t_limit": t_lim, "bonferroni": bonf,
            "aliased": fit.aliased}


# ------------------------------------------------------------------ equations
def coded_to_actual_coefficients(terms, beta, lows, highs):
    """Coded numeric polynomial coefficients -> actual units."""
    lows = np.asarray(lows, float)
    highs = np.asarray(highs, float)
    c = (lows + highs) / 2
    h = np.where(highs - lows == 0, 2.0, highs - lows) / 2
    a = 1 / h
    b = -c / h
    out = {}
    for t, coef in zip(terms, beta):
        partial = {tuple([0] * len(t)): float(coef)}
        for j, e in enumerate(t):
            if e == 0:
                continue
            new = {}
            for exps, val in partial.items():
                for m in range(e + 1):
                    ne = list(exps)
                    ne[j] += m
                    ne = tuple(ne)
                    new[ne] = new.get(ne, 0.0) + val * comb(e, m) * a[j] ** m * b[j] ** (e - m)
            partial = new
        for exps, val in partial.items():
            out[exps] = out.get(exps, 0.0) + val
    items = sorted(out.items(), key=lambda kv: _term_sort_key(kv[0]))
    return [t for t, _ in items], np.array([v for _, v in items])


def categoric_combos(space):
    cats = [i for i, L in enumerate(space.cat) if L]
    if not cats:
        return [{}]
    return [dict(zip(cats, combo)) for combo in itertools.product(*[range(space.cat[i]) for i in cats])]


def numeric_polynomial(fit, combo):
    """Collapse categorical columns at a given level combination -> (numeric terms, coefficients)."""
    sp = fit.space
    acc = {}
    for t, lv, b in zip(fit.col_terms, fit.col_levels, fit.coef):
        mult = 1.0
        for i, j in lv.items():
            L = sp.cat[i]
            lev = combo[i]
            mult *= 1.0 if lev == j else (-1.0 if lev == L - 1 else 0.0)
        if mult == 0:
            continue
        nt = tuple(0 if sp.cat[i] else e for i, e in enumerate(t)) if not is_delta(t) else t
        acc[nt] = acc.get(nt, 0.0) + mult * b
    items = sorted(acc.items(), key=lambda kv: _term_sort_key(kv[0]))
    return [t for t, _ in items], np.array([v for _, v in items])


def actual_equations(fit, lows, highs):
    """Actual equations per categorical level combination: list of (combo, terms, coefficients)."""
    out = []
    for combo in categoric_combos(fit.space):
        t, b = numeric_polynomial(fit, combo)
        at, ab = coded_to_actual_coefficients(t, b, lows, highs)
        out.append((combo, at, ab))
    return out


def mixture_actual_coefficients(terms, beta, lows, total):
    """Scheffé pseudo-component coefficients -> actual components. Only when all lower limits = 0."""
    if np.any(np.abs(np.asarray(lows, float)) > 1e-12):
        return None
    return terms, np.array([b / total ** degree(t) for t, b in zip(terms, beta)])


def format_items(lhs, items, digits=5):
    """items: list of (label, coefficient)."""
    lines = [f"{lhs} ="]
    for lab, b in items:
        val = f"{b:+.{digits}g}"
        lines.append(f"   {val}" if lab in ("Intercept", "") else f"   {val} * {lab}")
    return "\n".join(lines)


def format_equation(response_name, terms, beta, names=None, digits=5, lhs=None):
    items = [("" if is_intercept(t) else term_name(t, names), b) for t, b in zip(terms, beta)]
    return format_items(lhs or response_name, items, digits)


# ------------------------------------------------------------------ propagation of error
def poe(fit, coded, sd_coded, h=1e-4):
    """Propagation of error (original scale): sqrt(Σ (∂y/∂z_i)² σ_z,i² + σ_resid²).

    sd_coded: standard deviation of each factor in coded units (0 = not included).
    """
    coded = np.atleast_2d(np.asarray(coded, float))
    sd = np.asarray(sd_coded, float)
    var = np.zeros(len(coded))
    for i in np.where(sd > 0)[0]:
        up, dn = coded.copy(), coded.copy()
        up[:, i] += h
        dn[:, i] -= h
        g = (fit.predict(up) - fit.predict(dn)) / (2 * h)
        var += (g * sd[i]) ** 2
    z = fit.predict_t(coded)
    dinv = (inverse_transform(z + h, fit.transform) - inverse_transform(z - h, fit.transform)) / (2 * h)
    resid_var = getattr(fit, "resid_var_for_poe", fit.mse)
    var += (np.abs(dinv) ** 2) * (resid_var if np.isfinite(resid_var) else 0.0)
    return np.sqrt(var)


class PoeModel:
    """Wrapper so POE can be used like a model (for optimization & graphs)."""

    def __init__(self, fit, sd_coded):
        self.fit, self.sd = fit, np.asarray(sd_coded, float)

    def predict(self, coded, original=True):
        return poe(self.fit, coded, self.sd)
