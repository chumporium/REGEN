"""Linear constraints (e.g. "2A - B >= 0", "A + B <= 60") and design region geometry.

Constraints are stored in actual units: coef · x (op) rhs. For computation they are converted to
coded / pseudo-component form: G z <= h.
"""
import itertools
import re

import numpy as np

from .designs import LETTERS

_OPS = ("<=", ">=", "=<", "=>", "<", ">")


def _parse_side(text, k):
    coef = np.zeros(k)
    const = 0.0
    s = text.replace(" ", "").replace(",", ".").upper()
    if not s:
        return coef, const
    if s[0] not in "+-":
        s = "+" + s
    for tok in re.findall(r"[+-][^+-]+", s):
        sign = -1.0 if tok[0] == "-" else 1.0
        body = tok[1:]
        m = re.fullmatch(r"(\d*\.?\d*)\*?([A-Z])", body)
        if m:
            letter = m.group(2)
            if letter not in LETTERS[:k]:
                raise ValueError(f"Factor '{letter}' does not exist (use letters {LETTERS[0]}–{LETTERS[k - 1]}).")
            num = m.group(1)
            coef[LETTERS.index(letter)] += sign * (float(num) if num not in ("", ".") else 1.0)
            continue
        try:
            const += sign * float(body)
        except ValueError:
            raise ValueError(f"Cannot read '{tok}'.") from None
    return coef, const


def parse(text, k):
    """'2A - B >= 0' -> {'text', 'coef', 'op', 'rhs'} in the form coef·x (op) rhs."""
    t = text.strip()
    for op in _OPS:
        if op in t:
            left, right = t.split(op, 1)
            break
    else:
        raise ValueError("A constraint must use <= or >=.")
    op = "<=" if op in ("<=", "=<", "<") else ">="
    cl, kl = _parse_side(left, k)
    cr, kr = _parse_side(right, k)
    coef = cl - cr
    rhs = kr - kl
    if not np.any(coef):
        raise ValueError("The constraint contains no factors.")
    return {"text": t, "coef": coef.tolist(), "op": op, "rhs": float(rhs)}


def parse_many(text, k):
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                out.append(parse(line, k))
            except ValueError as exc:
                raise ValueError(f"Line {i}: {exc}") from None
    return out


def to_le(cons):
    """All constraints in the form a·x <= b (actual units)."""
    rows, rhs = [], []
    for c in cons:
        a = np.asarray(c["coef"], float)
        if c["op"] == ">=":
            rows.append(-a)
            rhs.append(-c["rhs"])
        else:
            rows.append(a)
            rhs.append(c["rhs"])
    k = len(cons[0]["coef"]) if cons else 0
    return np.array(rows).reshape(-1, k), np.array(rhs)


def coded_system(cons, offset, scale):
    """Convert a·x <= b with x = offset + scale*z into G z <= h."""
    A, b = to_le(cons)
    if not len(A):
        return np.zeros((0, len(offset))), np.zeros(0)
    return A * scale, b - A @ offset


def feasible(z, G, h, tol=1e-9):
    z = np.atleast_2d(z)
    if not len(G):
        return np.ones(len(z), bool)
    return np.all(z @ G.T <= h + tol, axis=1)


# ------------------------------------------------------------------ geometry
def region_system(lo, hi, G, h, mixture):
    """Combine bounds & constraints: inequalities (Gi z <= hi), equalities (E z = e)."""
    d = len(lo)
    Gi = [np.vstack([-np.eye(d), np.eye(d)])]
    hi_ = [np.concatenate([-np.asarray(lo, float), np.asarray(hi, float)])]
    if len(G):
        Gi.append(G)
        hi_.append(h)
    Gi, hv = np.vstack(Gi), np.concatenate(hi_)
    if mixture:
        E, e = np.ones((1, d)), np.array([1.0])
    else:
        E, e = np.zeros((0, d)), np.zeros(0)
    return Gi, hv, E, e


def vertices(lo, hi, G, h, mixture, tol=1e-7):
    """Vertices of the polytope {lo<=z<=hi, Gz<=h, (Σz=1)}. Returns (V, active_sets)."""
    Gi, hv, E, e = region_system(lo, hi, G, h, mixture)
    d = len(lo)
    need = d - E.shape[0]
    found, actives = [], []
    for S in itertools.combinations(range(len(Gi)), need):
        A = np.vstack([Gi[list(S)], E])
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        z = np.linalg.solve(A, np.concatenate([hv[list(S)], e]))
        if np.all(Gi @ z <= hv + tol):
            key = tuple(np.round(z, 7))
            if key not in {tuple(np.round(f, 7)) for f in found}:
                found.append(z)
                actives.append(frozenset(np.where(np.abs(Gi @ z - hv) < tol)[0]))
    if not found:
        raise ValueError("The design region is empty: the bounds and constraints contradict each other.")
    return np.array(found), actives


def candidate_points(lo, hi, G, h, mixture, grid_levels=3):
    """Candidate points: vertices, edge midpoints, constraint face centers, centroid, axial points,
    plus a grid (for process factors)."""
    V, act = vertices(lo, hi, G, h, mixture)
    d = len(lo)
    dim = d - (1 if mixture else 0)
    pts, types = [v for v in V], ["Vertex"] * len(V)
    # edge midpoints: two vertices that share >= dim-1 active constraints
    for i, j in itertools.combinations(range(len(V)), 2):
        if len(act[i] & act[j]) >= dim - 1:
            pts.append((V[i] + V[j]) / 2)
            types.append("Edge midpoint")
    # constraint face centers
    Gi, hv, _, _ = region_system(lo, hi, G, h, mixture)
    for c in range(len(Gi)):
        on = [V[i] for i in range(len(V)) if c in act[i]]
        if len(on) >= 3:
            pts.append(np.mean(on, axis=0))
            types.append("Face center")
    centroid = V.mean(axis=0)
    pts.append(centroid)
    types.append("Centroid")
    for v in V:
        pts.append((v + centroid) / 2)
        types.append("Axial")
    if not mixture:
        for g in itertools.product(np.linspace(-1, 1, grid_levels), repeat=d):
            g = np.array(g)
            if np.all(g >= np.asarray(lo) - 1e-9) and np.all(g <= np.asarray(hi) + 1e-9):
                pts.append(g)
                types.append("Grid")
    P = np.array(pts)
    ok = feasible(P, G, h, 1e-7)
    P, types = P[ok], [t for t, o in zip(types, ok) if o]
    _, first = np.unique(np.round(P, 6), axis=0, return_index=True)
    first = np.sort(first)
    return P[first], [types[i] for i in first]


def sample_region(lo, hi, G, h, mixture, n, rng):
    """Uniform random points in the design region (for the I-optimal moment matrix)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    out, tries = [], 0
    while sum(len(o) for o in out) < n and tries < 60:
        m = max(n * 4, 2000)
        if mixture:
            free = 1.0 - lo.sum()
            z = lo + free * rng.dirichlet(np.ones(len(lo)), size=m)
            z = z[np.all(z <= hi + 1e-12, axis=1)]
        else:
            z = lo + rng.random((m, len(lo))) * (hi - lo)
        out.append(z[feasible(z, G, h)])
        tries += 1
    z = np.vstack(out) if out else np.zeros((0, len(lo)))
    return z[:n]
