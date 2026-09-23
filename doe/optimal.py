"""Optimal (custom) designs: selecting runs from candidate points with an exchange algorithm.

The design structure follows the common pattern of DOE software:
- model points : chosen optimally (D or I) for the planned model
- lack of fit  : extra points farthest from the points already chosen
- replicates   : repeats of the model points with the highest leverage (for pure error)
"""
import itertools

import numpy as np

from . import models

CRITERIA = {"I": "I-optimal (best prediction)", "D": "D-optimal (best coefficients)"}


def _objective(X, W, criterion, ridge=1e-8):
    M = X.T @ X + ridge * np.eye(X.shape[1])
    if criterion == "D":
        sign, logdet = np.linalg.slogdet(M)
        return logdet if sign > 0 else -np.inf
    try:
        return -float(np.trace(np.linalg.solve(M, W)))
    except np.linalg.LinAlgError:
        return -np.inf


def exchange(C, n, W, criterion="I", rng=None, max_iter=300, ridge=1e-8):
    """Fedorov exchange: choose n rows from the candidate matrix C (Nc x p). Returns the indices."""
    rng = rng or np.random.default_rng()
    Nc, p = C.shape
    design = list(rng.choice(Nc, size=n, replace=Nc < n))
    for _ in range(max_iter):
        X = C[design]
        M = X.T @ X + ridge * np.eye(p)
        Minv = np.linalg.inv(M)
        A = C @ Minv                           # Nc x p
        djj = np.einsum("ij,ij->i", A, C)      # x_j' M^-1 x_j
        Dij = A @ X.T                          # Nc x n : x_j' M^-1 x_i
        dii = np.einsum("ij,ij->i", X @ Minv, X)
        s11 = 1 + djj[:, None]
        s22 = -1 + dii[None, :]
        det = s11 * s22 - Dij ** 2             # det(C + U'M^-1U), C = diag(1,-1)
        if criterion == "D":
            gain = -det                        # = det(M')/det(M)
            gain[design, :] = -np.inf          # avoid duplicate points
            j, i = np.unravel_index(np.argmax(gain), gain.shape)
            if gain[j, i] <= 1 + 1e-9:
                break
        else:
            G = Minv @ W @ Minv
            CG = C @ G
            gjj = np.einsum("ij,ij->i", CG, C)
            gij = CG @ X.T
            gii = np.einsum("ij,ij->i", X @ G, X)
            with np.errstate(divide="ignore", invalid="ignore"):
                red = (s22 * gjj[:, None] - 2 * Dij * gij + s11 * gii[None, :]) / det
            red[~np.isfinite(red)] = -np.inf
            red[np.abs(det) < 1e-12] = -np.inf
            red[design, :] = -np.inf
            j, i = np.unravel_index(np.argmax(red), red.shape)
            if red[j, i] <= 1e-10:
                break
        design[i] = j
    return design


def build(cands, terms, space, n_model, n_lof, n_rep, region, criterion="I", seed=None, n_starts=6,
          cand_types=None):
    """Returns (candidate_row_indices, point_types)."""
    rng = np.random.default_rng(seed)
    C = models.expand(cands, terms, space)[0]
    p = C.shape[1]
    if n_model < p:
        n_model = p
    kept = models.remove_aliased(C)
    if len(kept) < p:
        raise ValueError("There are not enough candidate points to estimate this model. "
                         "Relax the constraints or choose a simpler model.")
    F = models.expand(region, terms, space)[0]
    W = F.T @ F / len(F)
    best, best_val = None, -np.inf
    for _ in range(n_starts):
        d = exchange(C, n_model, W, criterion, rng)
        val = _objective(C[d], W, criterion)
        if val > best_val:
            best, best_val = d, val
    design = list(best)
    types = ["Model"] * len(design)

    # lack-of-fit points: farthest from the selected points
    chosen = set(design)
    P = _scaled(cands, space)
    for _ in range(n_lof):
        rest = [j for j in range(len(cands)) if j not in chosen]
        if not rest:
            break
        dist = np.min(np.linalg.norm(P[rest][:, None, :] - P[list(chosen)][None, :, :], axis=2), axis=1)
        j = rest[int(np.argmax(dist))]
        design.append(j)
        chosen.add(j)
        types.append("Lack of Fit")

    # replicates: model points with the highest leverage
    X = C[design]
    Minv = np.linalg.pinv(X.T @ X)
    lev = np.einsum("ij,jk,ik->i", X, Minv, X)
    order = [design[i] for i in np.argsort(-lev)]
    seen = []
    for j in order:
        if len(seen) >= n_rep:
            break
        if j not in seen:
            seen.append(j)
    design += seen
    types += ["Replicate"] * len(seen)
    return design, types


def _scaled(P, space):
    """Coordinates for distances: categoric factors are one-hot encoded so a level difference = distance 1."""
    P = np.asarray(P, float)
    if space is None or not space.has_categoric:
        return P
    cols = []
    for i, L in enumerate(space.cat):
        if L:
            cols.append(np.eye(L)[np.rint(P[:, i]).astype(int)] / np.sqrt(2))
        else:
            cols.append(P[:, [i]])
    return np.hstack(cols)


def cross_categoric(points, types, space):
    """Cross the numeric points with every combination of categoric levels."""
    cats = [i for i, L in enumerate(space.cat) if L]
    if not cats:
        return points, types
    out, out_t = [], []
    for combo in itertools.product(*[range(space.cat[i]) for i in cats]):
        P = points.copy()
        for i, lev in zip(cats, combo):
            P[:, i] = lev
        out.append(P)
        out_t += types
    return np.vstack(out), out_t
