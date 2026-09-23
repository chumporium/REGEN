"""Building experimental design matrices in coded units (-1 .. +1)."""
import itertools
from functools import lru_cache

import numpy as np

from .fraction_table import SEARCHED_GENERATORS

# Factor letters follow DOE convention (the letter I is skipped to avoid confusion with the number 1)
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"

DESIGN_TYPES = {
    "full_factorial": "2-Level Full Factorial",
    "fractional_factorial": "2-Level Fractional Factorial",
    "general_factorial": "General Factorial (Multilevel Categoric)",
    "ccd": "Central Composite Design (CCD)",
    "bbd": "Box-Behnken Design (BBD)",
    "optimal_rsm": "Optimal (Custom) - Process",
    "historical": "Historical Data (Process)",
    "historical_mixture": "Historical Data (Mixture)",
    "plackett_burman": "Screening: Plackett-Burman",
    "dsd": "Screening: Definitive Screening Design (DSD)",
    "taguchi": "Screening: Taguchi Orthogonal Array",
    "simplex_lattice": "Mixture: Simplex Lattice",
    "simplex_centroid": "Mixture: Simplex Centroid",
    "optimal_mixture": "Mixture: Optimal (Custom)",
    "combined_optimal": "Combined Mixture + Process (Optimal)",
    "lhs": "Space-filling: Latin Hypercube (for ANN)",
    "hybrid": "Hybrid RSM + Space-filling (for RSM & ANN)",
}

# number of NUMERIC factors allowed (categoric factors can be added separately)
FACTOR_LIMITS = {
    "full_factorial": (1, 9),
    "fractional_factorial": (3, 21),
    "general_factorial": (0, 0),
    "ccd": (2, 6),
    "bbd": (3, 7),
    "optimal_rsm": (1, 8),
    "historical": (1, 10),
    "historical_mixture": (2, 12),
    "plackett_burman": (2, 23),
    "dsd": (3, 20),
    "taguchi": (2, 15),
    "simplex_lattice": (3, 6),
    "simplex_centroid": (3, 6),
    "optimal_mixture": (2, 8),
    "combined_optimal": (2, 6),
    "lhs": (1, 20),
    "hybrid": (2, 7),
}

# design types that may include categoric factors
CATEGORIC_OK = ("full_factorial", "general_factorial", "ccd", "bbd", "optimal_rsm", "historical", "lhs", "hybrid")

# Standard fractional factorial 2^(k-p) generators: key (k, p). Up to 7 factors use textbook generators;
# the rest (up to 21 factors, 512 runs) come from doe/fraction_table.py.
FRACTION_GENERATORS = {
    (3, 1): ["AB"],
    (4, 1): ["ABC"],
    (5, 1): ["ABCD"],
    (5, 2): ["AB", "AC"],
    (6, 1): ["ABCDE"],
    (6, 2): ["ABC", "BCD"],
    (6, 3): ["AB", "AC", "BC"],
    (7, 1): ["ABCDEF"],
    (7, 2): ["ABCD", "ABDE"],
    (7, 3): ["ABC", "BCD", "ACD"],
    (7, 4): ["AB", "AC", "BC", "ABC"],
}
FRACTION_GENERATORS = {**SEARCHED_GENERATORS, **FRACTION_GENERATORS}
FACTORIAL_RUNS = (4, 8, 16, 32, 64, 128, 256, 512)

# Box-Behnken blocks for k >= 6 (factor indices start at 0)
BBD_BLOCKS = {
    6: [(0, 1, 3), (1, 2, 4), (2, 3, 5), (0, 3, 4), (1, 4, 5), (0, 2, 5)],
    7: [(3, 4, 5), (0, 5, 6), (1, 4, 6), (0, 1, 3), (2, 3, 6), (0, 2, 4), (1, 2, 5)],
}


def default_center_points(design_type, k):
    if design_type == "ccd":
        return {2: 5, 3: 6, 4: 6, 5: 6, 6: 9}.get(k, 6)
    if design_type == "bbd":
        return {3: 5, 4: 5, 5: 6, 6: 6, 7: 6}.get(k, 5)
    return 0


def full_factorial(k):
    """2^k factorial in standard (Yates) order: factor A changes fastest."""
    runs = np.array(list(itertools.product([-1.0, 1.0], repeat=k)))
    return runs[:, ::-1].copy()


def fraction_options(k):
    """Available p values for k factors, with their resolution (4 to 512 runs)."""
    opts = []
    for (kk, p), gens in sorted(FRACTION_GENERATORS.items()):
        if kk == k and 2 ** (k - p) <= FACTORIAL_RUNS[-1]:
            opts.append((p, resolution(k, p)))
    return opts


def _generator_words(k, p):
    base = LETTERS[: k - p]
    words = []
    for i, g in enumerate(FRACTION_GENERATORS[(k, p)]):
        words.append(frozenset(g) | {LETTERS[k - p + i]})
    return base, words


def _word_mask(word):
    m = 0
    for ch in word:
        m |= 1 << LETTERS.index(ch)
    return m


@lru_cache(maxsize=256)
def _group(k, p):
    """All defining relation words as bitmasks (without I), generated from the generators."""
    grp = np.zeros(1, np.int64)
    for _, w in zip(range(p), _generator_words(k, p)[1]):
        grp = np.concatenate([grp, grp ^ _word_mask(w)])
    return grp[1:]


def _popcount(a):
    a = np.asarray(a, np.int64)
    c = np.zeros(a.shape, np.int64)
    while np.any(a):
        c += a & 1
        a = a >> 1
    return c


def _mask_name(m):
    return "".join(LETTERS[i] for i in range(25) if m >> i & 1)


def defining_relation(k, p):
    g = _group(k, p)
    return sorted((_mask_name(int(m)) for m in g), key=lambda s: (len(s), s))


@lru_cache(maxsize=512)
def word_length_pattern(k, p):
    """(A1, ..., Ak): number of defining relation words of length 1..k."""
    L = _popcount(_group(k, p))
    return tuple(int(c) for c in np.bincount(L, minlength=k + 1)[1:])


def resolution(k, p):
    if p <= 0:
        return 0
    wlp = word_length_pattern(k, p)
    return next(i + 1 for i, c in enumerate(wlp) if c)


def roman(n):
    return {0: "full", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"}.get(n, str(n))


def generator_text(k, p):
    """['E = ABCD', 'F = ABC', ...]"""
    return [f"{LETTERS[k - p + i]} = {g}" for i, g in enumerate(FRACTION_GENERATORS[(k, p)])]


def factorial_table():
    """{(run, k): p} for every cell of the two-level factorial table (p = 0 means full factorial)."""
    out = {}
    for k in range(2, 22):
        for runs in FACTORIAL_RUNS:
            n = int(np.log2(runs))
            if n == k:
                out[(runs, k)] = 0
            elif n < k and (k, k - n) in FRACTION_GENERATORS:
                out[(runs, k)] = k - n
    return out


def alias_summary(k, p, max_len=3):
    """Aliases of main effects and two-factor interactions with other effects up to length `max_len`.
    Returns a list of (effect, [aliases...]) for the effects that have aliases."""
    if p <= 0:
        return []
    g = _group(k, p)
    out = []
    effects = [1 << i for i in range(k)] + [(1 << i) | (1 << j) for i in range(k) for j in range(i + 1, k)]
    for e in effects:
        al = e ^ g
        ln = _popcount(al)
        keep = sorted({int(a) for a, n in zip(al, ln) if n <= max_len}, key=lambda m: (bin(m).count("1"), m))
        if keep:
            out.append((_mask_name(e), [_mask_name(m) for m in keep]))
    return out


def min_runs_for_resolution(k, res):
    """Smallest number of two-level factorial runs for k factors with resolution >= res (None if there is none)."""
    for (runs, kk), p in sorted(factorial_table().items()):
        if kk == k and (p == 0 or resolution(k, p) >= res):
            return runs, p
    return None


def fractional_factorial(k, p):
    base_letters, _ = _generator_words(k, p)
    base = full_factorial(k - p)
    cols = [base[:, i] for i in range(k - p)]
    for g in FRACTION_GENERATORS[(k, p)]:
        col = np.ones(base.shape[0])
        for ch in g:
            col = col * base[:, base_letters.index(ch)]
        cols.append(col)
    return np.column_stack(cols)


CCD_ALPHA_TYPES = {
    "rotatable": "Rotatable",
    "face": "Face-centered (alpha = 1)",
    "orthogonal": "Orthogonal",
    "custom": "Custom",
}


def ccd_alpha(k, alpha_type, custom=1.0, center_points=None):
    n_f = 2 ** k
    if alpha_type == "rotatable":
        return n_f ** 0.25
    if alpha_type == "face":
        return 1.0
    if alpha_type == "orthogonal":
        n_c = default_center_points("ccd", k) if center_points is None else center_points
        n = n_f + 2 * k + n_c
        return (n_f * (np.sqrt(n) - np.sqrt(n_f)) ** 2 / 4) ** 0.25
    return float(custom)


def ccd_points(k, alpha):
    fact = full_factorial(k)
    axial = []
    for i in range(k):
        for s in (-alpha, alpha):
            row = np.zeros(k)
            row[i] = s
            axial.append(row)
    return fact, np.array(axial)


def bbd_points(k):
    if k in BBD_BLOCKS:
        groups = BBD_BLOCKS[k]
    else:
        groups = list(itertools.combinations(range(k), 2))
    rows = []
    for g in groups:
        for signs in itertools.product([-1.0, 1.0], repeat=len(g)):
            row = np.zeros(k)
            # the last sign is reversed so the first factor changes fastest
            for idx, s in zip(g, signs[::-1]):
                row[idx] = s
            rows.append(row)
    return np.array(rows)


MIXTURE_TYPES = ("simplex_lattice", "simplex_centroid", "optimal_mixture", "combined_optimal", "historical_mixture")

# 2^k factorial block generators (Montgomery): key (k, number_of_blocks)
FACTORIAL_BLOCK_GENERATORS = {
    (2, 2): ["AB"],
    (3, 2): ["ABC"], (3, 4): ["AB", "AC"],
    (4, 2): ["ABCD"], (4, 4): ["ABC", "ACD"], (4, 8): ["AB", "BC", "CD"],
    (5, 2): ["ABCDE"], (5, 4): ["ABC", "CDE"], (5, 8): ["ABE", "BCE", "CDE"],
    (6, 2): ["ABCDEF"], (6, 4): ["ABCF", "CDEF"], (6, 8): ["ABEF", "ABCD", "ACE"],
    (7, 2): ["ABCDEFG"], (7, 4): ["ABCFG", "CDEFG"], (7, 8): ["ABC", "DEF", "AFG"],
}

# 4-factor Box-Behnken in 3 orthogonal blocks: factor pair -> block
BBD4_BLOCKS = {(0, 1): 1, (2, 3): 1, (0, 3): 2, (1, 2): 2, (0, 2): 3, (1, 3): 3}


def is_mixture(design_type):
    return design_type in MIXTURE_TYPES


def blocking_options(design_type, k):
    """Numbers of blocks available for this design type and number of factors."""
    if design_type == "full_factorial":
        return [b for b in (1, 2, 4, 8) if b == 1 or (k, b) in FACTORIAL_BLOCK_GENERATORS]
    if design_type == "ccd":
        return [1, 2, 3] if k >= 3 else [1, 2]
    if design_type == "bbd" and k == 4:
        return [1, 3]
    return [1]


def _factorial_block_labels(fact, k, n_blocks):
    if n_blocks <= 1:
        return np.ones(len(fact), int)
    labels = np.zeros(len(fact), int)
    for j, word in enumerate(FACTORIAL_BLOCK_GENERATORS[(k, n_blocks)]):
        prod = np.ones(len(fact))
        for ch in word:
            prod = prod * fact[:, LETTERS.index(ch)]
        labels += (prod > 0).astype(int) << j
    # block numbers follow first appearance so block 1 holds the first standard run
    order = {}
    for lab in labels:
        order.setdefault(lab, len(order) + 1)
    return np.array([order[lab] for lab in labels])


def _spread(n, blocks_for):
    """Spread n center points evenly over the list of blocks `blocks_for`."""
    return sorted(blocks_for[i % len(blocks_for)] for i in range(n))


def ccd_block_alpha(k, n_blocks, center_points):
    """Alpha that makes the blocks orthogonal in a blocked CCD."""
    n_f = 2 ** k
    labels = _spread(center_points, list(range(1, n_blocks + 1)))
    f0 = sum(1 for b in labels if b < n_blocks)
    a0 = len(labels) - f0
    return float(np.sqrt(n_f * (2 * k + a0) / (2 * (n_f + f0))))


# ------------------------------------------------------------------ mixture
def _mixture_point_type(row):
    nz = int(np.sum(row > 1e-9))
    if nz == 1:
        return "Vertex"
    if nz == len(row) and np.allclose(row, row[0]):
        return "Centroid"
    if nz == 2:
        return "Edge"
    return "Interior"


def simplex_lattice(q, m):
    pts = [np.array(c, float) / m for c in itertools.product(range(m + 1), repeat=q) if sum(c) == m]
    pts.sort(key=lambda r: (int(np.sum(r > 1e-9)), tuple(-r)))
    return np.array(pts)


def simplex_centroid(q):
    pts = []
    for r in range(1, q + 1):
        for subset in itertools.combinations(range(q), r):
            row = np.zeros(q)
            row[list(subset)] = 1.0 / r
            pts.append(row)
    return np.array(pts)


def axial_check_blends(q):
    pts = []
    for i in range(q):
        row = np.full(q, 1.0 / (2 * q))
        row[i] = (q + 1) / (2 * q)
        pts.append(row)
    return np.array(pts)


def build_mixture(design_type, q, lattice_degree=2, augment=True, replicate_points=True):
    base = simplex_lattice(q, lattice_degree) if design_type == "simplex_lattice" else simplex_centroid(q)
    centroid = np.full(q, 1.0 / q)
    has_centroid = any(np.allclose(r, centroid) for r in base)
    rows = [base]
    n_axial_start = None
    if augment:
        n_axial_start = base.shape[0]
        rows.append(axial_check_blends(q))
        if not has_centroid:
            rows.append(centroid[None, :])
            has_centroid = True
    if replicate_points:
        # replicate the vertices (+ centroid) so pure error / lack of fit can be computed
        rows.append(np.eye(q))
        if has_centroid:
            rows.append(centroid[None, :])
    coded = np.vstack(rows)
    types = [_mixture_point_type(r) for r in coded]
    if n_axial_start is not None:
        for i in range(n_axial_start, n_axial_start + q):
            types[i] = "Axial"
    return coded, types


# ------------------------------------------------------------------ space-filling
def latin_hypercube(k, n, seed=None, iters=None):
    """Maximin Latin Hypercube in [-1, 1]^k: each factor has n equally spaced levels (including the bounds -1 and +1),
    each level used exactly once. Rearranged by swapping levels between runs so the smallest distance between points
    is as large as possible (Morris-Mitchell phi_p criterion), spreading the points evenly over the whole space."""
    rng = np.random.default_rng(seed)
    n = int(n)
    if n < 2:
        return np.zeros((max(n, 0), k))
    lev = np.linspace(-1, 1, n)
    if n > 2000:        # n x n distance matrix too large: random LHS (still one point per level) without optimization
        return np.column_stack([lev[rng.permutation(n)] for _ in range(k)])
    iters = int(iters or min(20000, 200 * n))
    pp = 15.0

    def phi_rows(X, i):
        d = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
        d[i] = np.inf
        return d

    best_X, best_phi = None, np.inf
    for _ in range(2 if n <= 60 else 1):
        X = np.column_stack([lev[rng.permutation(n)] for _ in range(k)])
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(D, np.inf)
        phi = float((D ** -pp).sum() / 2)
        temp = phi * 0.02
        for t in range(iters):
            j = rng.integers(k)
            a, b = rng.choice(n, 2, replace=False)
            X[[a, b], j] = X[[b, a], j]
            da, db = phi_rows(X, a), phi_rows(X, b)
            old = (D[a] ** -pp).sum() + (D[b] ** -pp).sum() - D[a, b] ** -pp
            new = (da ** -pp).sum() + (db ** -pp).sum() - da[b] ** -pp
            dphi = new - old
            if dphi < 0 or rng.random() < np.exp(-dphi / max(temp, 1e-300)):
                phi += dphi
                D[a], D[:, a], D[b], D[:, b] = da, da, db, db
                D[a, a] = D[b, b] = np.inf
            else:
                X[[a, b], j] = X[[b, a], j]
            temp *= 0.9995
        if phi < best_phi:
            best_X, best_phi = X.copy(), phi
    return best_X


def maximin_augment(existing, k, n_new, seed=None, n_cand=4000):
    """Add n_new points in the cube [-1, 1]^k, each as far as possible from the existing points."""
    rng = np.random.default_rng(seed)
    if n_new <= 0:
        return np.zeros((0, k))
    cand = np.vstack([latin_hypercube(k, max(n_new * 4, 20), seed, iters=2000), rng.uniform(-1, 1, (n_cand, k))])
    E = np.asarray(existing, float).reshape(-1, k)
    if len(E):
        from scipy.spatial import cKDTree
        d = cKDTree(E).query(cand)[0]
    else:
        d = np.full(len(cand), np.inf)
    out = []
    for _ in range(int(n_new)):
        i = int(np.argmax(d))
        out.append(cand[i])
        d = np.minimum(d, np.sqrt(((cand - cand[i]) ** 2).sum(axis=1)))
        d[i] = -np.inf
    return np.array(out)


HYBRID_BASES = {"ccd_face": "CCD face-centered (3 level)", "ccd": "CCD rotatable (5 level)",
                "bbd": "Box-Behnken (3 level)", "full_factorial": "Factorial (2 level)"}


# ------------------------------------------------------------------ main
def build_design(design_type, k, center_points=0, replicates=1, fraction_p=1,
                 alpha_type="rotatable", alpha_custom=1.0, blocks=1,
                 lattice_degree=2, augment=True, replicate_points=True, pb_runs=12, taguchi_array="L8",
                 lhs_runs=30, hybrid_base="ccd_face", extra_points=10, design_seed=None):
    """Returns (coded_matrix, point_type_list, block_labels)."""
    if is_mixture(design_type):
        coded, types = build_mixture(design_type, k, lattice_degree, augment, replicate_points)
        return coded, types, np.ones(len(coded), int)

    blocks = int(blocks) if int(blocks) in blocking_options(design_type, k) else 1
    parts = []  # (array, type, block label per row or None)
    if design_type == "full_factorial":
        fact = full_factorial(k)
        parts.append((fact, "Factorial", _factorial_block_labels(fact, k, blocks)))
    elif design_type == "fractional_factorial":
        parts.append((fractional_factorial(k, fraction_p), "Factorial", None))
    elif design_type == "ccd":
        if blocks > 1 and alpha_type == "orthogonal":
            alpha = ccd_block_alpha(k, blocks, center_points)
        else:
            alpha = ccd_alpha(k, alpha_type, alpha_custom, center_points)
        fact, axial = ccd_points(k, alpha)
        fb = _factorial_block_labels(fact, k, 2) if blocks == 3 else np.ones(len(fact), int)
        parts.append((fact, "Factorial", fb))
        parts.append((axial, "Axial", np.full(len(axial), blocks)))
    elif design_type == "plackett_burman":
        parts.append((plackett_burman(k, int(pb_runs)), "Screening", None))
    elif design_type == "dsd":
        parts.append((definitive_screening(k), "Screening", None))
    elif design_type == "taguchi":
        parts.append((taguchi(k, taguchi_array), "Orthogonal", None))
    elif design_type == "lhs":
        parts.append((latin_hypercube(k, int(lhs_runs), design_seed if design_seed is not None else 11),
                      "Space-filling", None))
    elif design_type == "hybrid":
        if hybrid_base == "bbd" and k < 3:
            raise ValueError("Box-Behnken requires at least 3 factors.")
        sub = {"ccd_face": ("ccd", {"alpha_type": "face"}), "ccd": ("ccd", {"alpha_type": "rotatable"}),
               "bbd": ("bbd", {}), "full_factorial": ("full_factorial", {})}[hybrid_base]
        base, btypes, _ = build_design(sub[0], k, 0, 1, **sub[1])
        extra = maximin_augment(np.vstack([base, np.zeros((1, k))]), k, int(extra_points),
                                design_seed if design_seed is not None else 11)
        rows_ = [base] + ([extra] if len(extra) else [])
        types_ = list(btypes) + ["Space-filling"] * len(extra)
        parts.append((np.vstack(rows_), None, None))
        parts[-1] = (parts[-1][0], types_, None)
    elif design_type == "bbd":
        pts = bbd_points(k)
        lab = None
        if blocks == 3 and k == 4:
            lab = np.array([BBD4_BLOCKS[tuple(int(i) for i in np.nonzero(r)[0])] for r in pts])
        parts.append((pts, "Edge", lab))
    else:
        raise ValueError(f"Unknown design type: {design_type}")

    rows, types, blk = [], [], []
    for _ in range(max(1, int(replicates))):
        for arr, label, lab in parts:
            rows.append(arr)
            types.extend(label if isinstance(label, list) else [label] * arr.shape[0])
            blk.extend(list(lab) if lab is not None else [1] * arr.shape[0])
    if center_points > 0:
        rows.append(np.zeros((int(center_points), k)))
        types.extend(["Center"] * int(center_points))
        blk.extend(_spread(int(center_points), list(range(1, blocks + 1))))
    coded = np.vstack(rows)
    coded[np.abs(coded) < 1e-12] = 0.0
    blk = np.array(blk, int)
    # sort by block; the standard order within each block is kept
    order = np.argsort(blk, kind="stable")
    return coded[order], [types[i] for i in order], blk[order]


def randomized_run_order(n, seed=None, blocks=None):
    """Random run order; when blocked, runs are randomized within blocks and the blocks run in sequence."""
    rng = np.random.default_rng(seed)
    blocks = np.ones(n, int) if blocks is None else np.asarray(blocks)
    run = np.empty(n, dtype=int)
    start = 1
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        run[idx[rng.permutation(len(idx))]] = np.arange(start, start + len(idx))
        start += len(idx)
    return run


def coded_to_actual(coded, lows, highs):
    lows = np.asarray(lows, float)
    highs = np.asarray(highs, float)
    center = (lows + highs) / 2
    half = (highs - lows) / 2
    return center + np.asarray(coded, float) * half


def actual_to_coded(actual, lows, highs):
    lows = np.asarray(lows, float)
    highs = np.asarray(highs, float)
    center = (lows + highs) / 2
    half = (highs - lows) / 2
    return (np.asarray(actual, float) - center) / half


# ------------------------------------------------------------------ augment
def axial_points(k, alpha):
    rows = []
    for i in range(k):
        for sgn in (-alpha, alpha):
            r = np.zeros(k)
            r[i] = sgn
            rows.append(r)
    return np.array(rows)


def foldover(coded, factor=None):
    """Full foldover (all signs reversed) or a single factor only."""
    out = np.array(coded, float).copy()
    if factor is None:
        out = -out
    else:
        out[:, factor] = -out[:, factor]
    out[np.abs(out) < 1e-12] = 0.0
    return out


# ------------------------------------------------------------------ screening
PB_GENERATORS = {
    12: "++-+++---+-",
    20: "++--++++-+-+----++-",
    24: "+++++-+-++--++--+-+----",
}


def plackett_burman(k, runs=12):
    if runs not in PB_GENERATORS:
        raise ValueError("Plackett-Burman is available for 12, 20, or 24 runs.")
    if k > runs - 1:
        raise ValueError(f"PB with {runs} runs allows at most {runs - 1} factors.")
    g = np.array([1.0 if c == "+" else -1.0 for c in PB_GENERATORS[runs]])
    rows = [np.roll(g, i) for i in range(runs - 1)] + [-np.ones(runs - 1)]
    return np.array(rows)[:, :k]


def _gf_elements(q):
    """Elements of GF(q) for prime q or q = 9 (a + b·i, i² = -1 mod 3)."""
    if q == 9:
        return [(a, b) for a in range(3) for b in range(3)]
    return list(range(q))


def _gf_sub(x, y, q):
    if q == 9:
        return ((x[0] - y[0]) % 3, (x[1] - y[1]) % 3)
    return (x - y) % q


def _gf_is_square(x, q):
    if q == 9:
        sq = {((a * a - b * b) % 3, (2 * a * b) % 3) for a in range(3) for b in range(3) if (a, b) != (0, 0)}
        return x in sq
    return any((t * t) % q == x for t in range(1, q))


def _gf_zero(x, q):
    return x == (0, 0) if q == 9 else x == 0


def conference_matrix(m):
    """Conference matrix of order m (C Cᵀ = (m-1) I) using the Paley construction, q = m - 1."""
    q = m - 1
    if q != 9 and not (q > 2 and all(q % d for d in range(2, int(q ** 0.5) + 1))):
        raise ValueError(f"A conference matrix of order {m} is not available.")
    el = _gf_elements(q)

    def chi(x):
        return 0 if _gf_zero(x, q) else (1 if _gf_is_square(x, q) else -1)

    C = np.zeros((m, m))
    minus_one = (2, 0) if q == 9 else q - 1
    s = chi(minus_one)
    for j in range(1, m):
        C[0, j] = 1
        C[j, 0] = s
    for i in range(1, m):
        for j in range(1, m):
            C[i, j] = chi(_gf_sub(el[j - 1], el[i - 1], q))
    return C


DSD_ORDERS = [4, 6, 8, 10, 12, 14, 18, 20]


def definitive_screening(k, extra_center=0):
    """DSD (Jones & Nachtsheim): [C; -C; 0] where C is a conference matrix, 2m+1 runs."""
    m = k if k % 2 == 0 else k + 1
    m = next((o for o in DSD_ORDERS if o >= m), None)
    if m is None:
        raise ValueError("DSD is available for 3–20 factors.")
    C = conference_matrix(m)[:, :k]
    rows = np.vstack([C, -C, np.zeros((1 + extra_center, k))])
    return rows


def _two_level_oa(nbase):
    base = full_factorial(nbase)[:, ::-1]  # column 1 changes slowest
    cols = []
    for j in range(1, 2 ** nbase):
        col = np.ones(len(base))
        for b in range(nbase):
            if j >> (nbase - 1 - b) & 1:
                col = col * base[:, b]
        cols.append(col)
    return np.column_stack(cols)


def _three_level_oa(nbase):
    base = np.array(list(itertools.product(range(3), repeat=nbase)))
    vecs = [v for v in itertools.product(range(3), repeat=nbase) if any(v) and v[next(i for i, x in enumerate(v) if x)] == 1]
    vecs.sort(key=lambda v: (sum(1 for x in v if x), v[::-1]))
    cols = [(base @ np.array(v)) % 3 for v in vecs]
    return np.column_stack(cols) - 1.0


L18 = """11111111 11222222 11333333 12112233 12223311 12331122 13121323 13232131 13313212
21133221 21211332 21322113 22123132 22231213 22312321 23132312 23213123 23321231"""

TAGUCHI = {
    "L4": ("L4 (2^3)", lambda: _two_level_oa(2), [2] * 3),
    "L8": ("L8 (2^7)", lambda: _two_level_oa(3), [2] * 7),
    "L9": ("L9 (3^4)", lambda: _three_level_oa(2), [3] * 4),
    "L12": ("L12 (2^11)", lambda: plackett_burman(11, 12), [2] * 11),
    "L16": ("L16 (2^15)", lambda: _two_level_oa(4), [2] * 15),
    "L18": ("L18 (2^1 × 3^7)", lambda: _l18(), [2] + [3] * 7),
    "L27": ("L27 (3^13)", lambda: _three_level_oa(3), [3] * 13),
}


def _l18():
    rows = [[int(c) for c in r] for r in L18.split()]
    A = np.array(rows, float)
    out = A.copy()
    out[:, 0] = np.where(A[:, 0] == 1, -1.0, 1.0)
    out[:, 1:] = A[:, 1:] - 2.0
    return out


def taguchi(k, array="L8"):
    name, fn, levels = TAGUCHI[array]
    if k > len(levels):
        raise ValueError(f"{name} allows at most {len(levels)} factors.")
    return fn()[:, :k]
