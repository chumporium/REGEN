"""Experiment project data structure + saving/opening files (.doe, JSON format)."""
import itertools
import json
from dataclasses import dataclass, field

import numpy as np

from . import constraints as cons_mod
from . import designs, models, optimal
from .optimize import optimize as run_optimize

FILE_VERSION = 3

# point-type labels written by older versions (Indonesian) -> current labels
LEGACY_POINT_TYPES = {"Faktorial": "Factorial", "Pusat": "Center", "Aksial": "Axial", "Historis": "Historical",
                      "Tepi": "Edge", "Replikasi": "Replicate", "Konfirmasi": "Confirmation", "Saran": "Suggested",
                      "Blok": "Block", "Skrining": "Screening", "Ortogonal": "Orthogonal"}


@dataclass
class Factor:
    name: str
    unit: str
    low: float
    high: float
    kind: str = "numeric"      # "numeric" | "categoric" | "process" (process factor in a combined design)
    levels: list = None        # level names for categoric factors
    upper: float = None        # upper bound of a mixture component (optimal design)
    sd: float = None           # factor standard deviation (actual units) for propagation of error

    @property
    def categoric(self):
        return self.kind == "categoric"


@dataclass
class Response:
    name: str
    unit: str = ""
    kind: str = "normal"       # "normal" | "binomial" | "poisson"
    trials: float = None       # number of trials per run (binomial)


RESPONSE_KEYS = ("name", "unit", "kind", "trials")


@dataclass
class Project:
    design_type: str
    options: dict
    factors: list          # process factors, or mixture components (low = lower bound)
    responses: list
    coded: np.ndarray      # coded (-1..+1), pseudo-components, or categoric level indices
    point_types: list
    run_order: np.ndarray
    data: np.ndarray       # n x r, NaN = not filled in yet
    blocks: np.ndarray = None
    mixture_total: float = 1.0
    constraints: list = field(default_factory=list)   # in actual units
    groups: np.ndarray = None                         # whole-plot id (split-plot)
    htc: list = field(default_factory=list)           # indices of hard-to-change factors
    model_specs: dict = field(default_factory=dict)   # response idx -> {"order", "terms", "transform"}
    opt_criteria: dict = field(default_factory=dict)
    path: str = None
    dirty: bool = False
    ann: dict = field(default_factory=dict, repr=False)             # response idx -> AnnModel
    model_source: dict = field(default_factory=dict)                 # response idx -> "rsm" | "ann"
    nsga_result: dict = None                                         # last NSGA-II result (not saved)
    nsga_compare: dict = None                                        # ANN vs RSM front comparison (not saved)
    learning: dict = None                                            # ANN learning curve per response (not saved)
    excluded: list = field(default_factory=list)                     # indices of runs excluded from the analysis
    notes: str = ""                                                  # free-form project notes
    solutions: list = field(default_factory=list, repr=False)       # last optimization results (not saved)
    criteria_used: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.blocks is None:
            self.blocks = np.ones(self.coded.shape[0], int)
        if self.is_mixture:
            span = self.mixture_total - sum(self.factors[i].low for i in self.comp_idx)
            for i in self.comp_idx:
                f = self.factors[i]
                f.high = f.low + span if f.upper is None else min(f.upper, f.low + span)
        for f in self.factors:
            if f.categoric:
                f.low, f.high = 0.0, float(len(f.levels) - 1)

    # ---------- creation ----------
    @classmethod
    def create(cls, design_type, factors, responses, seed=None, mixture_total=1.0, constraints=None,
               n_runs=None, **options):
        constraints = constraints or []
        dummy = cls(design_type=design_type, options=options, factors=factors, responses=responses,
                    coded=np.zeros((0, len(factors))), point_types=[], run_order=np.zeros(0, int),
                    data=np.zeros((0, len(responses))), mixture_total=float(mixture_total),
                    constraints=constraints)
        blocks = None
        split = options.pop("split_plot", None)
        if design_type == "combined_optimal":
            coded, types = dummy._build_combined(seed)
        elif design_type in ("optimal_rsm", "optimal_mixture"):
            coded, types = dummy._build_optimal(seed)
        elif design_type == "general_factorial":
            combos = list(itertools.product(*[range(len(f.levels)) for f in factors]))
            reps = max(1, int(options.get("replicates", 1)))
            coded = np.array([c for _ in range(reps) for c in combos], float)
            types = ["Factorial"] * len(coded)
        elif design_type in ("historical", "historical_mixture"):
            coded = np.full((int(n_runs or 10), len(factors)), np.nan)
            types = ["Historical"] * len(coded)
        else:
            num = [i for i, f in enumerate(factors) if not f.categoric]
            base, types, blocks = designs.build_design(design_type, len(num), **options)
            coded, types, blocks = dummy._cross(base, types, blocks)
        n = coded.shape[0]
        run_order = designs.randomized_run_order(n, seed, blocks)
        groups, htc = None, []
        if split and split.get("htc"):
            from .mixed import whole_plot_groups
            htc = sorted(split["htc"])
            groups, run_order = whole_plot_groups(coded, htc, split.get("wp_size"), seed)
            blocks = None
            options["split_plot"] = split
        return cls(design_type=design_type, options=options, factors=factors, responses=responses,
                   coded=coded, point_types=types, blocks=blocks, run_order=run_order,
                   data=np.full((n, len(responses)), np.nan), mixture_total=float(mixture_total),
                   constraints=constraints, groups=groups, htc=htc)

    @classmethod
    def from_table(cls, factors, responses, fvalues, yvalues, mixture_total=None):
        """Historical data project from an imported table.

        fvalues: n x k actual values (categoric = level index, NaN = empty); yvalues: n x r.
        Mixture components (kind 'numeric' when mixture_total is given) use lower/upper bounds from the data.
        """
        fvalues = np.asarray(fvalues, float).reshape(len(fvalues), len(factors))
        n = len(fvalues)
        mixture = mixture_total is not None
        p = cls.create("historical_mixture" if mixture else "historical", factors, responses, n_runs=n,
                       mixture_total=mixture_total or 1.0)
        p.coded = p.to_coded(fvalues)
        p.coded[:, p.cat_idx] = fvalues[:, p.cat_idx]
        p.data = np.asarray(yvalues, float).reshape(n, len(responses))
        p.run_order = np.arange(1, n + 1)
        p.point_types = ["Historical"] * n
        p.dirty = True
        return p

    def _cross(self, base, types, blocks):
        """Place the numeric design in its columns, then cross it with the categoric level combinations."""
        num = self.num_idx
        cats = self.cat_idx
        combos = list(itertools.product(*[range(len(self.factors[i].levels)) for i in cats])) or [()]
        rows, t_out, b_out = [], [], []
        for combo in combos:
            M = np.zeros((base.shape[0], self.k))
            M[:, num] = base
            for i, lev in zip(cats, combo):
                M[:, i] = lev
            rows.append(M)
            t_out += list(types)
            b_out += list(blocks)
        return np.vstack(rows), t_out, np.array(b_out, int)

    def _build_optimal(self, seed):
        o = self.options
        rng = np.random.default_rng(seed)
        sp = self.space
        num = self.num_idx
        lo, hi = self.coded_bounds()
        G, h = self.constraint_system(cols=num)
        cands_n, _ = cons_mod.candidate_points(lo[num], hi[num], G, h, self.is_mixture)
        region_n = cons_mod.sample_region(lo[num], hi[num], G, h, self.is_mixture, 3000, rng)
        if len(region_n) < 50:
            region_n = cands_n
        cands = np.zeros((len(cands_n), self.k))
        cands[:, num] = cands_n
        cands, _ = optimal.cross_categoric(cands, [""] * len(cands), sp)
        region = np.zeros((len(region_n), self.k))
        region[:, num] = region_n
        for i in self.cat_idx:
            region[:, i] = rng.integers(0, len(self.factors[i].levels), len(region))
        order = o.get("model_order", "quadratic")
        terms = self.terms_for_order(order)
        p = models.expand(cands[:1], terms, sp)[0].shape[1]
        n_model = p + int(o.get("extra_model", 0))
        idx, types = optimal.build(cands, terms, sp, n_model, int(o.get("n_lof", 5)), int(o.get("n_rep", 5)),
                                   region, o.get("criterion", "I"), seed)
        return cands[idx], types

    def _build_combined(self, seed):
        o = self.options
        rng = np.random.default_rng(seed)
        comp, proc = self.comp_idx, self.proc_idx
        if len(comp) < 2 or not proc:
            raise ValueError("A combined design requires at least 2 components and 1 process factor.")
        lo, hi = self.coded_bounds()
        G, h = self.constraint_system(cols=comp)
        mix_c, _ = cons_mod.candidate_points(lo[comp], hi[comp], G, h, True)
        grid = np.array(list(itertools.product((-1.0, 0.0, 1.0), repeat=len(proc))))
        cands = np.zeros((len(mix_c) * len(grid), self.k))
        cands[:, comp] = np.repeat(mix_c, len(grid), axis=0)
        cands[:, proc] = np.tile(grid, (len(mix_c), 1))
        reg_m = cons_mod.sample_region(lo[comp], hi[comp], G, h, True, 3000, rng)
        region = np.zeros((len(reg_m), self.k))
        region[:, comp] = reg_m
        region[:, proc] = rng.uniform(-1, 1, (len(reg_m), len(proc)))
        order = o.get("model_order", "quadratic|linear")
        terms = models.combined_terms(comp, proc, self.k, order)
        n_model = len(terms) + int(o.get("extra_model", 0))
        idx, types = optimal.build(cands, terms, self.space, n_model, int(o.get("n_lof", 5)),
                                   int(o.get("n_rep", 5)), region, o.get("criterion", "I"), seed)
        return cands[idx], types

    # ---------- properties ----------
    @property
    def is_mixture(self):
        return designs.is_mixture(self.design_type)

    @property
    def is_historical(self):
        return self.design_type in ("historical", "historical_mixture")

    @property
    def k(self):
        return len(self.factors)

    @property
    def n(self):
        return self.coded.shape[0]

    @property
    def n_blocks(self):
        return len(np.unique(self.blocks))

    @property
    def num_idx(self):
        return [i for i, f in enumerate(self.factors) if not f.categoric]

    @property
    def cat_idx(self):
        return [i for i, f in enumerate(self.factors) if f.categoric]

    @property
    def comp_idx(self):
        """Mixture component indices (empty for process designs)."""
        if not self.is_mixture:
            return []
        return [i for i, f in enumerate(self.factors) if f.kind == "numeric"]

    @property
    def proc_idx(self):
        return [i for i, f in enumerate(self.factors) if f.kind == "process"]

    @property
    def is_split_plot(self):
        return self.groups is not None

    @property
    def is_combined(self):
        return self.design_type == "combined_optimal"

    def sd_coded(self):
        """Factor standard deviations in coded units (for POE)."""
        _, sc = self._map()
        out = np.zeros(self.k)
        for i, f in enumerate(self.factors):
            if f.sd and not f.categoric and sc[i]:
                out[i] = float(f.sd) / sc[i]
        return out

    @property
    def has_poe(self):
        return bool(np.any(self.sd_coded() > 0))

    def response_family(self, j):
        return getattr(self.responses[j], "kind", "normal") or "normal"

    @property
    def space(self):
        return models.Space([len(f.levels) if f.categoric else 0 for f in self.factors],
                            [list(f.levels) if f.categoric else None for f in self.factors])

    @property
    def lows(self):
        return np.array([f.low for f in self.factors])

    @property
    def highs(self):
        return np.array([f.high for f in self.factors])

    @property
    def mixture_span(self):
        return self.mixture_total - float(sum(self.factors[i].low for i in self.comp_idx))

    def _map(self):
        """(offset, scale) per factor: actual = offset + scale * coded (categoric: identity)."""
        off, sc = np.zeros(self.k), np.ones(self.k)
        for i, f in enumerate(self.factors):
            if f.categoric:
                continue
            if self.is_mixture and f.kind != "process":
                off[i], sc[i] = f.low, self.mixture_span
            else:
                off[i], sc[i] = (f.low + f.high) / 2, (f.high - f.low) / 2
        return off, sc

    def to_actual(self, coded):
        off, sc = self._map()
        return off + np.asarray(coded, float) * sc

    def to_coded(self, actual):
        off, sc = self._map()
        return (np.asarray(actual, float) - off) / sc

    def factor_to_actual(self, i, coded_value):
        off, sc = self._map()
        return off[i] + np.asarray(coded_value, float) * sc[i]

    def factor_to_coded(self, i, actual_value):
        off, sc = self._map()
        return (np.asarray(actual_value, float) - off[i]) / sc[i]

    @property
    def actual(self):
        return self.to_actual(self.coded)

    def format_value(self, i, actual_value, digits=6):
        from math import isnan
        v = float(actual_value)
        if isnan(v):
            return ""
        f = self.factors[i]
        if f.categoric:
            j = int(round(v))
            return f.levels[j] if 0 <= j < len(f.levels) else "?"
        return f"{v:.{digits}g}"

    def coded_bounds(self):
        """Coded bounds of each factor for the design region / optimization."""
        lo, hi = np.full(self.k, -1.0), np.full(self.k, 1.0)
        for i, f in enumerate(self.factors):
            if f.categoric:
                lo[i], hi[i] = 0, len(f.levels) - 1
            elif self.is_mixture and f.kind != "process":
                lo[i], hi[i] = 0.0, (f.high - f.low) / self.mixture_span
        return lo, hi

    def constraint_system(self, cols=None):
        """Linear constraints in coded space: G z <= h (only the columns `cols` when given)."""
        if not self.constraints:
            d = len(cols) if cols is not None else self.k
            return np.zeros((0, d)), np.zeros(0)
        off, sc = self._map()
        for c in self.constraints:
            if any(abs(c["coef"][i]) > 0 for i in self.cat_idx):
                raise ValueError("Constraints cannot contain categoric factors.")
        G, h = cons_mod.coded_system(self.constraints, off, np.where(self.space.cat, 0.0, sc))
        if cols is not None:
            other = [i for i in range(self.k) if i not in cols]
            if len(G) and np.any(np.abs(G[:, other]) > 1e-12):
                raise ValueError("Constraints may only contain the relevant numeric components/factors.")
            return G[:, cols], h
        return G, h

    def feasible(self, coded):
        """True for points that satisfy the component bounds (mixture) & constraints."""
        z = np.atleast_2d(np.asarray(coded, float))
        ok = np.ones(len(z), bool)
        if self.is_mixture:
            lo, hi = self.coded_bounds()
            c = self.comp_idx
            ok &= np.all((z[:, c] >= lo[c] - 1e-9) & (z[:, c] <= hi[c] + 1e-9), axis=1)
        if self.constraints:
            G, h = self.constraint_system()
            ok &= cons_mod.feasible(z, G, h, 1e-7)
        return ok

    def region_samples(self, n=4000, seed=0):
        """Uniform random points in the design region (for FDS & evaluation)."""
        rng = np.random.default_rng(seed)
        lo, hi = self.coded_bounds()
        z = np.zeros((n * 3, self.k))
        comp = self.comp_idx
        for i, f in enumerate(self.factors):
            if f.categoric:
                z[:, i] = rng.integers(0, len(f.levels), len(z))
            elif i not in comp:
                z[:, i] = rng.uniform(lo[i], hi[i], len(z))
        if comp:
            free = 1.0 - lo[comp].sum()
            z[:, comp] = lo[comp] + free * rng.dirichlet(np.ones(len(comp)), size=len(z))
        z = z[self.feasible(z)]
        return z[:n]

    def factor_label(self, i):
        f = self.factors[i]
        unit = f" ({f.unit})" if f.unit else ""
        return f"{designs.LETTERS[i]}: {f.name}{unit}"

    def response_label(self, j):
        r = self.responses[j]
        unit = f" ({r.unit})" if r.unit else ""
        return f"R{j + 1}: {r.name}{unit}"

    def display_order(self):
        return np.argsort(self.run_order, kind="stable")

    def design_description(self):
        o = self.options
        name = designs.DESIGN_TYPES[self.design_type]
        extra = ""
        if self.design_type == "fractional_factorial":
            p = o.get("fraction_p", 1)
            nk = len(self.num_idx)
            extra = f" 2^({nk}-{p}), Resolution {designs.roman(designs.resolution(nk, p))}"
        elif self.design_type == "ccd":
            axial = np.abs(self.coded[np.array(self.point_types) == "Axial"][:, self.num_idx])
            if axial.size:
                extra = f", alpha = {axial.max():.4g}"
        elif self.design_type == "simplex_lattice":
            extra = f" {{{self.k}, {o.get('lattice_degree', 2)}}}"
        elif self.design_type in ("optimal_rsm", "optimal_mixture"):
            extra = f" ({o.get('criterion', 'I')}-optimal, model {models.order_label(o.get('model_order', 'quadratic'))})"
        if o.get("augmented"):
            extra += " + " + ", ".join(o["augmented"])
        unit = "components" if self.is_mixture else ("factor" if self.k == 1 else "factors")
        if self.is_combined:
            n_proc = len(self.proc_idx)
            unit = f"components + {n_proc} process factor{'s' if n_proc > 1 else ''}"
            extra = f" ({o.get('criterion', 'I')}-optimal, model {models.order_label(o.get('model_order', 'quadratic|linear'))})"
        if self.is_split_plot:
            extra += f", split-plot (HTC: {', '.join(designs.LETTERS[i] for i in self.htc)}; " \
                     f"{len(np.unique(self.groups))} whole plots)"
        ncat = len(self.cat_idx)
        cat = f" ({ncat} categoric)" if ncat else ""
        blk = f", {self.n_blocks} blocks" if self.n_blocks > 1 else ""
        total = f", total = {self.mixture_total:g}" if self.is_mixture else ""
        nc = len(self.constraints)
        ncons = f", {nc} constraint{'s' if nc > 1 else ''}" if nc else ""
        count = len(self.comp_idx) if self.is_combined else self.k
        return f"{name}{extra} - {count} {unit}{cat}{total}{ncons}, {self.n} run{'s' if self.n != 1 else ''}{blk}"

    # ---------- responses & rows ----------
    def add_response(self, name, unit=""):
        self.responses.append(Response(name, unit))
        self.data = np.column_stack([self.data, np.full(self.n, np.nan)])
        self.dirty = True

    def remove_response(self, j):
        # combined ANN models that include response j are removed too (their outputs cannot be separated)
        drop = {getattr(m, "gid", None) for i, m in self.ann.items() if i == j and getattr(m, "n_out", 1) > 1}
        keep = {i: m for i, m in self.ann.items() if i != j and not (drop and getattr(m, "gid", None) in drop)}
        for m in keep.values():
            m.resp_idx = [(r if r < j else r - 1) for r in getattr(m, "resp_idx", [])]
        self.ann = {(i if i < j else i - 1): v for i, v in keep.items()}
        self.model_source = {(i if i < j else i - 1): v for i, v in self.model_source.items() if i != j}
        del self.responses[j]
        self.data = np.delete(self.data, j, axis=1)
        self.model_specs = {(i if i < j else i - 1): v for i, v in self.model_specs.items() if i != j}
        self.opt_criteria = {(i if isinstance(i, str) or i < j else i - 1): v
                             for i, v in self.opt_criteria.items() if i != j}
        self.dirty = True

    def set_factor_value(self, idx, i, text):
        """Set a factor value (historical design). Categoric: level name or level number (1..L)."""
        f = self.factors[i]
        t = str(text).strip()
        if t == "":
            self.coded[idx, i] = np.nan
        elif f.categoric:
            names = [lv.lower() for lv in f.levels]
            if t.lower() in names:
                self.coded[idx, i] = names.index(t.lower())
            elif t.isdigit() and 1 <= int(t) <= len(f.levels):
                self.coded[idx, i] = int(t) - 1
            else:
                raise ValueError(f"Unknown level '{t}' for {f.name}.")
        else:
            self.coded[idx, i] = float(self.factor_to_coded(i, float(t.replace(",", "."))))
        self.dirty = True

    def add_rows(self, count, point_type="Historical"):
        start = int(self.run_order.max()) + 1 if self.n else 1
        self.coded = np.vstack([self.coded, np.full((count, self.k), np.nan)])
        self.data = np.vstack([self.data, np.full((count, len(self.responses)), np.nan)])
        self.point_types += [point_type] * count
        self.blocks = np.concatenate([self.blocks, np.ones(count, int)])
        self.run_order = np.concatenate([self.run_order, np.arange(start, start + count)])
        self.dirty = True

    @property
    def model_data(self):
        """Response data for analysis: excluded runs are treated as empty."""
        if not self.excluded:
            return self.data
        d = self.data.copy()
        d[[i for i in self.excluded if i < self.n]] = np.nan
        return d

    def set_excluded(self, rows, on=True):
        cur = set(self.excluded)
        cur = cur | set(int(r) for r in rows) if on else cur - set(int(r) for r in rows)
        self.excluded = sorted(i for i in cur if 0 <= i < self.n)
        self.dirty = True

    def add_runs(self, coded, point_type="Suggested", new_block=False, seed=None):
        """Add new runs (coded values) at the end of the run order; responses are left empty to be filled in."""
        new = np.atleast_2d(np.asarray(coded, float))
        m = len(new)
        if not m:
            return 0
        if self.is_split_plot:                 # each new run = its own whole plot (hard-to-change factors are reset)
            self.groups = np.concatenate([self.groups, int(self.groups.max()) + 1 + np.arange(m)])
        self.coded = np.vstack([self.coded, new])
        self.point_types = list(self.point_types) + [point_type] * m
        blk = int(self.blocks.max()) + 1 if (new_block and len(self.blocks)) else 1
        self.blocks = np.concatenate([self.blocks, np.full(m, blk)])
        start = int(self.run_order.max()) + 1 if len(self.run_order) else 1
        self.run_order = np.concatenate([self.run_order, start + np.random.default_rng(seed).permutation(m)])
        self.data = np.vstack([self.data, np.full((m, len(self.responses)), np.nan)])
        self.options.setdefault("augmented", []).append(f"{m} {point_type.lower()} run{'s' if m > 1 else ''}")
        self.dirty = True
        return m

    def remove_rows(self, rows):
        keep = np.setdiff1d(np.arange(self.n), rows)
        remap = {int(old): new for new, old in enumerate(keep)}
        self.excluded = sorted(remap[i] for i in self.excluded if i in remap)
        self.coded, self.data = self.coded[keep], self.data[keep]
        self.point_types = [self.point_types[i] for i in keep]
        self.blocks = self.blocks[keep]
        self.run_order = np.argsort(np.argsort(self.run_order[keep], kind="stable"), kind="stable") + 1
        self.dirty = True

    # ---------- augment ----------
    def augment_options(self):
        opts = ["center", "replicate"]
        if self.design_type in ("full_factorial", "fractional_factorial") and len(self.num_idx) >= 2:
            opts.append("ccd")
        if self.design_type == "fractional_factorial":
            opts.append("foldover")
        return opts

    def augment(self, kind, count=1, new_block=True, alpha_type="rotatable", factor=None, seed=None):
        if self.is_split_plot:
            raise ValueError("Augment is not supported for split-plot designs yet.")
        num = self.num_idx
        cats = self.cat_idx
        combos = list(itertools.product(*[range(len(self.factors[i].levels)) for i in cats])) or [()]

        def per_combo(base):
            out = []
            for combo in combos:
                M = np.zeros((len(base), self.k))
                M[:, num] = base
                for i, lev in zip(cats, combo):
                    M[:, i] = lev
                out.append(M)
            return np.vstack(out)

        label = ""
        if kind == "center":
            if self.is_mixture:
                new = np.repeat(self.coded.mean(axis=0, keepdims=True), count, axis=0)
                types = ["Centroid"] * count
            else:
                new = per_combo(np.zeros((count, len(num))))
                types = ["Center"] * len(new)
            label = f"{count} center point{'s' if count > 1 else ''}"
        elif kind == "replicate":
            new = np.vstack([self.coded] * count)
            types = list(self.point_types) * count
            label = f"replicate x{count}"
        elif kind == "ccd":
            fact_pts = np.array([t == "Factorial" for t in self.point_types])
            n_f = len(np.unique(np.round(self.coded[fact_pts][:, num], 6), axis=0))
            alpha = n_f ** 0.25 if alpha_type == "rotatable" else 1.0
            base = np.vstack([designs.axial_points(len(num), alpha), np.zeros((count, len(num)))])
            new = per_combo(base)
            types = (["Axial"] * (2 * len(num)) + ["Center"] * count) * len(combos)
            self.design_type = "ccd"
            self.options["alpha_type"] = alpha_type
            label = f"axial points (alpha {alpha:.3g})"
        elif kind == "foldover":
            new = self.coded.copy()
            sub = designs.foldover(new[:, num], None if factor is None else num.index(factor))
            new[:, num] = sub
            types = ["Foldover"] * len(new)
            label = "foldover" + ("" if factor is None else f" {designs.LETTERS[factor]}")
        else:
            raise ValueError(f"Unknown augment type: {kind}")
        m = len(new)
        blk = np.full(m, int(self.blocks.max()) + 1 if new_block else 1)
        self.coded = np.vstack([self.coded, new])
        self.point_types = list(self.point_types) + types
        self.blocks = np.concatenate([self.blocks, blk])
        start = int(self.run_order.max()) + 1 if len(self.run_order) else 1
        rng = np.random.default_rng(seed)
        self.run_order = np.concatenate([self.run_order, start + rng.permutation(m)])
        self.data = np.vstack([self.data, np.full((m, len(self.responses)), np.nan)])
        self.options.setdefault("augmented", []).append(label)
        self.dirty = True
        return m

    # ---------- model ----------
    def available_orders(self):
        return models.orders_for_design(self.design_type)

    def terms_for_order(self, order):
        if self.is_combined:
            if order == "mean":
                return [tuple([0] * self.k)]
            return models.combined_terms(self.comp_idx, self.proc_idx, self.k, order)
        if self.is_mixture:
            return models.mixture_terms(self.k, order)
        return models.model_terms(self.k, order, self.space)

    def model_spec(self, j):
        if j not in self.model_specs:
            order = self.options.get("model_order") or models.default_order(self.design_type)
            if order not in self.available_orders():
                order = models.default_order(self.design_type)
            self.model_specs[j] = {"order": order, "terms": None}
        self.model_specs[j].setdefault("transform", {"kind": "none"})
        return self.model_specs[j]

    def model_terms(self, j):
        spec = self.model_spec(j)
        if spec["terms"] is not None:
            return [models.normalize_term(t) for t in spec["terms"]]
        return self.terms_for_order(spec["order"])

    def _fit_kwargs(self, j):
        return {"mixture": self.is_mixture,
                "blocks": self.blocks if self.n_blocks > 1 else None,
                "transform": self.model_spec(j).get("transform"),
                "space": self.space}

    def fit_terms(self, j, terms, transform=True):
        kw = self._fit_kwargs(j)
        if not transform:
            kw["transform"] = None
        if not [t for t in terms if not models.is_intercept(models.normalize_term(t))]:
            kw["mixture"] = False
            terms = [tuple([0] * self.k)]
        fam = self.response_family(j)
        if fam in ("binomial", "poisson"):
            from .glm import fit_glm
            return fit_glm(self.coded, self.model_data[:, j], terms, fam, self.responses[j].trials or 1,
                           kw["mixture"], kw["blocks"], kw["space"])
        if self.is_split_plot:
            from .mixed import fit_reml
            return fit_reml(self.coded, self.model_data[:, j], terms, self.groups, self.htc, kw["mixture"],
                            kw["transform"], kw["space"])
        return models.fit_model(self.coded, self.model_data[:, j], terms, **kw)

    def analysis_kind(self, j):
        fam = self.response_family(j)
        if fam in ("binomial", "poisson"):
            return "glm"
        return "reml" if self.is_split_plot else "ols"

    def fit_with_error(self, j):
        y = self.model_data[:, j]
        ok = ~np.isnan(y) & np.all(np.isfinite(self.coded), axis=1)
        if np.sum(ok) < 3:
            return None, "Fill in at least 3 complete rows (factors + response) to start the analysis."
        try:
            return self.fit_terms(j, self.model_terms(j)), None
        except (ValueError, np.linalg.LinAlgError) as exc:
            return None, str(exc)

    def fit(self, j):
        return self.fit_with_error(j)[0]

    # ---------- model source (RSM / ANN) ----------
    def source(self, j):
        src = self.model_source.get(j, "rsm")
        return "ann" if src == "ann" and j in self.ann else "rsm"

    def model_with_error(self, j):
        """Model used by graphs, optimization, and prediction: RSM or ANN, as selected."""
        if self.source(j) == "ann":
            return self.ann[j], None
        return self.fit_with_error(j)

    def model(self, j):
        return self.model_with_error(j)[0]

    def ann_responses(self):
        """Responses that can be modeled with ANN (continuous)."""
        return [j for j in range(len(self.responses)) if self.response_family(j) == "normal"]

    def train_ann(self, j, **kw):
        """j: response index (separate network) or list of indices (one combined multi-output network)."""
        m = self.build_ann(j, **kw)
        self.set_ann(m)
        return m

    def build_ann(self, j, **kw):
        """Train without changing the project (safe to run in a separate thread)."""
        from . import ann
        js = list(j) if isinstance(j, (list, tuple)) else [j]
        return ann.train(self.coded, self.model_data[:, js], self.space, resp_idx=js, **kw)

    def set_ann(self, m):
        for k, j in enumerate(m.resp_idx):
            old = self.ann.get(j)
            if old is not None and getattr(old, "n_out", 1) > 1:        # break up the old combined network
                for i in [i for i, v in self.ann.items() if getattr(v, "gid", None) == old.gid]:
                    self.ann.pop(i, None)
            self.ann[j] = m.view(k)
        self.dirty = True

    def ann_group(self, j):
        """Indices of the responses that share one network with response j."""
        m = self.ann.get(j)
        return list(m.resp_idx) if m is not None and getattr(m, "n_out", 1) > 1 else [j]

    def rsm_eval(self, j):
        """Function (fit_rows, test_rows) -> test NMSE of the RSM model for response j (for the learning curve).
        None if the analysis is not OLS."""
        if self.analysis_kind(j) != "ols":
            return None
        y = self.model_data[:, j]
        ok = np.isfinite(y) & np.all(np.isfinite(self.coded), axis=1)
        var = max(float(np.var(y[ok])), 1e-300)
        terms = self.model_terms(j)
        kw = {**self._fit_kwargs(j), "blocks": None}

        def ev(fit_rows, test_rows):
            fit = models.fit_model(self.coded[fit_rows], y[fit_rows], terms, **kw)
            return float(np.mean((y[test_rows] - fit.predict(self.coded[test_rows])) ** 2)) / var
        return ev

    def rsm_cv(self, js, rows, splits):
        """RSM error (current model) on the same data splits as the ANN architecture search:
        fitted on training + validation data, scored on test data. The error is normalized by the response variance
        (NMSE) as in the ANN search. None if the analysis is not OLS or the model cannot be computed."""
        js = list(js) if isinstance(js, (list, tuple)) else [js]
        if any(self.analysis_kind(j) != "ols" for j in js):
            return None
        X = self.coded[rows]
        tests, trains, rmse = [], [], []
        for j in js:
            y = self.model_data[rows, j]
            var = max(float(y.var()), 1e-300)
            terms = self.model_terms(j)
            kw = {**self._fit_kwargs(j), "blocks": None}
            te_e, tr_e = [], []
            for tr, va, te in splits:
                fit_rows = np.concatenate([tr, va]).astype(int)
                try:
                    fit = models.fit_model(X[fit_rows], y[fit_rows], terms, **kw)
                except (ValueError, np.linalg.LinAlgError):
                    return None
                pt = fit.predict(X[te])
                if not np.all(np.isfinite(pt)):
                    return None
                te_e.append(float(np.mean((y[te] - pt) ** 2)) / var)
                tr_e.append(float(np.mean((y[fit_rows] - fit.predict(X[fit_rows])) ** 2)) / var)
            tests.append(np.mean(te_e))
            trains.append(np.mean(tr_e))
            rmse.append(float(np.sqrt(np.mean(te_e) * var)))
        return {"test": float(np.mean(tests)), "train": float(np.mean(trains)), "rmse": rmse,
                "q2": 1 - float(np.mean(tests))}

    # ---------- NSGA-II / NSGA-III ----------
    def _model_for(self, j, source=None):
        """Model for response j. source None = user's choice; "rsm" / "ann" = forced (ANN only if available)."""
        if source == "rsm":
            return self.fit(j)
        if source == "ann":
            return self.ann[j] if j in self.ann else self.fit(j)
        return self.model(j)

    def run_nsga(self, objectives, factor_bounds, pop=100, gens=150, seed=0, pc=0.9, eta_c=15, eta_m=20,
                 progress=None, algorithm="auto", auto_stop=True, stop_tol=1e-3, stop_window=20, source=None,
                 store=True, cancel=None):
        """objectives: list of dicts {index, goal ('maximize'|'minimize'|'target'|'none'), target, low, high, weight,
        poe(bool)}. low/high (may be NaN) = response limits (constraints). Returns the result with the Pareto solutions.

        algorithm: "auto" (NSGA-III for >= 3 objectives) | "nsga2" | "nsga3". auto_stop: stop on convergence.
        source: None (selected model of each response) | "rsm" | "ann" (to compare the fronts of both models)."""
        from . import nsga2
        lo, hi = self.coded_bounds()
        for i, (a, b) in factor_bounds.items():
            ca, cb = sorted(float(v) for v in self.factor_to_coded(i, np.array([a, b])))
            lo[i], hi[i] = (max(lo[i], ca), min(hi[i], cb)) if self.is_mixture and i in self.comp_idx else (ca, cb)
        G, h = self.constraint_system()
        comp = self.comp_idx
        is_int = np.array([f.categoric for f in self.factors])
        models_ = {}
        for o in objectives:
            m = self._model_for(o["index"], source)
            if m is None:
                raise ValueError(f"The model for {self.responses[o['index']].name} is not available yet.")
            if o.get("poe"):
                m = models.PoeModel(m, self.sd_coded())
            models_[id(o)] = m
        active = [o for o in objectives if o["goal"] != "none"]
        if len(active) < 2:
            raise ValueError("NSGA-II requires at least 2 objectives (maximize/minimize/target).")
        alg = algorithm if algorithm in ("nsga2", "nsga3") else ("nsga3" if len(active) >= 3 else "nsga2")
        # NSGA-II with 4+ objectives advances slowly and noisily: automatic stop was shown to stop too early
        stop_off = auto_stop and alg == "nsga2" and len(active) >= 4
        if stop_off:
            auto_stop = False

        def repair(X):
            if not comp:
                return X
            X = X.copy()
            Z = np.clip(X[:, comp], lo[comp], hi[comp])
            for _ in range(30):
                d = 1.0 - Z.sum(axis=1)
                if np.all(np.abs(d) < 1e-12):
                    break
                room = np.where(d[:, None] > 0, hi[comp] - Z, Z - lo[comp])
                tot = room.sum(axis=1)
                share = np.where(tot[:, None] > 0, room / np.where(tot > 0, tot, 1)[:, None], 0)
                Z = np.clip(Z + share * d[:, None], lo[comp], hi[comp])
            X[:, comp] = Z
            return X

        def evaluate(X):
            F, CV = [], np.zeros(len(X))
            for o in objectives:
                y = models_[id(o)].predict(X)
                lo_o, hi_o = o.get("low", np.nan), o.get("high", np.nan)
                scale = abs(hi_o - lo_o) if np.isfinite(lo_o) and np.isfinite(hi_o) and hi_o != lo_o else                     max(np.nanstd(y), 1e-9)
                if np.isfinite(lo_o):
                    CV += np.maximum(0, lo_o - y) / scale
                if np.isfinite(hi_o):
                    CV += np.maximum(0, y - hi_o) / scale
                if o["goal"] == "maximize":
                    F.append(-y)
                elif o["goal"] == "minimize":
                    F.append(y)
                elif o["goal"] == "target":
                    F.append(np.abs(y - o["target"]))
            if len(G):
                CV += np.maximum(0, X @ G.T - h).sum(axis=1)
            if comp:
                CV += np.abs(X[:, comp].sum(axis=1) - 1.0) * (np.abs(X[:, comp].sum(axis=1) - 1.0) > 1e-6)
            return np.column_stack(F), CV

        res = nsga2.run(evaluate, lo, hi, is_int, repair if comp else None, pop, gens, pc, eta_c, eta_m,
                        seed=seed, progress=progress, algorithm=alg, auto_stop=auto_stop, stop_tol=stop_tol,
                        stop_window=stop_window, cancel=cancel)
        front = res["front"]
        if not len(front):
            raise ValueError("No feasible solution. Relax the response limits or constraints.")
        X = res["X"][front]
        F = res["F"][front]
        order = np.argsort(F[:, 0])
        X, F = X[order], F[order]
        Y = {}
        for j in range(len(self.responses)):
            m = self._model_for(j, source)
            if m is not None:
                Y[j] = m.predict(X)
        Yobj = [models_[id(o)].predict(X) for o in active]
        weights = [o.get("weight", 1.0) for o in active]
        score = nsga2.topsis(F, weights)
        sources = {o["index"]: ("ANN" if isinstance(models_[id(o)], ann_model_types()) else "RSM") for o in active}
        result = {"X": X, "F": F, "Y": Y, "Yobj": Yobj, "objectives": objectives, "active": active, "topsis": score,
                  "i_topsis": int(np.argmax(score)), "i_knee": nsga2.knee(F), "history": res["history"],
                  "hv_curve": res["hv_curve"], "algorithm": res["algorithm"], "n_refs": res["n_refs"],
                  "gens_run": res["gens_run"], "stop_gen": res["stop_gen"], "stop_reason": res["stop_reason"],
                  "stop_tol": stop_tol, "stop_window": stop_window, "source": source, "sources": sources,
                  "stop_off": stop_off,
                  "bounds": (lo, hi), "factor_bounds": factor_bounds,
                  "params": {"pop": pop, "gens": gens, "pc": pc, "eta_c": eta_c, "eta_m": eta_m, "seed": seed,
                             "algorithm": res["algorithm"], "auto_stop": auto_stop}}
        result["check"] = nsga_convergence(result)
        if store:
            self.nsga_result = result
        return result

    def nsga_adequacy(self, base, runs=None, progress=None, cancel=None):
        """Population & generation adequacy check: repeat the optimization with other seeds (same population) and with
        2x the population, then compare the hypervolume (normalized jointly) and the TOPSIS compromise solution.
        Returns a dict {runs: [...], verdict: {...}}."""
        from . import nsga2
        prm = base["params"]
        runs = runs or [("Different seed (same population)", prm["pop"], prm["seed"] + 101),
                        ("Different seed 2 (same population)", prm["pop"], prm["seed"] + 202),
                        (f"Population 2x ({2 * prm['pop']})", 2 * prm["pop"], prm["seed"])]
        out = [("Current result", prm["pop"], prm["seed"], base)]
        for k, (label, pop, seed) in enumerate(runs):
            if cancel is not None and cancel():
                break
            if progress:
                progress(k, len(runs), label)
            r = self.run_nsga(base["objectives"], base["factor_bounds"], pop, prm["gens"], seed, prm["pc"],
                              prm["eta_c"], prm["eta_m"], algorithm=prm.get("algorithm", "auto"),
                              auto_stop=prm.get("auto_stop", True), stop_tol=base.get("stop_tol", 1e-3),
                              stop_window=base.get("stop_window", 20), source=base.get("source"), store=False,
                              cancel=cancel)
            out.append((label, pop, seed, r))
        allF = np.vstack([r["F"] for *_, r in out])
        ideal, nadir = allF.min(axis=0), allF.max(axis=0)
        m = allF.shape[1]
        rows = []
        lo, hi = self.coded_bounds()
        span_x = np.where(hi - lo > 0, hi - lo, 1.0)
        x0 = base["X"][base["i_topsis"]]
        for label, pop, seed, r in out:
            hv = nsga2.hypervolume(r["F"], ideal, nadir) / nsga2.hv_max(m)
            xt = r["X"][r["i_topsis"]]
            rows.append({"label": label, "pop": pop, "seed": seed, "hv": hv, "n": len(r["X"]),
                         "gens": r["gens_run"], "stop": r["stop_gen"] is not None and "converged" in
                         (r["stop_reason"] or ""), "dx": float(np.max(np.abs(xt - x0) / span_x)),
                         "y_topsis": [float(v[r["i_topsis"]]) for v in r["Yobj"]]})
        hv0 = rows[0]["hv"]
        same = [r for r in rows[1:] if r["pop"] == prm["pop"]]
        big = [r for r in rows[1:] if r["pop"] > prm["pop"]]
        spread = max([abs(r["hv"] - hv0) for r in same] + [0.0]) / max(hv0, 1e-12)
        gain_big = max([r["hv"] - hv0 for r in big] + [0.0]) / max(hv0, 1e-12)
        gens_ok = base["check"]["gens_ok"]
        pop_ok = spread < 0.01 and gain_big < 0.01
        verdict = {"spread": spread, "gain_big": gain_big, "gens_ok": gens_ok, "pop_ok": pop_ok,
                   "dx_max": max(r["dx"] for r in rows[1:]) if len(rows) > 1 else 0.0}
        return {"runs": rows, "verdict": verdict}

    def fit_summary(self, j):
        if self.analysis_kind(j) == "reml":
            return [], None
        order_fits = []
        for order in ["mean"] + self.available_orders():
            try:
                order_fits.append((order, self.fit_terms(j, self.terms_for_order(order))))
            except (ValueError, np.linalg.LinAlgError):
                break
        if self.analysis_kind(j) == "glm":
            from .glm import sequential_summary
            return sequential_summary(order_fits)
        return models.sequential_summary(order_fits)

    def backward_eliminate(self, j, alpha_out):
        return models.backward_elimination(lambda t: self.fit_terms(j, t), self.model_terms(j), self.k,
                                           alpha_out, mixture=self.is_mixture)

    def box_cox(self, j):
        if self.analysis_kind(j) != "ols":
            return None
        return models.box_cox(self.coded, self.model_data[:, j], self.model_terms(j), mixture=self.is_mixture,
                              blocks=self.blocks if self.n_blocks > 1 else None, space=self.space)

    @property
    def is_two_level_factorial(self):
        return self.design_type in ("full_factorial", "fractional_factorial", "plackett_burman") \
            and not self.cat_idx and not self.is_split_plot

    def effects(self, j):
        return models.factorial_effects(self.coded, self.model_data[:, j], self.space)

    # ---------- optimization ----------
    def optimize(self, criteria, factor_bounds):
        """factor_bounds: {i: (actual_low, actual_high)} for numeric factors."""
        lo, hi = self.coded_bounds()
        for i, (a, b) in factor_bounds.items():
            ca, cb = sorted(float(v) for v in self.factor_to_coded(i, np.array([a, b])))
            lo[i], hi[i] = max(lo[i], ca) if self.is_mixture else ca, min(hi[i], cb) if self.is_mixture else cb
        G, h = self.constraint_system()
        cats = self.cat_idx
        combos = list(itertools.product(*[range(len(self.factors[i].levels)) for i in cats])) or [()]
        sols = []
        errors = []
        for combo in combos:
            l2, h2 = lo.copy(), hi.copy()
            for i, lev in zip(cats, combo):
                l2[i] = h2[i] = lev
            try:
                sols += run_optimize(criteria, list(zip(l2, h2)), mixture=self.is_mixture, G=G, h=h,
                                     n_starts=max(8, 30 // len(combos)), mix_idx=self.comp_idx or None)
            except ValueError as exc:
                errors.append(str(exc))
        if not sols:
            raise ValueError(errors[0] if errors else "No solution found.")
        sols.sort(key=lambda s: -s[0])
        return sols

    # ---------- save / open ----------
    def to_dict(self):
        def term_list(ts):
            return None if ts is None else [list(t) for t in ts]
        return {
            "version": FILE_VERSION,
            "design_type": self.design_type,
            "options": self.options,
            "mixture_total": self.mixture_total,
            "factors": [dict(f.__dict__) for f in self.factors],
            "responses": [r.__dict__ for r in self.responses],
            "coded": [[None if np.isnan(v) else float(v) for v in row] for row in self.coded],
            "point_types": self.point_types,
            "blocks": self.blocks.tolist(),
            "run_order": self.run_order.tolist(),
            "constraints": self.constraints,
            "groups": None if self.groups is None else self.groups.tolist(),
            "htc": self.htc,
            "data": [[None if np.isnan(v) else float(v) for v in row] for row in self.data],
            "model_specs": {str(k): {**v, "terms": term_list(v["terms"])} for k, v in self.model_specs.items()},
            "opt_criteria": {str(k): v for k, v in self.opt_criteria.items()},
            "ann": {str(k): m.to_dict() for k, m in self.ann.items()},
            "model_source": {str(k): v for k, v in self.model_source.items()},
            "excluded": list(self.excluded),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d):
        n = len(d["coded"])

        def arr(rows):
            return np.array([[np.nan if v is None else v for v in row] for row in rows], float)

        return cls(
            design_type=d["design_type"], options=d.get("options", {}),
            factors=[Factor(**f) for f in d["factors"]],
            responses=[Response(**{k: v for k, v in r.items() if k in RESPONSE_KEYS}) for r in d["responses"]],
            coded=arr(d["coded"]).reshape(n, -1),
            point_types=[LEGACY_POINT_TYPES.get(t, t) for t in d["point_types"]],
            run_order=np.array(d["run_order"], int), data=arr(d["data"]).reshape(n, -1),
            blocks=np.array(d["blocks"], int) if "blocks" in d else None,
            mixture_total=float(d.get("mixture_total", 1.0)),
            constraints=d.get("constraints", []),
            groups=np.array(d["groups"], int) if d.get("groups") is not None else None,
            htc=d.get("htc", []),
            model_specs={int(k): v for k, v in d.get("model_specs", {}).items()},
            opt_criteria={(int(k) if k.isdigit() else k): v for k, v in d.get("opt_criteria", {}).items()},
        )

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1, ensure_ascii=False)
        self.path = path
        self.dirty = False

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        p = cls.from_dict(d)
        from .ann import AnnModel
        base = {}
        for k, md in sorted(d.get("ann", {}).items(), key=lambda kv: int(kv[0])):
            j = int(k)
            gid = md.get("gid")
            if gid and gid in base:                          # combined network: one object, several views
                p.ann[j] = base[gid].view(int(md.get("out", 0)))
                continue
            m = AnnModel.from_dict(md, p.space, p.coded, p.data, j)
            p.ann[j] = m
            if gid:
                base[gid] = m
        p.model_source = {int(k): v for k, v in d.get("model_source", {}).items()}
        p.excluded = [int(i) for i in d.get("excluded", [])]
        p.notes = d.get("notes", "") or ""
        p.path = path
        return p


# ------------------------------------------------------------------ sample data
def sample_project():
    """Example: extraction optimization with a 3-factor Box-Behnken design, 2 responses."""
    factors = [Factor("Temperature", "°C", 60, 80), Factor("Time", "min", 30, 90),
               Factor("Concentration", "%", 1, 3)]
    responses = [Response("Yield", "%"), Response("Purity", "%")]
    p = Project.create("bbd", factors, responses, seed=7, center_points=5)
    A, B, C = p.coded.T
    rng = np.random.default_rng(3)
    y1 = 78 + 4.2 * A + 3.1 * B + 1.8 * C + 1.6 * A * B - 0.6 * A * C + 0.4 * B * C \
        - 5.1 * A ** 2 - 3.2 * B ** 2 - 1.9 * C ** 2 + rng.normal(0, 0.6, p.n)
    y2 = 91 - 2.4 * A + 1.1 * B - 0.8 * C - 0.9 * A * B - 1.7 * A ** 2 - 0.6 * C ** 2 \
        + rng.normal(0, 0.5, p.n)
    p.data = np.column_stack([np.round(y1, 2), np.round(y2, 2)])
    return p


def sample_mixture_project():
    """Example: edible film formulation, simplex lattice {3,2} + axial points, total 100%."""
    factors = [Factor("Starch", "%", 50, 0), Factor("Glycerol", "%", 10, 0), Factor("Chitosan", "%", 10, 0)]
    responses = [Response("Tensile Strength", "MPa"), Response("Elongation", "%")]
    p = Project.create("simplex_lattice", factors, responses, seed=11, mixture_total=100,
                       lattice_degree=2, augment=True, replicate_points=True)
    A, B, C = p.coded.T
    rng = np.random.default_rng(5)
    y1 = 14.2 * A + 6.1 * B + 9.8 * C + 8.5 * A * B + 12.4 * A * C - 4.2 * B * C + rng.normal(0, 0.35, p.n)
    y2 = 18 * A + 42 * B + 25 * C + 16 * A * B - 10 * A * C + 6 * B * C + rng.normal(0, 1.0, p.n)
    p.data = np.column_stack([np.round(y1, 2), np.round(y2, 1)])
    return p


def sample_factorial_project():
    """Example: 2^4 factorial screening (unreplicated) - suitable for a half-normal plot."""
    factors = [Factor("Temperature", "°C", 30, 50), Factor("pH", "", 5, 7), Factor("Agitation", "rpm", 100, 200),
               Factor("Inoculum", "%", 2, 6)]
    p = Project.create("full_factorial", factors, [Response("Biomass", "g/L")], seed=4, center_points=0)
    A, B, C, D = p.coded.T
    rng = np.random.default_rng(8)
    p.data[:, 0] = np.round(12 + 2.1 * A + 0.2 * B + 1.6 * D + 1.3 * A * D - 0.9 * A * B
                            + rng.normal(0, 0.25, p.n), 2)
    return p


def sample_categoric_project():
    """Example: CCD with 2 numeric factors x 1 categoric factor (solvent type)."""
    factors = [Factor("Temperature", "°C", 40, 60), Factor("Time", "h", 2, 6),
               Factor("Solvent", "", 0, 0, kind="categoric", levels=["Ethanol", "Methanol", "Acetone"])]
    p = Project.create("ccd", factors, [Response("Yield", "%")], seed=9, center_points=3,
                       alpha_type="rotatable")
    A, B, C = p.coded.T
    shift = np.array([0.0, 2.5, -3.0])[C.astype(int)]
    rng = np.random.default_rng(12)
    p.data[:, 0] = np.round(30 + shift + 3 * A + 2 * B - 2.5 * A ** 2 - 1.5 * B ** 2 + 0.8 * A * B
                            + 1.2 * A * (C == 1) + rng.normal(0, 0.5, p.n), 2)
    return p


def sample_logistic_project():
    """Example: binomial response - number of seeds germinated out of 50 seeds per run (logistic regression)."""
    factors = [Factor("Temperature", "°C", 20, 35), Factor("Humidity", "%", 60, 90)]
    p = Project.create("ccd", factors, [Response("Germinated", "seeds", kind="binomial", trials=50)], seed=21,
                       center_points=5, alpha_type="rotatable")
    A, B = p.coded.T
    eta = 0.8 + 0.9 * A + 0.6 * B - 0.7 * A ** 2 - 0.3 * A * B
    p.data[:, 0] = np.random.default_rng(6).binomial(50, 1 / (1 + np.exp(-eta)))
    return p


def sample_splitplot_project():
    """Example: split-plot - oven temperature (hard to change) x time & concentration (easy to change)."""
    factors = [Factor("Oven Temperature", "°C", 150, 190, sd=2.0), Factor("Time", "min", 10, 20, sd=0.5),
               Factor("Concentration", "%", 1, 3, sd=0.1)]
    p = Project.create("full_factorial", factors, [Response("Hardness", "N")], seed=15, center_points=0,
                       replicates=2, split_plot={"htc": [0], "wp_size": 4})
    A, B, C = p.coded.T
    rng = np.random.default_rng(10)
    wp = rng.normal(0, 1.2, p.groups.max() + 1)[p.groups]
    p.data[:, 0] = np.round(50 + 4 * A + 2.5 * B - 1.5 * C + 1.2 * A * B + wp + rng.normal(0, 0.6, p.n), 2)
    return p


def sample_combined_project():
    """Example: combined mixture (3 dough components) + process (baking temperature)."""
    factors = [Factor("Flour", "%", 50, 0, upper=80), Factor("Sugar", "%", 10, 0, upper=30),
               Factor("Fat", "%", 10, 0, upper=30), Factor("Temperature", "°C", 160, 200, kind="process")]
    p = Project.create("combined_optimal", factors, [Response("Texture", "score")], seed=5, mixture_total=100,
                       model_order="quadratic|linear", criterion="I")
    x = p.coded
    rng = np.random.default_rng(4)
    p.data[:, 0] = np.round(6 * x[:, 0] + 4 * x[:, 1] + 3 * x[:, 2] + 5 * x[:, 0] * x[:, 1]
                            + x[:, 3] * (1.2 * x[:, 0] - 0.8 * x[:, 1]) + rng.normal(0, 0.15, p.n), 3)
    return p


def sample_ann_project():
    """ANN example: dye adsorption, 4 factors, 40 historical runs (Latin hypercube) with a nonlinear response."""
    rng = np.random.default_rng(2024)
    n, k = 40, 4
    u = (np.argsort(rng.random((k, n)), axis=1).T + rng.random((n, k))) / n        # Latin hypercube [0, 1]
    ph, dose, t_contact, c0 = 2 + 8 * u[:, 0], 0.5 + 3.5 * u[:, 1], 10 + 110 * u[:, 2], 20 + 180 * u[:, 3]
    qmax = 95 * np.exp(-((ph - 6.2) / 2.6) ** 2)                                       # pH optimum
    kin = 1 - np.exp(-t_contact / 28)                                                      # pseudo-first-order kinetics
    ce = c0 / (1 + 4.5 * dose)
    y = qmax * kin * (0.08 * ce / (1 + 0.08 * ce)) * (1 + 0.05 * dose) + rng.normal(0, 1.2, n)
    factors = [Factor("pH", "", 2, 10), Factor("Adsorbent Dose", "g/L", 0.5, 4),
               Factor("Contact Time", "min", 10, 120), Factor("Initial Concentration", "mg/L", 20, 200)]
    p = Project.from_table(factors, [Response("Removal Efficiency", "%")],
                           np.column_stack([np.round(ph, 2), np.round(dose, 2), np.round(t_contact, 1), np.round(c0, 1)]),
                           np.round(np.clip(y, 0, 100), 2)[:, None])
    p.dirty = False
    return p


# case study gallery: (key, title, category, function, description)
CASES = [
    ("rsm", "Extraction Optimization (Box-Behnken)", "Response Surface", sample_project,
     "3 process factors (temperature, time, concentration), 17 runs with 5 center points, 2 responses (yield and "
     "purity). Good for learning the full workflow: fit summary, ANOVA, contour/3D, desirability optimization, "
     "NSGA-II, and prediction."),
    ("mixture", "Edible Film Formulation (Mixture)", "Mixture", sample_mixture_project,
     "Simplex lattice {3,2} + axial points, total 100%. Components starch, glycerol, chitosan; responses tensile "
     "strength and elongation. Scheffé model, ternary contour, trace plot."),
    ("factorial", "Fermentation Screening (2^4 Factorial)", "Screening", sample_factorial_project,
     "Unreplicated 2-level full factorial, 4 factors. Learn to select effects with the half-normal plot and "
     "Pareto chart (Lenth)."),
    ("categoric", "Extraction with Solvent Type (Categoric)", "Response Surface", sample_categoric_project,
     "CCD with 2 numeric factors × 1 categoric factor (ethanol/methanol/acetone). Equations per level and "
     "optimization across levels."),
    ("logistic", "Seed Germination (Logistic Regression)", "GLM", sample_logistic_project,
     "Binomial response: number of seeds germinated out of 50 seeds per run. IRLS, likelihood-ratio tests, "
     "odds ratios."),
    ("splitplot", "Oven Product Hardness (Split-Plot)", "Split-Plot", sample_splitplot_project,
     "Oven temperature is hard to change (whole plot); time and concentration are easy to change. REML analysis "
     "and variance components."),
    ("combined", "Cake Dough + Baking Temperature (Combined)", "Mixture + Process", sample_combined_project,
     "I-optimal combined design with 3 mixture components and 1 process factor."),
    ("ann", "Dye Adsorption (ANN, 40 runs)", "Machine Learning", sample_ann_project,
     "Historical data, 40 runs (Latin hypercube) with a strongly nonlinear pattern: pH optimum, kinetics, Langmuir "
     "isotherm. Good for trying the data readiness assessment, automatic ANN architecture search, normalization, "
     "and the RSM vs ANN comparison."),
]


def ann_model_types():
    from .ann import AnnModel
    return (AnnModel,)


def nsga_convergence(res):
    """Quick assessment after one run: enough generations? large enough population? (no extra runs)."""
    from .nsga2 import stop_tol_eff
    hist = res["history"]
    tol, w = res.get("stop_tol", 1e-3), res.get("stop_window", 20)
    checks = [h for h in hist if h.get("conv")]
    last = checks[-1] if checks else None
    converged_at = None
    for h in checks:                     # first generation where the convergence criterion is met
        if h["gain"] < stop_tol_eff(h, tol) and h["shift"] < 0.01:
            converged_at = h["gen"]
            break
    gens_ok = bool(last is not None and last["gain"] < stop_tol_eff(last, tol) and last["shift"] < 0.01)
    pop = res["params"]["pop"]
    n_front = hist[-1]["n_front"] if hist else 0
    m = len(res["active"])
    notes = []
    if last is None:
        notes.append(f"Too few generations to assess convergence (at least {2 * w} needed).")
    elif gens_ok:
        notes.append(f"Enough generations: the mean hypervolume of the last {w} generations changed by only "
                     f"{100 * last['gain']:+.3f}% compared with the previous {w} generations (limit "
                     f"{100 * stop_tol_eff(last, tol):.2f}%) and the ideal point shifted {100 * last['shift']:.2f}% "
                     "(limit 1%).")
    else:
        notes.append(f"Not enough generations yet: the mean hypervolume of the last {w} generations is still rising "
                     f"{100 * last['gain']:+.3f}% and the ideal point shifted {100 * last['shift']:.2f}%. Increase "
                     "the maximum generations (automatic stop still ends the run once it converges).")
    sat = n_front >= pop
    if sat and m >= 3:
        notes.append(f"All {pop} individuals are already on the first front. For {m} objectives this population may "
                     "not be large enough to describe the whole front; try 2x the population and compare (Check "
                     "Population Size button).")
    elif sat:
        notes.append(f"All {pop} individuals are on the first front (normal for 2 objectives). The point density "
                     "is set by the population size; increase it if you need a denser front.")
    else:
        notes.append(f"{n_front} of {pop} individuals are on the first front; the population still has room "
                     "to explore.")
    if res.get("algorithm") == "nsga3":
        notes.append(f"NSGA-III uses {res.get('n_refs', 0)} reference points for {m} objectives "
                     f"(population {pop}).")
    elif m >= 4:
        notes.append(f"{m} objectives with NSGA-II: crowding distance is less effective. Consider NSGA-III.")
    if res.get("stop_off"):
        notes.append("Automatic stop was not used: with NSGA-II and 4 or more objectives the front advances very "
                     "slowly and noisily, and testing showed that automatic stop ends too early. All maximum "
                     "generations are run; use NSGA-III for a reliable automatic stop.")
    return {"gens_ok": gens_ok, "gain": last["gain"] if last else np.nan,
            "shift": last["shift"] if last else np.nan, "converged_at": converged_at, "saturated": sat,
            "n_front": n_front, "notes": notes}
