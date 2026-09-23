"""Plot drawing functions (used on screen and in the PDF report)."""
import matplotlib.cm as cm
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.tri import Triangulation
from scipy import stats

from doe import designs, models
from doe.optimize import overall_desirability

from .common import fmt

BLUE = "#1f5f99"
RED = "#e11d48"
COLORS = ["#1f5f99", "#ea580c", "#4f46e5", "#e11d48", "#ca8a04", "#475569", "#16a34a", "#9333ea"]
H = np.sqrt(3) / 2


def msize(n, base=20):
    """Marker size that shrinks for large data so the plot stays readable and light."""
    return base if n <= 400 else max(2.0, base * np.sqrt(400 / n))


def _stems(ax, x, y, color):
    """Vertical stems (Cook's, leverage, DFFITS); for large data, plain dots are enough."""
    if len(x) > 1500:
        ax.plot(x, y, ".", ms=2, color=color, rasterized=True)
    else:
        ax.vlines(x, 0, y, color=color, lw=2)


def fit_limits(ax, x, y, margin=0.05):
    """Set axis limits from data points, data lines, stems (vlines), and horizontal/vertical limit lines.

    The default autoscale can expand unreasonably when the plot is first drawn while the widget is still very
    small (marker size and axhline lines are read as data units), so the limits are computed here."""
    from matplotlib.collections import LineCollection
    xs, ys = [np.ravel(np.asarray(x, float))], [np.ravel(np.asarray(y, float))]
    ytr, xtr = ax.get_yaxis_transform(), ax.get_xaxis_transform()
    for ln in ax.lines:
        tr = ln.get_transform()
        xd, yd = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if tr == ax.transData:
            xs.append(xd)
            ys.append(yd)
        elif tr == ytr:
            ys.append(yd)
        elif tr == xtr:
            xs.append(xd)
    for c in ax.collections:
        if isinstance(c, LineCollection) and c.get_transform() == ax.transData:
            for seg in c.get_segments():
                if len(seg):
                    xs.append(seg[:, 0])
                    ys.append(seg[:, 1])

    def lim(parts):
        v = np.concatenate(parts)
        v = v[np.isfinite(v)]
        if not len(v):
            return None
        lo, hi = float(v.min()), float(v.max())
        pad = (hi - lo) * margin if hi > lo else (abs(lo) * 0.05 or 1.0)
        return lo - pad, hi + pad

    lx, ly = lim(xs), lim(ys)
    if lx:
        ax.set_xlim(*lx)
    if ly:
        ax.set_ylim(*ly)


def _message(fig, text):
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", color="#555", wrap=True)


# ------------------------------------------------------------------ diagnostics
def _limit_lines(ax, fit):
    ax.axhline(0, color="#888", lw=0.8)
    if fit.df_resid > 1:
        lim = stats.t.ppf(1 - 0.05 / (2 * fit.n), fit.df_resid - 1)
        for s in (-lim, lim):
            ax.axhline(s, color=RED, lw=1, ls="--")


def draw_diagnostics(fig, fit, run):
    """6 diagnostic plots. `run` = run number of each row used by the fit."""
    axs = fig.subplots(2, 4)
    r = fit.stud_ext
    valid = ~np.isnan(r)

    ax = axs[0, 0]
    if valid.sum() >= 3:
        rs = np.sort(r[valid])
        probs = (np.arange(1, len(rs) + 1) - 0.5) / len(rs)
        ax.scatter(rs, stats.norm.ppf(probs), s=msize(len(rs)), color=BLUE, zorder=3, rasterized=len(rs) > 2000)
        ax.plot([rs.min(), rs.max()], [rs.min(), rs.max()], color=RED, lw=1)
        ticks = [0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
        ax.set_yticks(stats.norm.ppf(ticks))
        ax.set_yticklabels([f"{100 * t:g}" for t in ticks])
    ax.set_title("Normal Plot of Residuals")
    ax.set_xlabel("Externally studentized residuals")
    ax.set_ylabel("Normal probability (%)")

    ax = axs[0, 1]
    ax.scatter(fit.yhat[valid], r[valid], s=msize(fit.n), color=BLUE, rasterized=fit.n > 2000)
    _limit_lines(ax, fit)
    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ext. studentized residuals")

    o = np.argsort(run)
    ax = axs[0, 2]
    if fit.n > 1500:
        ax.plot(run[o], r[o], ".", ms=2, color=BLUE, rasterized=True)
    else:
        ax.plot(run[o], r[o], "o-", ms=4, lw=0.8, color=BLUE)
    _limit_lines(ax, fit)
    ax.set_title("Residuals vs Run Order")
    ax.set_xlabel("Run number")
    ax.set_ylabel("Ext. studentized residuals")

    ax = axs[0, 3]
    ax.scatter(fit.y, fit.yhat, s=msize(fit.n), color=BLUE, rasterized=fit.n > 2000)
    lo, hi = min(fit.y.min(), fit.yhat.min()), max(fit.y.max(), fit.yhat.max())
    ax.plot([lo, hi], [lo, hi], color=RED, lw=1)
    ax.set_title("Predicted vs Actual")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    ax = axs[1, 0]
    _stems(ax, run[o], np.nan_to_num(fit.cooks[o]), BLUE)
    ax.axhline(1.0, color=RED, lw=1, ls="--", label="limit 1.0")
    ax.axhline(4 / fit.n, color="#e08e0b", lw=1, ls=":", label="4/n")
    ax.set_title("Cook's Distance")
    ax.set_xlabel("Run number")
    ax.legend(fontsize=7)

    ax = axs[1, 1]
    _stems(ax, run[o], fit.leverage[o], BLUE)
    ax.axhline(2 * fit.p / fit.n, color=RED, lw=1, ls="--", label="2p/n")
    ax.set_ylim(0, 1.05)
    ax.set_title("Leverage")
    ax.set_xlabel("Run number")
    ax.legend(fontsize=7)

    ax = axs[1, 2]
    _stems(ax, run[o], np.nan_to_num(fit.dffits[o]), BLUE)
    lim = 2 * np.sqrt(fit.p / fit.n)
    for s in (-lim, lim):
        ax.axhline(s, color=RED, lw=1, ls="--")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("DFFITS")
    ax.set_xlabel("Run number")

    for a in axs.flat[:-1]:
        a.grid(alpha=0.3)
        a.title.set_fontsize(10)

    ax = axs[1, 3]
    ax.axis("off")
    sw = fit.stats.get("shapiro_p", np.nan)
    n_out = int(np.sum(np.abs(np.nan_to_num(r)) > (stats.t.ppf(1 - 0.05 / (2 * fit.n), fit.df_resid - 1)
                                                    if fit.df_resid > 1 else np.inf)))
    lines = ["Diagnostics summary", "",
             f"n = {fit.n},  p = {fit.p}",
             f"Shapiro-Wilk p = {sw:.3f}" if not np.isnan(sw) else "Shapiro-Wilk: -",
             f"Max Cook's = {np.nanmax(fit.cooks):.3f}" if np.isfinite(np.nanmax(fit.cooks)) else "",
             f"Max leverage = {fit.leverage.max():.3f}",
             f"Outliers (|t| > Bonferroni limit): {n_out}"]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", fontsize=9, family="monospace")


def draw_leverage(fig, fit, run):
    ax = fig.add_subplot(111)
    o = np.argsort(run)
    _stems(ax, run[o], fit.leverage[o], BLUE)
    ax.axhline(2 * fit.p / fit.n, color=RED, lw=1, ls="--", label="2p/n")
    ax.set_ylim(0, 1.05)
    ax.set_title("Leverage")
    ax.set_xlabel("Run number")
    ax.legend()
    ax.grid(alpha=0.3)


def draw_box_cox(fig, bc, current_lam=None):
    if bc is None:
        _message(fig, "The Box-Cox plot needs all response values > 0\n(and at least 3 data points).")
        return
    ax = fig.add_subplot(111)
    ax.plot(bc["lambdas"], bc["lnss"], color=BLUE, lw=2)
    if not np.isnan(bc["threshold"]):
        ax.axhline(bc["threshold"], color=RED, ls="--", lw=1, label="95% CI limit")
        lo, hi = bc["ci"]
        ax.axvspan(lo, hi, color="#1a7f37", alpha=0.08)
    ax.axvline(bc["best"], color="#1a7f37", lw=1.5, label=f"Best λ = {bc['best']:.2f}")
    if current_lam is not None:
        ax.axvline(current_lam, color="#333", lw=1.2, ls=":", label=f"Current λ = {current_lam:g}")
    lo, hi = bc["ci"]
    ax.set_title(f"Box-Cox - 95% CI for λ: {lo:.2f} to {hi:.2f}")
    ax.set_xlabel("λ (lambda)")
    ax.set_ylabel("ln(SS residual)")
    ax.grid(alpha=0.3)
    ax.legend()


# ------------------------------------------------------------------ factorial effects
def draw_half_normal(fig, eff, selected, normal=False):
    """Half-normal (or normal) plot of effects. Selected points (in the model) are colored orange."""
    ax = fig.add_subplot(111)
    e = np.asarray(eff["effects"])
    names = [models.term_name(t) for t in eff["terms"]]
    m = len(e)
    if normal:
        order = np.argsort(e)
        q = stats.norm.ppf((np.arange(1, m + 1) - 0.5) / m)
        xs = e[order]
        ax.set_xlabel("Standardized effect")
        ax.set_ylabel("Normal probability (%)")
        ticks = [0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
        ax.set_yticks(stats.norm.ppf(ticks))
        ax.set_yticklabels([f"{100 * t:g}" for t in ticks])
    else:
        order = np.argsort(np.abs(e))
        q = stats.halfnorm.ppf((np.arange(1, m + 1) - 0.5) / m)
        xs = np.abs(e)[order]
        ax.set_xlabel("|Standardized effect|")
        ax.set_ylabel("Half-normal (%)")
        ticks = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        ax.set_yticks(stats.halfnorm.ppf(ticks))
        ax.set_yticklabels([f"{100 * t:g}" for t in ticks])
    sel = np.array([eff["terms"][i] in selected for i in order])
    ax.scatter(xs[~sel], q[~sel], s=36, color=BLUE, zorder=3, picker=5, label="Error")
    ax.scatter(xs[sel], q[sel], s=46, color="#e08e0b", edgecolor="k", zorder=4, picker=5, label="Selected (model)")
    for x, y, i in zip(xs, q, order):
        if eff["terms"][i] in selected or abs(e[i]) >= eff["me"]:
            ax.annotate(names[i], (x, y), xytext=(5, 2), textcoords="offset points", fontsize=9)
    err = ~sel
    if err.sum() >= 2:
        slope = np.median(xs[err] / np.where(q[err] == 0, np.nan, q[err]))
        if np.isfinite(slope) and slope > 0:
            qq = np.array([q.min() if normal else 0, q.max()])
            ax.plot(qq * slope, qq, color=RED, lw=1, ls="--")
    ax.set_title("Normal Plot of Effects" if normal else "Half-Normal Plot of Effects")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    return ax, xs, q, order


def draw_pareto(fig, eff, selected):
    ax = fig.add_subplot(111)
    t = np.asarray(eff["t"])
    order = np.argsort(-t)
    names = [models.term_name(eff["terms"][i]) for i in order]
    colors = ["#e08e0b" if eff["terms"][i] in selected else BLUE for i in order]
    ax.bar(range(len(order)), t[order], color=colors)
    ax.axhline(eff["t_limit"], color=RED, ls="--", lw=1, label=f"t limit = {eff['t_limit']:.3g}")
    ax.axhline(eff["bonferroni"], color="#333", ls=":", lw=1, label=f"Bonferroni = {eff['bonferroni']:.3g}")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(names, rotation=90 if len(order) > 12 else 0)
    ax.set_ylabel("|t| of effect")
    ax.set_title("Pareto Chart" + (" (based on Lenth's PSE)" if eff["df_resid"] <= 0 else ""))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


# ------------------------------------------------------------------ slice grid
class Grid:
    """Point grid for 2D plots: square (process) or ternary triangle (mixture)."""

    def __init__(self, project, axes, base, n=60):
        self.p = project
        self.axes = list(axes)
        self.base = np.array(base, float)
        self.tern = project.is_mixture
        self.empty = False
        if self.tern:
            ia, ib, ic = self.axes
            others = [i for i in project.comp_idx if i not in self.axes]
            self.rest = 1.0 - float(sum(self.base[i] for i in others))
            if self.rest <= 1e-9:
                self.empty = True
                return
            for res in (n, 3 * n):
                ii, jj = np.meshgrid(np.arange(res + 1), np.arange(res + 1), indexing="ij")
                keep = ii + jj <= res
                aa, cc = ii[keep] / res, jj[keep] / res
                bb = 1 - aa - cc
                pts = np.tile(self.base, (len(aa), 1))
                pts[:, ia], pts[:, ib], pts[:, ic] = aa * self.rest, bb * self.rest, cc * self.rest
                self.ok = project.feasible(pts)
                if self.ok.mean() > 0.2:
                    break  # feasible region is large enough; no need for a finer grid
            self.X, self.Y = cc + aa / 2, aa * H
            self.tri = Triangulation(self.X, self.Y)
            self.pts = pts
            # triangles entirely outside the region are hidden; edges are clipped with an exact polygon
            self.tri.set_mask(np.all(~self.ok[self.tri.triangles], axis=1))
            self.empty = not self.ok.any()
            self.poly = self._feasible_polygon()
        else:
            xi, yi = self.axes
            g = np.linspace(-1, 1, n)
            GX, GY = np.meshgrid(g, g)
            pts = np.tile(self.base, (GX.size, 1))
            pts[:, xi], pts[:, yi] = GX.ravel(), GY.ravel()
            self.shape = GX.shape
            self.AX = project.factor_to_actual(xi, GX)
            self.AY = project.factor_to_actual(yi, GY)
            self.pts = pts
            self.ok = project.feasible(pts)
            self.empty = not self.ok.any()
            self.poly = self._feasible_polygon()

    # --- feasible region polygon (exact, from bounds and linear constraints)
    def _halfplanes(self):
        """Half-planes a·u <= b in 2D plot coordinates (ternary: u = (a, c); square: u = coded (x, y))."""
        p = self.p
        k = p.k
        rows = []
        G, h = p.constraint_system() if p.constraints else (np.zeros((0, k)), np.zeros(0))
        lo, hi = p.coded_bounds()
        cons = [(G[r], h[r]) for r in range(len(G))]
        if self.tern:
            for i in self.axes:
                e = np.zeros(k)
                e[i] = 1
                cons += [(e, hi[i]), (-e, -lo[i])]
            ia, ib, ic = self.axes
            for g, hv in cons:
                other = sum(g[i] * self.base[i] for i in range(k) if i not in self.axes)
                a1 = self.rest * (g[ia] - g[ib])
                a2 = self.rest * (g[ic] - g[ib])
                rows.append((a1, a2, hv - other - self.rest * g[ib]))
            rows += [(-1, 0, 0), (0, -1, 0), (1, 1, 1)]
        else:
            xi, yi = self.axes
            for g, hv in cons:
                other = sum(g[i] * self.base[i] for i in range(k) if i not in self.axes)
                rows.append((g[xi], g[yi], hv - other))
            rows += [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)]
        return np.array(rows, float)

    def _feasible_polygon(self):
        if self.ok.all() or not self.ok.any():
            return None
        R = self._halfplanes()
        verts = []
        for i in range(len(R)):
            for j in range(i + 1, len(R)):
                A = R[[i, j], :2]
                if abs(np.linalg.det(A)) < 1e-12:
                    continue
                u = np.linalg.solve(A, R[[i, j], 2])
                if np.all(R[:, :2] @ u <= R[:, 2] + 1e-9):
                    verts.append(u)
        if len(verts) < 3:
            return None
        V = np.unique(np.round(np.array(verts), 10), axis=0)
        c = V.mean(axis=0)
        V = V[np.argsort(np.arctan2(V[:, 1] - c[1], V[:, 0] - c[0]))]
        if self.tern:
            return np.column_stack([V[:, 1] + V[:, 0] / 2, V[:, 0] * H])
        return np.column_stack([self.p.factor_to_actual(self.axes[0], V[:, 0]),
                                self.p.factor_to_actual(self.axes[1], V[:, 1])])

    def _clip(self, ax, artist):
        if self.poly is None or artist is None:
            return artist
        from matplotlib.patches import Polygon
        patch = Polygon(self.poly, closed=True, transform=ax.transData, fill=False, visible=False)
        ax.add_patch(patch)
        artist.set_clip_path(patch)
        return artist

    def clabel(self, ax, cs, **kw):
        labels = ax.clabel(cs, **kw)
        if self.poly is not None:
            from matplotlib.path import Path
            path = Path(self.poly)
            for t in labels:
                if not path.contains_point(t.get_position(), radius=1e-9):
                    t.set_visible(False)
        return labels

    def masked(self, Z):
        Z = np.asarray(Z, float)
        return Z if self.tern else Z.reshape(self.shape)

    def fill(self, ax, Z, levels=12, cmap="viridis", **kw):
        if "colors" in kw:
            cmap = None
        if self.tern:
            return self._clip(ax, ax.tricontourf(self.tri, Z, levels=levels, cmap=cmap, **kw))
        return self._clip(ax, ax.contourf(self.AX, self.AY, self.masked(Z), levels=levels, cmap=cmap, **kw))

    def lines(self, ax, Z, levels=12, **kw):
        if self.tern:
            return self._clip(ax, ax.tricontour(self.tri, Z, levels=levels, **kw))
        return self._clip(ax, ax.contour(self.AX, self.AY, self.masked(Z), levels=levels, **kw))

    def xy(self, coded):
        coded = np.atleast_2d(coded)
        if self.tern:
            s = coded[:, self.axes] / self.rest
            return s[:, 2] + s[:, 0] / 2, s[:, 0] * H
        return (self.p.factor_to_actual(self.axes[0], coded[:, self.axes[0]]),
                self.p.factor_to_actual(self.axes[1], coded[:, self.axes[1]]))

    def slice_mask(self, coded, tol=1e-3):
        coded = np.atleast_2d(coded)
        m = np.ones(len(coded), bool)
        for i in range(self.p.k):
            if i not in self.axes:
                m &= np.abs(coded[:, i] - self.base[i]) < tol
        if not self.tern:
            m &= np.all(np.abs(coded[:, self.axes]) <= 1 + 1e-9, axis=1)
        return m

    def decorate(self, ax):
        p = self.p
        if not self.tern:
            ax.set_xlabel(p.factor_label(self.axes[0]))
            ax.set_ylabel(p.factor_label(self.axes[1]))
            if self.poly is not None:
                ax.set_facecolor("#d0d4da")
            return
        ax.plot([0, 1, 0.5, 0], [0, 0, H, 0], color="k", lw=1.2)
        if self.poly is not None:
            ax.fill([0, 1, 0.5], [0, 0, H], color="#d9dde2", zorder=0)
        # scale lines per component in actual units
        okp = self.pts[self.ok]
        for n_ax, i in enumerate(self.axes):
            col = COLORS[n_ax]
            vals = p.factor_to_actual(i, okp[:, i])
            ticks = [t for t in MaxNLocator(5).tick_values(vals.min(), vals.max())
                     if vals.min() - 1e-9 <= t <= vals.max() + 1e-9]
            for v in ticks:
                f = float(p.factor_to_coded(i, v)) / self.rest
                u = np.linspace(0, 1 - f, 60)
                others = [n for n in range(3) if n != n_ax]
                parts = {n_ax: np.full_like(u, f), others[0]: u, others[1]: 1 - f - u}
                a, b, c = parts[0], parts[1], parts[2]
                x, y = c + a / 2, a * H
                ax.plot(x, y, color=col, lw=0.6, alpha=0.55, zorder=1)
                pts = np.tile(self.base, (len(u), 1))
                pts[:, self.axes[0]], pts[:, self.axes[1]], pts[:, self.axes[2]] = (a * self.rest, b * self.rest,
                                                                                     c * self.rest)
                inside = np.where(p.feasible(pts))[0]
                if len(inside):
                    m = inside[0]
                    ax.text(x[m], y[m], fmt(v, 4), color=col, fontsize=7, ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7), zorder=6)
        names = [p.factor_label(i) for i in self.axes]
        xs, ys = self.X[self.ok], self.Y[self.ok]
        zoom = len(xs) and (np.ptp(xs) < 0.8 or np.ptp(ys) < 0.7)
        if zoom:
            mx = max(np.ptp(xs), np.ptp(ys)) * 0.12 + 0.01
            ax.set_xlim(xs.min() - mx, xs.max() + mx)
            ax.set_ylim(ys.min() - mx, ys.max() + mx)
            ax.set_aspect("equal")
            ax.axis("off")
            for n_ax, nm in enumerate(names):
                ax.text(0.01, 0.99 - 0.04 * n_ax, f"- {nm}", color=COLORS[n_ax], transform=ax.transAxes,
                        va="top", fontsize=8, fontweight="bold")
            ax.text(0.5, 0.005, "Zoomed to the feasible region. Numbers on lines = component amount (actual units).",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color="#555")
            return
        tops = [p.factor_to_actual(i, self.rest) for i in self.axes]
        ax.text(0.5, H + 0.04, f"{names[0]}\n{fmt(float(tops[0]))}", ha="center", va="bottom", fontsize=9,
                color=COLORS[0])
        ax.text(-0.03, -0.04, f"{names[1]}\n{fmt(float(tops[1]))}", ha="right", va="top", fontsize=9,
                color=COLORS[1])
        ax.text(1.03, -0.04, f"{names[2]}\n{fmt(float(tops[2]))}", ha="left", va="top", fontsize=9,
                color=COLORS[2])
        ax.text(0.5, -0.06, "Numbers at corners = component amount at the vertex (actual units). Gray area = outside "
                "bounds/constraints.", ha="center", va="top", fontsize=7, color="#555")
        ax.set_aspect("equal")
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(-0.2, H + 0.18)
        ax.axis("off")

    def design_points(self, ax, fit_or_none=None, coded=None):
        p = self.p
        coded = p.coded if coded is None else coded
        ok = np.all(np.isfinite(coded), axis=1)
        pts = coded[ok][self.slice_mask(coded[ok])]
        if not len(pts):
            return
        sub = pts[:, self.axes]
        uniq, idx, counts = np.unique(np.round(sub, 6), axis=0, return_index=True, return_counts=True)
        x, y = self.xy(pts[idx])
        big = len(x) > 300
        ax.scatter(x, y, color="red", edgecolor="none" if big else "white", s=msize(len(x), 40), zorder=5,
                   clip_on=False, rasterized=big)
        for xx, yy, c in zip(x, y, counts):
            if c > 1 and not big:
                ax.annotate(str(c), (xx, yy), textcoords="offset points", xytext=(6, 4), color="red",
                            fontweight="bold")

    def mark(self, ax, coded, label="Optimum"):
        if coded is None:
            return
        if not self.slice_mask(coded, tol=2e-2)[0]:
            return
        x, y = self.xy(coded)
        ax.scatter(x, y, marker="*", s=260, color="#ffd400", edgecolor="k", zorder=7)
        ax.annotate(label, (x[0], y[0]), textcoords="offset points", xytext=(8, -12), fontsize=9,
                    fontweight="bold")


# ------------------------------------------------------------------ response graphs
def draw_response_2d(fig, project, fit, axes, base, resp_label, kind="contour", levels=12, mark=None):
    grid = Grid(project, axes, base)
    if grid.empty:
        _message(fig, "This slice lies entirely outside the design region.\n"
                      "Change the values of the other factors/components.")
        return
    Z = fit.predict(grid.pts)
    if kind == "contour":
        ax = fig.add_subplot(111)
        grid.decorate(ax)
        cf = grid.fill(ax, Z, levels)
        cs = grid.lines(ax, Z, levels, colors="k", linewidths=0.5)
        grid.clabel(ax, cs, inline=True, fontsize=8, fmt=lambda v: fmt(v, 4))
        fig.colorbar(cf, ax=ax, label=resp_label, shrink=0.85 if grid.tern else 1.0)
        grid.design_points(ax)
        grid.mark(ax, mark)
    else:
        ax = fig.add_subplot(111, projection="3d")
        if grid.tern:
            surf = ax.plot_trisurf(grid.tri, Z, cmap="viridis", edgecolor="none", alpha=0.92)
            ax.plot([0, 1, 0.5, 0], [0, 0, H, 0], [np.nanmin(Z)] * 4, color="k", lw=1)
            for pos, i in zip(((0.5, H), (0, 0), (1, 0)), grid.axes):
                ax.text(pos[0], pos[1], np.nanmin(Z), project.factor_label(i), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            Zs = np.where(grid.ok, Z, np.nan).reshape(grid.shape)
            surf = ax.plot_surface(grid.AX, grid.AY, Zs, cmap="viridis", edgecolor="none", alpha=0.92)
            ax.set_xlabel(project.factor_label(axes[0]))
            ax.set_ylabel(project.factor_label(axes[1]))
        fig.colorbar(surf, ax=ax, shrink=0.6, label=resp_label)
        coded = project.coded[fit.rows_used]
        m = grid.slice_mask(coded)
        if m.any():
            x, y = grid.xy(coded[m])
            ax.scatter(x, y, fit.y_orig[m], color="red", s=25, depthshade=False)
        ax.set_zlabel(resp_label)
    ax.set_title(resp_label)


# compatibility with the old name
def draw_process_2d(fig, project, fit, xi, yi, base, resp_label, kind="contour", levels=12, mark=None):
    draw_response_2d(fig, project, fit, [xi, yi], base, resp_label, kind, levels, mark)


def draw_ternary(fig, project, fit, comps, fixed, resp_label, kind="contour", levels=12, mark=None):
    base = np.zeros(project.k)
    for i, v in fixed.items():
        base[i] = v
    draw_response_2d(fig, project, fit, comps, base, resp_label, kind, levels, mark)


def slice_points(project, fit, used, base):
    coded = project.coded[fit.rows_used]
    mask = np.ones(len(coded), bool)
    for i in range(project.k):
        if i not in used:
            mask &= np.abs(coded[:, i] - base[i]) < 1e-3
    return coded[mask], fit.y_orig[mask]


def _levels_of(project, i):
    """Coded points along factor i: numeric -> continuous -1..1, categoric -> all levels."""
    f = project.factors[i]
    if f.categoric:
        return np.arange(len(f.levels), dtype=float), list(f.levels)
    return np.linspace(-1, 1, 100), None


def draw_oneway(fig, project, fit, xi, base, resp_label):
    ax = fig.add_subplot(111)
    g, labels = _levels_of(project, xi)
    pts = np.tile(base, (g.size, 1))
    pts[:, xi] = g
    yhat = fit.predict(pts)
    pts_d, yv = slice_points(project, fit, [xi], base)
    if labels:
        lo, hi = fit.confidence_band(pts) if fit.df_resid > 0 else (yhat, yhat)
        ax.errorbar(g, yhat, yerr=[yhat - lo, hi - yhat], fmt="o", color=BLUE, capsize=6, ms=8,
                    label="Predicted ± 95% CI")
        if len(pts_d):
            ax.scatter(pts_d[:, xi] + 0.08, yv, color="red", s=24, zorder=5, label="Design data")
        ax.set_xticks(g)
        ax.set_xticklabels(labels)
    else:
        xa = project.factor_to_actual(xi, g)
        ax.plot(xa, yhat, color=BLUE, lw=2, label="Predicted")
        if fit.df_resid > 0:
            lo, hi = fit.confidence_band(pts)
            ax.fill_between(xa, lo, hi, color=BLUE, alpha=0.15, label="95% confidence interval")
        if len(pts_d):
            ax.scatter(project.factor_to_actual(xi, pts_d[:, xi]), yv, color="red", s=30, zorder=5,
                       label="Design data")
    ax.set_xlabel(project.factor_label(xi))
    ax.set_ylabel(resp_label)
    ax.set_title(f"Effect of {project.factors[xi].name}")
    ax.grid(alpha=0.3)
    ax.legend()


def draw_all_factors(fig, project, fit, base, resp_label):
    """One-factor effects for all factors at once (shared Y axis so effect sizes can be compared)."""
    k = project.k
    ncol = min(3, k)
    nrow = int(np.ceil(k / ncol))
    axs = np.atleast_1d(fig.subplots(nrow, ncol, sharey=True, squeeze=False)).ravel()
    ci = fit.df_resid > 0
    for xi, ax in zip(range(k), axs):
        g, labels = _levels_of(project, xi)
        pts = np.tile(base, (g.size, 1))
        pts[:, xi] = g
        yhat = fit.predict(pts)
        lo, hi = fit.confidence_band(pts) if ci else (yhat, yhat)
        col = COLORS[xi % len(COLORS)]
        if labels:
            ax.errorbar(np.arange(len(g)), yhat, yerr=[yhat - lo, hi - yhat], fmt="o", color=col, capsize=5, ms=6)
            ax.set_xticks(np.arange(len(g)), labels)
        else:
            xa = project.factor_to_actual(xi, g)
            ax.plot(xa, yhat, color=col, lw=2)
            if ci:
                ax.fill_between(xa, lo, hi, color=col, alpha=0.15)
            ax.axvline(project.factor_to_actual(xi, base[xi]), color="#94a3b8", ls=":", lw=1)
        ax.set_xlabel(project.factor_label(xi), fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axs[k:]:
        ax.axis("off")
    for r in range(nrow):
        axs[r * ncol].set_ylabel(resp_label, fontsize=9)
    ref = ", ".join(f"{project.factors[i].name}={project.format_value(i, project.factor_to_actual(i, base[i]), 4)}"
                    for i in range(k))
    fig.suptitle(f"All Factors - other factors at the reference point: {ref}" + (" (band = 95% CI)" if ci else ""),
                 fontsize=10)


def draw_interaction(fig, project, fit, xi, ti, base, resp_label):
    """Interaction plot: X axis = factor xi, lines = levels of factor ti (low/high or each level)."""
    ax = fig.add_subplot(111)
    g, xlabels = _levels_of(project, xi)
    ft = project.factors[ti]
    if ft.categoric:
        traces = [(j, lv) for j, lv in enumerate(ft.levels)]
    else:
        traces = [(-1.0, f"{ft.name} low ({fmt(ft.low)})"), (1.0, f"{ft.name} high ({fmt(ft.high)})")]
    xa = g if xlabels else project.factor_to_actual(xi, g)
    for n, (v, lab) in enumerate(traces):
        pts = np.tile(base, (g.size, 1))
        pts[:, xi] = g
        pts[:, ti] = v
        yhat = fit.predict(pts)
        col = COLORS[n % len(COLORS)]
        if xlabels:
            lo, hi = fit.confidence_band(pts) if fit.df_resid > 0 else (yhat, yhat)
            ax.errorbar(xa + (n - len(traces) / 2) * 0.05, yhat, yerr=[yhat - lo, hi - yhat], fmt="o-",
                        color=col, capsize=5, label=str(lab))
        else:
            ax.plot(xa, yhat, color=col, lw=2, label=str(lab))
            if fit.df_resid > 0:
                ends = pts[[0, -1]]
                lo, hi = fit.confidence_band(ends)
                ye = fit.predict(ends)
                ax.errorbar(xa[[0, -1]], ye, yerr=[ye - lo, hi - ye], fmt="none", ecolor=col, capsize=5)
    if xlabels:
        ax.set_xticks(g)
        ax.set_xticklabels(xlabels)
    ax.set_xlabel(project.factor_label(xi))
    ax.set_ylabel(resp_label)
    ax.set_title(f"Interaction {designs.LETTERS[xi]}{designs.LETTERS[ti]}")
    ax.grid(alpha=0.3)
    ax.legend()


def draw_cube(fig, project, fit, axes3, base, resp_label):
    """Cube plot: predictions at the 8 cube corners (3 numeric factors)."""
    ax = fig.add_subplot(111)
    i, j, k = axes3
    ox, oy = 0.45, 0.35  # oblique projection for the third axis
    corners = {}
    for a in (-1, 1):
        for b in (-1, 1):
            for c in (-1, 1):
                pt = base.copy()
                pt[i], pt[j], pt[k] = a, b, c
                x = (a + 1) / 2 + ((c + 1) / 2) * ox
                y = (b + 1) / 2 + ((c + 1) / 2) * oy
                corners[(a, b, c)] = (x, y, float(fit.predict(pt[None, :])[0]))
    for (a, b, c), (x, y, _) in corners.items():
        for d in range(3):
            nb = [a, b, c]
            if nb[d] == -1:
                nb[d] = 1
                x2, y2, _ = corners[tuple(nb)]
                ax.plot([x, x2], [y, y2], color="#555", lw=1.2)
    vals = np.array([v for _, _, v in corners.values()])
    norm = (vals - vals.min()) / (np.ptp(vals) or 1)
    for ((a, b, c), (x, y, v)), nv in zip(corners.items(), norm):
        ax.scatter([x], [y], s=900, color=cm.viridis(nv), edgecolor="k", zorder=3)
        ax.text(x, y, fmt(v, 4), ha="center", va="center", fontsize=9, zorder=4,
                color="white" if nv < 0.55 else "black", fontweight="bold")
    fi, fj, fk = (project.factors[q] for q in (i, j, k))
    ax.text(0.5, -0.12, f"{project.factor_label(i)}: {fmt(fi.low)} → {fmt(fi.high)}", ha="center")
    ax.text(-0.14, 0.5, f"{project.factor_label(j)}: {fmt(fj.low)} → {fmt(fj.high)}", ha="center",
            va="center", rotation=90)
    ax.text(1 + ox / 2 + 0.12, oy / 2 - 0.05, f"{project.factor_label(k)}:\n{fmt(fk.low)} → {fmt(fk.high)}",
            ha="left", va="center", rotation=0)
    ax.set_xlim(-0.25, 1 + ox + 0.45)
    ax.set_ylim(-0.25, 1 + oy + 0.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Cube Plot - {resp_label}")


def draw_perturbation(fig, project, fit, base, resp_label):
    """Perturbation plot: each numeric factor is moved from the reference point, other factors stay fixed."""
    ax = fig.add_subplot(111)
    g = np.linspace(-1, 1, 81)
    for n, i in enumerate(project.num_idx):
        pts = np.tile(base, (g.size, 1))
        pts[:, i] = base[i] + g if project.is_mixture else g
        col = COLORS[n % len(COLORS)]
        ax.plot(g, fit.predict(pts), lw=2, color=col, label=project.factor_label(i))
        ax.annotate(designs.LETTERS[i], (1, fit.predict(pts[-1:])[0]), color=col, fontweight="bold",
                    xytext=(4, 0), textcoords="offset points")
    ax.axvline(0, color="#888", lw=0.8)
    ref = ", ".join(f"{project.factors[i].name}={project.format_value(i, project.factor_to_actual(i, base[i]), 4)}"
                    for i in range(project.k))
    ax.set_xlabel("Deviation from reference point (coded units)")
    ax.set_ylabel(resp_label)
    ax.set_title(f"Perturbation - reference: {ref}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()


# ------------------------------------------------------------------ graphical optimization
def draw_overlay(fig, project, items, axes, base, mark=None):
    """items: list of (label, fit, lower_limit, upper_limit). Yellow area = all criteria met."""
    grid = Grid(project, axes, base)
    if grid.empty:
        _message(fig, "This slice lies entirely outside the design region.")
        return
    ax = fig.add_subplot(111)
    grid.decorate(ax)
    good = grid.ok.copy()
    for n, (lab, fit, lo, hi) in enumerate(items):
        Z = fit.predict(grid.pts)
        good &= (Z >= lo) & (Z <= hi)
        col = COLORS[n % len(COLORS)]
        for lim, name in ((lo, "lower"), (hi, "upper")):
            if np.nanmin(Z) < lim < np.nanmax(Z):
                cs = grid.lines(ax, Z, levels=[lim], colors=[col], linewidths=1.8)
                grid.clabel(ax, cs, fmt=lambda v, lab=lab: f"{lab}: {fmt(v, 4)}", fontsize=8)
    grid.fill(ax, good.astype(float), levels=[-0.5, 0.5, 1.5], colors=["#e4e7eb", "#ffe066"], alpha=0.95,
              zorder=0.5)
    if not grid.tern and grid.poly is not None:
        ax.set_facecolor("#8d96a0")
    from matplotlib.patches import Patch
    handles = [Patch(color="#ffe066", label="Meets all criteria"),
               Patch(color="#e4e7eb", label="Does not meet")]
    if not grid.ok.all():
        handles.append(Patch(color="#8d96a0" if not grid.tern else "#bdbdbd", label="Outside bounds/constraints"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    grid.design_points(ax)
    grid.mark(ax, mark)
    ax.set_title("Overlay Plot - yellow = meets all criteria" + ("" if good.any() else " (NONE)"))


def draw_desirability(fig, project, criteria, axes, base, levels=12, mark=None):
    grid = Grid(project, axes, base)
    if grid.empty:
        _message(fig, "This slice lies entirely outside the design region.")
        return
    ax = fig.add_subplot(111)
    grid.decorate(ax)
    D = overall_desirability(grid.pts, criteria)
    lv = np.linspace(0, 1, levels + 1)
    cf = grid.fill(ax, D, lv, cmap="RdYlGn")
    cs = grid.lines(ax, D, lv[1:-1], colors="k", linewidths=0.5)
    grid.clabel(ax, cs, inline=True, fontsize=8, fmt=lambda v: f"{v:.2f}")
    fig.colorbar(cf, ax=ax, label="Desirability", shrink=0.85 if grid.tern else 1.0)
    grid.design_points(ax)
    grid.mark(ax, mark, "Optimum")
    ax.set_title("Desirability Contour")


# ------------------------------------------------------------------ mixture
def draw_trace(fig, project, fit, resp_label, ref=None):
    """Trace plot in Piepel direction from the reference blend (default: design centroid)."""
    ax = fig.add_subplot(111)
    comp = project.comp_idx
    ref = np.nanmean(project.coded, axis=0) if ref is None else np.asarray(ref, float)
    t = np.linspace(0, 1, 201)
    for i in comp:
        pts = np.tile(ref, (len(t), 1))
        scale = (1 - t) / max(1 - ref[i], 1e-9)
        for c in comp:
            pts[:, c] = ref[c] * scale
        pts[:, i] = t
        ok = project.feasible(pts)
        if not ok.any():
            continue
        dev = t - ref[i]
        yv = np.where(ok, fit.predict(pts), np.nan)
        col = COLORS[i % len(COLORS)]
        ax.plot(dev, yv, lw=2, color=col, label=project.factor_label(i))
        last = np.where(ok)[0][-1]
        ax.annotate(designs.LETTERS[i], (dev[last], yv[last]), color=col, fontweight="bold",
                    xytext=(4, 0), textcoords="offset points")
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("Deviation from reference blend (pseudo-components)")
    ax.set_ylabel(resp_label)
    ax.set_title("Trace Plot (Piepel direction, reference = design centroid)")
    ax.grid(alpha=0.3)
    ax.legend()


def draw_poe(fig, project, fit, axes, base, resp_label, levels=12, mark=None):
    """Propagation of error contour: response variation transmitted from factor variation."""
    grid = Grid(project, axes, base)
    if grid.empty:
        _message(fig, "This slice lies entirely outside the design region.")
        return
    Z = models.poe(fit, grid.pts, project.sd_coded())
    ax = fig.add_subplot(111)
    grid.decorate(ax)
    cf = grid.fill(ax, Z, levels, cmap="magma_r")
    cs = grid.lines(ax, Z, levels, colors="k", linewidths=0.5)
    grid.clabel(ax, cs, inline=True, fontsize=8, fmt=lambda v: fmt(v, 3))
    fig.colorbar(cf, ax=ax, label=f"POE {resp_label}", shrink=0.85 if grid.tern else 1.0)
    grid.design_points(ax)
    grid.mark(ax, mark)
    sds = ", ".join(f"{project.factors[i].name} σ={fmt(project.factors[i].sd)}" for i in range(project.k)
                    if project.factors[i].sd)
    ax.set_title(f"Propagation of Error - {resp_label}\n({sds})", fontsize=10)
