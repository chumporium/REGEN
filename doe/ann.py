"""Artificial Neural Network (MLP, 1–5 hidden layers, one or more outputs) trained with Levenberg-Marquardt /
Bayesian Regularization.

Follows common practice (MATLAB `feedforwardnet` + `trainlm` / `trainbr`):
- inputs & outputs are normalized (default mapminmax to [-1, 1]; also [0, 1], z-score, or none); normalization
  parameters are computed from the training (+ validation) data only so the test data does not "leak"
- data is split into training / validation / test; early stopping: training stops when the validation error has
  not improved for `max_fail` consecutive epochs, then the best weights (lowest validation error) are used - for
  both trainlm and trainbr
- several random initializations (restarts); the one with the lowest training+validation error is kept
- one network can model several responses at once (multi-output); `model.view(k)` gives a single-response
  interface (output k) so it can be used by graphs, optimization, and prediction like an RSM model
"""
import copy
import uuid

import numpy as np

from . import parallel

ACTIVATIONS = {"tansig": "tansig (tanh)", "logsig": "logsig (sigmoid)"}
NORMS = {"mapminmax": "Min-max to [-1, 1] (mapminmax) - recommended",
         "minmax01": "Min-max to [0, 1]",
         "zscore": "Z-score standardization (mapstd): mean 0, SD 1",
         "none": "No normalization (for comparison)"}
LM_MAX_WEIGHTS = 2000       # above this LM/Bayesian is too heavy (W x W matrix) -> switch to Adam automatically
LM_MAX_JACOBIAN = 30e6      # Jacobian element limit (training rows x outputs x weights) for LM/Bayesian
ALGORITHMS = {"trainbr": "Bayesian Regularization (trainbr) - recommended for small DOE data",
              "trainlm": "Levenberg-Marquardt (trainlm)",
              "adam": "Adam mini-batch - for very large networks / very large data"}
STATUS = {"good": "Good fit", "under": "Underfitting", "over": "Overfitting", "?": "Cannot be assessed yet"}
LARGE_N = 400          # from this many rows on, architecture search uses a hold-out instead of k-fold
SEARCH_MAX_ROWS = 4000  # larger data: architecture search uses a random sample of this size
PARALLEL_MIN_WORK = 3e7  # minimum estimated workload (rows x weights x epochs x restarts) for parallel to pay off


def _act(a, kind):
    if kind == "logsig":
        s = 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500)))
        return s, s * (1 - s)
    t = np.tanh(a)
    return t, 1 - t ** 2


def encode(coded, space):
    """Coded factors -> network input columns (categorical: one-hot)."""
    coded = np.atleast_2d(np.asarray(coded, float))
    cols, names = [], []
    for i in range(coded.shape[1]):
        L = space.cat[i] if space is not None else 0
        if L:
            idx = np.rint(coded[:, i]).astype(int)
            for j in range(L):
                cols.append((idx == j).astype(float))
                names.append((i, j))
        else:
            cols.append(coded[:, i])
            names.append((i, None))
    return np.column_stack(cols), names


def norm_fit(A, method="mapminmax"):
    """Per-column normalization parameters: xn = base + (x - off) / scale."""
    A = np.asarray(A, float)
    if method == "mapminmax":
        off, scale, base = A.min(axis=0), (A.max(axis=0) - A.min(axis=0)) / 2, -1.0
    elif method == "minmax01":
        off, scale, base = A.min(axis=0), A.max(axis=0) - A.min(axis=0), 0.0
    elif method == "zscore":
        off = A.mean(axis=0)
        scale, base = (A.std(axis=0, ddof=1) if len(A) > 1 else np.zeros(A.shape[1])), 0.0
    elif method == "none":
        off, scale, base = np.zeros(A.shape[1]), np.ones(A.shape[1]), 0.0
    else:
        raise ValueError(f"Unknown normalization method '{method}'.")
    scale = np.where(scale > 0, scale, 1.0)
    return off, scale, base


def normalize(A, method="mapminmax"):
    """Normalize a table (columns = variables). -> (normalized table, off, scale, base)."""
    A = np.asarray(A, float)
    ok = np.isfinite(A)
    out = np.full(A.shape, np.nan)
    offs, scales = np.zeros(A.shape[1]), np.ones(A.shape[1])
    base = 0.0
    for c in range(A.shape[1]):
        col = A[ok[:, c], c]
        if not len(col):
            continue
        o, sc, base = norm_fit(col[:, None], method)
        offs[c], scales[c] = o[0], sc[0]
        out[ok[:, c], c] = base + (col - o[0]) / sc[0]
    return out, offs, scales, base


def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    n = len(y)
    if n == 0:
        return {"n": 0}
    e = y - yhat
    sst = float(((y - y.mean()) ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        aad = float(np.mean(np.abs(e) / np.abs(y)) * 100) if np.all(y != 0) else np.nan
        r = float(np.corrcoef(y, yhat)[0, 1]) if n > 1 and np.ptp(y) > 0 and np.ptp(yhat) > 0 else np.nan
    return {"n": n, "r2": 1 - float(e @ e) / sst if sst > 0 else np.nan, "r": r, "mse": float(np.mean(e ** 2)),
            "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))), "aad": aad}


def n_weights_for(n_in, layers, n_out=1):
    sz = [n_in] + list(layers)
    return sum(sz[l + 1] * (sz[l] + 1) for l in range(len(layers))) + n_out * (layers[-1] + 1)


def complete_rows(coded, Y):
    coded, Y = np.asarray(coded, float), np.asarray(Y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    return np.where(np.all(np.isfinite(Y), axis=1) & np.all(np.isfinite(coded), axis=1))[0]


def cv_folds(n, k, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return [np.sort(f) for f in np.array_split(idx, k)]


def diagnose(train_nmse, val_nmse=np.nan, test_nmse=np.nan, noise_nmse=None):
    """Fit status from normalized errors (MSE / response variance; 0 = perfect, 1 = no better than the mean).

    - underfitting: the model cannot follow the training data (low training R²) and the error on new data is similar
    - overfitting : the error on new data is far above the training error, or the training error is below the
                    measurement noise level
    """
    gen = np.nanmean([v for v in (val_nmse, test_nmse) if v is not None and np.isfinite(v)] or [np.nan])
    msgs = []
    noise_ok = noise_nmse is not None and np.isfinite(noise_nmse) and noise_nmse > 0
    if noise_ok and train_nmse < 0.5 * noise_nmse:
        msgs.append("training error is below the measurement noise - the network is memorizing noise")
        return "over", msgs
    if np.isfinite(gen):
        if gen > 2.0 * train_nmse and gen - train_nmse > 0.02:
            msgs.append(f"error on new data is {gen / max(train_nmse, 1e-12):.1f}× the training error")
            return "over", msgs
        limit = max(3.0 * noise_nmse, 0.1) if noise_ok else 0.2
        if train_nmse > limit and gen < 1.5 * train_nmse + 0.02:
            msgs.append(f"training R² is only {1 - train_nmse:.3f} and the error on new data is similar - "
                        "network capacity is too low")
            return "under", msgs
        return "good", msgs
    if train_nmse > 0.2:
        msgs.append(f"training R² is only {1 - train_nmse:.3f}")
        return "under", msgs
    return "?", ["no validation/test data to assess overfitting"]


class AnnModel:
    """Trained ANN model. Its interface mimics FitResult so it can be used by graphs, optimization, and prediction.

    For multi-output networks, the object stored per response is `view(k)` (a shallow copy sharing the weights),
    with `out` = the output index for that response.
    """

    kind = "ann"
    transform = None
    df_resid = 0
    block_idx = []
    out = 0
    n_out = 1
    early_stop = False
    diagnosis = None
    cv_diag = None          # diagnosis of this architecture from cross-validation during the search (more reliable)

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def view(self, k):
        v = copy.copy(self)
        v.out = int(k)
        return v

    # ---- weight structure
    @property
    def sizes(self):
        return [self.n_in] + list(self.layers)

    @property
    def arch(self):
        return "-".join(str(v) for v in self.sizes + [self.n_out])

    @property
    def n_weights(self):
        return n_weights_for(self.n_in, self.layers, self.n_out)

    @property
    def combined(self):
        return self.n_out > 1

    def _unpack(self, w):
        """-> [(W1, b1), (W2, b2), ...], W_out (n_out x h_L), b_out (n_out)"""
        sz = self.sizes
        out, pos = [], 0
        for l in range(len(self.layers)):
            n_in, n_o = sz[l], sz[l + 1]
            W = w[pos:pos + n_o * n_in].reshape(n_o, n_in)
            pos += n_o * n_in
            b = w[pos:pos + n_o]
            pos += n_o
            out.append((W, b))
        h = self.layers[-1]
        W_out = w[pos:pos + self.n_out * h].reshape(self.n_out, h)
        pos += self.n_out * h
        return out, W_out, w[pos:pos + self.n_out]

    # ---- normalization
    def _xn(self, X):
        return self.x_base + (X - self.x_off) / self.x_scale

    def _yn(self, Y):
        return self.y_base + (Y - self.y_off) / self.y_scale

    def _yn_inv(self, Yn):
        return (Yn - self.y_base) * self.y_scale + self.y_off

    def _forward(self, Xn, w):
        layers, W_out, b_out = self._unpack(w)
        h = Xn
        for W, b in layers:
            h, _ = _act(h @ W.T + b, self.activation)
        return h @ W_out.T + b_out

    # ---- model interface (one response = output `out`)
    def predict_all(self, coded):
        X, _ = encode(coded, self.space)
        return self._yn_inv(self._forward(self._xn(X), self.weights))

    def predict(self, coded, original=True):
        return self.predict_all(coded)[:, self.out]

    def predict_t(self, coded):
        return self.predict(coded)

    def confidence_band(self, coded, level=0.95):
        y = self.predict(coded)
        return y, y

    def intervals(self, coded, n_obs=1, level=0.95):
        y = self.predict(coded)
        nan = np.full(len(y), np.nan)
        return {"pred": y, "pred_t": y, "se_mean": nan, "se_pred": nan, "t": np.nan, "ci": (nan, nan),
                "pi": (nan, nan)}

    def t_crit(self, level=0.95):
        return np.nan

    @property
    def metrics(self):
        return self.metrics_all[self.out]

    @property
    def y_orig(self):
        return self.Y_used[:, self.out]

    @property
    def mse(self):
        return self.metrics["all"]["mse"]

    @property
    def resid_var_for_poe(self):
        return self.metrics["all"]["mse"]

    # ---- interpretation
    def importance(self, names=None):
        """Relative input importance (%) for this output, summed per original factor.

        1 hidden layer: Garson's method. More than 1 layer: extended Garson - each input's share is propagated
        through every layer in proportion to |weight|.
        """
        layers, W_out, _ = self._unpack(self.weights)
        W1 = np.abs(layers[0][0])
        share = W1 / np.maximum(W1.sum(axis=1, keepdims=True), 1e-300)      # layer 1 neurons x inputs
        for W, _ in layers[1:]:
            A = np.abs(W)
            share = (A / np.maximum(A.sum(axis=1, keepdims=True), 1e-300)) @ share
        v = np.abs(W_out[self.out])
        if len(layers) == 1:
            share = share * (v / max(v.sum(), 1e-300))[:, None]            # classic Garson
            s = share.sum(axis=0)
        else:
            s = (v / max(v.sum(), 1e-300)) @ share
        per = {}
        for (i, _), val in zip(self.input_names, s):
            per[i] = per.get(i, 0.0) + val
        tot = sum(per.values()) or 1.0
        return {i: 100 * val / tot for i, val in per.items()}

    # ---- save
    KEYS = ["layers", "activation", "n_in", "n_out", "split", "seed", "restarts", "max_epochs", "max_fail",
            "idx_train", "idx_val", "idx_test", "epochs", "best_epoch", "stop_reason", "algorithm", "gamma", "norm",
            "early_stop", "resp_idx", "out", "gid", "diagnosis", "nmse", "cv_diag", "auto_adam", "workers"]

    def to_dict(self):
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items() if k in self.KEYS}
        d["out"] = self.out
        for k in ("weights", "xmin", "xmax", "x_off", "x_scale", "ymin", "ymax", "y_off", "y_scale", "rows_used"):
            d[k] = np.asarray(getattr(self, k)).tolist()
        d["x_base"], d["y_base"] = float(self.x_base), float(self.y_base)
        d["input_names"] = [[i, j] for i, j in self.input_names]
        d["history"] = {k: list(map(float, v)) for k, v in self.history.items()}
        d["metrics_all"] = self.metrics_all
        return d

    @classmethod
    def from_dict(cls, d, space, coded, data, j=0):
        """data: project response matrix (n x r) - or a response vector for old files; j = response index."""
        d = dict(d)
        if "layers" not in d:
            d["layers"] = [int(d.pop("hidden"))]      # version 5.0 files (one hidden layer)
        d.pop("hidden", None)
        data = np.asarray(data, float)
        data = data[:, None] if data.ndim == 1 else data
        resp_idx = d.get("resp_idx") or [j]
        if data.shape[1] == 1:
            resp_idx_local = [0]
        else:
            resp_idx_local = resp_idx
        arrays = ("weights", "xmin", "xmax", "x_off", "x_scale", "ymin", "ymax", "y_off", "y_scale", "input_names",
                  "rows_used", "metrics", "metrics_all")
        m = cls(**{k: v for k, v in d.items() if k not in arrays})
        for key in ("diagnosis", "cv_diag"):             # older files use the Indonesian status key "pas"
            dg = getattr(m, key, None)
            if isinstance(dg, dict) and dg.get("status") == "pas":
                setattr(m, key, dict(dg, status="good"))
        m.resp_idx = list(resp_idx)
        m.n_out = int(d.get("n_out", 1))
        m.out = int(d.get("out", 0))
        m.weights = np.array(d["weights"], float)
        m.xmin, m.xmax = np.array(d["xmin"], float), np.array(d["xmax"], float)
        m.ymin = np.atleast_1d(np.array(d["ymin"], float))
        m.ymax = np.atleast_1d(np.array(d["ymax"], float))
        if "x_off" in d:
            m.x_off, m.x_scale = np.array(d["x_off"], float), np.array(d["x_scale"], float)
            m.y_off = np.atleast_1d(np.array(d["y_off"], float))
            m.y_scale = np.atleast_1d(np.array(d["y_scale"], float))
        else:                                            # files before 6.0: always mapminmax
            m.norm = "mapminmax"
            m.x_off, m.x_scale, m.x_base = m.xmin, np.where(m.xmax > m.xmin, (m.xmax - m.xmin) / 2, 1.0), -1.0
            m.y_off, m.y_scale, m.y_base = m.ymin, (m.ymax - m.ymin) / 2, -1.0
        m.metrics_all = d["metrics_all"] if "metrics_all" in d else [d["metrics"]]
        m.input_names = [tuple(x) for x in d["input_names"]]
        m.space = space
        m.rows_used = np.array(d["rows_used"], int)
        m.Y_used = data[m.rows_used][:, resp_idx_local]
        m.coded_used = np.asarray(coded, float)[m.rows_used]
        if not getattr(m, "gid", None):
            m.gid = uuid.uuid4().hex
        return m


def _jacobian(model, Xn, w):
    """Output (n x n_out) & Jacobian ∂output/∂weights ((n·n_out) x W) by backpropagation."""
    layers, W_out, b_out = model._unpack(w)
    hs, ds = [Xn], []
    h = Xn
    for W, b in layers:
        h, d = _act(h @ W.T + b, model.activation)
        hs.append(h)
        ds.append(d)
    out = h @ W_out.T + b_out
    n, n_out = len(Xn), model.n_out
    offs, pos = [], 0
    for W, _ in layers:
        offs.append(pos)
        pos += W.size + W.shape[0]
    p_wout = pos
    p_bout = pos + W_out.size
    J = np.zeros((n, n_out, model.n_weights))
    for o in range(n_out):
        g = ds[-1] * W_out[o]                              # ∂out_o/∂a at the last layer (n x n_L)
        for l in range(len(layers) - 1, -1, -1):
            W, _ = layers[l]
            prev = hs[l]
            gW = (g[:, :, None] * prev[:, None, :]).reshape(n, -1)
            J[:, o, offs[l]:offs[l] + gW.shape[1]] = gW
            J[:, o, offs[l] + gW.shape[1]:offs[l] + gW.shape[1] + g.shape[1]] = g
            if l > 0:
                g = (g @ W) * ds[l - 1]
        hL = hs[-1].shape[1]
        J[:, o, p_wout + o * hL:p_wout + (o + 1) * hL] = hs[-1]
        J[:, o, p_bout + o] = 1.0
    return out, J.reshape(n * n_out, -1)


def _init_weights(model, rng):
    """Nguyen-Widrow initialization for each hidden layer."""
    sz = model.sizes
    parts = []
    for l in range(len(model.layers)):
        n_in, n_o = sz[l], sz[l + 1]
        beta = 0.7 * n_o ** (1.0 / max(n_in, 1))
        W = rng.uniform(-1, 1, (n_o, n_in))
        W = beta * W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
        parts += [W.ravel(), rng.uniform(-beta, beta, n_o)]
    parts += [rng.uniform(-0.5, 0.5, model.n_out * model.layers[-1]), rng.uniform(-0.5, 0.5, model.n_out)]
    return np.concatenate(parts)


def _mse(model, Xn, Yn, idx, w):
    if len(idx) == 0:
        return np.nan
    e = Yn[idx] - model._forward(Xn[idx], w)
    return float(np.mean(e ** 2))


def _train_lm(model, Xn, Yn, tr, va, te, w, max_epochs, max_fail, mu=1e-3, min_grad=1e-7, cancel=None):
    hist = {"train": [], "val": [], "test": []}
    best_w, best_val, best_ep, fails = w.copy(), np.inf, 0, 0
    reason = "Maximum epochs"
    I = np.eye(len(w))
    for ep in range(max_epochs):
        out, J = _jacobian(model, Xn[tr], w)
        e = (Yn[tr] - out).ravel()
        sse = float(e @ e)
        hist["train"].append(sse / len(e))
        hist["val"].append(_mse(model, Xn, Yn, va, w))
        hist["test"].append(_mse(model, Xn, Yn, te, w))
        if len(va):
            if hist["val"][-1] < best_val - 1e-15:
                best_val, best_w, best_ep, fails = hist["val"][-1], w.copy(), ep, 0
            else:
                fails += 1
                if fails >= max_fail:
                    reason = f"Early stopping (no validation improvement {max_fail}×)"
                    break
        else:
            best_w, best_ep = w.copy(), ep
        if cancel is not None and ep % 20 == 0 and cancel():
            reason = "Stopped by user"
            break
        g = J.T @ e
        if np.max(np.abs(g)) < min_grad:
            reason = "Minimum gradient"
            break
        H = J.T @ J
        while True:
            try:
                dw = np.linalg.solve(H + mu * I, g)
            except np.linalg.LinAlgError:
                mu *= 10
                if mu > 1e10:
                    break
                continue
            w_new = w + dw
            e_new = Yn[tr] - model._forward(Xn[tr], w_new)
            if float((e_new ** 2).sum()) < sse:
                mu = max(mu * 0.1, 1e-20)
                w = w_new
                break
            mu *= 10
            if mu > 1e10:
                break
        if mu > 1e10:
            reason = "Maximum μ"
            break
    if not len(va):
        best_w = w
    tr_mse = _mse(model, Xn, Yn, tr, best_w)
    score = (tr_mse + best_val) / 2 if len(va) else tr_mse
    return best_w, hist, best_ep, reason, score, None


def _train_br(model, Xn, Yn, tr, va, te, w, max_epochs, max_fail=6, early_stop=False, mu=5e-3, min_grad=1e-7,
              warmup=15, tol=0.0, cancel=None):
    """Bayesian regularization (MacKay) inside Levenberg-Marquardt, like MATLAB trainbr.

    With early_stop and validation data: the weights with the lowest validation error are used; training stops when
    validation has not improved for `max_fail` consecutive epochs (counted after the warm-up period).
    """
    hist = {"train": [], "val": [], "test": []}
    N, W = len(tr) * model.n_out, len(w)
    alpha, beta = 0.01, 1.0          # small initial α (not 0) so the first γ is computed from the Hessian
    I = np.eye(W)
    use_val = early_stop and len(va) > 0
    best_w, best_val, best_ep, fails, best_gamma = w.copy(), np.inf, 0, 0, float(W)
    reason, gamma = "Maximum epochs", float(W)
    cached = None
    for ep in range(max_epochs):
        out, J = cached if cached is not None else _jacobian(model, Xn[tr], w)
        cached = None
        e = (Yn[tr] - out).ravel()
        sse, ssw = float(e @ e), float(w @ w)
        perf = beta * sse + alpha * ssw
        hist["train"].append(sse / N)
        hist["val"].append(_mse(model, Xn, Yn, va, w))
        hist["test"].append(_mse(model, Xn, Yn, te, w))
        if use_val:
            if hist["val"][-1] < best_val - 1e-15:
                best_val, best_w, best_ep, fails, best_gamma = hist["val"][-1], w.copy(), ep, 0, gamma
            elif ep >= warmup:
                fails += 1
                if fails >= max_fail:
                    reason = f"Early stopping (no validation improvement {max_fail}×)"
                    break
        if cancel is not None and ep % 20 == 0 and cancel():
            reason = "Stopped by user"
            break
        jj, je = J.T @ J, J.T @ e
        g = beta * je - alpha * w
        if np.max(np.abs(g)) < min_grad:
            reason = "Minimum gradient"
            break
        while True:
            try:
                dw = np.linalg.solve(beta * jj + (mu + alpha) * I, g)
            except np.linalg.LinAlgError:
                mu *= 10
                if mu > 1e10:
                    break
                continue
            w_new = w + dw
            e_new = Yn[tr] - model._forward(Xn[tr], w_new)
            perf_new = beta * float((e_new ** 2).sum()) + alpha * float(w_new @ w_new)
            if perf_new < perf:
                w = w_new
                mu = max(mu * 0.1, 1e-20)
                break
            mu *= 10
            if mu > 1e10:
                break
        if mu > 1e10:
            reason = "Maximum μ"
            break
        # update the hyperparameters (evidence framework) after the warm-up, so training does not get stuck in an
        # over-regularized solution before the network has had a chance to learn the data
        if ep < warmup:
            continue
        h_tr = hist["train"]
        if tol > 0 and len(h_tr) > warmup + 10 and abs(h_tr[-1] - h_tr[-11]) <= tol * max(h_tr[-11], 1e-300):
            reason = "Converged"
            break
        out, J = cached = _jacobian(model, Xn[tr], w)          # reused at the start of the next epoch
        e = (Yn[tr] - out).ravel()
        sse, ssw = max(float(e @ e), 1e-300), max(float(w @ w), 1e-300)
        try:
            gamma = W - alpha * float(np.trace(np.linalg.inv(beta * (J.T @ J) + alpha * I)))
        except np.linalg.LinAlgError:
            gamma = float(W)
        gamma = min(max(gamma, 1.0), N - 1.0)
        alpha = gamma / (2 * ssw)
        beta = (N - gamma) / (2 * sse)
    if use_val:
        w, gamma, ep_best = best_w, best_gamma, best_ep
        score = (_mse(model, Xn, Yn, tr, w) + best_val) / 2
    else:
        ep_best = len(hist["train"]) - 1
        score = _mse(model, Xn, Yn, tr, w)
    return w, hist, ep_best, reason, score, gamma


def _loss_grad(model, Xn, Yn, w):
    """MSE and its gradient with respect to all weights (backpropagation), without building the full Jacobian."""
    layers, W_out, b_out = model._unpack(w)
    hs, ds = [Xn], []
    h = Xn
    for W, b in layers:
        h, d = _act(h @ W.T + b, model.activation)
        hs.append(h)
        ds.append(d)
    E = h @ W_out.T + b_out - Yn
    m = E.size
    dout = 2.0 * E / m
    parts = []
    g = (dout @ W_out) * ds[-1]
    grads = []
    for l in range(len(layers) - 1, -1, -1):
        grads.append((g.T @ hs[l], g.sum(axis=0)))
        if l > 0:
            g = (g @ layers[l][0]) * ds[l - 1]
    for gW, gb in reversed(grads):
        parts += [gW.ravel(), gb]
    parts += [(dout.T @ hs[-1]).ravel(), dout.sum(axis=0)]
    return float((E ** 2).sum() / m), np.concatenate(parts)


def _train_adam(model, Xn, Yn, tr, va, te, w, max_epochs, max_fail, rng, lr=2e-3, cancel=None):
    """Mini-batch Adam with early stopping (validation) and learning-rate decay on plateaus.
    Memory and time per step scale with the number of weights, so networks of any size can be trained."""
    hist = {"train": [], "val": [], "test": []}
    n = len(tr)
    batch = int(min(256, max(16, n // 20)))
    m1, m2 = np.zeros_like(w), np.zeros_like(w)
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    patience = max(max_fail, 20)
    best_w, best_v, best_ep, fails, stale = w.copy(), np.inf, 0, 0, 0
    reason = "Maximum epochs"
    for ep in range(max_epochs):
        perm = rng.permutation(tr)
        for s0 in range(0, n, batch):
            idx = perm[s0:s0 + batch]
            _, g = _loss_grad(model, Xn[idx], Yn[idx], w)
            t += 1
            m1 = b1 * m1 + (1 - b1) * g
            m2 = b2 * m2 + (1 - b2) * g * g
            w = w - lr * (m1 / (1 - b1 ** t)) / (np.sqrt(m2 / (1 - b2 ** t)) + eps)
        hist["train"].append(_mse(model, Xn, Yn, tr, w))
        hist["val"].append(_mse(model, Xn, Yn, va, w))
        hist["test"].append(_mse(model, Xn, Yn, te, w))
        v = hist["val"][-1] if len(va) else hist["train"][-1]
        if not np.isfinite(v):
            reason = "Diverged"
            break
        if v < best_v * (1 - 1e-4):
            best_v, best_w, best_ep, fails, stale = v, w.copy(), ep, 0, 0
        else:
            fails += 1
            stale += 1
            if stale >= 8:                     # plateau: reduce the step size
                lr *= 0.5
                stale = 0
            if fails >= patience:
                reason = (f"Early stopping (no validation improvement for {patience} epochs)" if len(va)
                          else "Converged")
                break
        if cancel is not None and cancel():
            reason = "Stopped by user"
            break
    tr_mse = _mse(model, Xn, Yn, tr, best_w)
    score = (tr_mse + best_v) / 2 if len(va) else tr_mse
    return best_w, hist, best_ep, reason, score, None


def _restart_job(model, Xn, Yn, tr, va, te, algorithm, max_epochs, max_fail, early_stop, tol, seed, r,
                 cancel=None):
    """One restart (random initialization r). Each restart has its own random generator, so the results are
    identical whether run sequentially or in parallel on many cores."""
    rng = np.random.default_rng([int(seed), int(r) + 1])
    w0 = _init_weights(model, rng)
    if algorithm == "trainbr":
        res = _train_br(model, Xn, Yn, tr, va, te, w0, max_epochs, max_fail, early_stop, tol=tol, cancel=cancel)
        if len(va) and not early_stop:          # validation (if any) is still used to pick the best restart
            vm = _mse(model, Xn, Yn, va, res[0])
            res = res[:4] + ((res[4] + vm) / 2, res[5])
    elif algorithm == "adam":
        res = _train_adam(model, Xn, Yn, tr, va, te, w0, max_epochs, max_fail, rng, cancel=cancel)
    else:
        res = _train_lm(model, Xn, Yn, tr, va, te, w0, max_epochs, max_fail, cancel=cancel)
    return res


def work_estimate(n_train, n_weights, n_out=1, epochs=300, jobs=1):
    return float(n_train) * n_weights * n_out * epochs * jobs


def split_indices(n, split, rng):
    p_tr, p_va, p_te = split
    idx = rng.permutation(n)
    n_va = int(round(n * p_va))
    n_te = int(round(n * p_te))
    if p_va > 0 and n_va == 0 and n >= 6:
        n_va = 1
    if p_te > 0 and n_te == 0 and n >= 6:
        n_te = 1
    n_tr = n - n_va - n_te
    return np.sort(idx[:n_tr]), np.sort(idx[n_tr:n_tr + n_va]), np.sort(idx[n_tr + n_va:])


def train(coded, y, space, hidden=8, activation="tansig", split=(0.70, 0.15, 0.15), seed=0, restarts=10,
          max_epochs=500, max_fail=6, fixed_split=None, algorithm="trainlm", progress=None, norm="mapminmax",
          tol=0.0, early_stop=True, resp_idx=None, noise=None, cancel=None, n_jobs=1, restart0=0):
    """Train a network. y: vector (one response) or n x r matrix (one network for r responses).

    noise: standard measurement uncertainty of each response (may be None) - used for the overfitting diagnosis.
    n_jobs: number of CPU cores; restarts run in parallel when the workload is large enough.
    restart0: index of the first restart (used by the architecture search, which spreads restarts over cores).
    """
    coded = np.asarray(coded, float)
    Y = np.asarray(y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    n_out = Y.shape[1]
    rows = complete_rows(coded, Y)
    if len(rows) < 4:
        raise ValueError("Too little data to train an ANN (at least 4 complete runs required).")
    X, names = encode(coded[rows], space)
    YY = Y[rows]
    rng = np.random.default_rng(seed)
    if fixed_split is not None:
        tr, va, te = (np.asarray(a, int) for a in fixed_split)
    else:
        tr, va, te = split_indices(len(rows), split, rng)
    if not early_stop and fixed_split is None and len(va):
        tr = np.sort(np.concatenate([tr, va]))                 # no early stopping: train on validation data too
        va = np.array([], int)
    if len(tr) < 2:
        raise ValueError("Too little training data. Increase the training data percentage.")
    fitrows = np.concatenate([tr, va]).astype(int)          # normalization from training + validation data only
    xmin, xmax = X[fitrows].min(axis=0), X[fitrows].max(axis=0)
    ymin, ymax = YY[fitrows].min(axis=0), YY[fitrows].max(axis=0)
    if np.any(ymax == ymin):
        raise ValueError("Constant response - the ANN cannot be trained.")
    x_off, x_scale, x_base = norm_fit(X[fitrows], norm)
    y_off, y_scale, y_base = norm_fit(YY[fitrows], norm)
    layers = [int(v) for v in (hidden if isinstance(hidden, (list, tuple)) else [hidden])]
    if not layers or min(layers) < 1:
        raise ValueError("Each hidden layer needs at least 1 neuron.")
    model = AnnModel(layers=layers, activation=activation, n_in=X.shape[1], n_out=n_out, space=space, xmin=xmin,
                     xmax=xmax, ymin=ymin, ymax=ymax, input_names=names, split=list(split), seed=seed,
                     restarts=restarts, max_epochs=max_epochs, max_fail=max_fail, algorithm=algorithm, norm=norm,
                     x_off=x_off, x_scale=x_scale, x_base=float(x_base), y_off=y_off, y_scale=y_scale,
                     y_base=float(y_base), early_stop=bool(early_stop and len(va) > 0),
                     resp_idx=list(resp_idx) if resp_idx is not None else list(range(n_out)), out=0,
                     gid=uuid.uuid4().hex)
    Xn = model._xn(X)
    Yn = model._yn(YY)
    # very large network: LM/Bayesian needs a weights x weights matrix -> switch to Adam automatically
    model.auto_adam = False
    if algorithm in ("trainlm", "trainbr") and (model.n_weights > LM_MAX_WEIGHTS or
                                                len(tr) * n_out * model.n_weights > LM_MAX_JACOBIAN):
        algorithm = "adam"
        model.algorithm = "adam"
        model.auto_adam = True
    nr = max(1, int(restarts))
    jobs = [(model, Xn, Yn, tr, va, te, algorithm, max_epochs, max_fail, early_stop, tol, seed, restart0 + r)
            for r in range(nr)]
    workers = int(n_jobs or 1) if nr > 1 and (parallel.is_warm() or work_estimate(
        len(tr), model.n_weights, n_out, max_epochs, nr) >= PARALLEL_MIN_WORK) else 1
    done = [0]

    def on_result(i, res):
        done[0] += 1
        if progress and workers > 1:
            progress(done[0], nr)

    if progress:
        progress(0, nr)
    if workers > 1:
        results = parallel.run_jobs(_restart_job, jobs, workers, cancel, on_result)
    else:
        results = []
        for r, job in enumerate(jobs):
            if cancel is not None and cancel() and results:
                break
            if progress and r:
                progress(r, nr)
            results.append(_restart_job(*job, cancel=cancel))
    results = [res for res in results if res is not None]
    if not results:
        raise ValueError("Training was stopped before any restart finished.")
    best = min(results, key=lambda res: res[4])
    model.workers = workers
    model.score = float(best[4])
    model.weights, model.history, model.best_epoch, model.stop_reason = best[0], best[1], best[2], best[3]
    model.gamma = best[5]
    model.epochs = len(best[1]["train"])
    model.idx_train, model.idx_val, model.idx_test = tr.tolist(), va.tolist(), te.tolist()
    model.rows_used = rows
    model.Y_used = YY
    model.coded_used = coded[rows]
    pred = model._yn_inv(model._forward(Xn, model.weights))
    model.metrics_all = [{"train": metrics(YY[tr, k], pred[tr, k]), "val": metrics(YY[va, k], pred[va, k]),
                          "test": metrics(YY[te, k], pred[te, k]), "all": metrics(YY[:, k], pred[:, k])}
                         for k in range(n_out)]
    var = np.maximum(YY.var(axis=0), 1e-300)

    def nmse(idx):
        return float(np.mean(np.mean((YY[idx] - pred[idx]) ** 2, axis=0) / var)) if len(idx) else np.nan

    model.nmse = {"train": nmse(tr), "val": nmse(va), "test": nmse(te)}
    noise_nmse = None
    if noise is not None:
        nz = [(float(u) ** 2 / var[k]) for k, u in enumerate(noise) if u]
        noise_nmse = float(np.mean(nz)) if nz else None
    st, msgs = diagnose(model.nmse["train"], model.nmse["val"], model.nmse["test"], noise_nmse)
    model.diagnosis = {"status": st, "notes": msgs, "noise_nmse": noise_nmse}
    return model


# ------------------------------------------------------------------ architecture search
def search_splits(N, folds=5, seed=0, large_n=LARGE_N):
    """Architecture evaluation scheme: list of (training, validation, test).

    - small N: k-fold (each fold is the test data once); 15% of the remaining data is used for early-stopping
      validation
    - large N: a single 70 / 15 / 15 hold-out (stable enough and much faster)
    """
    rng = np.random.default_rng(seed + 7)
    if N >= large_n:
        tr, va, te = split_indices(N, (0.70, 0.15, 0.15), np.random.default_rng(seed))
        return [(tr, va, te)], "holdout"
    out = []
    for f in cv_folds(N, int(min(folds, N)), seed):
        rest = np.setdiff1d(np.arange(N), f)
        n_va = max(1, int(round(0.15 * len(rest)))) if len(rest) >= 6 else 0
        perm = rng.permutation(rest)
        out.append((np.sort(perm[n_va:]), np.sort(perm[:n_va]), f))
    return out, "kfold"


def _arch_job(Xc, Yc, space, layers, split, r, opts, cancel=None):
    """One architecture search job: architecture `layers`, data split `split`, restart r.
    Returns (restart selection score, predictions for all data, best epoch)."""
    m = train(Xc, Yc, space, list(layers), opts["activation"], seed=opts["seed"], restarts=1, restart0=r,
              max_epochs=opts["max_epochs"], max_fail=opts["max_fail"], fixed_split=split,
              algorithm=opts["algorithm"], norm=opts["norm"], tol=1e-5, early_stop=True, cancel=cancel)
    return float(m.score), m.predict_all(Xc), m.best_epoch


def _summarize_architecture(layers, splits, var, Yc, runs):
    """runs[s] = list of _arch_job results for split s (one per restart). The best restart per split is chosen by
    the training + validation score (the test data takes no part in the choice)."""
    tr_e, va_e, te_e, eps, sq = [], [], [], [], []
    for (tr, va, te), rs in zip(splits, runs):
        rs = [r for r in rs if r is not None]
        if not rs:
            return None
        _, P, ep = min(rs, key=lambda r: r[0])

        def nm(idx):
            return float(np.mean(np.mean((Yc[idx] - P[idx]) ** 2, axis=0) / var)) if len(idx) else np.nan

        tr_e.append(nm(tr))
        va_e.append(nm(va))
        te_e.append(nm(te))
        sq.append(np.mean((Yc[te] - P[te]) ** 2 / var, axis=1))
        eps.append(ep)
    te_e = np.array(te_e)
    if len(splits) > 1:
        se = float(te_e.std(ddof=1) / np.sqrt(len(te_e)))
    else:
        s = sq[0]
        se = float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0
    test = float(te_e.mean())
    return {"layers": list(layers), "depth": len(layers), "train": float(np.mean(tr_e)),
            "val": float(np.nanmean(va_e)) if np.isfinite(va_e).any() else np.nan, "test": test, "se": se,
            "q2": 1 - test, "r2_train": 1 - float(np.mean(tr_e)), "epochs": float(np.mean(eps)),
            "fold_test": te_e.tolist(),
            "rmse": [float(np.sqrt(max(test, 0) * v)) for v in var]}


def evaluate_architectures(Xc, Yc, space, archs, splits, var, activation="tansig", algorithm="trainbr", restarts=3,
                           max_epochs=300, max_fail=6, seed=0, norm="mapminmax", cancel=None, n_jobs=1):
    """Evaluate several architectures at once. Every combination (architecture x data split x restart) is a
    separate job spread over `n_jobs` CPU cores. Errors are normalized by the response variance (NMSE)."""
    opts = {"activation": activation, "algorithm": algorithm, "max_epochs": max_epochs, "max_fail": max_fail,
            "seed": seed, "norm": norm}
    nr = max(1, int(restarts))
    keys, jobs = [], []
    for a, layers in enumerate(archs):
        for si, split in enumerate(splits):
            for r in range(nr):
                keys.append((a, si))
                jobs.append((Xc, Yc, space, list(layers), split, r, opts))
    results = parallel.run_jobs(_arch_job, jobs, n_jobs, cancel)
    runs = [[[] for _ in splits] for _ in archs]
    for (a, si), res in zip(keys, results):
        runs[a][si].append(res)
    return [_summarize_architecture(layers, splits, var, Yc, runs[a]) for a, layers in enumerate(archs)]


def evaluate_architecture(Xc, Yc, space, layers, splits, var, activation="tansig", algorithm="trainbr", restarts=3,
                          max_epochs=300, max_fail=6, seed=0, norm="mapminmax", cancel=None, n_jobs=1):
    return evaluate_architectures(Xc, Yc, space, [layers], splits, var, activation, algorithm, restarts, max_epochs,
                                  max_fail, seed, norm, cancel, n_jobs)[0]


SEARCH_SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 64, 80, 100, 128, 160, 200)
EQUIV_R2 = 0.002            # test R2 differences below this count as equal (choose the simpler network)


def neuron_sizes(max_neurons=None):
    """Neuron counts to try, increasing step by step. max_neurons None = no limit (the sequence keeps growing by
    ~25%; the search stops on its own through early stopping)."""
    for h in SEARCH_SIZES:
        if max_neurons is not None and h > max_neurons:
            break
        yield h
    if max_neurons is not None and max_neurons in SEARCH_SIZES:
        return
    if max_neurons is not None and max_neurons < SEARCH_SIZES[-1]:
        yield int(max_neurons)
        return
    h = SEARCH_SIZES[-1]
    while True:
        h = int(round(h * 1.25))
        if max_neurons is not None and h >= max_neurons:
            yield int(max_neurons)
            return
        yield h


def architecture_search(coded, y, space, max_layers=3, max_neurons=10, activation="tansig", algorithm="trainbr",
                        folds=5, restarts=3, max_epochs=300, seed=0, norm="mapminmax", weight_ratio=None,
                        patience=2, max_fail=6, noise=None, progress=None, cancel=None, max_rows=SEARCH_MAX_ROWS,
                        n_jobs=1):
    """Automatically search the number of hidden layers AND the number of neurons per layer (may differ), with early
    stopping.

    Early stopping at three levels:
    1. Weights - each training run stops when the validation error has not improved (max_fail epochs); the best
                 weights are used.
    2. Neurons - the neuron count of a layer is increased step by step; stops when the test error has not improved
                 by ≥ 1% for `patience` consecutive steps.
    3. Layers  - hidden layers are added only while they improve the test error by ≥ 1%.
    Each architecture is diagnosed (good / underfitting / overfitting) from training error vs error on new data.
    Final choice: the simplest architecture with status "good" whose error is no worse than the best + 1 standard
    error.

    n_jobs > 1: all training runs (architecture x fold x restart) are spread over many CPU cores. When cores are
    left over, the next few neuron counts are evaluated together; the order of early-stopping decisions stays the
    same, so the results are identical to the single-core version.
    """
    coded = np.asarray(coded, float)
    Y = np.asarray(y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    rows = complete_rows(coded, Y)
    N_all = len(rows)
    if N_all < 8:
        raise ValueError(f"Only {N_all} complete runs. Architecture search needs at least 8 runs.")
    notes = []
    if N_all > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_rows, replace=False))
        notes.append(f"Large data ({N_all} runs): the search uses {max_rows} random runs for speed; the final model "
                     "is trained on all data.")
    Xc, Yc = coded[rows], Y[rows]
    N = len(rows)
    splits, scheme = search_splits(N, folds, seed)
    var = np.maximum(Yc.var(axis=0), 1e-300)
    n_train = min(len(s[0]) for s in splits)
    n_in = encode(Xc[:1], space)[0].shape[1]
    n_out = Y.shape[1]
    # weight_ratio: None = automatic (2x / 1x training rows), inf or <= 0 = no weight limit
    if weight_ratio is None:
        ratio = 2.0 if algorithm == "trainbr" else 1.0
    else:
        ratio = float(weight_ratio) if weight_ratio and weight_ratio > 0 else np.inf
    limit = max(ratio * n_train, n_weights_for(n_in, [1], n_out))
    max_neurons = int(max_neurons) if max_neurons else None
    max_layers = int(max_layers) if max_layers else None
    noise_nmse = None
    if noise is not None:
        nz = [float(u) ** 2 / var[k] for k, u in enumerate(noise) if u]
        noise_nmse = float(np.mean(nz)) if nz else None
    table, seen = [], set()
    state = {"stopped": False, "count": 0}
    per_arch = len(splits) * max(1, int(restarts))
    workers = max(1, int(n_jobs or 1))     # search = tens to hundreds of training runs: parallel always pays off
    batch_size = max(1, min(patience + 2, workers // per_arch)) if workers > 1 else 1

    def run_many(lays):
        if cancel is not None and cancel():
            state["stopped"] = True
            return None
        c0 = state["count"] + 1
        state["count"] += len(lays)
        if progress:
            label = ", ".join("-".join(map(str, lay)) for lay in lays)
            progress(state["count"] if len(lays) == 1 else f"{c0}-{state['count']}", len(lays[0]), label)
        rs = evaluate_architectures(Xc, Yc, space, lays, splits, var, activation, algorithm, restarts, max_epochs,
                                    max_fail, seed, norm, cancel, workers)
        if cancel is not None and cancel():
            state["stopped"] = True
        return rs

    def record(lay, r):
        r["n_weights"] = n_weights_for(n_in, lay, n_out)
        r["arch"] = "-".join(map(str, [n_in, *lay, n_out]))
        table.append(r)
        seen.add(tuple(lay))

    def grow(prefix, depth):
        """Add neurons to layer `depth` (on top of `prefix`) until the test error stops improving."""
        best, fails = np.inf, 0
        sizes = neuron_sizes(max_neurons)
        exhausted = False
        while not exhausted:
            lays = []
            while len(lays) < batch_size:
                h = next(sizes, None)
                if h is None:
                    exhausted = True
                    break
                lay = prefix + [h]
                if tuple(lay) in seen:
                    continue
                if n_weights_for(n_in, lay, n_out) > limit:
                    exhausted = True
                    break
                lays.append(lay)
            if not lays:
                return None
            try:
                rs = run_many(lays)
            except MemoryError:
                notes.append(f"Architecture {'/'.join(map(str, lays[0]))} is too large for the computer's memory; "
                             "adding neurons was stopped.")
                return None
            if rs is None:
                return None
            for lay, r in zip(lays, rs):
                if r is None:
                    return None
                record(lay, r)
                if r["test"] < best * 0.99:
                    best, fails = r["test"], 0
                else:
                    fails += 1
                    if fails >= patience:
                        return f"stopped at {lay[-1]} neurons"
            if state["stopped"]:
                return None
        return None

    msg = grow([], 1)
    if msg:
        notes.append(f"Layer 1: {msg} (test error did not improve {patience}× in a row).")
    best_so_far = min((r["test"] for r in table), default=np.inf)
    depth = 1
    while max_layers is None or depth < max_layers:
        depth += 1
        if state["stopped"]:
            break
        prev = sorted((r for r in table if r["depth"] == depth - 1), key=lambda r: r["test"])[:2]
        if not prev or n_weights_for(n_in, prev[0]["layers"] + [1], n_out) > limit:
            notes.append(f"{depth} hidden layers not tried: the minimum number of weights exceeds the limit of "
                         f"{limit:.0f} weights (= {ratio:g} × {n_train} training rows).")
            break
        for r in prev:
            grow(r["layers"], depth)
            if state["stopped"]:
                break
        new_best = min((r["test"] for r in table if r["depth"] == depth), default=np.inf)
        if new_best >= best_so_far * 0.99:
            notes.append(f"Layer early stopping: {depth} hidden layers did not improve the test error by ≥ 1%, "
                         "adding layers was stopped.")
            break
        best_so_far = new_best
    if not table:
        raise ValueError("The search was stopped before any architecture finished evaluating.")

    best_test = min(r["test"] for r in table)
    for r in table:
        st = "good"
        if noise_nmse and r["train"] < 0.5 * noise_nmse and r["test"] > best_test * 1.05:
            st = "over"
        elif r["test"] > 2.0 * r["train"] and r["test"] > 1.2 * best_test and r["test"] - r["train"] > 0.02:
            st = "over"
        elif (r["train"] > 1.5 * best_test and r["test"] < 1.5 * r["train"] + 0.02 and r["test"] > 1.2 * best_test
              and r["test"] - best_test > 0.01):
            st = "under"            # test R2 at least 0.01 below the best AND the training error is also high
        r["status"] = st
    pool = [r for r in table if r["status"] == "good"] or table
    best_raw = min(pool, key=lambda r: r["test"])
    thr = best_raw["test"] + max(best_raw["se"], EQUIV_R2)
    best = min((r for r in pool if r["test"] <= thr), key=lambda r: (r["n_weights"], r["depth"], r["test"]))
    depth_best = {}
    for r in table:
        if r["depth"] not in depth_best or r["test"] < depth_best[r["depth"]]["test"]:
            depth_best[r["depth"]] = r
    return {"table": table, "best": best["layers"], "best_raw": best_raw["layers"], "threshold": thr,
            "depth_best": depth_best, "limit": limit, "ratio": ratio, "folds": len(splits), "scheme": scheme,
            "n": N, "n_all": N_all, "n_train": n_train, "n_in": n_in, "n_out": n_out, "notes": notes,
            "cancelled": state["stopped"], "algorithm": algorithm, "activation": activation, "norm": norm,
            "max_neurons": max_neurons, "max_layers": max_layers, "patience": patience, "var": var.tolist(),
            "rows": rows, "splits": splits, "noise_nmse": noise_nmse, "workers": workers}


# ------------------------------------------------------------------ learning curve
LC_FRACTIONS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _lc_job(Xc, Yc, space, layers, tr, va, te, opts, cancel=None):
    """One learning-curve point: train on tr (+ va for early stopping), evaluate on te."""
    m = train(Xc, Yc, space, list(layers), opts["activation"], seed=opts["seed"], restarts=opts["restarts"],
              max_epochs=opts["max_epochs"], max_fail=6, fixed_split=(tr, va, te), algorithm=opts["algorithm"],
              norm=opts["norm"], tol=1e-5, early_stop=len(va) > 0, cancel=cancel)
    P = m.predict_all(Xc)
    var = opts["var"]
    fit = np.concatenate([tr, va]).astype(int)
    return (float(np.mean(np.mean((Yc[te] - P[te]) ** 2, axis=0) / var)),
            float(np.mean(np.mean((Yc[fit] - P[fit]) ** 2, axis=0) / var)))


def _power_law(n, e):
    """e(n) = a + b n^-c (a, b >= 0): a = the error that remains however much data is added (noise/model limit)."""
    n, e = np.asarray(n, float), np.asarray(e, float)
    ok = np.isfinite(e)
    n, e = n[ok], e[ok]
    if len(n) < 3:
        return None
    best = None
    for c in np.linspace(0.2, 2.5, 47):
        Z = np.column_stack([np.ones(len(n)), n ** -c])
        coef, *_ = np.linalg.lstsq(Z, e, rcond=None)
        a, b = coef
        if a < 0:
            a = 0.0
            b = float(np.dot(Z[:, 1], e) / np.dot(Z[:, 1], Z[:, 1]))
        b = max(b, 0.0)
        sse = float(((a + b * n ** -c - e) ** 2).sum())
        if best is None or sse < best[3]:
            best = (float(a), float(b), float(c), sse)
    return best


def learning_curve(coded, y, space, layers, activation="tansig", algorithm="trainbr", norm="mapminmax",
                   restarts=2, max_epochs=300, repeats=None, test_frac=0.2, fractions=LC_FRACTIONS, seed=0,
                   rsm_eval=None, tol_r2=0.01, n_jobs=1, cancel=None, progress=None):
    """Learning curve: the ANN is trained on part of the data (30%, 40%, ... 100% of the training data) and evaluated
    on a fixed test set that is never used for training. Repeated `repeats` times with different random splits.

    If the test error still drops as data is added, more data still helps. The curve is fitted with
    e(n) = a + b n^-c to estimate the amount of data needed for the test R2 to be within `tol_r2` of its upper
    limit (1 - a).
    rsm_eval(fit_rows, test_rows) -> RSM test NMSE on the same split (optional, for comparison).
    """
    coded = np.asarray(coded, float)
    Y = np.asarray(y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    rows = complete_rows(coded, Y)
    N = len(rows)
    if N < 12:
        raise ValueError(f"The learning curve needs at least 12 complete runs (found {N}).")
    Xc, Yc = coded[rows], Y[rows]
    var = np.maximum(Yc.var(axis=0), 1e-300)
    opts = {"activation": activation, "algorithm": algorithm, "norm": norm, "restarts": restarts,
            "max_epochs": max_epochs, "seed": seed, "var": var}
    repeats = int(repeats or (10 if N < 60 else 6 if N < 400 else 4))
    n_te = max(3, int(round(test_frac * N)))
    rng = np.random.default_rng(seed + 31)
    jobs, keys, sizes = [], [], []
    for r in range(repeats):
        perm = rng.permutation(N)
        te, pool = np.sort(perm[:n_te]), perm[n_te:]
        for fi, f in enumerate(fractions):
            n_fit = max(5, int(round(f * len(pool))))
            sub = pool[:n_fit]
            n_va = max(1, int(round(0.15 * n_fit))) if n_fit >= 10 else 0
            va, tr = np.sort(sub[:n_va]), np.sort(sub[n_va:])
            jobs.append((Xc, Yc, space, list(layers), tr, va, te, opts))
            keys.append((r, fi, np.sort(sub), te))
            if r == 0:
                sizes.append(n_fit)
    done = [0]

    def on_result(i, res):
        done[0] += 1
        if progress:
            progress(done[0], len(jobs))

    results = parallel.run_jobs(_lc_job_safe, jobs, n_jobs, cancel, on_result)
    A = np.full((repeats, len(fractions)), np.nan)
    T = np.full((repeats, len(fractions)), np.nan)
    R = np.full((repeats, len(fractions)), np.nan)
    for (r, fi, sub, te), res in zip(keys, results):
        if res is not None:
            A[r, fi], T[r, fi] = res
        if rsm_eval is not None:
            try:
                R[r, fi] = rsm_eval(rows[sub], rows[te])
            except Exception:  # noqa: BLE001 - too little data for the RSM model
                R[r, fi] = np.nan

    def med(M):
        """Median and quartiles across repeats (robust to failed training on very little data)."""
        m, lo, hi = (np.full(M.shape[1], np.nan) for _ in range(3))
        for i in range(M.shape[1]):
            v = M[:, i][np.isfinite(M[:, i])]
            if len(v):
                m[i], lo[i], hi[i] = np.median(v), np.percentile(v, 25), np.percentile(v, 75)
        return m, lo, hi

    a_m, a_lo, a_hi = med(A)
    t_m, _, _ = med(T)
    r_m, r_lo, r_hi = med(R)
    sizes = np.array(sizes, float)
    n_full = int(sizes[-1])
    frac_train = 1 - n_te / N
    k0 = 2 if len(sizes) >= 6 else 0          # the 2 smallest points are too noisy to estimate the curve shape

    def needed(curve):
        f = _power_law(sizes[k0:], curve[k0:])
        if f is None:
            return None, None
        a, b, c, _ = f
        n_star = int(np.ceil((b / tol_r2) ** (1 / c))) if b > 0 else n_full
        return f, n_star

    fit, n_star = needed(a_m)
    out = {"sizes": sizes.tolist(), "ann": a_m.tolist(), "ann_lo": a_lo.tolist(), "ann_hi": a_hi.tolist(),
           "ann_train": t_m.tolist(), "rsm": r_m.tolist(), "rsm_lo": r_lo.tolist(), "rsm_hi": r_hi.tolist(),
           "N": N, "n_test": n_te, "repeats": repeats,
           "layers": list(layers), "fit": fit, "tol_r2": tol_r2, "test_frac": test_frac}
    status = "?"
    if fit is not None:
        # estimate uncertainty: bootstrap across repeats (curve median recomputed from resampled repeats)
        brng = np.random.default_rng(seed + 97)
        boots = []
        for _ in range(200):
            Ab = A[brng.integers(0, repeats, repeats)]
            cur = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan for c in Ab.T])
            _, nb = needed(cur)
            if nb is not None:
                boots.append(nb)
        n_med = int(np.median(boots)) if boots else n_star
        q1, q3 = (np.percentile(boots, [25, 75]) if boots else (n_star, n_star))
        out["r2_limit"] = 1 - fit[0]
        out["n_star_fit"] = n_med
        out["total_needed"] = int(np.ceil(n_med / frac_train))
        out["total_range"] = (int(np.ceil(q1 / frac_train)), int(np.ceil(q3 / frac_train)))
        status = "enough" if n_med <= n_full else ("far" if n_med > 5 * n_full else "more")
    if N < 25:
        status = "few"                      # too few for a reliable curve
    last = a_m[np.isfinite(a_m)]
    out["slope"] = float((last[-2] - last[-1]) / max(last[-2], 1e-12)) if len(last) >= 2 else np.nan
    out["status"] = status
    return out


def _lc_job_safe(*a, cancel=None):
    try:
        return _lc_job(*a, cancel=cancel)
    except (ValueError, np.linalg.LinAlgError):
        return None
