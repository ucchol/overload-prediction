"""Decision time from choice models, docs/time_models_plan.md (exploratory).

Every model gives each trial a predicted log time. Scores use log time centered within participant, so they say
which trials take a given person longer. Values are rescaled to 0 to 1 by each set's scale.

Three kinds of model.
- Baselines, least squares laws on trial features (S1 log2 n, S2 the four structure features).
- Laws derived from a choice model, a choice sensitivity fitted to choices and one time slope on a quantity the
  model computes (entropy of its choice probabilities, or the search model's expected looks).
- Process models that predict a full time distribution and are fitted jointly to choices and times (LBA, MLBA LBA,
  Wald race, relative Wald) or to times given the fitted search (sample until return timing). Their trial
  prediction is their own expected log time, with no further calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln, log_ndtr, ndtr, softmax
from scipy.stats import gamma as gamma_dist

from .cdm import static_scores
from .search import scaled_symmetric

LOG2 = np.log(2.0)
GRID = np.geomspace(1e-3, 60.0, 360)  # decision time grid in relative units (person geometric mean time is 1)


# ----------------------------------------------------------------------------- data
@dataclass
class Trials:
    x: np.ndarray  # trials by nmax, values on 0 to 1, NaN past n
    n: np.ndarray
    choice: np.ndarray
    logt: np.ndarray  # log seconds
    pid: np.ndarray  # participant index within the set

    def take(self, m: np.ndarray) -> "Trials":
        return Trials(self.x[m], self.n[m], self.choice[m], self.logt[m], self.pid[m])

    @property
    def mask(self) -> np.ndarray:
        return np.isfinite(self.x)


def from_timeset(ts, scale: tuple[float, float]) -> Trials:
    lo, hi = scale
    nmax = max(b.values.shape[1] for b in ts.blocks)
    xs, ns, cs, ls, ps = [], [], [], [], []
    for b in ts.blocks:
        k = b.values.shape[1]
        x = np.full((len(b.choice), nmax), np.nan)
        x[:, :k] = (b.values - lo) / (hi - lo)
        xs.append(x); ns.append(np.full(len(b.choice), k)); cs.append(b.choice); ls.append(b.logt); ps.append(b.participant)
    return Trials(np.vstack(xs), np.concatenate(ns), np.concatenate(cs), np.concatenate(ls), np.concatenate(ps))


def center(y: np.ndarray, pid: np.ndarray) -> np.ndarray:
    s = np.bincount(pid, weights=y); c = np.bincount(pid)
    return y - (s / np.maximum(c, 1))[pid]


def r2(y: np.ndarray, yhat: np.ndarray, pid: np.ndarray, w: np.ndarray | None = None) -> float:
    yc, hc = center(y, pid), center(yhat, pid)
    w = np.ones_like(yc) if w is None else w
    return float(1.0 - np.sum(w * (yc - hc) ** 2) / np.sum(w * yc ** 2))


def relative_time(tr: Trials) -> np.ndarray:
    """Seconds divided by the person's geometric mean time (over the trials given)."""
    return np.exp(center(tr.logt, tr.pid))


# ----------------------------------------------------------------------------- features
def structure(tr: Trials) -> np.ndarray:
    v = np.sort(np.where(tr.mask, tr.x, -np.inf), axis=1)[:, ::-1]
    best, second = v[:, 0], v[:, 1]
    rest = np.where(np.isfinite(v[:, 1:]), v[:, 1:], np.nan)
    return np.column_stack([np.log2(tr.n), best, best - second, np.nanmean(rest, axis=1)])


def rich_features(tr: Trials) -> np.ndarray:
    v = np.sort(np.where(tr.mask, tr.x, -np.inf), axis=1)[:, ::-1]
    top = np.where(np.isfinite(v[:, :5]), v[:, :5], np.nan)
    rest = np.where(np.isfinite(v[:, 1:]), v[:, 1:], np.nan)
    near = (np.where(tr.mask, tr.x, -np.inf) >= v[:, :1] - 0.1).sum(axis=1)
    ent = [entropy_bits(softmax_masked(lam * tr.x)) for lam in (2.0, 6.0, 15.0)]
    return np.column_stack([np.log2(tr.n), top, v[:, 0] - v[:, 1], np.nanmean(rest, axis=1), np.nanstd(rest, axis=1),
                            near, *ent])


def lsq_center(F: np.ndarray, y: np.ndarray, pid: np.ndarray) -> np.ndarray:
    """Least squares coefficients without intercept on data centered within participant (minimum norm)."""
    Fc = np.column_stack([center(F[:, j], pid) for j in range(F.shape[1])])
    return np.linalg.lstsq(Fc, center(y, pid), rcond=None)[0]


# ----------------------------------------------------------------------------- choice models and their quantities
def softmax_masked(z: np.ndarray) -> np.ndarray:
    z = np.where(np.isfinite(z), z, -np.inf)
    return softmax(z, axis=1)


def entropy_bits(p: np.ndarray) -> np.ndarray:
    return -np.sum(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0), axis=1)


def scores(tr: Trials, family: str) -> np.ndarray:
    if family == "CL":
        return tr.x
    return static_scores(tr.x, family, mlba_scale=6.0)


def fit_lambda(s: np.ndarray, choice: np.ndarray) -> float:
    def nll(loglam):
        z = np.where(np.isfinite(s), np.exp(loglam) * s, -np.inf)
        m = z.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
        return -np.sum(z[np.arange(choice.size), choice] - lse)
    r = minimize_scalar(nll, bounds=(-6, 6), method="bounded", options={"xatol": 1e-8})
    return float(np.exp(r.x))


def sr_look_survival(q: np.ndarray) -> np.ndarray:
    """P(L > k) for k = 0 .. n, where L is the number of looks of sample until return. P(L > k) = k! e_k(q)."""
    return scaled_symmetric(q)


def sr_choice_log2(lam: float, tr: Trials) -> np.ndarray:
    from .search import race_probabilities
    out = np.empty(len(tr.n))
    for i in range(len(tr.n)):
        k = tr.n[i]
        q = softmax(lam * tr.x[i, :k])
        out[i] = np.log2(max(race_probabilities(q)[tr.choice[i]], 1e-300))
    return out


def fit_sr_lambda(tr: Trials) -> float:
    r = minimize_scalar(lambda a: -sr_choice_log2(np.exp(a), tr).sum(), bounds=(-4, 5), method="bounded",
                        options={"xatol": 1e-6})
    return float(np.exp(r.x))


def sr_expected_looks(lam: float, tr: Trials) -> np.ndarray:
    out = np.empty(len(tr.n))
    for i in range(len(tr.n)):
        q = softmax(lam * tr.x[i, : tr.n[i]])
        out[i] = sr_look_survival(q).sum()
    return out


def tls_expected_looks(lam: float, tr: Trials) -> np.ndarray:
    from .search import two_look_satisficing
    out = np.empty(len(tr.n))
    for i in range(len(tr.n)):
        out[i] = two_look_satisficing(lam, tr.x[i, : tr.n[i]], v_top=1.0)[1]
    return out


# ----------------------------------------------------------------------------- accumulator kernels
def lba_fS(u: np.ndarray, v: np.ndarray, A: float, b: float, s: float = 1.0):
    """Density and survival of one LBA accumulator's finishing time u (>0), drift N(v, s), start U[0, A], threshold b."""
    su = s * u
    z1 = (b - A - u * v) / su
    z2 = (b - u * v) / su
    phi1, phi2 = np.exp(-0.5 * z1 ** 2) / np.sqrt(2 * np.pi), np.exp(-0.5 * z2 ** 2) / np.sqrt(2 * np.pi)
    P1, P2 = ndtr(z1), ndtr(z2)
    F = 1 + (b - A - u * v) / A * P1 - (b - u * v) / A * P2 + su / A * phi1 - su / A * phi2
    f = (-v * P1 + s * phi1 + v * P2 - s * phi2) / A
    return np.maximum(f, 1e-300), np.clip(1 - F, 1e-300, 1.0)


def wald_fS(u: np.ndarray, v: np.ndarray, b: float):
    """Density and survival of an inverse Gaussian first passage, drift v > 0, threshold b, unit noise."""
    su = np.sqrt(u)
    logf = np.log(b) - 0.5 * np.log(2 * np.pi) - 1.5 * np.log(u) - (b - v * u) ** 2 / (2 * u)
    F = ndtr((v * u - b) / su) + np.exp(np.minimum(2 * v * b + log_ndtr(-(v * u + b) / su), 700))
    return np.exp(logf) + 1e-300, np.clip(1 - F, 1e-300, 1.0)


@dataclass
class Race:
    """A race of accumulators whose drifts come from a trial's values. kind is lba, mlba, wald or rwald, and a
    suffix _n lets the threshold grow with set size, b n^gamma (gamma >= 0, one more parameter)."""
    kind: str
    theta: np.ndarray = field(default_factory=lambda: np.zeros(0))

    names = {"lba": ("d0", "log d1", "log A", "log k", "t0 logit"),
             "mlba": ("d0", "log g", "log l", "log A", "log k", "t0 logit"),
             "wald": ("log d0", "log d1", "log b", "t0 logit"),
             "rwald": ("log v", "log tau", "log b", "t0 logit")}
    start = {"lba": [0.5, 1.0, 0.0, 0.0, 0.0], "mlba": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
             "wald": [0.0, 0.5, 0.5, 0.0], "rwald": [0.5, 1.5, 0.5, 0.0]}

    @property
    def base(self) -> str:
        return self.kind.removesuffix("_n")

    @property
    def nscale(self) -> bool:
        return self.kind.endswith("_n")

    def _split(self, th):
        if self.nscale:
            return np.r_[th[:-2], th[-1]], float(np.exp(th[-2]))
        return th, 0.0

    def drifts(self, x: np.ndarray, th: np.ndarray) -> np.ndarray:
        th, _ = self._split(th)
        ok = np.isfinite(x)
        x0 = np.where(ok, x, 0.0)
        if self.base == "lba":
            return th[0] + np.exp(th[1]) * x0
        if self.base == "mlba":
            d = x0[:, :, None] - x0[:, None, :]
            pair = ok[:, :, None] & ok[:, None, :]
            n = ok.sum(axis=1, keepdims=True)
            m = np.where(pair, np.exp(-np.exp(th[2]) * np.abs(d)) * d, 0.0).sum(axis=2) / np.maximum(n - 1, 1)
            return th[0] + np.exp(th[1]) * m
        if self.base == "wald":
            return np.exp(th[0]) + np.exp(th[1]) * x0
        if self.base == "rwald":
            xm = np.where(ok, x, -np.inf)
            top2 = -np.sort(-xm, axis=1)[:, :2]
            other = np.where(xm >= top2[:, :1], top2[:, 1:2], top2[:, :1])
            return np.exp(th[0]) / (1 + np.exp(-np.exp(th[1]) * (x0 - other)))
        raise ValueError(self.kind)

    def t0(self, th: np.ndarray, tmin: float) -> float:
        return tmin / (1 + np.exp(-th[-1]))

    def fS(self, u: np.ndarray, v: np.ndarray, th: np.ndarray, n=None):
        th, g = self._split(th)
        mult = 1.0 if (n is None or g == 0.0) else np.asarray(n, float) ** g
        if self.base in ("lba", "mlba"):
            A = np.exp(th[-3]); b = A + np.exp(th[-2]) * mult
            return lba_fS(u, v, A, b)
        return wald_fS(u, v, np.exp(th[-2]) * mult)

    def loglik(self, th: np.ndarray, tr: Trials, rt: np.ndarray, tmin: float, eps: float = 0.0,
               deadline: np.ndarray | None = None) -> np.ndarray:
        """Log density of the chosen option finishing first at the observed (relative) time, per trial. With a
        deadline (relative units, per trial), the density is conditioned on a response before it."""
        ok = tr.mask
        v = self.drifts(tr.x, th)
        u = np.maximum(rt - self.t0(th, tmin), 1e-6)[:, None]
        f, S = self.fS(u, v, th, tr.n[:, None])
        logS = np.where(ok, np.log(S), 0.0)
        idx = np.arange(len(rt))
        ll = np.log(f[idx, tr.choice]) + logS.sum(axis=1) - logS[idx, tr.choice]
        if self.base in ("lba", "mlba"):  # condition on some accumulator finishing
            pnone = np.where(ok, log_ndtr(-v), 0.0).sum(axis=1)
            ll = ll - np.log1p(-np.exp(np.minimum(pnone, -1e-12)))
        if deadline is not None:
            ud = np.maximum(deadline - self.t0(th, tmin), 1e-6)[:, None]
            _, Sd = self.fS(ud, v, th, tr.n[:, None])
            smin = np.exp(np.where(ok, np.log(Sd), 0.0).sum(axis=1))
            if self.base in ("lba", "mlba"):
                pn = np.exp(np.where(ok, log_ndtr(-v), 0.0).sum(axis=1))
                pin = (1 - smin) / np.maximum(1 - pn, 1e-300)
            else:
                pin = 1 - smin
            ll = ll - np.log(np.clip(pin, 1e-300, 1.0))
        if eps > 0:
            ll = np.logaddexp(np.log1p(-eps) + ll, np.log(eps) - np.log(tr.n) - np.log(20.0))
        return ll

    def fit(self, tr: Trials, starts: int = 3, seed: int = 0, eps: float = 0.0, deadline_s: float | None = None) -> "Race":
        rt = relative_time(tr)
        dl = None if deadline_s is None else rt * deadline_s / np.exp(tr.logt)
        tmin = float(np.min(rt))
        rng = np.random.default_rng(seed)
        best = None
        if self.theta.size:
            th0 = np.array(self.theta, float)
        else:
            b0 = list(self.start[self.base])
            th0 = np.array(b0[:-1] + [np.log(0.3), b0[-1]] if self.nscale else b0, float)
        for k in range(starts):
            x0 = th0 if k == 0 else th0 + rng.normal(0, 0.7, th0.size)
            def obj(th):
                ll = self.loglik(th, tr, rt, tmin, eps, dl)
                val = -np.sum(ll)
                return val if np.isfinite(val) else 1e12
            r = minimize(obj, x0, method="L-BFGS-B", options={"maxiter": 800})
            if best is None or r.fun < best.fun:
                best = r
        out = Race(self.kind, best.x.copy())
        out.tmin, out.nll, out.eps = tmin, float(best.fun), eps
        return out

    def expected_logt_deadline(self, tr: Trials, deadline_rel: np.ndarray) -> np.ndarray:
        """Expected log relative time given a response before a per trial deadline (relative units)."""
        S, pn = self.min_survival(tr)
        t0 = self.t0(self.theta, self.tmin)
        S = np.column_stack([np.ones(len(S)), S])
        mass = (S[:, :-1] - S[:, 1:])
        mid = np.concatenate([[GRID[0] / 2], np.sqrt(GRID[:-1] * GRID[1:])])
        keep = (t0 + GRID)[None, :] <= deadline_rel[:, None]
        mass = np.where(keep, mass, 0.0)
        return (mass @ np.log(t0 + mid)) / np.maximum(mass.sum(axis=1), 1e-300)

    def min_survival(self, tr: Trials, chunk: int = 400):
        """Survival of the race's finishing time on GRID, per trial, and the probability nothing finishes."""
        th = self.theta
        out = np.empty((len(tr.n), GRID.size)); pn = np.zeros(len(tr.n))
        for a in range(0, len(tr.n), chunk):
            sl = slice(a, a + chunk)
            x = tr.x[sl]; ok = np.isfinite(x)
            v = self.drifts(x, th)
            _, S = self.fS(GRID[None, None, :], v[:, :, None], th, ok.sum(axis=1)[:, None, None])
            out[sl] = np.exp(np.where(ok[:, :, None], np.log(S), 0.0).sum(axis=1))
            if self.base in ("lba", "mlba"):
                pn[sl] = np.exp(np.where(ok, log_ndtr(-v), 0.0).sum(axis=1))
        return out, pn

    def expected_logt(self, tr: Trials) -> np.ndarray:
        """Expected log relative time, log(t0 + U), U the race's finishing time, by integration over GRID."""
        S, pn = self.min_survival(tr)
        t0 = self.t0(self.theta, self.tmin)
        S = np.column_stack([np.ones(len(S)), S])
        mass = (S[:, :-1] - S[:, 1:])
        mid = np.concatenate([[GRID[0] / 2], np.sqrt(GRID[:-1] * GRID[1:])])
        e = mass @ np.log(t0 + mid) + np.clip(S[:, -1] - pn, 0, None) * np.log(t0 + GRID[-1])
        return e / np.maximum(1 - pn, 1e-12)

    def choice_log2(self, tr: Trials, chunk: int = 300) -> np.ndarray:
        """Bits of the chosen option under the race's choice probabilities (integrated over time)."""
        th = self.theta
        out = np.empty(len(tr.n))
        du = np.diff(np.concatenate([[0.0], GRID]))
        for a in range(0, len(tr.n), chunk):
            sl = slice(a, a + chunk)
            x = tr.x[sl]; ok = np.isfinite(x); c = tr.choice[sl]
            v = self.drifts(x, th)
            f, S = self.fS(GRID[None, None, :], v[:, :, None], th, ok.sum(axis=1)[:, None, None])
            logS = np.where(ok[:, :, None], np.log(S), 0.0)
            tot = logS.sum(axis=1)
            idx = np.arange(len(c))
            dens = f[idx, c, :] * np.exp(tot - logS[idx, c, :])
            p = dens @ du
            allp = (np.where(ok[:, :, None], f, 0.0) * np.exp(tot[:, None, :] - logS)) @ du
            out[sl] = np.log2(np.maximum(p, 1e-300) / np.maximum(allp.sum(axis=1), 1e-300))
        return out


# ----------------------------------------------------------------------------- sample until return timing
@dataclass
class SRTiming:
    lam: float = 1.0
    theta: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))  # log kappa, log theta, t0 logit

    def look_pmf(self, tr: Trials, kmax: int) -> np.ndarray:
        out = np.zeros((len(tr.n), kmax + 1))
        for i in range(len(tr.n)):
            q = softmax(self.lam * tr.x[i, : tr.n[i]])
            S = sr_look_survival(q)  # P(L > k), k = 0..n, and P(L > n + 1) = 0
            S = np.concatenate([S, [0.0]])
            pm = S[:-1] - S[1:]  # P(L = k), k = 1 .. n + 1
            out[i, 1 : pm.size + 1] = pm
        return out

    def loglik(self, th, pmf, rt, tmin):
        kap, sc = np.exp(th[0]), np.exp(th[1]); t0 = tmin / (1 + np.exp(-th[2]))
        u = np.maximum(rt - t0, 1e-6)
        k = np.arange(pmf.shape[1])[None, :]
        with np.errstate(divide="ignore"):
            lg = (k * kap - 1) * np.log(u[:, None]) - u[:, None] / sc - gammaln(np.maximum(k * kap, 1e-9)) - k * kap * np.log(sc)
        lg = np.where(k > 0, lg, -np.inf) + np.log(np.maximum(pmf, 1e-300))
        m = lg.max(axis=1, keepdims=True)
        return m[:, 0] + np.log(np.exp(lg - m).sum(axis=1))

    def fit(self, tr: Trials, lam: float | None = None, starts: int = 3, seed: int = 0) -> "SRTiming":
        lam = fit_sr_lambda(tr) if lam is None else lam
        out = SRTiming(lam, self.theta.copy())
        rt = relative_time(tr); tmin = float(rt.min())
        pmf = out.look_pmf(tr, int(tr.n.max()) + 1)
        rng = np.random.default_rng(seed); best = None
        for s in range(starts):
            x0 = out.theta if s == 0 else out.theta + rng.normal(0, 0.7, 3)
            r = minimize(lambda th: -np.sum(out.loglik(th, pmf, rt, tmin)), x0, method="L-BFGS-B")
            if best is None or r.fun < best.fun:
                best = r
        out.theta, out.tmin, out.nll = best.x, tmin, float(best.fun)
        return out

    def expected_logt(self, tr: Trials) -> np.ndarray:
        kap, sc = np.exp(self.theta[0]), np.exp(self.theta[1]); t0 = self.tmin / (1 + np.exp(-self.theta[2]))
        pmf = self.look_pmf(tr, int(tr.n.max()) + 1)
        e = np.zeros(len(tr.n))
        cdf_grid = np.concatenate([[0.0], GRID])
        mid = np.concatenate([[GRID[0] / 2], np.sqrt(GRID[:-1] * GRID[1:])])
        for k in range(1, pmf.shape[1]):
            if not pmf[:, k].any():
                continue
            c = gamma_dist.cdf(cdf_grid, a=k * kap, scale=sc)
            mass = np.diff(c); tail = 1 - c[-1]
            ek = mass @ np.log(t0 + mid) + tail * np.log(t0 + GRID[-1])
            e += pmf[:, k] * ek
        return e
