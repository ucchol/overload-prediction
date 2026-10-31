"""Mixed logit with a random sensitivity, the two parameter rival of the one parameter models (exploratory).

Every person chooses by CL, softmax(lambda s) over the same scores, but people differ in their sensitivity,
log lambda ~ Normal(mu, sigma). The two parameters are fitted by maximum likelihood of each person's choices taken
together. A person who was not fitted is predicted by the population average, the mean of softmax(lambda s) over the
fitted spread, as every other model predicts.

The integral over a person's sensitivity is a sum over a fixed fine grid of log lambda (spacing 0.02), with weights
from the normal density renormalised over the grid. People who made many choices (183 on average in the snack data)
have a likelihood far narrower than the gaps between Gauss-Hermite nodes, which made a 20 node rule wrong by several
log likelihood units and gave false local optima. On the grid each person's likelihood is computed once per fit.

This asks whether differences between people, rather than a change within people, explain how choices change with
the number of options. Added on the study author's request after Amendment 8.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HALF_WIDTH = 7.0  # grid of log lambda, centre plus or minus this
STEP = 0.02
MU_RANGE = 2.5  # mu within the centre plus or minus this, so mu plus or minus 3 sigma stays on the grid
SIGMA_BOUNDS = (0.05, 1.5)
STARTS = (np.log(0.1), np.log(0.5), np.log(1.2))  # starting log spreads
PREDICT_NODES = 101


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = np.where(np.isfinite(z), z, -np.inf)
    return z - logsumexp(z, axis=-1, keepdims=True)


def _grid_weights(grid: np.ndarray, mu: float, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """log weights of the grid points under Normal(mu, sigma), renormalised, and their derivatives in mu and log
    sigma."""
    d = (grid - mu) / sigma
    a = -0.5 * d**2
    logw = a - logsumexp(a)
    p = np.exp(logw)
    da_mu = d / sigma
    da_ls = d**2
    return logw, da_mu - p @ da_mu, da_ls - p @ da_ls


def person_loglik(S: np.ndarray, choice: np.ndarray, inv: np.ndarray, people: int, grid: np.ndarray) -> np.ndarray:
    """Sum of log probabilities of each person's choices at every grid sensitivity (grid points by people)."""
    rows = np.arange(choice.size)
    out = np.empty((grid.size, people))
    for k, g in enumerate(grid):
        out[k] = np.bincount(inv, weights=_log_softmax(np.exp(g) * S)[rows, choice], minlength=people)
    return out


def fit_mxl(
    S: np.ndarray, choice: np.ndarray, person: np.ndarray, w: np.ndarray, start: float | None = None
) -> tuple[float, float]:
    """Maximum likelihood (mu, sigma) from trials weighted by w, where every trial of a person carries the same weight
    (the person's draw count). A person's trials share one sensitivity. start is the sensitivity of a plain CL fit,
    the centre of the grid."""
    keep = w > 0
    S, choice, person, w = S[keep], choice[keep], person[keep], w[keep]
    uniq, inv = np.unique(person, return_inverse=True)
    mult = np.zeros(uniq.size)
    mult[inv] = w
    centre = float(np.log(start)) if start is not None and start > 0 else 0.0
    grid = centre + np.arange(-HALF_WIDTH, HALF_WIDTH + STEP / 2, STEP)
    L = person_loglik(S, choice, inv, uniq.size, grid)

    def nll(theta: np.ndarray) -> tuple[float, np.ndarray]:
        mu, log_sigma = theta
        logw, dmu, dls = _grid_weights(grid, mu, np.exp(log_sigma))
        a = L + logw[:, None]
        lse = logsumexp(a, axis=0)
        r = np.exp(a - lse) @ mult  # posterior weight of each grid point, summed over people
        return -float(mult @ lse), -np.array([r @ dmu, r @ dls])

    best = None
    for log_sigma in STARTS:
        res = minimize(
            nll,
            np.array([centre, log_sigma]),
            method="L-BFGS-B",
            jac=True,
            bounds=[(centre - MU_RANGE, centre + MU_RANGE), tuple(np.log(SIGMA_BOUNDS))],
            options={"ftol": 1e-13, "gtol": 1e-8},
        )
        if best is None or res.fun < best.fun:
            best = res
    return float(best.x[0]), float(np.exp(best.x[1]))


def predict_mxl(mu: float, sigma: float, S: np.ndarray) -> np.ndarray:
    """Population average choice probabilities, trials by options (zero past a set), by Gauss-Hermite quadrature
    (smooth in the sensitivity, so a fixed rule is accurate)."""
    x, wts = np.polynomial.hermite.hermgauss(PREDICT_NODES)
    out = np.zeros(S.shape)
    for xk, wk in zip(np.sqrt(2.0) * x, wts / np.sqrt(np.pi), strict=True):
        out += wk * np.exp(_log_softmax(np.exp(mu + sigma * xk) * S))
    return out
