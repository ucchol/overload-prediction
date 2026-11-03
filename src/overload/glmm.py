"""Logistic mixed model with correlated random effects per group, by the Laplace approximation.

The same estimator as lme4's glmer with nAGQ = 1. For group i, random effects b_i ~ N(0, Sigma) enter the linear
predictor through the columns Z (an intercept, and optionally one slope), so

    logit P(y = 1) = X beta + offset + Z b_i.

For given beta and Sigma, each group's mode of b_i is found by Newton's method, and the marginal likelihood is
approximated by the Laplace formula. Beta and the Cholesky factor of Sigma are fitted by L-BFGS-B, and standard
errors come from a finite difference Hessian of the Laplace log likelihood at the optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


@dataclass
class GLMMFit:
    beta: np.ndarray
    se: np.ndarray
    names: list[str]
    sigma: np.ndarray  # random effects covariance
    loglik: float
    converged: bool

    def z(self) -> np.ndarray:
        return self.beta / self.se

    def p(self) -> np.ndarray:
        return 2 * norm.sf(np.abs(self.z()))

    def table(self) -> list[dict]:
        return [{"term": n, "estimate": float(b), "se": float(s), "z": float(b / s), "p": float(2 * norm.sf(abs(b / s))),
                 "odds_ratio": float(np.exp(b)), "or_low": float(np.exp(b - 1.959964 * s)),
                 "or_high": float(np.exp(b + 1.959964 * s))} for n, b, s in zip(self.names, self.beta, self.se, strict=True)]


def _chol(theta: np.ndarray, q: int) -> np.ndarray:
    t = np.clip(theta, -12.0, 5.0)
    if q == 1:
        return np.array([[np.exp(t[0])]])
    return np.array([[np.exp(t[0]), 0.0], [t[1], np.exp(t[2])]])


def _laplace(beta, theta, X, Z, y, off, g, G, q, b_init=None):
    L = _chol(theta, q)
    S = L @ L.T + 1e-10 * np.eye(q)
    Si = np.linalg.inv(S)
    eta0 = X @ beta + off
    b = np.zeros((G, q)) if b_init is None else b_init.copy()
    for _ in range(50):
        eta = eta0 + np.einsum("nk,nk->n", Z, b[g])
        mu = expit(eta)
        w = mu * (1 - mu)
        r = y - mu
        grad = np.zeros((G, q)); H = np.zeros((G, q, q))
        for a in range(q):
            grad[:, a] = np.bincount(g, weights=Z[:, a] * r, minlength=G)
            for c in range(a, q):
                H[:, a, c] = np.bincount(g, weights=Z[:, a] * Z[:, c] * w, minlength=G)
                H[:, c, a] = H[:, a, c]
        grad -= b @ Si.T
        H += Si[None]
        step = np.linalg.solve(H, grad[..., None])[..., 0]
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    eta = eta0 + np.einsum("nk,nk->n", Z, b[g])
    ll_obs = np.sum(y * eta - np.logaddexp(0, eta))
    quad = np.einsum("gi,ij,gj->", b, Si, b)
    mu = expit(eta); w = mu * (1 - mu)
    H = np.zeros((G, q, q))
    for a in range(q):
        for c in range(a, q):
            H[:, a, c] = np.bincount(g, weights=Z[:, a] * Z[:, c] * w, minlength=G)
            H[:, c, a] = H[:, a, c]
    H += Si[None]
    _, logdetH = np.linalg.slogdet(H)
    _, logdetS = np.linalg.slogdet(S)
    ll = ll_obs - 0.5 * quad - 0.5 * G * logdetS - 0.5 * logdetH.sum()
    return ll, b


def fit_glmm(y, X, group, Z=None, offset=None, names=None, slope_random=True) -> GLMMFit:
    """y 0/1, X fixed effects design (include an intercept column), group labels, Z the random effects design
    (default, the intercept and, when slope_random, the second column of X)."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    _, g = np.unique(group, return_inverse=True)
    G = int(g.max()) + 1
    if Z is None:
        Z = X[:, :2] if slope_random else X[:, :1]
    Z = np.asarray(Z, float)
    q = Z.shape[1]
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    p = X.shape[1]
    # start from the logistic regression without random effects
    r0 = minimize(lambda bb: -np.sum(y * (X @ bb + off) - np.logaddexp(0, X @ bb + off)), np.zeros(p), method="BFGS")
    th0 = np.array([np.log(0.5)]) if q == 1 else np.array([np.log(0.5), 0.0, np.log(0.2)])
    state = {"b": None}

    def nll(par):
        ll, b = _laplace(par[:p], par[p:], X, Z, y, off, g, G, q, state["b"])
        if np.isfinite(ll):
            state["b"] = b
        return -ll if np.isfinite(ll) else 1e12

    x0 = np.concatenate([r0.x, th0])
    best = None
    for start in (x0, x0 + np.r_[np.zeros(p), np.full(th0.size, -1.0)]):
        state["b"] = None
        r = minimize(nll, start, method="L-BFGS-B", options={"maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r
    par = best.x
    state["b"] = None
    # finite difference Hessian of the negative log likelihood
    k = par.size
    h = 1e-4 * np.maximum(1.0, np.abs(par))
    Hm = np.zeros((k, k))
    f0 = nll(par)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h[i]
            ej = np.zeros(k); ej[j] = h[j]
            fpp = nll(par + ei + ej); fpm = nll(par + ei - ej); fmp = nll(par - ei + ej); fmm = nll(par - ei - ej)
            Hm[i, j] = Hm[j, i] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(Hm)
        se = np.sqrt(np.maximum(np.diag(cov)[:p], 0))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    L = _chol(par[p:], q)
    names = names or [f"x{i}" for i in range(p)]
    return GLMMFit(par[:p], se, list(names), L @ L.T, -float(f0), bool(best.success))
