"""Separate check of the Laplace GLMM (src/overload/glmm.py) on the real growth models.

The growth model with a random intercept only is refitted here by adaptive Gauss Hermite quadrature (25 nodes per
participant, centred at each participant's mode), written without the package's GLMM code. With many trials per
person the Laplace and quadrature estimates should agree closely. Compared, the slope of log2 n and its SE.
Usage, python3 -I verification/check_glmm_quadrature.py
"""
import sys
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, roots_hermite

sys.path.insert(0, "src")
from overload.experiments.overload_premonition import groups  # noqa: E402
from overload.glmm import fit_glmm  # noqa: E402

x_gh, w_gh = roots_hermite(25)
G = groups({"cdm": "data/cdm", "reutskaja": "data/reutskaja", "snack": "data/snack"})
sets = {"snack": G["snack"][0], "reutskaja": G["reutskaja"][0]}


def agq_nll(par, y, x, g, ng):
    b0, b1, ls = par
    s = np.exp(ls)
    tot = 0.0
    for i in range(ng):
        m = g == i
        yi, xi = y[m], x[m]
        eta0 = b0 + b1 * xi
        f = lambda u: -(np.sum(yi * (eta0 + u) - np.logaddexp(0, eta0 + u)) - 0.5 * u * u / s / s)
        r = minimize_scalar(f, bounds=(-10 * s - 5, 10 * s + 5), method="bounded")
        u0 = r.x
        mu = expit(eta0 + u0)
        h = np.sum(mu * (1 - mu)) + 1 / s / s
        sd = 1 / np.sqrt(h)
        u = u0 + np.sqrt(2) * sd * x_gh
        ll = np.array([np.sum(yi * (eta0 + uu) - np.logaddexp(0, eta0 + uu)) for uu in u])
        logint = ll - 0.5 * u * u / s / s - np.log(np.sqrt(2 * np.pi) * s) + np.log(np.sqrt(2) * sd) + x_gh ** 2
        mx = logint.max()
        tot += mx + np.log(np.sum(w_gh * np.exp(logint - mx)))
    return -tot


for ds, panels in sets.items():
    p = panels[0]
    y = p.miss().astype(float); x = np.log2(p.n); x = x - x.mean()
    _, g = np.unique(p.participant, return_inverse=True); ng = g.max() + 1
    X = np.column_stack([np.ones(y.size), x])
    lap = fit_glmm(y, X, g, Z=np.ones((y.size, 1)))
    r = minimize(agq_nll, np.r_[lap.beta, 0.5 * np.log(lap.sigma[0, 0])], args=(y, x, g, ng), method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
    # SE of the slope by a finite difference second derivative of the profile in b1
    h = 1e-3
    f0 = r.fun
    fp = agq_nll(r.x + [0, h, 0], y, x, g, ng); fm = agq_nll(r.x - [0, h, 0], y, x, g, ng)
    se_curv = 1 / np.sqrt((fp - 2 * f0 + fm) / h / h)
    print(ds, "slope Laplace", round(lap.beta[1], 5), "AGQ", round(r.x[1], 5),
          "| intercept SD Laplace", round(float(np.sqrt(lap.sigma[0, 0])), 4), "AGQ", round(float(np.exp(r.x[2])), 4),
          "| slope SE Laplace", round(lap.se[1], 5), "AGQ (curvature, conditional)", round(se_curv, 5))
