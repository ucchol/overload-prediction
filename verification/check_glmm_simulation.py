"""Simulation check of src/overload/glmm.py. Known fixed effects and random effect covariance, 40 datasets,
recovery (mean estimate), coverage of 95 percent Wald intervals, and the mean estimated random effect SDs."""
import sys
import numpy as np
from scipy.special import expit
sys.path.insert(0, "src")
from overload.glmm import fit_glmm

rng = np.random.default_rng(4242)
beta = np.array([-1.5, 0.35]); sd = np.array([0.6, 0.25]); rho = 0.3
S = np.array([[sd[0] ** 2, rho * sd[0] * sd[1]], [rho * sd[0] * sd[1], sd[1] ** 2]])
G, per = 40, 80
est, cover, sds, ses = [], [], [], []
for r in range(40):
    g = np.repeat(np.arange(G), per)
    x = rng.choice([2.0, 3.0, 4.0, 5.0], size=g.size) - 3.5
    b = rng.multivariate_normal([0, 0], S, size=G)
    eta = beta[0] + b[g, 0] + (beta[1] + b[g, 1]) * x
    y = rng.random(g.size) < expit(eta)
    X = np.column_stack([np.ones(g.size), x])
    f = fit_glmm(y, X, g)
    est.append(f.beta); cover.append(np.abs(f.beta - beta) < 1.959964 * f.se); sds.append(np.sqrt(np.diag(f.sigma))); ses.append(f.se)
est, cover, sds, ses = map(np.array, (est, cover, sds, ses))
print("true beta", beta, "mean estimate", est.mean(0).round(3), "sd of estimates", est.std(0).round(3))
print("mean SE", ses.mean(0).round(3))
print("coverage of 95% intervals", cover.mean(0))
print("true random SDs", sd, "mean estimate", sds.mean(0).round(3))
