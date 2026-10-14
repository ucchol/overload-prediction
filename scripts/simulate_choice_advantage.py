"""How much better would the search model predict choices than CL if it were the true process?

Design check for Amendment 5, test S5. Displays of 9 to 36 items with ratings drawn uniformly from -3 to 3, as on the
snack data. Choices come from sample until return at sensitivity lambda. CL's sensitivity is then fitted to the
search model's own choice probabilities (expected log likelihood, per set size, which favors CL), and the difference
in expected log likelihood of the chosen item is reported in bits and as a percent change of the likelihood.

The answer is a fraction of one percent. Even when the search model is true, CL with a refitted sensitivity predicts
the choices almost as well, so choices cannot tell the two processes apart and S5 is framed as non inferiority.
No data are used.

Usage
    python scripts/simulate_choice_advantage.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from overload.search import sample_until_return

SET_SIZES = (9, 16, 25, 36)
DISPLAYS = 1500


def main() -> None:
    rng = np.random.default_rng(5)
    print("lambda of the search model | set size | CL sensitivity fitted | search minus CL, bits | percent")
    for lam in (0.5, 1.0, 1.5):
        for n in SET_SIZES:
            V = rng.integers(-3, 4, size=(DISPLAYS, n)).astype(float)
            P = np.array([sample_until_return(lam, v) for v in V])

            def cl_nll(g: float, V: np.ndarray = V, P: np.ndarray = P) -> float:
                return -float(np.mean(np.sum(P * (g * V - logsumexp(g * V, axis=1, keepdims=True)), axis=1)))

            fit = minimize_scalar(cl_nll, bounds=(0.0, 20.0), method="bounded", options={"xatol": 1e-10})
            d = (float(np.mean(np.sum(P * np.log(P), axis=1))) + fit.fun) / np.log(2.0)
            print(f"{lam:.1f} | {n} | {fit.x:.2f} | {d:+.5f} | {100 * (2**d - 1):+.3f}")


if __name__ == "__main__":
    main()
