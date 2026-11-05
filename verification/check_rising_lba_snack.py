"""Is the corrected rising threshold LBA fit for snacks at a glance (gamma about 0.01) the optimum? Refits on the
participants outside the vault from ten cold starts, and from the ten start fixed threshold optimum with gamma 0.02,
0.3 and 0.7, and prints each negative log likelihood and gamma. Usage, python3 -I verification/check_rising_lba_snack.py DEV_CSV
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from overload import time_models as tm  # noqa: E402
from overload.experiments.time_structure import SCALE, load_all  # noqa: E402
from time_models_run import vault_mask  # noqa: E402

np.seterr(all="ignore")
S = load_all(Path(sys.argv[1]), Path("data/snack"), Path("data/reutskaja"), Path("data/cdm"), Path("data/gaze"))
T = tm.from_timeset(S["snack"], SCALE["snack"])
D = T.take(~vault_mask(int(T.pid.max()) + 1)[T.pid])
R0 = tm.Race("lba").fit(D, starts=10)
print("fixed, ten starts, nll %.1f" % R0.nll, flush=True)
Rc = tm.Race("lba_n").fit(D, starts=10)
print("rising, ten cold starts, nll %.1f gamma %.3f" % (Rc.nll, np.exp(Rc.theta[-2])), flush=True)
for g in (0.02, 0.3, 0.7):
    th0 = np.r_[R0.theta[:-1], np.log(g), R0.theta[-1]]
    R = tm.Race("lba_n", th0).fit(D, starts=3)
    print("rising, warm start gamma %.2f, nll %.1f gamma %.3f" % (g, R.nll, np.exp(R.theta[-2])), flush=True)
