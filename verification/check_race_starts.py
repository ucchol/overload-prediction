"""Do the frozen vault fits of the fixed threshold races reach their optimum? Each race is refitted on the
participants outside the vault with 10 starts instead of 3 (same first start, seven more random restarts), and its
negative log likelihood compared with the frozen fit's. Usage, python3 -I verification/check_race_starts.py DEV_CSV VAULT_DIR SET...
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from overload import time_models as tm  # noqa: E402
from overload.experiments.time_structure import SCALE, load_all  # noqa: E402
from time_models_run import vault_mask  # noqa: E402

np.seterr(all="ignore")
dev, vdir, sets = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
S = load_all(dev, Path("data/snack"), Path("data/reutskaja"), Path("data/cdm"), Path("data/gaze"))
KIND = {"LBA": "lba", "MLBA_LBA": "mlba", "Wald": "wald", "RWald": "rwald"}
for name in sets:
    T = tm.from_timeset(S[name], SCALE[name])
    D = T.take(~vault_mask(int(T.pid.max()) + 1)[T.pid])
    ex = json.load(open(vdir / f"{name}_summary.json"))["extras"]
    for m, k in KIND.items():
        R = tm.Race(k).fit(D, starts=10, deadline_s=3.0 if name == "reutskaja" else None)
        print(name, m, "frozen %.1f" % ex[m][0]["nll"], "ten starts %.1f" % R.nll,
              "difference %+.1f" % (ex[m][0]["nll"] - R.nll), flush=True)
