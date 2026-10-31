'''Compare the blind MXL check (verification/independent_mxl) with this package, original sample, every value.
Usage: python -I verification/compare_mxl.py OUT_JSON   (run from the repository root)'''
import json, sys
from pathlib import Path
import numpy as np
np.seterr(all="ignore")
from overload.cdm import load_cdm
from overload.overload_forecast import EXPLORATORY_MODELS, cdm_panels, load_reutskaja, panel_measures
from overload.experiments.overload_premonition import snack_panel
from overload.experiments import overload_dev as dev

ind = json.load(open(sys.argv[1]))
G = cdm_panels(load_cdm("data/cdm"))
panels = [p for ps, _ in G.values() for p in ps] + [load_reutskaja("data/reutskaja")[0], snack_panel(Path("data/snack"))[0]]
worst = {}
def note(k, a, b):
    d = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
    worst[k] = max(worst.get(k, 0.0), d)
for p in panels:
    I = ind["open"][p.name]
    for mode, fitted in (("held_out", False), ("fitted", True)):
        r = panel_measures(p, np.ones(p.n.size), EXPLORATORY_MODELS, fitted)
        note("observed growth", r["growth"]["observed"], I["observed_growth"])
        for m in EXPLORATORY_MODELS:
            J = I["models"][m][mode]
            note(f"{mode} miss rates", r["rate"][m], J["miss_by_size"])
            note(f"{mode} growth", r["growth"][m], J["growth"])
            note(f"{mode} bits", r["loglik"][m]["all"], J["bits"])
            note(f"{mode} bits by size", r["loglik"][m]["by_size"], J["bits_by_size"])
            if m == "MXL":
                note(f"{mode} MXL mu sigma", r["lambda"]["MXL"], J["mu_sigma"])
    print(p.name, "done", flush=True)
# development data, MXL
dev.run.__globals__  # noqa
from overload.data import load_study
from overload.engine import precompute_bases
tasks = load_study(Path("/mnt/user-data/uploads/Updated_Choice_Overload.csv"))
ms = [m for m in dev.models() if m.name == "CL"]
dev._STATE.update(tasks=tasks, models=ms, bases=precompute_bases(tasks, ms))
dev._STATE["mxl"] = {t: dev._mxl_rows(t) for t in tasks}
from overload import mixed_logit
from overload.metrics import log_score
for t in ("file", "laptop"):
    fits = dev._mxl(t, None)
    I = ind["development"][t]
    S = dev._STATE["mxl"][t]["S"]
    counts = tasks[t].counts(None)
    for mode in ("fitted", "held_out"):
        miss, bits, par = [], [], []
        for j, trial in enumerate(tasks[t].trials):
            mu, sg = fits["fitted"] if mode == "fitted" else fits[j]
            q = mixed_logit.predict_mxl(mu, sg, S[j:j+1])[0, :trial.U.size]
            miss.append(1 - q[trial.U == trial.U.max()].sum()); bits.append(log_score(counts[j], q))
            par.append([mu, sg])
        note(f"dev {mode} miss", miss, I[mode]["miss"])
        note(f"dev {mode} bits", bits, I[mode]["bits_above_uniform"])
        note(f"dev {mode} mu sigma", par if mode == "held_out" else par[:1], I[mode]["mu_sigma"])
for k, v in worst.items():
    print(f"{k:28s} max abs diff {v:.2e}")
