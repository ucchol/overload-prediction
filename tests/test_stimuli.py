"""Checks of the stimulus definitions against the study's own scoring rule (plan, Amendment 4).

The designers scored every laptop with a documented rule, the weighted sum of point levels for memory (4),
graphics (4), storage (3), processor (3), battery (2), brand (2) and price (1). Their scores must follow that
rule on the option definitions the analysis uses. No participant data are needed.
"""

from __future__ import annotations

import json
from importlib import resources

import numpy as np
import pytest

from overload.coding import code_laptop_trial

WEIGHTS = np.array([4, 4, 3, 3, 2, 2, 1.0])  # memory, graphics, storage, processor, battery, brand, price


def _load(name: str) -> dict:
    return json.loads(resources.files("overload.stimuli").joinpath(name).read_text())


def _rule_correlation(spec: dict, rubric: dict) -> dict[str, float]:
    out = {}
    for key, s in spec.items():
        X = code_laptop_trial(s["options"])
        U = np.array([rubric[s["rubric_key"]][o] for o in s["options"]], dtype=float)
        out[key] = float(np.corrcoef(U, X @ WEIGHTS)[0, 1])
    return out


def test_designers_scores_follow_the_rule_on_the_storyboard_definitions():
    corr = _rule_correlation(_load("laptop_trials.json"), _load("rubric.json")["laptop"])
    assert all(r > 0.95 for r in corr.values()), corr


def test_earlier_definitions_break_the_rule_only_in_trials_5_and_6():
    corr = _rule_correlation(_load("laptop_trials_aamas.json"), _load("rubric.json")["laptop"])
    assert all(corr[k] > 0.95 for k in ("LT1_L8", "LT2_H8", "LT3_L16", "LT4_H16")), corr
    assert all(corr[k] < 0.8 for k in ("LT5_L32", "LT6_H32")), corr


@pytest.mark.parametrize("key", ["LT1_L8", "LT2_H8", "LT3_L16", "LT4_H16"])
def test_both_definitions_agree_outside_trials_5_and_6(key):
    assert _load("laptop_trials.json")[key] == _load("laptop_trials_aamas.json")[key]


@pytest.mark.parametrize("key", ["LT5_L32", "LT6_H32"])
def test_trials_5_and_6_keep_their_options_and_layout(key):
    new, old = _load("laptop_trials.json")[key], _load("laptop_trials_aamas.json")[key]
    assert list(new["options"]) == list(old["options"])
    assert {k: v for k, v in new.items() if k != "options"} == {k: v for k, v in old.items() if k != "options"}
    assert new["options"] != old["options"]
