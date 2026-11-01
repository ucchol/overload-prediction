"""Independent filename cue for the H3 test A3b (plan, Amendment 9).

The cue of a file is the cosine similarity between the mean GloVe vector of the words in its name and
the mean GloVe vector of the anchor, the task's question as the study describes it. It is computed from
the filenames and the frozen vectors alone, never from choices.

Words. The name loses its extension (the text after the last dot), then splits at every character that
is not a letter and at every lower to upper case boundary, and at the last capital of a run of capitals
followed by a lower case letter. Words are lower cased. Words missing from the vectors are dropped. A
name with no word in the vectors takes the mean cue of the other names of its trial.

Vectors. GloVe 6B, 300 dimensions (Pennington, Socher and Manning, 2014, Public Domain Dedication and
License), as distributed in gensim-data release glove-wiki-gigaword-300. Only the vectors of the words
used are kept, in stimuli/filename_glove.json, extracted by scripts/extract_filename_glove.py.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from importlib import resources

import numpy as np

ANCHOR = "which file is most worth taking"
GLOVE_SOURCE = "glove-wiki-gigaword-300.gz"
GLOVE_SHA256 = "0a7aebbe49097dc6e5ffff7a25e9aa20181a6e862050ca68bd2f36e056739e00"

_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def stem(name: str) -> str:
    """The name without its extension."""
    return name.rsplit(".", 1)[0] if "." in name else name


def words(text: str) -> list[str]:
    out: list[str] = []
    for chunk in re.split(r"[^A-Za-z]+", text):
        if chunk:
            out.extend(w.lower() for w in _CAMEL.split(chunk) if w)
    return out


def file_words(name: str) -> list[str]:
    return words(stem(name))


def all_words(names: Iterable[str]) -> set[str]:
    out = set(words(ANCHOR))
    for n in names:
        out.update(file_words(n))
    return out


def load_vectors() -> dict[str, np.ndarray]:
    raw = json.loads(resources.files("overload.stimuli").joinpath("filename_glove.json").read_text())
    assert raw["source_sha256"] == GLOVE_SHA256
    return {w: np.asarray(v, dtype=float) for w, v in raw["vectors"].items()}


def _mean(ws: list[str], vec: Mapping[str, np.ndarray]) -> np.ndarray | None:
    found = [vec[w] for w in ws if w in vec]
    return np.mean(found, axis=0) if found else None


def cue(names: Iterable[str], vec: Mapping[str, np.ndarray]) -> np.ndarray:
    """Cue of every name of one trial, in order."""
    names = list(names)
    a = _mean(words(ANCHOR), vec)
    assert a is not None
    vals: list[float | None] = []
    for n in names:
        m = _mean(file_words(n), vec)
        vals.append(None if m is None else float(m @ a / (np.linalg.norm(m) * np.linalg.norm(a))))
    known = [v for v in vals if v is not None]
    fill = float(np.mean(known)) if known else 0.0
    return np.array([fill if v is None else v for v in vals])
