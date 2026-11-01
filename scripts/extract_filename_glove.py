"""Keep the GloVe vectors of the words in the file task's names and the anchor (plan, Amendment 9).

Usage: python -I scripts/extract_filename_glove.py PATH/glove-wiki-gigaword-300.gz
Writes src/overload/stimuli/filename_glove.json. Reads no choices.
"""

import gzip
import hashlib
import json
import sys
from pathlib import Path

from overload.filename_cue import GLOVE_SHA256, all_words

src = Path(sys.argv[1])
digest = hashlib.sha256(src.read_bytes()).hexdigest()
assert digest == GLOVE_SHA256, digest
trials = json.loads((Path(__file__).resolve().parents[1] / "src/overload/stimuli/file_trials.json").read_text())
names = [n for t in trials.values() for n in t["options"]]
need = all_words(names)
vectors = {}
with gzip.open(src, "rt", encoding="utf-8") as f:
    next(f)
    for line in f:
        w, rest = line.split(" ", 1)
        if w in need:
            vectors[w] = [float(x) for x in rest.split()]
out = Path(__file__).resolve().parents[1] / "src/overload/stimuli/filename_glove.json"
out.write_text(json.dumps({"source": "gensim-data glove-wiki-gigaword-300", "source_sha256": digest,
                           "license": "PDDL", "vectors": dict(sorted(vectors.items()))}))
missing = sorted(need - set(vectors))
print(f"words {len(need)}, found {len(vectors)}, missing {missing}")
