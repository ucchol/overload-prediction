"""Append analysis sections to notebooks/02_overload_time_search.ipynb with their outputs.

Each section is a markdown cell, a code cell and a findings cell. The code is run here, from the repository root,
and its printed text and figures are stored as the cell's outputs, so the notebook shows what was run without
re-executing the earlier sections. Usage, python3 -I scripts/append_notebook.py SECTIONS_JSON
SECTIONS_JSON is a list of {"title", "intro", "code", "findings"}.
"""
import base64
import contextlib
import io
import json
import sys
from pathlib import Path

NB = Path("notebooks/02_overload_time_search.ipynb")


def run(code):
    buf = io.StringIO()
    images = []
    ns = {"REPO": Path(".").resolve(), "Path": Path}

    def show_png(path):
        images.append(base64.b64encode(Path(path).read_bytes()).decode())

    ns["show_png"] = show_png
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<cell>", "exec"), ns)
    out = []
    if buf.getvalue():
        out.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(keepends=True)})
    for im in images:
        out.append({"output_type": "display_data", "data": {"image/png": im, "text/plain": ["<Figure>"]}, "metadata": {}})
    return out


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


if __name__ == "__main__":
    nb = json.load(open(NB))
    secs = json.load(open(sys.argv[1]))
    count = max([c.get("execution_count") or 0 for c in nb["cells"] if c["cell_type"] == "code"] + [0])
    for s in secs:
        count += 1
        nb["cells"].append(md(f"## {s['title']}\n\n{s['intro']}"))
        nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": count,
                            "source": s["code"].splitlines(keepends=True), "outputs": run(s["code"])})
        nb["cells"].append(md("**Findings.**\n\n" + s["findings"]))
    json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
    print("cells", len(nb["cells"]))
