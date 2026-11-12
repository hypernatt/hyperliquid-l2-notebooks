"""Execute notebooks/four_resolutions.ipynb in place (outputs kept).

    python tools/run_notebook.py            execute and save the notebook
    python tools/run_notebook.py --check    same, then fail if results/summary.json moved away from the committed one
"""
import json
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "four_resolutions.ipynb"
SUMMARY = ROOT / "results" / "summary.json"
TOLERANCE = 2e-6          # summary.json is rounded to 6 decimals


def first_difference(a, b, path="summary"):
    """Path of the first value that differs between two JSON structures, or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path}: keys differ"
        for k in a:
            if (found := first_difference(a[k], b[k], f"{path}.{k}")):
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: lengths differ"
        for i, (x, y) in enumerate(zip(a, b)):
            if (found := first_difference(x, y, f"{path}[{i}]")):
                return found
        return None
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b))
    if numbers:
        return None if abs(a - b) <= TOLERANCE + 1e-9 * max(abs(a), abs(b)) else f"{path}: {a} != {b}"
    return None if a == b else f"{path}: {a!r} != {b!r}"


def main() -> int:
    committed = SUMMARY.read_bytes() if SUMMARY.exists() else None
    nb = nbformat.read(NOTEBOOK, as_version=4)
    started = time.time()
    NotebookClient(nb, timeout=1800, kernel_name="python3", record_timing=False,
                   resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    with open(NOTEBOOK, "w", encoding="utf-8", newline=chr(10)) as f:
        nbformat.write(nb, f)
    print(f"executed in {time.time() - started:.0f} s")
    if "--check" not in sys.argv:
        return 0
    if committed is None:
        print("no committed results/summary.json to compare with")
        return 1
    difference = first_difference(json.loads(committed), json.loads(SUMMARY.read_bytes()))
    if difference:
        print("results/summary.json changed:", difference)
        return 1
    SUMMARY.write_bytes(committed)
    print("results/summary.json: same results as the committed file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
