"""Checks on the published text: wording, characters, and the two fields the code must never read."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "four_resolutions.ipynb"
CODE = [ROOT / "hlbook.py", ROOT / "hlplots.py", ROOT / "tools" / "run_notebook.py", ROOT / "tools" / "render_readme.py"]

EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
CLAIMS = re.compile(
    r"\b(unique|unrivall?ed|unmatched|revolutionary|world[- ]class|best[- ]in[- ]class|state[- ]of[- ]the[- ]art|"
    r"cutting[- ]edge|game[- ]chang\w+|guaranteed?|the only (dataset|archive|source|provider)|"
    r"first (ever|of its kind)|the first (dataset|archive|provider))\b", re.IGNORECASE)
ORDER_COUNT_OR_COUNTERPARTIES = re.compile(
    r"""\[\s*["']n["']\s*\]|\.get\(\s*["']n["']|\[\s*["']users["']\s*\]|\.get\(\s*["']users["']""")
ADDRESS = re.compile(r"0x[0-9a-fA-F]{40}(?![0-9a-fA-F])")


def notebook_cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def notebook_text(kinds=("markdown", "code"), outputs=True) -> str:
    parts = []
    for cell in notebook_cells():
        if cell["cell_type"] in kinds:
            parts.append("".join(cell["source"]))
        for out in cell.get("outputs", []) if outputs else []:
            parts.append("".join(out.get("text", [])))
            data = out.get("data", {})
            parts.append("".join(data.get("text/plain", [])))
            parts.append("".join(data.get("text/html", [])))
    return "\n".join(parts)


def published_texts() -> dict[str, str]:
    texts = {p.name: p.read_text(encoding="utf-8") for p in CODE}
    texts["README.md"] = (ROOT / "README.md").read_text(encoding="utf-8")
    texts["notebook"] = notebook_text()
    texts["summary.json"] = (ROOT / "results" / "summary.json").read_text(encoding="utf-8")
    return texts


def test_no_emoji_anywhere():
    for name, text in published_texts().items():
        assert not EMOJI.search(text), name


def test_wording_stays_factual():
    prose = {"README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
             "notebook": notebook_text(kinds=("markdown",)),
             "hlplots.py": (ROOT / "hlplots.py").read_text(encoding="utf-8")}
    for name, text in prose.items():
        found = CLAIMS.search(text)
        assert not found, f"{name}: {found.group(0) if found else ''}"


def test_order_counts_and_counterparties_are_never_read():
    sources = {p.name: p.read_text(encoding="utf-8") for p in CODE[:2]}
    sources["notebook code"] = notebook_text(kinds=("code",), outputs=False)
    for name, text in sources.items():
        assert not ORDER_COUNT_OR_COUNTERPARTIES.search(text), name


def test_no_wallet_address_is_published():
    for name, text in published_texts().items():
        assert not ADDRESS.search(text), name


def test_the_notebook_text_cites_no_result():
    """Markdown cells explain; numbers come from the code cells. Digits allowed in prose: the fixed facts below."""
    allowed = re.compile(
        r"20 (price )?levels|24-hour|SHA-256|CC BY 4\.0|1e-5|at least 1 BTC|[Cc]hapter \d|^#+ \d\.|^\d\. |`[^`]*`|"
        r"\(\.\./[^)]*\)|https?://\S+", re.MULTILINE)
    for cell in notebook_cells():
        if cell["cell_type"] != "markdown":
            continue
        text = allowed.sub("", "".join(cell["source"]))
        assert not re.search(r"\d", text), text[:200]


def test_the_notebook_and_the_readme_carry_the_same_badges():
    """Five badges: tests, DOI, data licence, code licence, Hugging Face copy; every one is a link."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    line = next(row for row in readme if row.startswith("[![CI]("))
    assert line.count("[![") == 5 and line.count("](https://") == 10
    first_cell = "".join(notebook_cells()[0]["source"])
    assert line in first_cell
    summary = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    assert f"https://doi.org/{summary['release']['doi']})" in line


def test_the_notebook_is_executed_and_clean():
    cells = notebook_cells()
    code = [c for c in cells if c["cell_type"] == "code"]
    assert code and all(c["execution_count"] for c in code)
    for c in code:
        for out in c["outputs"]:
            assert out["output_type"] != "error"
            assert out.get("name") != "stderr", "".join(out.get("text", []))[:200]


@pytest.mark.parametrize("name", ["00_day_mid", "01_one_snapshot", "02_depth_day", "03_cost_by_size",
                                  "04_cost_by_hour", "05_order_sizes", "06_realized_vs_book"])
def test_every_figure_is_written(name):
    assert (ROOT / "results" / f"{name}.png").stat().st_size > 20_000
