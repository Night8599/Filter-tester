"""Every Python block in the README runs, in order, as one script."""

import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_readme_python_blocks_run(monkeypatch):
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / "README.md").read_text(), re.S)
    assert len(blocks) >= 8
    monkeypatch.chdir(ROOT)  # the CSV example uses paths relative to the repository
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    namespace: dict = {}
    for i, code in enumerate(blocks):
        try:
            exec(compile(code, f"README block {i}", "exec"), namespace)
        except Exception as err:
            pytest.fail(f"README Python block {i} failed: {type(err).__name__}: {err}\n{code}")
    plt.close("all")
