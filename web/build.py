"""Build the static web interface into web/dist/ (what GitHub Pages serves).

    python web/build.py

Copies the page files and packs the Python package (src/filter_tester) into
dist/filter_tester.zip, which the page loads into Pyodide.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
PACKAGE = ROOT / "src" / "filter_tester"
PAGE_FILES = ["index.html", "style.css", "app.js", "backend.js", "charts.js"]


def build(out: Path = WEB / "dist") -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in PAGE_FILES:
        shutil.copy2(WEB / name, out / name)
    (out / ".nojekyll").write_text("")  # serve files as they are
    with zipfile.ZipFile(out / "filter_tester.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PACKAGE.rglob("*.py")):
            zf.write(path, path.relative_to(PACKAGE.parent).as_posix())
    return out


if __name__ == "__main__":
    out = build(Path(sys.argv[1]) if len(sys.argv) > 1 else WEB / "dist")
    print(f"built {out}")
