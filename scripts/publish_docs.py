"""Build the GitBook site output into site/.

Copies public docs into site/, excluding internal developer docs.
The contents of site/ are pushed to the gitbook-docs branch by CI.

Usage:
    python scripts/publish_docs.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

DOCS_SRC = Path("docs")
SITE_DIR = Path("site")

# Internal developer docs — not published to GitBook
EXCLUDE_PATHS = {
    "AGENTS.md",
    "commands",
    "skills",
}

GITBOOK_YAML = """\
root: ./

structure:
  readme: index.md
  summary: SUMMARY.md
"""


def main() -> None:
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir()

    for src in DOCS_SRC.iterdir():
        if src.name in EXCLUDE_PATHS:
            continue
        dst = SITE_DIR / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    (SITE_DIR / ".gitbook.yaml").write_text(GITBOOK_YAML, encoding="utf-8")

    md_files = list(SITE_DIR.rglob("*.md"))
    print(f"Done — {len(md_files)} markdown files written to {SITE_DIR}/")


if __name__ == "__main__":
    main()
