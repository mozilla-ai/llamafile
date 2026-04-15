"""Validate the GitBook site/ output before deployment.

Checks:
  1. site/.gitbook.yaml exists and contains required keys
  2. Every file linked in site/SUMMARY.md exists on disk
  3. Every relative .md link inside site/**/*.md resolves to a real file

Exits non-zero on any failure so CI catches problems before deployment.

Usage:
    python scripts/validate_docs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SITE = Path("site")

# ── helpers ──────────────────────────────────────────────────────────────────

def error(msg: str) -> None:
    print(f"  ERROR  {msg}", file=sys.stderr)


def ok(msg: str) -> None:
    print(f"  ok     {msg}")


# ── 1. .gitbook.yaml ─────────────────────────────────────────────────────────

def check_gitbook_yaml() -> list[str]:
    failures: list[str] = []
    path = SITE / ".gitbook.yaml"
    if not path.exists():
        failures.append("site/.gitbook.yaml is missing")
        return failures

    try:
        import yaml  # type: ignore[import]
        cfg = yaml.safe_load(path.read_text())
    except Exception:
        # yaml not installed — parse minimally by hand
        text = path.read_text()
        cfg = {}
        for line in text.splitlines():
            if ":" in line and not line.startswith(" "):
                k, _, v = line.partition(":")
                cfg[k.strip()] = v.strip()

    if "root" not in cfg:
        failures.append("site/.gitbook.yaml: missing 'root' key")
    else:
        ok(".gitbook.yaml has 'root'")

    return failures


# ── 2. SUMMARY.md links ───────────────────────────────────────────────────────

_SUMMARY_LINK = re.compile(r"\]\(([^)#]+\.md)(?:#[^)]*)?\)")


def check_summary() -> list[str]:
    failures: list[str] = []
    summary = SITE / "SUMMARY.md"
    if not summary.exists():
        failures.append("site/SUMMARY.md is missing")
        return failures

    text = summary.read_text()
    linked: list[str] = _SUMMARY_LINK.findall(text)
    if not linked:
        failures.append("SUMMARY.md: no markdown links found — is it empty?")
        return failures

    for rel in linked:
        target = SITE / rel
        if target.exists():
            ok(f"SUMMARY -> {rel}")
        else:
            failures.append(f"SUMMARY.md links to missing file: {rel}")

    return failures


# ── 3. Relative links inside markdown files ───────────────────────────────────

_MD_LINK = re.compile(r"\]\(([^)#:]+\.md)(?:#[^)]*)?\)")


def check_relative_links() -> list[str]:
    failures: list[str] = []
    md_files = [p for p in SITE.rglob("*.md") if p.name != "SUMMARY.md"]

    for md_file in sorted(md_files):
        text = md_file.read_text()
        for m in _MD_LINK.finditer(text):
            rel = m.group(1)
            # Resolve relative to the file's directory
            target = (md_file.parent / rel).resolve()
            site_abs = SITE.resolve()
            if not str(target).startswith(str(site_abs)):
                failures.append(f"{md_file.relative_to(SITE)}: link escapes site/: {rel}")
            elif target.exists():
                ok(f"{md_file.relative_to(SITE)} -> {rel}")
            else:
                failures.append(f"{md_file.relative_to(SITE)}: broken link -> {rel}")

    return failures


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    if not SITE.exists():
        print("ERROR: site/ directory not found — run publish_docs.py first", file=sys.stderr)
        return 1

    print("Validating site/ ...")
    failures: list[str] = []
    failures += check_gitbook_yaml()
    failures += check_summary()
    failures += check_relative_links()

    print()
    if failures:
        print(f"FAILED — {len(failures)} issue(s):", file=sys.stderr)
        for f in failures:
            error(f)
        return 1

    print(f"All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
