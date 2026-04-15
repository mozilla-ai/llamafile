# GitBook Documentation Branch

The `gitbook-docs` branch contains **published** GitBook-compatible documentation,
automatically updated by GitHub Actions on every push to `main`.

**Do not edit this branch manually** — all changes will be overwritten.

## How it works

1. `scripts/publish_docs.py` copies public docs from `docs/` into `site/`, excluding
   internal developer docs (`AGENTS.md`, `commands/`, `skills/`)
2. The contents of `site/` are pushed to this branch
3. GitBook syncs from this branch

## Editing docs

Edit markdown files in [`docs/`](https://github.com/mozilla-ai/llamafile/tree/main/docs)
on the `main` branch. The `gitbook-docs` branch will update automatically on the next
push to `main` that touches `docs/`.
