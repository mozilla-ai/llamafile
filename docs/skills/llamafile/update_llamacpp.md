# Keeping llamafile updated with upstream llama.cpp

llamafile relies on llama.cpp for many of its functionalities. Keeping it up-to-date
with the latest version upstream is generally a good practice, as it brings both
bugfixes and support for recent models and features.

This document is the canonical procedure for a llama.cpp bump. It is built by
**composing small, single-purpose tools** — do not improvise around them.

## Tools and their one job each

| Tool | One job | Run from |
|------|---------|----------|
| `make reset-repo` | Clean slate: drop all local changes, reset submodules | repo root |
| `make setup` | Pull submodules **and** apply patches (+ fetch UI assets). Needs a clean tree (can't pull onto a dirty one). Doubles as the patch-**application** test. | repo root |
| `tools/check_patches.sh` | Triage **only**: of the pre-existing patches, which still apply to the freshly-bumped submodule (line-number fuzz tolerated) and which need hand-work. | repo root |
| `llamafile:generate-patches` | Regenerate **all** patches from in-place submodule edits. The only sanctioned way to produce patches. | (wraps the `cd`) |
| `llamafile:verify-clean` | Clean round-trip: `reset-repo` → `setup` → clean build → `check`. The post-generate verification. | repo root |
| `llamafile:build` / `llamafile:check` | Build all targets / run unit tests. | repo root |

### DO / DON'T

- **DON'T** craft or edit patches with `git diff` / `git apply`. Patch *production*
  is `llamafile:generate-patches`; patch *triage* is `check_patches.sh`; patch
  *round-trip verification* is `llamafile:verify-clean`.
- **DON'T** hand-roll `for p in patches/*.patch; do git apply --check ...` loops.
  That is what `check_patches.sh` (forward, pre-edit) and `verify-clean` (full
  round-trip, post-generate) already do.
- **DON'T** rebuild incrementally after a reset/setup — always clean build
  (`verify-clean` does this). Stale objects link silently otherwise.
- **DON'T** run `generate-patches` until the in-place edits are *proven* (clean
  build succeeds and llamafile runs as expected).
- **DO** use `git diff $OLD_ID..$COMMIT_ID` for **upstream-drift recon only**
  (seeing what changed upstream to drive BUILD.mk / integration work). That is
  the one legitimate ad-hoc `git diff` use.

## The procedure

### Step 1 — Bump the submodule

Creates a new branch with the submodule at its latest commit. The tree is now
clean and patches are **not** yet applied (fresh upstream code).

```bash
git submodule update --init llama.cpp

cd llama.cpp
OLD_ID=`git rev-parse HEAD`
git fetch origin master
COMMIT_ID=`git rev-parse origin/master`
git checkout origin/master
cd ..

git checkout -b llamacpp_$COMMIT_ID
git add llama.cpp
git commit -m "Update llama.cpp submodule to $COMMIT_ID"
```

Keep `$OLD_ID` and `$COMMIT_ID` — you need them for drift recon below.

### Step 2 — Triage the existing patches

Run `tools/check_patches.sh` from the repo root. It reports, for each existing
patch, whether it still applies to the bumped submodule. This is *triage only*:
it tells you which patches are free and which need reconciliation. A patch may
be accepted despite line shifts — that fuzz is fine and welcome.

### Step 3 — Reconcile (edit llama.cpp in place)

Make the submodule build and work against the new upstream, editing files
**in place** (never editing patch files):

- Apply the patches that triage showed as clean; for the conflicting ones,
  hand-edit the new llama.cpp code to reproduce each patch's intent.
- **BUILD.mk:** add new upstream sources, drop deleted ones, fix renames. Drive
  this from drift recon: `cd llama.cpp && git diff --stat --summary $OLD_ID..$COMMIT_ID -- src/ common/ ggml/ tools/`
  and cross-check against `llama.cpp/CMakeLists.txt`. Mirror BUILD.mk source
  changes into the **GPU runtime build scripts** (see "Recurring breakage").
- **llamafile integration:** reconcile any API changes llamafile's code relies
  on (entry points, chat/reasoning wiring, `include/` API).

### Step 4 — Prove the reconciliation works (on the dirty tree)

Before generating any patches, prove the in-place edits actually work:

- Clean build (`llamafile:clean` then `llamafile:build`) and `llamafile:check`.
- Run llamafile as expected — ideally the integration tests. Runtime / GPU /
  platform validation is your hardware's job (see handoff checklist).

Only proceed past this gate once the build is green and llamafile runs. Patches
generated from unproven edits bake in breakage.

### Step 5 — Regenerate the patches

Run `llamafile:generate-patches`. It rewrites the full patch set from your
proven in-place edits (refreshing line numbers — this resilience is welcome).
New/untracked files (including `BUILD.mk`) are routed to `llamafile-files/`.

Update `llama.cpp.patches/README.md` for any patch added/removed/materially
reworked.

### Step 6 — Verify by clean round-trip

Run `llamafile:verify-clean`. `reset-repo` → `setup` re-applies the **new**
patches onto a clean tree (erroring if any patch is broken), then a clean build
and `check`. A green round-trip proves the committed patch set is internally
consistent.

## Host verification is necessary, not sufficient

`verify-clean` only exercises the CPU build on the host. It does **not** cover
the things that broke in every past bump. Hand these off for testing on real
hardware/platforms and report them in the PR:

- [ ] CUDA / ROCm smoke test on a Linux GPU box
- [ ] Windows smoke test (incl. GPU DSO extraction — see "Permission denied" class)
- [ ] macOS Metal runtime compile + run
- [ ] Web UI serves: `/`, `/bundle.js`, `/bundle.css` return 200
- [ ] Long-run stability (the `cv.wait` / futex class only shows after hours)

## Recurring breakage (check these proactively)

Distilled from PRs #941, #951, #983 — these recur almost every bump and are the
bulk of the manual follow-up. Check them in Step 3/4 instead of waiting for them
to surface during your testing:

- **GPU runtime build scripts** — the host build never exercises them (GPU DSOs
  compile at runtime on the target). When BUILD.mk gains/renames a ggml source,
  mirror it into `llamafile/build-functions.sh`, `cuda.sh`/`cuda.bat`,
  `rocm.sh`/`rocm.bat`, `vulkan.sh`/`vulkan.bat`, and the Metal runtime-compile
  bundle. This is the #1 recurring miss.
- **Web UI shipping** changes upstream most cycles (e.g. the move to
  `tools/ui/` + HF-bucket assets in #983). Check `tools/server/public/` and
  `tools/ui/` early; expect to re-derive the asset pipeline.
- **TinyBLAS vs ggml quant block formats / cuBLAS API** (`QK*` block sizes,
  strided-batched gemm). Diff ggml quant headers and cuBLAS call sites.
- **`GGML_CALL` annotations** on any new/renamed backend callback (the
  meta-backend / vulkan callback breakage).
- **Untimed `cv.wait()`** added upstream on server threads → needs the
  `wait_for(30s)` loop + widened sigmask treatment (the futex/EINTR family,
  hit in both #941 and #983). Only reproduces after a long run — your bug, not
  the build's.
- **chat-template / reasoning refactors** → re-check llamafile's
  `--reasoning`/thinking wiring in `chatbot_*`.

When a runtime bug can't be reproduced in-session (GPU / long-run / platform),
state hypotheses *as* hypotheses, don't turn guessed numbers into evidence, and
propose a measurement/instrumentation step before a fix.

## Reference

- **Upstream changes:** https://github.com/ggerganov/llama.cpp/compare/$OLD_ID...$COMMIT_ID
- **Example PRs:** [#941](https://github.com/mozilla-ai/llamafile/pull/941),
  [#951](https://github.com/mozilla-ai/llamafile/pull/951),
  [#983](https://github.com/mozilla-ai/llamafile/pull/983)
