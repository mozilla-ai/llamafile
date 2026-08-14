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
| `tools/check_patches.sh` | Triage **only**: of the pre-existing patches, which still apply to the freshly-bumped submodule (line-number fuzz tolerated) and which need hand-work. Prints the conflicting patch's name and file:line. | repo root |
| `apply-patches.sh --tolerant` | Reconcile-apply during a bump: apply every patch that fits, leave one `.rej` per drifted hunk (instead of aborting like strict `setup`), and do the non-patch steps (copy `llamafile-files/`, remove Makefile, fetch UI). | repo root |
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
- **DO** reconcile with `apply-patches.sh --tolerant` for the whole set at once,
  or `git apply --reject <patch>` for a single drifted patch (Step 3):
  each applies every hunk that still fits and drops only the stragglers as `.rej`
  files, so a 50-hunk mechanical patch (e.g. the `GGML_CALL` ones) collapses to
  hand-editing the 1–2 hunks that actually moved. This is distinct from — and
  doesn't violate — the DON'Ts above: `check_patches.sh` triages (file-level,
  pre-edit), `generate-patches` produces, and `--reject` is just a precise way to
  *apply* during reconciliation. Delete the `.rej` files once reconciled (they'd
  otherwise be picked up as untracked files).

## The procedure

### Step 0 — Starting state (fresh clone vs. already-set-up worktree)

The steps below assume a **clean** tree: submodules at their pinned commits with
no patches applied. A fresh clone is clean; a worktree you (or a previous
session) already ran `make setup` on is **not** — the applied patches live as
uncommitted working-tree changes inside the submodules (`git status` shows
`m llama.cpp`, `m whisper.cpp`, …), and Step 1's `git submodule update` will not
run over them.

If the tree is already set up, reset to a clean slate first — this is the common
case when resuming a bump in an existing worktree:

```bash
make reset-repo   # drops applied patches, resets all submodules
```

`reset-repo` and `setup` use **bare** `make` — both are exempt from the cosmocc
version check, and `setup` must run bare on a fresh clone since it downloads
cosmocc (every *other* target uses `.cosmocc/4.0.2/bin/make`). `reset-repo` is
destructive (it `rm -rf`s each submodule dir before restoring it), so a
permission-gated environment may prompt or block it; if bare `make` is blocked,
the equivalent `.cosmocc/4.0.2/bin/make reset-repo` usually goes through.

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

**To bump to a specific tag/release** (e.g. syncing to `b10083`) instead of the
latest master, resolve the tag to a commit for `$COMMIT_ID` — replace the
`fetch`/`checkout` lines above with:

```bash
git -C llama.cpp fetch origin --tags
COMMIT_ID=`git -C llama.cpp rev-list -n1 b10083`   # <tag> -> commit sha
git -C llama.cpp checkout $COMMIT_ID
```

Branch/commit naming: the repo convention is release-based, not raw-SHA — branch
`llamacpp-bNNNN` and commit `Update llama.cpp to bNNNN (<short-sha>)` (e.g.
`llamacpp-b10083`, `Update llama.cpp to b10083 (846e991)`). Use that in place of
the `llamacpp_$COMMIT_ID` form above when bumping to a tagged release; the branch
may already exist, in which case just switch to it instead of `checkout -b`.

Keep `$OLD_ID` and `$COMMIT_ID` — you need them for drift recon below.

### Step 2 — Triage the existing patches

Run `tools/check_patches.sh` from the repo root. It reports, for each existing
patch, whether it still applies to the bumped submodule. This is *triage only*:
it tells you which patches are free and which need reconciliation. A patch may
be accepted despite line shifts — that fuzz is fine and welcome.

`check_patches.sh` uses `git apply --check`, which tolerates line **offset** but
not changed context — stricter than the actual apply (`patch -p1`, via `setup` or
`--tolerant`, which applies with fuzz). So triage tends to *over*-report: a patch
flagged here may still apply cleanly under `--tolerant` and leave no `.rej`.

A patch that **fails** triage has three possible fates, not one — decide which
before editing:
- **Reconcile** — upstream moved the code; reproduce the patch's intent against
  the new source (the common case).
- **Drop as obsolete** — upstream **absorbed** the change, so the patch is now
  redundant (this bump: upstream added the `<algorithm>` include `ngram-mod`
  patched in, and replaced the async shader-compile the Vulkan patch was
  rewriting). Applying it would duplicate/conflict. Don't reconcile it — delete
  it (see Step 5 for the cleanup gotcha).
- **Split** — part still applies, part is obsolete; keep the live hunks, drop the
  rest (`git apply --reject` makes this visible).

### Step 3 — Reconcile (edit llama.cpp in place)

Make the submodule build and work against the new upstream, editing files
**in place** (never editing patch files):

- Apply the patches that triage showed as clean; for the conflicting ones,
  hand-edit the new llama.cpp code to reproduce each patch's intent (use
  `git apply --reject` to isolate just the drifted hunks — see DO/DON'T).
- **BUILD.mk source lists:** add new upstream sources, drop deleted ones, fix
  renames. Drive this from drift recon:
  `cd llama.cpp && git diff --stat --summary $OLD_ID..$COMMIT_ID -- src/ common/ ggml/ tools/ ':(exclude)tools/ui'`
  (exclude `tools/ui` — the web UI ships via `fetch-ui-assets.sh`, not patches or
  BUILD.mk, and its churn otherwise dominates the diff)
  and cross-check against `llama.cpp/CMakeLists.txt` (the `src/models/` and
  `tools/mtmd/models/` dirs are globbed there, so *every* new `.cpp` is a real
  TU). A new source usually has to be listed in **more than one place** — see the
  "keep-in-sync" list under "Recurring breakage". Missing one fails as a
  link-time `undefined reference`, often only at `verify-clean`, not as a compile
  error.
- **llamafile integration:** reconcile any upstream API change that llamafile's
  **own** code calls — these live outside the patch set (in `llamafile/`) and
  surface as *compile errors at build*, not in triage. The usual suspects are the
  `chatbot_*` files and the server bridge; this bump, `mtmd_helper_bitmap_init_*`
  gained a `placeholder` arg and changed return type, breaking
  `chatbot_eval.cpp`/`chatbot_cli.cpp`. When the build errors in a `llamafile/`
  file, grep `llamafile/` for the changed symbol.
- **Buildable dirty tree:** `make setup` does more than apply patches — it copies
  `llamafile-files/` into the submodule (`BUILD.mk`, `common/license.cpp`),
  removes the upstream `Makefile`, and fetches the UI. If you reconcile by hand
  (applying patches directly rather than via `setup`, e.g. because some patches
  don't yet apply), replicate those steps or the build fails on a missing
  `BUILD.mk`/`license.cpp`. Also: a full build needs **all** submodules
  initialized (`reset-repo` deinits them all); to prove *just* llama.cpp,
  `.cosmocc/4.0.2/bin/make o/$(MODE)/llama.cpp`.

Reconcile with the pre-built tools — don't hand-roll `git apply` loops. After
the bump the tree is pristine upstream with **none** of llamafile's changes;
re-applying the patch set *replays* those changes onto the new base (a rebase),
and Step 5 regenerates the patches from the result. Run on the clean, freshly
bumped tree:

- **Triage** already named (Step 2) the conflicting patches and their file:line —
  `check_patches.sh` prints the `git apply --check` diagnostic under each failing
  patch's name.
- **No conflicts (the common case):** just run `make setup`. It applies every
  patch and does the non-patch steps (copies `llamafile-files/`, removes the
  Makefile, fetches the UI) deterministically.
- **Conflicts:** strict `make setup`/`apply-patches.sh` is fail-early (aborts on
  the first reject), so use the tolerant variant, which applies every hunk that
  fits, does the same non-patch steps, and leaves one `*.rej` per drifted hunk
  (listing them at the end):

  ```bash
  ./llama.cpp.patches/apply-patches.sh --tolerant
  ```

  Edit each `*.rej` into place — the one irreducibly-manual step (see the
  `git apply --reject` note in DO/DON'T) — then delete them:

  ```bash
  find llama.cpp -name '*.rej' -delete
  ```

### Step 4 — Prove the reconciliation works (on the dirty tree)

Before generating any patches, prove the in-place edits actually work:

- Clean build (`llamafile:clean` then `llamafile:build`) and `llamafile:check`.
- Run llamafile as expected — ideally the integration tests. Runtime / GPU /
  platform validation is your hardware's job (see handoff checklist).
- **Capturing build output:** a full clean build takes minutes — run it
  backgrounded. Never write logs or relevant artifacts under `o/`: the clean
  step does `rm -rf o`, so a redirect to `o/build.log` fails *before* the build
  starts and reads like a build failure (and any other artifact left there is
  wiped). Keep them in a scratch dir outside `o/`.

Only proceed past this gate once the build is green and llamafile runs. Patches
generated from unproven edits bake in breakage.

### Step 5 — Regenerate the patches

Run `llamafile:generate-patches`. It rewrites the full patch set from your
proven in-place edits (refreshing line numbers — this resilience is welcome).
New/untracked files (including `BUILD.mk`) are routed to `llamafile-files/`.

**Gotcha — it only writes, never deletes.** A patch you dropped as obsolete
(Step 2) has no corresponding edit, so `generate-patches` simply doesn't
regenerate it — but the **old `.patch` file lingers** and keeps getting applied
by `setup`. Manually `git rm llama.cpp.patches/patches/<dropped>.patch` for each
one. Sanity-check: `ls llama.cpp.patches/patches | wc -l` should equal what you
expect (old count − dropped + added).

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
- [ ] macOS Metal runtime compile + run — first bump the llamafile version or
      `rm -rf ~/.llamafile/v/<VERSION>`, else you test a **stale cached Metal
      dylib** from before the bump (looks like a Metal regression; see
      testing.md "stale per-version cache")
- [ ] Web UI serves — **host-checkable, do it in-session**: `llama-server` starts
      in *router mode with no model*, so launch it and `curl` the routes. `GET /`
      → 200 `index.html`; the hashed bundles under `/_app/immutable/...` (read the
      paths out of `index.html` — they're content-hashed, not `/bundle.js`) → 200.
      With a gzip build (`llama_ui_use_gzip()`), send `Accept-Encoding: gzip` or
      assets return **415**, and expect `Content-Encoding: gzip` on the response.
- [ ] Long-run stability (the `cv.wait` / futex class only shows after hours)

**Reconciled a GPU-only patch?** The host build compiles none of the GPU backends
— CUDA, Vulkan, and Metal DSOs build at runtime on the target — so any hunk you
hand-edited in `ggml-cuda/*`, `ggml-vulkan/*`, or `ggml-metal/*` is **not**
exercised by `verify-clean`; a green round-trip only proves the CPU-path patches.
Call such reconciliations out in the PR and make sure the matching GPU smoke test
above is run before merge.

## Recurring breakage (check these proactively)

Distilled from PRs #941, #951, #983 — these recur almost every bump and are the
bulk of the manual follow-up. Check them in Step 3/4 instead of waiting for them
to surface during your testing:

- **Keep-in-sync coupling points** — a new/renamed source often must be declared
  in several places that don't validate each other. When you touch BUILD.mk
  source lists, walk this whole registry:
  - **`llamafile/BUILD.mk`** — the TUI binary (`o/$(MODE)/llamafile/llamafile`)
    relinks an *explicit* server-object list (`LLAMAFILE_SERVER_SUPPORT_OBJS`)
    that is **separate** from llama.cpp's `TOOL_SERVER_SRCS`. A new
    `tools/server/*` (or mtmd) source must be added to **both**; missing the
    second is a link-time `undefined reference` that only shows at the
    `verify-clean` link step, not as a compile error (this bump:
    `server-schema.cpp`).
  - **GPU runtime build scripts** — the host build never exercises them (GPU DSOs
    compile at runtime on the target). The cuda/rocm scripts collect sources by
    glob (`collect_gpu_sources` over `ggml-cuda/*.cu`) and `vulkan.sh` globs
    `*.comp`, so *top-level* sources are auto-picked — but verify the glob still
    covers any new **subdir** / non-globbed path, and mirror anything explicit
    into `build-functions.sh`, `cuda.sh`/`.bat`, `rocm.sh`/`.bat`,
    `vulkan.sh`/`.bat`, and the Metal runtime-compile bundle.
  - **`fetch-ui-assets.sh` ↔ `tools/ui/embed.cpp`** — fetch's required-asset
    check must mirror embed.cpp's `required_check[]` (see Web UI below).
- **Web UI shipping** changes upstream most cycles, but the *structure* is
  durable: the embed + serve halves are **upstream** files — `tools/ui/embed.cpp`
  and `tools/server/server-http.cpp` — which llamafile patches **neither** (they
  usually already handle the new format). Only **two llamafile glue pieces** ever
  need touching: `fetch-ui-assets.sh` (download/verify/extract the assets) and
  the UI block in `llamafile-files/BUILD.mk` (invoke `embed`). Each bump, re-read
  embed.cpp's CLI and its `required_check[]` and make fetch match — the interface
  drifts: b9747 changed `embed` from `<name> <path>` pairs to
  `<out_cpp> <out_h> [<asset_dir>]` (recursive over a dir), and moved the HF
  bucket from 4 flat files to a SvelteKit `dist.tar.gz` of hashed
  `_app/immutable/*`. fetch now extracts the tarball and builds a `dist/_gzip/`
  mirror so embed emits gzip-encoded assets (binary growth ~5MB vs ~17MB raw).
  Symptom of a missed bump: `fetch-ui-assets.sh` 404s and the server builds
  UI-less (degrades gracefully — easy to not notice).
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
