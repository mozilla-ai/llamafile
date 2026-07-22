---
description: Clean round-trip — reset, re-apply patches, clean build, and test
---

# Verify Clean

Verify the repository from a clean slate: reset, re-pull submodules and
re-apply patches, then do a **clean** build and run the tests. Use this:

- as the final verification after `llamafile:generate-patches` (confirms the
  regenerated patches re-apply cleanly and still build), and
- any time patches or submodules change and you need a trustworthy result.

Why a clean build: after a `reset-repo`/`setup`, submodule sources change
without their timestamps necessarily moving, so an incremental `make` can link
stale objects. Always `make clean` first here — do not rebuild incrementally.

`make setup` both pulls submodules and applies patches, and it cannot pull
submodules onto a dirty tree — that is exactly why `reset-repo` runs first.

Note: `reset-repo` is destructive (it `rm -rf`s the submodule dirs before
restoring them), so a permission-gated environment may prompt or block it. Run
it through the `$MAKE` (`.cosmocc/.../make`) form below rather than raw
`git clean`/`reset`; allowlist `make reset-repo` if round-trips keep stalling.

```bash
# ensure the toolchain is available
if [ ! -d .cosmocc/4.0.2 ]; then
  build/download-cosmocc.sh .cosmocc/4.0.2 4.0.2 85b8c37a406d862e656ad4ec14be9f6ce474c1b436b9615e91a55208aced3f44
fi
MAKE=.cosmocc/4.0.2/bin/make

$MAKE reset-repo     # clean: drop all local changes, reset submodules
$MAKE setup          # pull submodules + apply patches (+ fetch UI assets)
$MAKE clean          # drop stale build outputs
$MAKE -j$(nproc)     # clean build; on mac use -j$(sysctl -n hw.physicalcpu)
$MAKE check          # unit tests
```

A successful round-trip proves the committed patch set is internally
consistent. It does NOT cover GPU runtime backends, non-host platforms, the
web UI, or long-run stability — see the handoff checklist in
`update_llamacpp.md`.
