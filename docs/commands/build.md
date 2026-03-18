---
description: Build llamafile using the cosmocc toolchain
---

# Build Llamafile

Build the project using the Cosmopolitan toolchain.

First, ensure the toolchain is available:

```bash
if [ ! -d .cosmocc/4.0.2 ]; then
  build/download-cosmocc.sh .cosmocc/4.0.2 4.0.2 85b8c37a406d862e656ad4ec14be9f6ce474c1b436b9615e91a55208aced3f44
fi
```

Then build:

```bash
.cosmocc/4.0.2/bin/make -j $(nproc)
```
Adapt `nproc` to the OS where you are building, (e.g. `sysctl -n hw.physicalcpu` on mac)

Build outputs will be in `o/$(MODE)/`.
