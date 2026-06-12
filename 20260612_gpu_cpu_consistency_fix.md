# GPU/CPU numerical consistency: root cause and fixes

**Date:** 2026-06-12
**Branch:** `vulkan-ggml-fix`
**Status:** Four CPU-side bugs found and fixed; Vulkan (NVIDIA), Metal
(Apple Silicon) verified clean against both the fixed and vanilla CPU.
**Supersedes the hypothesis in:**
`../address-issue-938/20260610_vulkan_cpu_consistency_check.md`

## TL;DR — the attribution flipped

The iq4_xs/bf16 "Vulkan failures" reported by `test-backend-ops` on shadow
were not Vulkan failures. The GPU results were correct; the **CPU
reference** (llamafile's tinyBLAS/iqk fast path, `GGML_USE_LLAMAFILE`) was
wrong. The decisive experiment: with `LLAMAFILE_DISABLE_SGEMM=1` (vanilla
ggml CPU), the same binary + same DLL passes **947/947** MUL_MAT tests.
A native upstream llama.cpp build (same commit dbe9c0c, same machine, MSVC
+ CMake) also passes 947/947 — the Vulkan stack is clean on NVIDIA.

## The four CPU bugs (all fixed in this branch)

| # | Code | Affected | Mechanism | Blast radius |
|---|------|----------|-----------|--------------|
| 1 | `iqk_mul_mat.inc` vanilla-AVX2 `DequantizerIQ4XS` | iq4_xs MUL_MAT, NMSE 0.016–0.08 | LUT stored with +128 offset (values 1..241); shared `multiply_add` does `_mm256_maddubs_epi16` directly → int16 pair sums up to 2·241·127 = 61214 **saturate** | every x86 CPU except zen4-class (only zen4 takes the dpbusd FANCY path; even AVX512-VNNI Xeons route to the vanilla-AVX2 object) |
| 2 | `tinyblas_cpu_sgemm.inc` AVX512F bf16 branch | bf16 MUL_MAT with non-contiguous src1 → NaN/garbage (NMSE ~2) | only branch without a Btype check: reads bf16 wdata (vec_dot_type conversion) as f32 | AVX512F-without-BF16 CPUs (Skylake-X … Cascade/Ice Lake) |
| 3 | `tinyblas_cpu_sgemm.inc` iqk shortcut | all iqk-handled quants on batched+permuted src0 (NMSE ~2) | `iqk_mul_mat` has no lda/ldb params, assumes dense rows; hook forwarded permuted tensors anyway | x86 **and** ARM; masked on Vulkan (backend reports those shapes unsupported → tests skipped), exposed by Metal |
| 4 | `tinyblas_cpu_mixmul.inc` q1_0 case | all q1_0 MUL_MAT_ID (MoE) cases | `mixmat<32,32,…>` but QK1_0 = 128; kernel gets 4× the real block count | all ISAs with the mixmul path |

Fixes:

1. signed iq4_nl LUT + sign-trick multiply (`multiply_add_iq4xs`,
   |v| ≤ 127 keeps pair sums in int16 range; same technique as ggml's
   own AVX2 iq4 vec_dot); the −128·d mins compensation is dropped with
   the offset. FANCY (dpbusd) and ARM (signed vdot) paths were already
   correct and are untouched.
2. proper `tinyBLAS<…, ggml_bf16_t, ggml_bf16_t, …>` instantiation for
   bf16 B (the plain-AVX512F bf16 loader already existed).
3. gate the iqk shortcuts on `lda == k && ldb == k`.
4. `mixmat<128, 128, …>` for q1_0.

## Verification matrix

| Config | MUL_MAT | MUL_MAT_ID | full suite |
|---|---|---|---|
| shadow (Xeon W-3235 + Quadro RTX 6000, Vulkan DLL rebuilt from pin) — before | 868/947 | — | — |
| shadow, `LLAMAFILE_DISABLE_SGEMM=1`, before CPU fixes | 947/947 | — | — |
| shadow, native upstream control (CMake+MSVC, same commit) | 947/947 | — | — |
| shadow, after fixes #1+#2 | **947/947** | pending | pending (run in flight when shadow became unreachable) |
| local (Apple Silicon, Metal) — before #3/#4 | 1032/1062 | 618/692 | — |
| local, after all fixes | **1062/1062** | **692/692** | running |

The same-pin DLL rebuild also removed the version-skew caveat from the
original document: the failure signature was identical with the freshly
built DLL (79 failures: 68 bf16 + 11 iq4_xs).

## Vulkan build pipeline gap (separate fix, same branch)

`vulkan.bat`/`vulkan.sh` built `vulkan-shaders-gen` and `ggml-vulkan.cpp`
without the `GGML_VULKAN_{COOPMAT,COOPMAT2,INTEGER_DOT,BFLOAT16}_GLSLC_SUPPORT`
defines that upstream's CMake probes glslc for. Shipped DSOs therefore
silently dropped cooperative-matrix, integer-dot and native-bf16 pipelines
on every GPU and ran scalar fallback shaders everywhere — a configuration
upstream does not test on real hardware (their CI builds enable the
probes). Both scripts now run the same feature probe and invalidate the
build cache when the feature set changes. shadow's glslc (SDK 1.4.341.1)
supports all four.

This was not the numerical bug (the scalar fallbacks are *correct*), but
it is a real perf/feature divergence, and on Intel Arc it changes which
shader paths run (KHR coopmat is supported there) — relevant to the
still-open jmtor looping issue below.

## What this means for the original e2e symptom (jmtor's looping)

Still open, and now *more* puzzling in one sense, less in another:

- The op-level iq4_xs failure that plausibly explained the looping turned
  out to be CPU-side. CPU generation was the *coherent* one on jmtor, so
  ~3% CPU matmul error on 5 tensors evidently didn't destroy coherence —
  while the GPU, whose matmuls test clean on NVIDIA, looped.
- So the Arc looping is either (a) an Intel-driver/shader issue our
  NVIDIA runs can't see, or (b) something above the op level (scheduler,
  flash attention path, fp16 accumulation policy on that device).
- Next step unchanged from the original plan but with better tooling:
  run `backend_ops_test test -o MUL_MAT` (and `-o FLASH_ATTN_EXT`) **on
  the Arc machine** with the new harness + a DLL rebuilt with the new
  vulkan.bat (now with coopmat etc. — both with and without
  `GGML_VK_DISABLE_COOPMAT=1` to isolate). The fixed-CPU binary now gives
  a trustworthy reference there.

## Tooling / process notes

- `tests/backend_ops_harness.cpp` now probes all four backends (Metal,
  CUDA, ROCm, Vulkan) through llamafile's real loader, so one cosmocc
  binary is the consistency gate for every shipped DSO. Usage documented
  in `docs/building_dlls.md` ("Verifying numerical consistency").
- `LLAMAFILE_DISABLE_SGEMM=1` is the attribution tool: failures that
  disappear with it are CPU-side (tinyBLAS/iqk), not GPU-side.
- Coverage asymmetry matters: Vulkan rejects batched+permuted quantized
  MUL_MAT (`not supported`), so those CPU code paths were never tested by
  Vulkan runs. Metal accepts them and immediately exposed bug #3. Run the
  harness against the *most permissive* backend available when hunting
  CPU bugs.
- The upstream-native control (CMake+MSVC at the submodule pin,
  `build-native` in `C:\Users\Shadow\vkconsist\`) is reusable for future
  "ours vs upstream" comparisons; `GGML_VK_DISABLE_{COOPMAT,COOPMAT2,
  INTEGER_DOT_PRODUCT,BFLOAT16}` force upstream onto our fallback paths.

## Artifacts

| Artifact | Location |
|---|---|
| CPU fixes | commits on `vulkan-ggml-fix`: tinyblas/iqk (2 commits) |
| harness + BUILD.mk | commit on `vulkan-ggml-fix` |
| glslc probes in build scripts | commit on `vulkan-ggml-fix` |
| shadow test area (untouched repo) | `C:\Users\Shadow\vkconsist\` (same-pin DLL, harness exe, native control build, run logs) |
| backed-up vulkan build cache (other branch's) | `C:\Users\Shadow\.cache\llamafile-vulkan-build.bak-938` |
| run logs (local copies) | `/tmp/run1-mulmat.log`, `/tmp/run2.log` (nosgemm), `/tmp/run3.log` (fixed), `/tmp/native-stock.log`, `/tmp/metal-*.log` |

## Follow-ups

- [ ] Rerun shadow full suite + MUL_MAT_ID with the fixed binary (was in
      flight when shadow went unreachable).
- [ ] CUDA leg on shadow: build `ggml-cuda.dll` (cuda.bat; caches exist at
      `~/.cache/llamafile-cuda*-build`), run the harness against it.
- [ ] ROCm: needs an AMD box; harness is ready.
- [ ] Arc run (jmtor): new DLL (coopmat-enabled) + fixed harness; both
      `-o MUL_MAT` and `-o FLASH_ATTN_EXT`, each with/without
      `GGML_VK_DISABLE_COOPMAT=1`; then the deterministic e2e A/B from the
      original doc.
- [ ] Consider upstreaming fix #1 context to ik_llama (their current AVX2
      iq4_xs uses the sign trick already; llamafile's port predates it).
- [ ] Wire a fast-subset run into the release checklist (done in
      docs/building_dlls.md; needs to be habitual).
