# transcribefile

A single-file, cross-platform speech-to-text CLI built on
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) and the
same Cosmopolitan packaging as llamafile and whisperfile. One Actually
Portable Executable runs on macOS, Linux, Windows and BSD, on x86-64 and
ARM64, with no installation.

transcribe.cpp supports modern GGUF speech models from more than 16
different families, including Parakeet, Whisper, Canary, Voxtral, Moonshine,
and more (see `transcribe.cpp/docs/models/`).

## Quick start

```sh
make setup                                   # once: submodules + patches
.cosmocc/4.0.2/bin/make -j8 o//transcribefile

wget https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q4_K_M.gguf
o//transcribefile/transcribefile -m parakeet-tdt-0.6b-v3-Q4_K_M.gguf transcribe.cpp/samples/jfk.wav
```

Input audio is 16 kHz mono WAV (`transcribe.cpp/samples/` has many examples).
Run with `--help` for the full option list.


## GPU support

Backend selection is done via transcribe.cpp's `--backend` flag
(`auto|cpu|cpu_accel|metal|vulkan|cuda`, plus `--device` and
`--list-devices`). What this build actually wires up:

| backend | status |
|---|---|
| metal | supported on macOS/Apple Silicon |
| vulkan, cuda | not wired up yet (follow-up work) |

Metal uses llamafile's runtime loader (`llamafile/metal.c`): on first use
the bundled ggml Metal sources are extracted from the executable's zip
store, compiled into `ggml-metal.dylib` with the system compiler (needs
the Xcode command-line tools), cached under `~/.transcribefile/v/<ver>/`,
and loaded with `cosmo_dlopen`. The cache is deliberately separate from
llamafile's `~/.llamafile` so the two products' ggml trees can diverge
without overwriting each other's artifacts. With `--backend auto` (the
default), Metal is used when available and the run falls back to CPU
otherwise; only an explicit `--backend metal` makes a missing Metal
device an error, reported by transcribe.cpp.

GPU-side logging (device init banners, per-run Metal pipeline-state
creation) is suppressed by default; pass `--verbose` — same flag as
llamafile — to see it, along with the loader's own diagnostics, when
debugging GPU problems.

## Self-contained model bundles

Models can be embedded in the executable, llamafile-style, using a
`.args` file for default arguments:

```sh
printf -- '-m\n/zip/parakeet-tdt-0.6b-v3-Q4_K_M.gguf\n...\n' > .args
cp o//transcribefile/transcribefile parakeet-tdt-0.6b-v3-Q4_K_M.transcribefile
o//third_party/zipalign/zipalign -j0 parakeet-tdt-0.6b-v3-Q4_K_M.transcribefile parakeet-tdt-0.6b-v3-Q4_K_M.gguf .args

./parakeet-tdt-0.6b-v3-Q4_K_M.transcribefile audio.wav      # no -m needed
```

Command-line arguments still override the embedded defaults.

## Testing

```sh
tests/transcribefile_smoke.sh o//transcribefile/transcribefile
```

Model-free probes always run; set `TRANSCRIBEFILE_PARAKEET_GGUF` to a
parakeet GGUF to enable the end-to-end layer, and on Apple Silicon a
third layer checks that Metal is selected and produces transcripts
identical to the CPU backend.

## How it fits together

- `transcribefile/main.cpp` — entry point: crash reports, `--version`,
  `/zip/.args` defaults, GPU backend registration, then hands argv to
  upstream's CLI (`transcribe.cpp/examples/cli/main.cpp`, whose `main`
  is renamed via `-DTRANSCRIBEFILE`).
- `transcribe.cpp.patches/` — llamafile patches for the submodule: the
  cosmocc `BUILD.mk` plus the host-side ggml ABI patches (`GGML_CALL`,
  `free_struct`) that keep the backend interface structs identical to
  the GPU dylibs built from llama.cpp's ggml (both vendor ggml 0.15.2).
- `llamafile/gpu_backend.c`, `llamafile/metal.c` — shared GPU probe core
  and Metal runtime build, linked in via `o/$(MODE)/llamafile/gpu.a`.
