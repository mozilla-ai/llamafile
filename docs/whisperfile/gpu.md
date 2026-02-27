# Using Whisperfile with GPUs

GPU acceleration is most beneficial for the medium and large models. The
tiny model is already fast on CPU, so the speedup there is minimal.

Pass `--gpu auto` to let whisperfile detect and use the best available GPU
on your system. If no supported GPU is found, it falls back to CPU silently:

```bash
whisperfile -m models/ggml-medium.en.bin -f audio.wav --gpu auto
```

You can also target a specific backend:

- `--gpu apple` — Apple Metal (macOS, works on Apple Silicon and AMD GPUs)
- `--gpu nvidia` — NVIDIA CUDA (requires CUDA Toolkit to be installed)
- `--gpu amd` — AMD ROCm (requires ROCm to be installed on Linux)

To disable GPU acceleration entirely:

```bash
whisperfile -m models/ggml-medium.en.bin -f audio.wav --no-gpu
```

## Troubleshooting

**`ggml_backend_load_best: search path does not exist` warnings**

These are benign. They appear when whisperfile searches for GPU backend
libraries and doesn't find them — usually because no GPU is present or
configured. Transcription will continue on CPU. To suppress them, redirect
stderr:

```bash
whisperfile -m models/ggml-medium.en.bin -f audio.wav 2>/dev/null
```
