# Using Whisperfile with GPUs

Pass the `--gpu auto` flag to use GPU mode. This can be particularly
helpful in speeding up the medium and large models.

You can also specify a specific GPU backend:

- `--gpu apple` - Use Apple Metal (macOS)
- `--gpu amd` - Use AMD ROCm
- `--gpu nvidia` - Use NVIDIA CUDA

To disable GPU acceleration entirely, use `--no-gpu` or `--gpu disable`.
