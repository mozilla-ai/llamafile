# Whisperfile

Whisperfile is a high-performance speech-to-text tool built on
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) by Georgi
Gerganov, et al., and [OpenAI's Whisper](https://github.com/openai/whisper)
model weights.

Whisperfile bundles the binary and model weights into a **single 
self-contained executable** that runs on Linux, macOS, and Windows without
installation.

## Quick Start

```sh
# transcribe a local audio file
whisperfile -m whisper-tiny.en-q5_1.bin audio.wav

# translate non-English speech to English
whisperfile -m ggml-medium-q5_0.bin -f audio.ogg --translate

# start the HTTP server
whisper-server -m whisper-tiny.en-q5_1.bin --port 8080
```

## Features

- Transcribes WAV, MP3, FLAC, and Ogg Vorbis audio
- GPU acceleration via Apple Metal, NVIDIA CUDA, and AMD ROCm
- Translates speech from any language into English
- HTTP server with a REST API for remote transcription
- Pack the binary and model weights into a single portable executable

## Documentation

- [Getting Started](getting-started.md)
- [Packaging](packaging.md)
- [Using GPUs](gpu.md)
- [Speech Translation](translate.md)
- [Server](server.md)
