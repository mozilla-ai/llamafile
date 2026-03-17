# Getting Started with Whisperfile

This tutorial will explain how to turn speech from audio files into plain text, using the whisperfile software and OpenAI's whisper model.

## (0) Setup the repo

```bash
git clone https://github.com/mozilla-ai/llamafile.git
cd llamafile

# initialise all submodules - this step is required,
# as the submodules need to be pulled and patched first!
make setup
```

## (1) Download Model

First, you need to obtain the model weights. For this tutorial, we'll use the tiny quantized model, since
it is the smallest and fastest to get started with and works reasonably well. The transcribed output is readable, even though it may misspell or misunderstand some words.


```bash
curl -L -o models/whisper-tiny.en-q5_1.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin
```

## (2) Build Software

Now build the whisperfile software from source. 

```bash
.cosmocc/4.0.2/bin/make -j8 o//whisperfile
```

## (3) Run Program

Now that the software is compiled, here's an example of how to turn speech into text. Included in this repository is a .wav file holding a short clip of John F. Kennedy speaking. You can transcribe it using:

```bash
o//whisperfile/whisperfile -m models/whisper-tiny.en-q5_1.bin whisperfile/jfk.wav --no-prints
```

The `--no-prints` is optional. It's helpful in avoiding a lot of verbose logging and statistical information from being printed, which is useful when writing shell scripts.

## Supported Audio Formats

Whisperfile prefers that the input file be a 16khz .wav file with 16-bit signed linear samples that's stereo or mono. Otherwise it'll attempt to convert your audiofile automatically using an internal library. The MP3,
FLAC, and Ogg Vorbis formats are supported across platforms.

For example, here's an audio recording of a famous poem in MP3 format:

```bash
curl -LO https://archive.org/download/raven/raven_poe_64kb.mp3
o//whisperfile/whisperfile -m models/whisper-tiny.en-q5_1.bin -f raven_poe_64kb.mp3 -pc
```

Here we passed the `-pc` flag to get color-coded terminal output which communicates the confidence of transcription.

## Higher Quality Models

The tiny model may get some words wrong. For example, it might think
"quoth" is "quof". You can solve that using the medium model, which
enables whisperfile to decode The Raven perfectly. However it's slower.

```bash
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin
o//whisperfile/whisperfile -m ggml-medium.en.bin -f raven_poe_64kb.mp3 --no-prints
```

Lastly, there's the large model, which is the best, but also slowest.

```bash
curl -L -o models/whisper-large-v3.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
o//whisperfile/whisperfile -m models/whisper-large-v3.bin -f raven_poe_64kb.mp3 --no-prints
```

> [!NOTE]
> Here are how different model sizes compared in terms of size and performance:
>
> | Model | Download Size | Speed | Accuracy |
> |-------|--------------|-------|----------|
> | tiny | ~31 MB | fastest | good |
> | medium | ~1.5 GB | moderate | better |
> | large | ~3.1 GB | slowest | best |
>
> See [Higher Quality Models](#higher-quality-models) for download instructions.

## Installation

If you like whisperfile, you can also install it as a systemwide command by the llamafile project.

```bash
.cosmocc/4.0.2/bin/make -j8
sudo make install
```
