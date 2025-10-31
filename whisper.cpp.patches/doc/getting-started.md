## Getting Started with Whisperfile

This tutorial will explain how to turn speech from audio files into
plain text, using the whisperfile software and OpenAI's whisper model.

### (1) Download Model

First, you need to obtain the model weights. The tiny quantized weights
are the smallest and fastest to get started with. They work reasonably
well. The transcribed output is readable, even though it may misspell or
misunderstand some words.

```
wget -O whisper-tiny.en-q5_1.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin
```

### (2) Build Software

Now build the whisperfile software from source. You need to have modern
GNU Make installed. On Debian you can say `sudo apt install make`. On
other platforms like Windows and MacOS (where Apple distributes a very
old version of make) you can download a portable pre-built executable
from <https://cosmo.zip/pub/cosmos/bin/>.

```
make -j o//whisper.cpp/main
```

### (3) Run Program

Now that the software is compiled, here's an example of how to turn
speech into text. Included in this repository is a .wav file holding a
short clip of John F. Kennedy speaking. You can transcribe it using:

```
o//whisper.cpp/main -m whisper-tiny.en-q5_1.bin -f whisper.cpp/jfk.wav --no-prints
```

The `--no-prints` is optional. It's helpful in avoiding a lot of verbose
logging and statistical information from being printed, which is useful
when writing shell scripts.

## Supported Audio Formats

Whisperfile prefers that the input file be a 16khz .wav file with 16-bit
signed linear samples that's stereo or mono. Otherwise it'll attempt to
convert your audiofile automatically using an internal library. The MP3,
FLAC, and Ogg Vorbis Theora formats are supported across platforms.

For example, here's an audio recording of a famous poem in MP3 format:

```
wget https://archive.org/download/raven/raven_poe_64kb.mp3
o//whisper.cpp/main -m whisper-tiny.en-q5_1.bin -f raven_poe_64kb.mp3 -pc
```

Here we also passed the `-pc` flag to get color-coded terminal output
which communicates the confidence of transcription.

## Higher Quality Models

The tiny model may get some words wrong. For example, it might think
"quoth" is "quof". You can solve that using the medium model, which
enables whisperfile to decode The Raven perfectly. However it's slower.

```
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin
o//whisper.cpp/main -m ggml-medium.en.bin -f raven_poe_64kb.mp3 --no-prints
```

Lastly, there's the large model, which is the best, but also slowest.

```
wget -O whisper-large-v3.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
o//whisper.cpp/main -m whisper-large-v3.bin -f raven_poe_64kb.mp3 --no-prints
```

## Installation

If you like whisperfile, you can also install it as a systemwide command
named `whisperfile` along with other useful tools and utilities provided
by the llamafile project.

```
make -j
sudo make install
```
