# Speech Translation with Whisperfile

Whisperfile is not only able to transcribe speech to text. It's able to
translate that speech into English too, at the same time. All you have
to do is pass the `-tr` or `--translate` flag.

### Choosing a Model

In order for translation to work, you need to be using a multilingual
model. On <https://huggingface.co/ggerganov/whisper.cpp/> the files that
have `.en` in the name are English-only; you can't use those for
translation. One model that does work well in translation mode is
[`ggml-medium-q5_0.bin`](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin?download=true).

### Language Override

By default, the source language will be auto-detected. This works great
except for recordings with multiple languages. For example, if you have
a recording with a little bit of English at the beginning, but the rest
is in French, then you may want to pass the `-l fr` flag, to explicitly
specify the source language as French.
