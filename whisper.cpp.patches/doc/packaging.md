# How to make a Whisperfile

Whisperfile is designed to be a single-file solution for speech-to-text.
This tutorial will explain how you can merge the whisperfile executable
and OpenAI's model weights into a unified executable.

We'll be using Cosmopolitan Libc's "ZipOS" read-only filesystem and its
`.args` file convention for specifying default arguments.

## Prerequisites

To get started you first need the `zipalign` command. This is our
recommended solution for adding artifacts to a .zip file (note: the
whisperfile executable is both a zip file and executable). If `zipalign`
isn't already installed on your system, then you can build it from
source as follows:

```
make -j o//llamafile/zipalign
```

Now you can either obtain a prebuilt `whisperfile` executable from our
GitHub releases page, or build one from source yourself.

```
make -j o//whisper.cpp/main
cp o//whisper.cpp/main whisperfile
```

Next, choose your favorite weights. For the purposes of this tutorial,
we'll be using the tiny q5\_1 quantized whisper weights.

```
wget -O whisper-tiny.en-q5_1.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin
```

## Instructions


First, embed your weights inside your whisperfile. We're going to pass
the `-0` flag to disable PKZIP DEFLATE compression since it isn't
profitable on weights files.

```
zipalign -0 whisperfile whisper-tiny.en-q5_1.bin
```

Your weights are now embedded. You can list the embedded files using
`unzip -vl whisperfile` to confirm that it's there. Once the asset is
inside, Cosmopolitan Libc will make it available under the synthetic
`/zip/...` root directory, via the standard I/O APIs. So if `unzip -vl`
lists the asset as being named `whisper-tiny.en-q5_1.bin` then the
executable may access it at the path `/zip/whisper-tiny.en-q5_1.bin`.

```
./whisperfile -m /zip/whisper-tiny.en-q5_1.bin -f whisper.cpp/jfk.wav
```

It's now safe to delete the original weights file.

```
rm -f whisper-tiny.en-q5_1.bin
```

Next, to avoid needing to say `-m whisper-tiny.en-q5_1.bin` each time
you run your whisperfile, you can embed an additional file with the name
`.args` which specifies your default program arguments. You can do that
by first creating a file named `.args` in the current directory with the
following contents:

```
-m
/zip/whisper-tiny.en-q5_1.bin
...
```

As we can see, it's a simple file format that accepts one argument per
line. This way arguments can have spaces if needed, in which case quotes
should not be used (this isn't a shell script). The format also has one
special argument which is `...`. This should go at the end of the file.
It's where any additional CLI-specified args will be filled-in later on.

Now embed the `.args` file inside your whisperfile using `zipalign`.

```
zipalign whisperfile .args
rm -f .args
```

And congratulations, you now have a self-contained whisperfile!

```
./whisperfile -f whisper.cpp/jfk.wav
```
