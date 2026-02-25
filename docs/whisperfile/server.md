# whisper-server HTTP API

The whisper-server provides an HTTP API for speech-to-text transcription.
Audio files are passed to the inference model via HTTP requests. MP3,
FLAC, and OGG files are automatically converted to WAV format.

## Usage

Build and run the server with a model:

```bash
.cosmocc/4.0.2/bin/make -j8 o//whisperfile
o//whisperfile/whisper-server -m models/whisper-tiny.en-q5_1.bin
```

The server accepts the following options:

```text
whisper-server options:
  -m FNAME, --model FNAME     Path of Whisper model weights
  --host ADDR                 Hostname or IP address to bind to (default: 127.0.0.1)
  --port PORT                 Port number (default: 8080)
  -l LANG, --language LANG    Default spoken language ('auto' for auto-detect)
  -tr, --translate            Translate audio into English text
  -t N, --threads N           Number of threads to use during computation
  -ng, --no-gpu               Disable GPU acceleration
  --gpu VALUE                 Select GPU backend (auto, apple, amd, nvidia, disable)
  --log-disable               Suppress logging output
```

Run `whisper-server --help` for the complete list of options.

> [!WARNING]
> **Do not run the server with administrative privileges and ensure it's operated in a sandbox environment, especially since it involves risky operations like accepting user file uploads. Always validate and sanitize inputs to guard against potential security threats.**

## HTTP Endpoints

### GET /health

Returns server health status as JSON. Returns HTTP 503 if the model
is still loading.

```bash
curl http://localhost:8080/health
```

Response when ready (HTTP 200):

```json
{"status": "ok"}
```

Response while model is loading (HTTP 503):

```json
{"status": "loading model"}
```

### POST /inference

Transcribe an audio file. Send as multipart/form-data with the audio
file in a field named "file".

Optional form fields:

- `response_format` - Output format: json, text, srt, vtt, verbose_json (default: json)
- `language` - Spoken language or 'auto' for detection
- `translate` - Set to 'true' to translate to English
- `temperature` - Sampling temperature
- `prompt` - Initial prompt for the model

Example:

```bash
curl http://localhost:8080/inference \
  -F "file=@whisper.cpp/samples/jfk.wav" \
  -F "response_format=json"
```

Response (HTTP 200):

```json
{"text": " And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."}
```

### POST /load

Load a different model at runtime.

```bash
curl http://localhost:8080/load \
  -F "model=/path/to/model.bin"
```

Response (HTTP 200):

```text
Load was successful!
```
