# Working with Models

This page covers how to use models downloaded by third-party applications with llamafile.

## Using models from third-party applications

This section answers the question *"I already have a model downloaded locally by application X, can I use it with llamafile?"*

The general answer is **yes, as long as those models are locally stored in GGUF format**, but its implementation can be more or less straightforward depending on the application. A few examples (tested on a Mac) follow.

## LM Studio

[LM Studio](https://lmstudio.ai/) stores downloaded models in `~/.cache/lm-studio/models`, in subdirectories with the same name of the models (following HuggingFace's `account_name/model_name` format), with the same filename you saw when you chose to download the file.

So if you have downloaded e.g. the `llama-2-7b.Q2_K.gguf` file for `TheBloke/Llama-2-7B-GGUF`, you can run llamafile as follows:

```sh
cd ~/.cache/lm-studio/models/TheBloke/Llama-2-7B-GGUF
llamafile -m llama-2-7b.Q2_K.gguf
```

## Ollama

When you download a new model with [ollama](https://ollama.com), all its metadata will be stored in a manifest file under `~/.ollama/models/manifests/registry.ollama.ai/library/`. The directory and manifest file name are the model name as returned by `ollama list`. For instance, for `llama3:latest` the manifest file will be named `.ollama/models/manifests/registry.ollama.ai/library/llama3/latest`.

The manifest maps each file related to the model (e.g. GGUF weights, license, prompt template, etc) to a sha256 digest. The digest corresponding to the element whose `mediaType` is `application/vnd.ollama.image.model` is the one referring to the model's GGUF file.

Each sha256 digest is also used as a filename in the `~/.ollama/models/blobs` directory (if you look into that directory you'll see *only* those sha256-* filenames). This means you can directly run llamafile by passing the sha256 digest as the model filename.

So if e.g. the `llama3:latest` GGUF file digest is `sha256-00e1317cbf74d901080d7100f57580ba8dd8de57203072dc6f668324ba545f29`, you can run llamafile as follows:

```sh
cd ~/.ollama/models/blobs
llamafile -m sha256-00e1317cbf74d901080d7100f57580ba8dd8de57203072dc6f668324ba545f29
```

## Finding GGUF models online

Most open-source LLM models are available in GGUF format on [Hugging Face](https://huggingface.co/). You can search for models and download the `.gguf` files directly, then use them with llamafile:

```sh
# Download a model
curl -L -o model.gguf https://huggingface.co/[model-path]/resolve/main/[model-file].gguf

# Run it with llamafile
llamafile -m model.gguf
```

See the [Quickstart](quickstart.md#other-example-llamafiles) page for a list of example llamafiles provided by Mozilla.
