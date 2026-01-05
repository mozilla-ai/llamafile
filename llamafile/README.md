# llamafile_new - TUI Integration

This directory contains a new implementation of the llamafile TUI (Text User Interface) that integrates with the new llama.cpp build system.

## Overview

The TUI provides an interactive chatbot interface using:
- **bestline**: Terminal line input library with history and completion
- **highlight**: Syntax highlighting for code output
- **llama.cpp**: Core LLM inference via the new common API

## Building

```sh
# From the root directory
make llamafile_new
```

The binary will be built at `o/$(MODE)/llamafile_new/llamafile`.

## Usage

```sh
# Basic usage
./llamafile -m model.gguf

# With vision model (multimodal)
./llamafile -m model.gguf --mmproj vision.gguf

# TUI-only mode (no server)
./llamafile -m model.gguf --chat
```

## File Structure

### Core TUI Files
- `main.cpp` - Entry point
- `chatbot.h` - Main TUI header with globals
- `chatbot_main.cpp` - Initialization, model loading
- `chatbot_repl.cpp` - Main REPL loop, sampling
- `chatbot_eval.cpp` - Token evaluation
- `chatbot_hist.cpp` - Context/history management
- `chatbot_comm.cpp` - Slash commands
- `chatbot_file.cpp` - File upload handling
- `chatbot_help.cpp` - Help text
- `chatbot_hint.cpp` - Input hints
- `chatbot_comp.cpp` - Tab completion
- `chatbot_logo.cpp` - ASCII logo

### Utility Files
- `llama.h/cpp` - Wrapper functions for llama.cpp
- `string.h/cpp` - String utilities
- `color.h` - ANSI color codes
- `compute.h/cpp` - CPU description
- `datauri.h/cpp` - Data URI parsing
- `image.h/cpp` - Image type detection
- `xterm.h/cpp` - Terminal image display
- `bestline.h/c` - Line input library
- `macros.h` - MIN/MAX macros

### Highlight Library
- `highlight/` - Syntax highlighting for many languages

### Build
- `BUILD.mk` - Makefile for building llamafile_new

## API Changes from Old llamafile

The code has been updated to use the new llama.cpp API:

| Old API | New API |
|---------|---------|
| `gpt_params` | `common_params` |
| `gpt_params_parse()` | `common_params_parse()` |
| `llama_model_params_from_gpt_params()` | `common_model_params_to_llama()` |
| `llama_context_params_from_gpt_params()` | `common_context_params_to_llama()` |
| `llama_sampling_context` | `common_sampler` |
| `llama_sampling_init()` | `common_sampler_init()` |
| `llama_sampling_sample()` | `common_sampler_sample()` |
| `llama_sampling_accept()` | `common_sampler_accept()` |
| `llama_sampling_free()` | `common_sampler_free()` |
| `clip_ctx` / llava | `mtmd_context` |
| `clip_model_load()` | `mtmd_init_from_file()` |
| `clip_free()` | `mtmd_free()` |
| `llama_token_*()` | `llama_vocab_*()` |
| `llama_token_is_eog()` | `llama_vocab_is_eog()` |
| `llama_load_model_from_file()` | `llama_model_load_from_file()` |
| `llama_new_context_with_model()` | `llama_init_from_model()` |
| `llama_free_model()` | `llama_model_free()` |
| `llama_should_add_bos_token()` | `llama_model_add_bos_token()` |

## Current Limitations

1. **Image Processing**: The multimodal (mtmd) API integration is stubbed out. Images can be loaded but processing returns an error. Full implementation requires adapting to the new mtmd chunk-based API.

2. **Server Integration**: The background server feature is not yet implemented. Use the llama.cpp server separately for now.

## Cosmopolitan FLAG System

The code uses Cosmopolitan's FLAG system for configuration. This is a global variable system where flags are declared in `llamafile.h` and parsed at startup.

### Flag Declarations (llamafile.h)

```c
extern bool FLAG_log_disable;
extern bool FLAG_nologo;
extern const char *FLAG_mmproj;
// ... etc
```

### How Flags Work

1. Flags are global variables prefixed with `FLAG_`
2. They can be set via command-line arguments (`--flag-name`)
3. The `llamafile_get_flags()` function parses arguments and sets flags
4. Some flags map to `common_params` fields after parsing

### Migration Path (Removing Cosmopolitan Dependency)

To migrate away from Cosmopolitan's FLAG system:

1. **Replace FLAG_* with common_params fields**: Most flags have equivalents in `common_params`. For example:
   - `FLAG_log_disable` → use `common_log_pause()`
   - `FLAG_nologo` → use `params.nologo`
   - `FLAG_mmproj` → use `params.mmproj.path`

2. **Remove cosmo.h include**: Replace with standard headers:
   ```cpp
   // Before
   #include <cosmo.h>

   // After
   #include <stdlib.h>
   #include <string.h>
   // etc.
   ```

3. **Replace unassert() with assert()**: `unassert()` is a Cosmopolitan macro that can be replaced with standard `assert()`.

4. **Replace llamafile_has()**: Use standard argument parsing or add arguments to `common_params`.

5. **Replace Cosmopolitan-specific headers**: Some headers like `<__fwd/string.h>` are Cosmopolitan-specific forward declaration headers. Replace with standard includes:
   ```cpp
   // Before
   #include <__fwd/string.h>

   // After
   #include <string>
   ```

6. **Replace llamafile_describe_cpu()**: Use CPUID directly or a portable CPU detection library.

7. **Replace llamafile_has_gpu(), llamafile_has_metal()**: These check for GPU availability at runtime. Use llama.cpp's backend system or check `llama_n_gpu_layers()`.

### Example Migration

```cpp
// Before (Cosmopolitan)
#include <cosmo.h>
if (!llamafile_has(argv, "--verbose"))
    FLAG_log_disable = true;

// After (Standard)
bool verbose = false;
for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--verbose") == 0) {
        verbose = true;
        break;
    }
}
if (!verbose) {
    common_log_pause(common_log_main());
}
```

## Future Work

1. Implement full mtmd (multimodal) support for image processing
2. Add background server integration
3. Add embeddings support
4. Test with various model architectures
