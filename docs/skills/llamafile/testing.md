# Testing Llamafile

Guide to running and writing tests.

## Running Tests

### Manually testing the executable

#### TUI mode

Run a newly compiled llamafile executable this way:

```sh
./o/llamafile/llamafile --model gguf_model.gguf
```

where `gguf_model.gguf` is a file holding a model's weights in GGUF format. For
instance:

```sh
./o/llamafile/llamafile --model ~/llamafiles/gpt-oss-20b-MXFP4.gguf
```

#### Server mode

Run a newly compiled llamafile executable this way:

```sh
./o/llamafile/llamafile --model gguf_model.gguf --server
```

#### CLI mode (single prompt, non-interactive)

For a scriptable one-shot generation (e.g. smoke tests), use `--cli`:

```sh
./o/llamafile/llamafile --cli -m gguf_model.gguf -p "What is the capital of France?"
```

The wrapper's flags differ from raw llama.cpp — modes are `--cli`, `--server`,
`--chat`, and the default combined TUI+server. `--cli` requires `-p`; `--gpu`
takes `auto|apple|amd|nvidia|disable`. When unsure, run `--help` (and
`--cli --help` etc.) rather than guessing flags.

#### Verbose mode

When debugging, the `--verbose` argument is particularly useful as it adds
more verbose logging.


#### GPU/Metal: stale per-version runtime cache

llamafile compiles GPU backends at runtime and caches them **keyed by the
version string** under `~/.llamafile/v/<VERSION>/` (on macOS this holds the
extracted `ggml-metal.*` sources and the compiled `ggml-metal.dylib`). The key
is the version alone — not the binary's contents or mtime.

Consequence: if you rebuild after changing GPU/Metal code (e.g. a llama.cpp
bump) **without bumping the llamafile version**, the new binary finds the
existing `v/<VERSION>/` dir and reuses the **stale** dylib instead of
recompiling. That ABI/code mismatch looks exactly like a regression — the
server never reaches `/health`, GPU CLI runs fail — while a `--gpu disable`
(CPU) run works fine, masking the cause.

Before validating GPU/Metal on a dev machine, force a fresh compile:

- bump the llamafile version (preferred — a release does this anyway), or
- `rm -rf ~/.llamafile/v/<VERSION>` to drop the stale cache.

Corollary: a CPU-only smoke test (`--gpu disable`) does **not** exercise this
path. Always run at least one default/GPU invocation when verifying a build on
GPU-capable hardware.

#### Validating a prebuilt GPU dylib — verify by observation, never infer

A clean *compile* does not prove the dylib loads or runs on the GPU. Confirm it
with observed runtime behavior, and never conclude "it won't use the GPU" from
indirect signals (an `ldconfig` grep, an ICD filename) — enumerate/observe the
real thing.

1. **Deploy** the dylib where llamafile loads it: `~/.llamafile/v/<VERSION>/`
   (keyed by `version.h`; the same dir as the stale-cache note above), e.g.
   `cp ggml-vulkan.so ~/.llamafile/v/0.10.5/`.
2. **Force the backend** so you test the dylib you mean to (not a fallback):
   `--gpu cuda` / `--gpu vulkan` / `--gpu amd`. Run with `-ngl 99 --verbose`.
3. **Read the device line** — this is the proof the backend loaded and bound the
   GPU. For Vulkan: `ggml_vulkan: Found 1 Vulkan devices:` /
   `0 = NVIDIA L40S (...) | fp16: 1 | bf16: 0 | int dot: 0 | matrix cores: KHR_coopmat`
   / `register_device: registered device Vulkan0` / `load_tensors: layer N
   assigned to device Vulkan0`. That capability line also tells you which speed
   paths are live (`bf16: 0` / `int dot: 0` / `KHR_coopmat` vs coopmat2 = the
   build's glslc lacked them — see building.md).
4. **Cross-check VRAM**: `nvidia-smi --query-compute-apps=process_name,used_memory
   --format=csv,noheader` should show the llamafile/`.ape` process holding memory.
5. **Confirm output**: one chat/completion returns sane text.
6. To check whether a device is even visible to a backend, run the real probe —
   `vulkaninfo --summary` (Vulkan), `nvidia-smi` (CUDA) — not a library-file grep.

Gotcha: the llamafile process's `comm` is `.ape-*`, not `llamafile`, so
`pkill -x llamafile` misses it — kill by PID (from `nvidia-smi
--query-compute-apps=pid`), and never `pkill -f <model-or-cmd-substring>` from a
shell whose own command line contains that substring (it matches itself).

#### Where can I find GGUF model weights files?

Look for available gguf files in `~/ggufs/`. If you don't find any or the
directory is not present, ask the user where you can find them.


### Run All Unit Tests

Run `llamafile:check` to run all unit tests from the test suite.

### Run Integration Tests

```sh
# pre-bundled llamafile
./tests/integration/run_tests.sh --executable model_name.llamafile

# direct build: executable + model (+ mmproj for multimodal)
./tests/integration/run_tests.sh \
    --executable ./o/llamafile/llamafile \
    --model ~/path/to/model.gguf \
    --mmproj ~/path/to/mmproj.gguf
```

- executable can be a pre-bundled llamafile or just the server executable
- if running the server executable, `--model` (and `--mmproj` for multimodal models) can be specified too
- different tests are run to verify the model/server capabilities
- select categories with `-m` (markers: `cli`, `tui`, `server`, `combined`,
  `multimodal`, `tool_calling`, `thinking`, `gpu`, `cpu`)
- **reasoning models** (e.g. Qwen3.x) need `-m "not thinking"` — otherwise the
  thinking-block assertions derail unrelated tests. Such a model run this way
  is expected to pass the whole suite.
- the suite drives the **default GPU path** by default, so it also exercises
  the Metal/GPU cache caveat above — see "stale per-version cache"
- more information and a user manual are available in `tests/integration/README.md`

### Run Specific Test

Tests are defined as `.runs` targets in BUILD.mk:

```sh
.cosmocc/4.0.2/bin/make o/$(MODE)/llamafile/json_test.runs  # run a specific test target
```

Replace `$(MODE)` with the actual mode (e.g., `opt`, `dbg`).

## Test System Overview

### Test Pattern

Tests in llamafile use the `.runs` suffix convention:

```makefile
# In build/rules.mk
%.runs: %
    $<
    @touch $@

# In tests/BUILD.mk
.PHONY: o/$(MODE)/tests
o/$(MODE)/tests: \
    o/$(MODE)/tests/extract_data_uris_test.runs 
```

The `.runs` file is a timestamp marker indicating the test passed. The build system:
1. Compiles the test binary
2. Executes it
3. Creates `.runs` file if successful

### Test Dependencies

Tests should be run when:
- Their source changes
- Dependencies change
- `.runs` file is missing

The `llamafile:check` command depends on all `.runs` files, ensuring all tests run.

## Test Locations

### Submodule Tests

Each submodule may have its own tests:

```
llama.cpp/
└── tests/            # llama.cpp test suite

whisper.cpp/
└── tests/            # whisper.cpp tests
```

These tests are currently not run (as they are assumed valid when pulling from
an approved commit), but future plans include introducing them to verify the
cosmo build has the same behavior as the native one.

### llamafile Tests

These tests are saved in:

```
tests/
└── sgemm
     └── *_test.c     # Optimized CPU kernels tests
...
```

## Writing Tests

### Basic Test Structure

```c
// myfeature_test.c
#include "myfeature.h"
#include <assert.h>
#include <stdio.h>

void test_basic_functionality(void) {
    // Arrange
    int input = 42;

    // Act
    int result = my_function(input);

    // Assert
    assert(result == expected_value);
}

void test_edge_case(void) {
    assert(my_function(0) == 0);
    assert(my_function(-1) == handle_negative());
}

int main(void) {
    test_basic_functionality();
    test_edge_case();
    printf("All tests passed!\n");
    return 0;
}
```

### Adding to BUILD.mk

- Tests for a new feature are usually added in a separate directory under `tests`.

- Each directory holds a `BUILD.mk` file for specific dependencies and local tests
building.

- The `tests/BUILD.mk` file includes build files from each subdirectory and adds
phony targets for them. Refer to the current version of this file for an example.

- Test files which are manual (i.e. not unit or integration tests, that are used
as exemplifications of issues or performance comparisons) are added to the build
files of their respective directories. They are not added as `.runs` targets to 
the `tests/BUILD.mk` file, thus they need to be manually compiled and run.


## Debugging Failed Tests

### Running Single Test Manually

```sh
# Build a specific test
.cosmocc/4.0.2/bin/make o//tests/extract_data_uris_test

# Run directly
./o/tests/extract_data_uris_test
```

### Debug Build

For debugging, use debug mode:

```sh
.cosmocc/4.0.2/bin/make MODE=dbg o/dbg/llamafile/json_test
```

Debug builds include:
- Debug symbols
- Assertions enabled
- No optimization

### Verbose Output

Add printf/fprintf statements for debugging:

```c
#ifdef DEBUG
    fprintf(stderr, "Debug: value = %d\n", value);
#endif
```

## Test Categories

### Unit Tests

Test individual functions/modules, e.g.:
- JSON parsing
- String utilities
- Data structures

### Integration Tests

Test component interactions, e.g.:
- Server endpoints
- Model loading
- API responses

### Performance Tests

Benchmark critical paths:
- Inference speed
- Memory usage
- Startup time


## Continuous Integration

Tests should run automatically on:
- Pull requests
- Commits to main branches

### Local CI Simulation

Before pushing, run full test suite:

```sh
make reset-repo
make setup
# llamafile:clean
# llamafile:build
# llamafile:check
```

## Test Coverage

### Identifying Untested Code

Review critical paths:
- Error handling
- Edge cases
- Platform-specific code

### Adding Coverage

When adding features:
1. Write tests for happy path
2. Write tests for error cases
3. Write tests for edge cases
4. Update BUILD.mk

### Priority Areas

Focus testing on:
- Public API functions
- Security-sensitive code
- Complex algorithms
- Cross-platform behavior
