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

#### Verbose mode

When debugging, the `--verbose` argument is particularly useful as it adds
more verbose logging.


#### Where can I find GGUF model weights files?

Look for available gguf files in `~/llamafiles/`. Depending on the kind of
test, prefer:

- `gpt-oss-20b-MXFP4.gguf` for agentic tests
- `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` for multimodal tests
(also look for corresponding `mmproj` projector weights or ask for them)
- `Qwen3-0.6B-Q8_0.gguf` for any other tests



### Run All Unit Tests

Run `llamafile:check` to run all unit tests from the test suite.

### Run Integration Tests

```sh
./tests/integration/run_tests.sh --executable model_name.llamafile
```

- executable can be a pre-bundled llamafile or just the server executable
- if running the server executable, `--model` (and `--mmproj` for multimodal models) can be specified too
- different tests are run to verify the model/server capabilities
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
