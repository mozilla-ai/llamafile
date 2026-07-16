# Llamafile Integration Tests

Integration tests for llamafile covering CLI, TUI, and server modes.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A llamafile executable or pre-built `.llamafile`

## Running Tests

### Single model

```bash
cd tests/integration

# Run all tests with a pre-built llamafile
./run_tests.sh --executable ~/path/to/model.llamafile

# Run all tests with a direct build
./run_tests.sh --executable ./o/llamafile/llamafile --model /path/to/model.gguf

# Run with verbose output
./run_tests.sh --executable ~/model.llamafile -v
```

### Full suite across all models

`tests/run_integration_tests.sh` runs the full suite across a configurable list
of models, then runs the `help` and `ssl` tests once against the bare executable.
Edit the `TESTS` array and path variables at the top of the script to match your
local model collection.

```bash
# From the repo root — runs every model in the TESTS list, then help + ssl
./tests/run_integration_tests.sh
```

> **Network access**: the script includes the `ssl online` tests, which download
> a small model from Hugging Face. Make sure you have internet access before
> running it, or remove the `ssl and online` invocation from the script.

## Test Categories

Use `-m` to select test categories:

| Marker | Description |
|--------|-------------|
| `cli` | CLI mode tests |
| `tui` | TUI/chat mode tests |
| `server` | Server mode tests |
| `combined` | Combined (TUI/chat + server) mode tests |
| `multimodal` | Vision/image tests (requires multimodal model) |
| `tool_calling` | Tool use tests (requires tool-capable model) |
| `thinking` | Thinking model tests (QwQ, DeepSeek-R1, etc.) |
| `gpu` | GPU acceleration tests |
| `cpu` | CPU-only tests |
| `help` | `--help` output tests; no model needed |
| `bare_executable` | Tests that run once against the bare binary (no embedded model); automatically skipped when the executable is a `.llamafile` bundle. Includes `help` and `ssl`. |
| `ssl` | HTTPS/TLS tests: serving with `--ssl-cert-file` and model download over HTTPS. Needs `openssl` in PATH. Run against the bare executable with `--model`. |
| `determinism` | Tests requiring reproducible output at temperature=0. Unreliable on gpt-oss models (see `tests/run_integration_tests.sh`); deselect with `-m "not determinism"`. |
| `online` | Tests that need network access (the HTTPS model download from Hugging Face). Skip with `-m "not online"` if offline. |

Examples:

```bash
# Run only CLI tests
./run_tests.sh --executable ~/model.llamafile -m cli

# Run server and TUI tests
./run_tests.sh --executable ~/model.llamafile -m "server or tui"

# Skip multimodal and tool_calling tests
./run_tests.sh --executable ~/model.llamafile -m "not multimodal and not tool_calling"

# Run the ssl tests (bare executable required; --model for the serving tests)
./run_tests.sh --executable ./o/llamafile/llamafile --model model.gguf -m "ssl and not online"
./run_tests.sh --executable ./o/llamafile/llamafile -m "ssl and online"
```

## Options

| Option | Description |
|--------|-------------|
| `--executable PATH` | Path to llamafile binary or `.llamafile` |
| `--model PATH` | Path to model file (for direct builds) |
| `--gpu MODE` | GPU mode: `auto`, `apple`, `amd`, `nvidia`, `disable` |
| `--timeout-multiplier N` | Multiply all timeouts by N (e.g., `2.0` for slower models) |
| `-v` | Verbose output |
| `-x` | Stop on first failure |

Example with timeout multiplier for large models:

```bash
./run_tests.sh --executable ~/large-model.llamafile --timeout-multiplier 3.0
```

## Viewing Model Outputs

Use `--log-cli-level` to see what the model returns:

```bash
# Show commands and exit codes
./run_tests.sh --executable ~/model.llamafile --log-cli-level=INFO

# Show full model outputs
./run_tests.sh --executable ~/model.llamafile --log-cli-level=DEBUG
```

## Test Structure

```
tests/
├── run_integration_tests.sh  # Full-suite runner across all models (edit to customise)
└── integration/
    ├── run_tests.sh          # Single-model test runner
    ├── conftest.py           # Pytest fixtures
    ├── pyproject.toml        # Dependencies and pytest config
    ├── utils/
    │   └── llamafile.py      # LlamafileRunner utility class and shared helpers
    ├── fixtures/
    │   └── test_image.png    # Test image for multimodal tests
    └── tests/
        ├── test_cli.py           # CLI mode tests
        ├── test_tui.py           # TUI/chat mode tests
        ├── test_server.py        # Server mode tests
        ├── test_combined.py      # TUI+Server simultaneous mode
        ├── test_multimodal.py    # Image description tests
        ├── test_tool_calling.py  # Tool use tests
        ├── test_gpu.py           # GPU/CPU execution tests
        ├── test_help.py          # --help output tests (bare executable, no model)
        ├── test_ssl.py           # HTTPS/TLS serving and model download tests
        └── test_sandbox.py       # pledge/SECCOMP sandboxing tests
```
