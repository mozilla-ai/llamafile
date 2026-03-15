# Llamafile Integration Tests

Integration tests for llamafile covering CLI, TUI, and server modes.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A llamafile executable or pre-built `.llamafile`

## Running Tests

```bash
cd tests/integration

# Run all tests with a pre-built llamafile
./run_tests.sh --executable ~/path/to/model.llamafile

# Run all tests with a direct build
./run_tests.sh --executable ./o/llamafile/llamafile --model /path/to/model.gguf

# Run with verbose output
./run_tests.sh --executable ~/model.llamafile -v
```

## Test Categories

Use `-m` to select test categories:

| Marker | Description |
|--------|-------------|
| `cli` | CLI mode tests |
| `tui` | TUI/chat mode tests |
| `server` | Server mode tests |
| `server` | Combined (TUI/chat + server) mode tests |
| `multimodal` | Vision/image tests (requires multimodal model) |
| `tool_calling` | Tool use tests (requires tool-capable model) |
| `thinking` | Thinking model tests (QwQ, DeepSeek-R1, etc.) |
| `gpu` | GPU acceleration tests |
| `cpu` | CPU-only tests |

Examples:

```bash
# Run only CLI tests
./run_tests.sh --executable ~/model.llamafile -m cli

# Run server and TUI tests
./run_tests.sh --executable ~/model.llamafile -m "server or tui"

# Skip multimodal and tool_calling tests
./run_tests.sh --executable ~/model.llamafile -m "not multimodal and not tool_calling"
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
tests/integration/
├── run_tests.sh          # Test runner script
├── conftest.py           # Pytest fixtures
├── pyproject.toml        # Dependencies and pytest config
├── utils/
│   └── llamafile.py      # LlamafileRunner utility class
├── fixtures/
│   └── test_image.png    # Test image for multimodal tests
└── tests/
    ├── test_cli.py       # CLI mode tests
    ├── test_tui.py       # TUI mode tests
    ├── test_server.py    # Server mode tests
    ├── test_combined.py  # TUI+Server simultaneous mode
    ├── test_multimodal.py    # Image description tests
    ├── test_tool_calling.py  # Tool use tests
    └── test_gpu.py       # GPU/CPU execution tests
```
