#!/bin/bash
# Run llamafile integration tests
#
# Usage:
#   # With direct build
#   ./run_tests.sh --executable ./o/llamafile/llamafile --model /path/to/model.gguf
#
#   # With pre-built llamafile
#   ./run_tests.sh --executable ./Qwen-QwQ.llamafile
#
#   # Run specific test categories
#   ./run_tests.sh --executable ./model.llamafile -m "cli"
#   ./run_tests.sh --executable ./model.llamafile -m "server"
#   ./run_tests.sh --executable ./model.llamafile -m "multimodal"
#
#   # Skip slow tests
#   ./run_tests.sh --executable ./model.llamafile -m "not slow"
#
#   # Show model outputs (debug level logging)
#   ./run_tests.sh --executable ./model.llamafile --log-cli-level=DEBUG

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run pytest using uv
exec uv run pytest tests/ "$@"
