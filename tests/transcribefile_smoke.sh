#!/usr/bin/env bash
# transcribefile_smoke.sh — regression smoke for the transcribefile APE.
#
# Two layers:
#   1. Model-free probes — always run. Catches CLI argv / WAV-loader /
#      help-text regressions without needing any model artifact.
#   2. Parakeet end-to-end — gated on TRANSCRIBEFILE_PARAKEET_GGUF. Skipped
#      (with a warning, not a failure) when the model isn't available, so
#      `make check` stays green on machines without the model.
#
# Usage:
#   tests/transcribefile_smoke.sh <path-to-transcribefile-binary>
#
# Env:
#   TRANSCRIBEFILE_PARAKEET_GGUF  Path to a parakeet GGUF; if unset or the
#                                 file is missing, layer 2 is skipped.

set -euo pipefail

APE="${1:-}"
if [ -z "$APE" ] || [ ! -x "$APE" ]; then
    echo "FAIL: argv[1] must point at an executable transcribefile binary" >&2
    echo "      got: '$APE'" >&2
    exit 2
fi

# Resolve repo root from this script's location so samples/ paths work
# no matter the cwd `make` invokes us from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SAMPLE="$REPO_ROOT/transcribe.cpp/samples/jfk.wav"

if [ ! -f "$SAMPLE" ]; then
    echo "FAIL: sample wav not found at $SAMPLE" >&2
    echo "      (is the transcribe.cpp submodule initialized?)" >&2
    exit 2
fi

pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1" >&2; exit 1; }

echo "[smoke] layer 1 — model-free probes"

# --help should exit 0 and print the usage banner.
out=$("$APE" --help 2>&1) || fail "--help exited non-zero"
echo "$out" | grep -q '^usage:' || fail "--help missing 'usage:' banner"
pass "--help prints usage banner"

# Running on a wav without -m should still exit 0 and print the
# parsed-WAV duration line. This mirrors upstream's transcribe_cli_smoke
# CTest: it exercises argv parsing + WAV loader without needing a model.
out=$("$APE" "$SAMPLE" 2>&1) || fail "wav-only run exited non-zero"
echo "$out" | grep -q 'duration:' || fail "wav-only run missing 'duration:' line"
echo "$out" | grep -qE 'duration:.*11\.0' || fail "duration should report ~11.0 s for jfk.wav"
pass "wav-only run reports duration (no model required)"

echo "[smoke] layer 2 — parakeet end-to-end"

MODEL="${TRANSCRIBEFILE_PARAKEET_GGUF:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "  SKIP parakeet end-to-end: TRANSCRIBEFILE_PARAKEET_GGUF" \
         "not set or file missing" >&2
    echo "       (set it to a parakeet-tdt GGUF to run this layer)" >&2
    exit 0
fi

# Real transcription run. Asserts:
#   - exit 0
#   - timings / realtime lines present (so we know it actually decoded)
#   - the JFK quote contains 'country' (sanity-check the decoder didn't
#     silently produce garbage)
out=$("$APE" -m "$MODEL" "$SAMPLE" 2>&1) || fail "parakeet run exited non-zero"
echo "$out" | grep -q 'realtime:' || fail "parakeet output missing 'realtime:' line"
echo "$out" | grep -qi 'country' || fail "parakeet transcription missing 'country'"
realtime=$(echo "$out" | grep -oE 'realtime:[[:space:]]+[0-9]+x' | head -1 || true)
pass "parakeet transcribes jfk.wav (${realtime:-realtime: ?})"
