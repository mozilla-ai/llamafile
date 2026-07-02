#!/usr/bin/env bash
# transcribefile_smoke.sh — regression smoke for the transcribefile APE.
#
# Three layers:
#   1. Model-free probes — always run. Catches CLI argv / WAV-loader /
#      help-text regressions without needing any model artifact.
#   2. Parakeet end-to-end — gated on TRANSCRIBEFILE_PARAKEET_GGUF. Skipped
#      (with a warning, not a failure) when the model isn't available, so
#      `make check` stays green on machines without the model.
#   3. Metal backend — needs layer 2's model plus macOS on Apple Silicon
#      with a registered Metal device; skipped elsewhere. Asserts that
#      --backend metal selects an MTL device and that the Metal and CPU
#      transcripts match exactly.
#
# Usage:
#   tests/transcribefile_smoke.sh <path-to-transcribefile-binary>
#
# Env:
#   TRANSCRIBEFILE_PARAKEET_GGUF  Path to a parakeet GGUF; if unset or the
#                                 file is missing, layers 2 and 3 are skipped.

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

echo "[smoke] layer 3 — metal backend"

# Metal is macOS/Apple-Silicon only. Elsewhere (and on macs where the
# runtime dylib build isn't possible, e.g. no Xcode CLT) the contract is
# graceful degradation to CPU, so absence of a metal device is a SKIP,
# not a failure.
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "  SKIP metal: requires macOS on Apple Silicon" >&2
    exit 0
fi
if ! "$APE" --list-devices 2>/dev/null | grep -q 'kind=metal'; then
    echo "  SKIP metal: no metal device registered" \
         "(Xcode command-line tools missing?)" >&2
    exit 0
fi

# Explicit --backend metal must actually select an MTL device and produce
# the same transcript TEXT as the CPU backend. Only the text: fields are
# compared: word timestamps and token probabilities may legitimately
# differ across backends — different kernels and accumulation orders give
# slightly different logits, so near-tie decisions (a timestamp on a
# frame boundary, a probability rounding) can flip, and quantized models
# amplify this. The decoded text is expected to be stable.
mtl=$("$APE" --backend metal -q -m "$MODEL" "$SAMPLE" 2>/dev/null) \
    || fail "metal run exited non-zero"
echo "$mtl" | grep -qE 'backend:[[:space:]]+MTL' || fail "metal run did not select an MTL device"
echo "$mtl" | grep -qi 'country' || fail "metal transcription missing 'country'"
pass "parakeet transcribes jfk.wav on metal"

cpu=$("$APE" --backend cpu -q -m "$MODEL" "$SAMPLE" 2>/dev/null) \
    || fail "cpu run exited non-zero"
mtl_text=$(echo "$mtl" | grep '^text:' || true)
cpu_text=$(echo "$cpu" | grep '^text:' || true)
[ -n "$mtl_text" ] || fail "metal output has no text: line"
if [ "$mtl_text" != "$cpu_text" ]; then
    printf 'metal:\n%s\ncpu:\n%s\n' "$mtl_text" "$cpu_text" >&2
    fail "metal and cpu transcript text differs"
fi
pass "metal and cpu transcripts match"
