#!/bin/sh
# Script to package LLaVA models into a single llamafile
# This demonstrates the new multi-GGUF support

set -e

usage() {
    cat << EOF
Usage: $0 <language_model.gguf> <vision_encoder.gguf> <output.llamafile>

Package a LLaVA model (language model + vision encoder) into a single llamafile.

Arguments:
  language_model.gguf   Path to the language model GGUF file (e.g., llava-v1.5-7b-q4.gguf)
  vision_encoder.gguf   Path to the vision encoder GGUF file (e.g., mmproj-model-f16.gguf)
  output.llamafile      Output llamafile name

Example:
  $0 llava-v1.5-7b-q4.gguf mmproj-model-f16.gguf llava-v1.5-7b.llamafile

The resulting llamafile can be used without specifying --mmproj:
  ./llava-v1.5-7b.llamafile --image photo.jpg -p "What's in this image?"
EOF
    exit 1
}

if [ $# -ne 3 ]; then
    usage
fi

LANGUAGE_MODEL="$1"
VISION_ENCODER="$2"
OUTPUT_FILE="$3"

# Check if input files exist
if [ ! -f "$LANGUAGE_MODEL" ]; then
    echo "Error: Language model file not found: $LANGUAGE_MODEL"
    exit 1
fi

if [ ! -f "$VISION_ENCODER" ]; then
    echo "Error: Vision encoder file not found: $VISION_ENCODER"
    exit 1
fi

# Check if zipalign exists
if ! command -v zipalign >/dev/null 2>&1; then
    echo "Error: zipalign not found. Please build it first."
    echo "Run: make -j8"
    exit 1
fi

# Check if llamafile binary exists
if [ ! -f "o/$(uname -m)/bin/llamafile" ]; then
    echo "Error: llamafile binary not found. Please build it first."
    echo "Run: make -j8"
    exit 1
fi

echo "Packaging LLaVA model..."
echo "  Language model: $LANGUAGE_MODEL"
echo "  Vision encoder: $VISION_ENCODER"
echo "  Output file: $OUTPUT_FILE"

# Copy the llamafile binary
cp "o/$(uname -m)/bin/llamafile" "$OUTPUT_FILE"

# Use zipalign to add both GGUF files
echo "Adding GGUF files to llamafile..."
./o/$(uname -m)/bin/zipalign -j0 "$OUTPUT_FILE" \
    "$LANGUAGE_MODEL" \
    "$VISION_ENCODER"

# Make it executable
chmod +x "$OUTPUT_FILE"

echo ""
echo "Successfully created $OUTPUT_FILE!"
echo ""
echo "You can now use it without specifying --mmproj:"
echo "  ./$OUTPUT_FILE --image photo.jpg -p \"What's in this image?\""
echo ""
echo "The vision encoder will be auto-detected from the embedded files."