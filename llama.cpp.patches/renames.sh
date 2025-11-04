#!/usr/bin/env bash
set -euo pipefail

mv common/base64.hpp base64.h
mv common/common.cpp common.cpp
mv common/common.h common.h
mv common/console.cpp console.cpp
mv common/console.h console.h
mv common/grammar-parser.cpp grammar-parser.cpp
mv common/grammar-parser.h grammar-parser.h
mv common/json-schema-to-grammar.cpp json-schema-to-grammar.cpp
mv common/json-schema-to-grammar.h json-schema-to-grammar.h
mv common/json.hpp json.h
mv common/log.h log.h
mv common/ngram-cache.cpp ngram-cache.cpp
mv common/ngram-cache.h ngram-cache.h
mv common/sampling.cpp sampling.cpp
mv common/sampling.h sampling.h

mkdir -p imatrix
mv examples/imatrix/imatrix.cpp imatrix/imatrix.cpp

mkdir -p llama-bench
mv examples/main/README.md llama-bench/README.md
mv examples/llama-bench/llama-bench.cpp llama-bench/llama-bench.cpp

mkdir -p main
mv examples/main/main.cpp main/main.cpp

mkdir -p perplexity
mv examples/perplexity/perplexity.cpp perplexity/perplexity.cpp

mkdir -p quantize
mv examples/quantize/quantize.cpp quantize/quantize.cpp

mkdir -p llava
mv examples/llava/clip.cpp llava/clip.cpp
mv examples/llava/clip.h llava/clip.h
mv examples/llava/convert_image_encoder_to_gguf.py llava/convert-image-encoder-to-gguf.py
mv examples/llava/llava-cli.cpp llava/llava-cli.cpp
mv examples/llava/llava_surgery.py llava/llava-surgery.py
mv examples/llava/llava.cpp llava/llava.cpp
mv examples/llava/llava.h llava/llava.h

mkdir -p server/public server/themes
mv examples/server/chat-llama2.sh server/chat-llama2.sh
mv examples/server/chat.mjs server/chat.mjs
mv examples/server/chat.sh server/chat.sh
mv examples/server/deps.sh server/deps.sh
mv examples/server/httplib.h server/httplib.h
mv examples/server/public/completion.js server/public/completion.js
mv examples/server/public/index.js server/public/index.js
mv examples/server/public/json-schema-to-grammar.mjs server/public/json-schema-to-grammar.mjs
mv examples/server/themes/buttons-top/index.html server/public/index.html
mv examples/server/server.cpp server/server.cpp
mv examples/server/utils.hpp server/utils.h

mv examples/llama.android/README.md server/public/history-template.txt
mv examples/llama.android/llama/consumer-rules.pro server/public/prompt-template.txt

mv ggml/src/ggml-aarch64.c ggml-aarch64.c
mv ggml/src/ggml-aarch64.h ggml-aarch64.h
mv ggml/src/ggml-alloc.c ggml-alloc.c
mv ggml/include/ggml-alloc.h ggml-alloc.h
mv ggml/src/ggml-backend-impl.h ggml-backend-impl.h
mv ggml/src/ggml-backend.c ggml-backend.c
mv ggml/include/ggml-backend.h ggml-backend.h
mv ggml/src/ggml-common.h ggml-common.h
mv ggml/src/ggml-cuda.cu ggml-cuda.cu
mv ggml/include/ggml-cuda.h ggml-cuda.h
mv ggml/src/ggml-impl.h ggml-impl.h
mv ggml/include/ggml-metal.h ggml-metal.h
mv ggml/src/ggml-metal.m ggml-metal.m
mv ggml/src/ggml-metal.metal ggml-metal.metal
mv ggml/src/ggml-quants.h ggml-quants.h
mv ggml/src/ggml-quants.c ggml-quants.inc
mv ggml/src/ggml.c ggml.c
mv ggml/include/ggml.h ggml.h

mv src/llama-grammar.cpp llama-grammar.cpp
mv src/llama-grammar.h llama-grammar.h
mv src/llama-impl.h llama-impl.h
mv src/llama-sampling.cpp llama-sampling.cpp
mv src/llama-sampling.h llama-sampling.h
mv src/llama-vocab.cpp llama-vocab.cpp
mv src/llama-vocab.h llama-vocab.h
mv src/llama.cpp llama.cpp
mv include/llama.h llama.h
mv src/unicode-data.cpp unicode-data.cpp
mv src/unicode-data.h unicode-data.h
mv src/unicode.cpp unicode.cpp
mv src/unicode.h unicode.h

echo "Renames completed. Review with: git status"
