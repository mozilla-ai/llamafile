#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMA_CPP

# ==============================================================================
# Version information
# ==============================================================================
# GGML_VERSION and GGML_COMMIT are inherited from build/config.mk

LLAMA_VERSION := $(shell cd llama.cpp 2>/dev/null && git describe --tags --always 2>/dev/null || echo "unknown")
LLAMA_COMMIT := $(shell cd llama.cpp 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# ==============================================================================
# GGML Library (Core tensor operations)
# ==============================================================================

GGML_SRCS_C := \
	llama.cpp/ggml/src/ggml-alloc.c \
	llama.cpp/ggml/src/ggml-quants.c \
	llama.cpp/ggml/src/ggml.c \
	llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c \
	llama.cpp/ggml/src/ggml-cpu/quants.c

GGML_SRCS_CPP := \
	llama.cpp/ggml/src/ggml-backend-dl.cpp \
	llama.cpp/ggml/src/ggml-backend-meta.cpp \
	llama.cpp/ggml/src/ggml-backend-reg.cpp \
	llama.cpp/ggml/src/ggml-backend.cpp \
	llama.cpp/ggml/src/ggml-opt.cpp \
	llama.cpp/ggml/src/ggml-threading.cpp \
	llama.cpp/ggml/src/ggml.cpp \
	llama.cpp/ggml/src/gguf.cpp \
	llama.cpp/ggml/src/ggml-cpu/binary-ops.cpp \
	llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp \
	llama.cpp/ggml/src/ggml-cpu/hbm.cpp \
	llama.cpp/ggml/src/ggml-cpu/ops.cpp \
	llama.cpp/ggml/src/ggml-cpu/repack.cpp \
	llama.cpp/ggml/src/ggml-cpu/traits.cpp \
	llama.cpp/ggml/src/ggml-cpu/unary-ops.cpp \
	llama.cpp/ggml/src/ggml-cpu/vec.cpp \
	llama.cpp/ggml/src/ggml-cpu/amx/amx.cpp \
	llama.cpp/ggml/src/ggml-cpu/amx/mmq.cpp

GGML_OBJS := \
	$(GGML_SRCS_C:%.c=o/$(MODE)/%.c.o) \
	$(GGML_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)

# ==============================================================================
# LLAMA Library (LLM inference)
# ==============================================================================

LLAMA_SRCS_CPP := \
	llama.cpp/src/llama.cpp \
	llama.cpp/src/models/afmoe.cpp \
	llama.cpp/src/models/apertus.cpp \
	llama.cpp/src/models/arcee.cpp \
	llama.cpp/src/models/arctic.cpp \
	llama.cpp/src/models/arwkv7.cpp \
	llama.cpp/src/models/baichuan.cpp \
	llama.cpp/src/models/bailingmoe.cpp \
	llama.cpp/src/models/bailingmoe2.cpp \
	llama.cpp/src/models/bert.cpp \
	llama.cpp/src/models/bitnet.cpp \
	llama.cpp/src/models/bloom.cpp \
	llama.cpp/src/models/chameleon.cpp \
	llama.cpp/src/models/chatglm.cpp \
	llama.cpp/src/models/codeshell.cpp \
	llama.cpp/src/models/cogvlm.cpp \
	llama.cpp/src/models/cohere2.cpp \
	llama.cpp/src/models/cohere2moe.cpp \
	llama.cpp/src/models/command-r.cpp \
	llama.cpp/src/models/dbrx.cpp \
	llama.cpp/src/models/deci.cpp \
	llama.cpp/src/models/deepseek.cpp \
	llama.cpp/src/models/deepseek2.cpp \
	llama.cpp/src/models/deepseek2ocr.cpp \
	llama.cpp/src/models/deepseek32.cpp \
	llama.cpp/src/models/deepseek4.cpp \
	llama.cpp/src/models/delta-net-base.cpp \
	llama.cpp/src/models/dflash.cpp \
	llama.cpp/src/models/dots1.cpp \
	llama.cpp/src/models/dream.cpp \
	llama.cpp/src/models/eagle3.cpp \
	llama.cpp/src/models/ernie4-5-moe.cpp \
	llama.cpp/src/models/ernie4-5.cpp \
	llama.cpp/src/models/eurobert.cpp \
	llama.cpp/src/models/exaone.cpp \
	llama.cpp/src/models/exaone4.cpp \
	llama.cpp/src/models/exaone-moe.cpp \
	llama.cpp/src/models/falcon-h1.cpp \
	llama.cpp/src/models/falcon.cpp \
	llama.cpp/src/models/gemma-embedding.cpp \
	llama.cpp/src/models/gemma.cpp \
	llama.cpp/src/models/gemma2.cpp \
	llama.cpp/src/models/gemma3.cpp \
	llama.cpp/src/models/gemma3n.cpp \
	llama.cpp/src/models/gemma4-assistant.cpp \
	llama.cpp/src/models/gemma4.cpp \
	llama.cpp/src/models/glm-dsa.cpp \
	llama.cpp/src/models/glm4-moe.cpp \
	llama.cpp/src/models/glm4.cpp \
	llama.cpp/src/models/gpt2.cpp \
	llama.cpp/src/models/gptneox.cpp \
	llama.cpp/src/models/granite-hybrid.cpp \
	llama.cpp/src/models/granite-moe.cpp \
	llama.cpp/src/models/granite.cpp \
	llama.cpp/src/models/mamba-base.cpp \
	llama.cpp/src/models/grok.cpp \
	llama.cpp/src/models/grovemoe.cpp \
	llama.cpp/src/models/hunyuan-dense.cpp \
	llama.cpp/src/models/hunyuan-moe.cpp \
	llama.cpp/src/models/hunyuan-vl.cpp \
	llama.cpp/src/models/hy-v3.cpp \
	llama.cpp/src/models/internlm2.cpp \
	llama.cpp/src/models/jais.cpp \
	llama.cpp/src/models/jais2.cpp \
	llama.cpp/src/models/jamba.cpp \
	llama.cpp/src/models/jina-bert-v2.cpp \
	llama.cpp/src/models/jina-bert-v3.cpp \
	llama.cpp/src/models/kimi-linear.cpp \
	llama.cpp/src/models/lfm2.cpp \
	llama.cpp/src/models/lfm2moe.cpp \
	llama.cpp/src/models/llada-moe.cpp \
	llama.cpp/src/models/llada.cpp \
	llama.cpp/src/models/llama-embed.cpp \
	llama.cpp/src/models/llama4.cpp \
	llama.cpp/src/models/llama.cpp \
	llama.cpp/src/models/maincoder.cpp \
	llama.cpp/src/models/mamba.cpp \
	llama.cpp/src/models/mamba2.cpp \
	llama.cpp/src/models/mellum.cpp \
	llama.cpp/src/models/mimo2.cpp \
	llama.cpp/src/models/minicpm.cpp \
	llama.cpp/src/models/minicpm3.cpp \
	llama.cpp/src/models/minimax-m2.cpp \
	llama.cpp/src/models/mistral3.cpp \
	llama.cpp/src/models/mistral4.cpp \
	llama.cpp/src/models/modern-bert.cpp \
	llama.cpp/src/models/mpt.cpp \
	llama.cpp/src/models/nemotron-h-moe.cpp \
	llama.cpp/src/models/nemotron-h.cpp \
	llama.cpp/src/models/nemotron.cpp \
	llama.cpp/src/models/neo-bert.cpp \
	llama.cpp/src/models/nomic-bert-moe.cpp \
	llama.cpp/src/models/nomic-bert.cpp \
	llama.cpp/src/models/olmo.cpp \
	llama.cpp/src/models/olmo2.cpp \
	llama.cpp/src/models/olmoe.cpp \
	llama.cpp/src/models/openai-moe.cpp \
	llama.cpp/src/models/openelm.cpp \
	llama.cpp/src/models/orion.cpp \
	llama.cpp/src/models/paddleocr.cpp \
	llama.cpp/src/models/pangu-embed.cpp \
	llama.cpp/src/models/phi2.cpp \
	llama.cpp/src/models/phi3.cpp \
	llama.cpp/src/models/phimoe.cpp \
	llama.cpp/src/models/plamo.cpp \
	llama.cpp/src/models/plamo2.cpp \
	llama.cpp/src/models/plamo3.cpp \
	llama.cpp/src/models/plm.cpp \
	llama.cpp/src/models/qwen.cpp \
	llama.cpp/src/models/qwen2.cpp \
	llama.cpp/src/models/qwen2moe.cpp \
	llama.cpp/src/models/qwen2vl.cpp \
	llama.cpp/src/models/qwen3.cpp \
	llama.cpp/src/models/qwen3moe.cpp \
	llama.cpp/src/models/qwen3next.cpp \
	llama.cpp/src/models/qwen35.cpp \
	llama.cpp/src/models/qwen35moe.cpp \
	llama.cpp/src/models/qwen3vl.cpp \
	llama.cpp/src/models/qwen3vlmoe.cpp \
	llama.cpp/src/models/refact.cpp \
	llama.cpp/src/models/rnd1.cpp \
	llama.cpp/src/models/rwkv6-base.cpp \
	llama.cpp/src/models/rwkv6.cpp \
	llama.cpp/src/models/rwkv6qwen2.cpp \
	llama.cpp/src/models/rwkv7-base.cpp \
	llama.cpp/src/models/rwkv7.cpp \
	llama.cpp/src/models/seed-oss.cpp \
	llama.cpp/src/models/smallthinker.cpp \
	llama.cpp/src/models/smollm3.cpp \
	llama.cpp/src/models/stablelm.cpp \
	llama.cpp/src/models/starcoder.cpp \
	llama.cpp/src/models/starcoder2.cpp \
	llama.cpp/src/models/step35.cpp \
	llama.cpp/src/models/t5.cpp \
	llama.cpp/src/models/t5encoder.cpp \
	llama.cpp/src/models/talkie.cpp \
	llama.cpp/src/models/wavtokenizer-dec.cpp \
	llama.cpp/src/models/xverse.cpp \
	llama.cpp/src/llama-adapter.cpp \
	llama.cpp/src/llama-arch.cpp \
	llama.cpp/src/llama-batch.cpp \
	llama.cpp/src/llama-chat.cpp \
	llama.cpp/src/llama-context.cpp \
	llama.cpp/src/llama-cparams.cpp \
	llama.cpp/src/llama-grammar.cpp \
	llama.cpp/src/llama-graph.cpp \
	llama.cpp/src/llama-hparams.cpp \
	llama.cpp/src/llama-impl.cpp \
	llama.cpp/src/llama-io.cpp \
	llama.cpp/src/llama-kv-cache-dsa.cpp \
	llama.cpp/src/llama-kv-cache-dsv4.cpp \
	llama.cpp/src/llama-kv-cache-iswa.cpp \
	llama.cpp/src/llama-kv-cache.cpp \
	llama.cpp/src/llama-memory-hybrid.cpp \
	llama.cpp/src/llama-memory-hybrid-iswa.cpp \
	llama.cpp/src/llama-memory-recurrent.cpp \
	llama.cpp/src/llama-memory.cpp \
	llama.cpp/src/llama-mmap.cpp \
	llama.cpp/src/llama-model-loader.cpp \
	llama.cpp/src/llama-model-saver.cpp \
	llama.cpp/src/llama-model.cpp \
	llama.cpp/src/llama-quant.cpp \
	llama.cpp/src/llama-sampler.cpp \
	llama.cpp/src/llama-vocab.cpp \
	llama.cpp/src/unicode-data.cpp \
	llama.cpp/src/unicode.cpp

LLAMA_OBJS := $(LLAMA_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)

# ==============================================================================
# Common Library (Utilities shared across tools)
# ==============================================================================

COMMON_SRCS_CPP := \
	llama.cpp/common/arg.cpp \
	llama.cpp/common/chat-auto-parser-generator.cpp \
	llama.cpp/common/chat-auto-parser-helpers.cpp \
	llama.cpp/common/chat-diff-analyzer.cpp \
	llama.cpp/common/chat-peg-parser.cpp \
	llama.cpp/common/chat.cpp \
	llama.cpp/common/common.cpp \
	llama.cpp/common/console.cpp \
	llama.cpp/common/debug.cpp \
	llama.cpp/common/download.cpp \
	llama.cpp/common/fit.cpp \
	llama.cpp/common/hf-cache.cpp \
	llama.cpp/common/imatrix-loader.cpp \
	llama.cpp/common/jinja/caps.cpp \
	llama.cpp/common/jinja/lexer.cpp \
	llama.cpp/common/jinja/parser.cpp \
	llama.cpp/common/jinja/runtime.cpp \
	llama.cpp/common/jinja/string.cpp \
	llama.cpp/common/jinja/value.cpp \
	llama.cpp/common/json-schema-to-grammar.cpp \
	llama.cpp/common/license.cpp \
	llama.cpp/common/llguidance.cpp \
	llama.cpp/common/log.cpp \
	llama.cpp/common/ngram-cache.cpp \
	llama.cpp/common/ngram-map.cpp \
	llama.cpp/common/ngram-mod.cpp \
	llama.cpp/common/peg-parser.cpp \
	llama.cpp/common/preset.cpp \
	llama.cpp/common/reasoning-budget.cpp \
	llama.cpp/common/sampling.cpp \
	llama.cpp/common/speculative.cpp \
	llama.cpp/common/unicode.cpp

# Build info generation
LLAMA_BUILD_NUMBER := $(shell date +%s)
LLAMA_BUILD_COMMIT := $(shell cd llama.cpp 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
LLAMA_BUILD_COMPILER := cosmocc
LLAMA_BUILD_TARGET := cosmopolitan

o/$(MODE)/llama.cpp/common/build-info.cpp: llama.cpp/common/build-info.cpp.in
	@mkdir -p $(dir $@)
	sed -e 's/@LLAMA_BUILD_NUMBER@/$(LLAMA_BUILD_NUMBER)/g' \
	    -e 's/@LLAMA_BUILD_COMMIT@/$(LLAMA_BUILD_COMMIT)/g' \
	    -e 's/@BUILD_COMPILER@/$(LLAMA_BUILD_COMPILER)/g' \
	    -e 's/@BUILD_TARGET@/$(LLAMA_BUILD_TARGET)/g' \
	    $< > $@

COMMON_SRCS_CPP += o/$(MODE)/llama.cpp/common/build-info.cpp

COMMON_OBJS := $(COMMON_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)

# build-info.cpp #includes "build-info.h" from llama.cpp/common; tests build the
# single-prefix object directly via the generic rule, so add the include path.
o/$(MODE)/llama.cpp/common/build-info.cpp.o: private CPPFLAGS += -iquote llama.cpp/common

# ==============================================================================
# Additional support files
# ==============================================================================

GGUF_SRCS := llama.cpp/examples/gguf/gguf.cpp
GGUF_OBJS := $(GGUF_SRCS:%.cpp=o/$(MODE)/%.cpp.o)

# ==============================================================================
# Combined library (just llama.cpp, equivalent to cmake build)
# ==============================================================================

LLAMA_CPP_OBJS := \
	$(GGML_OBJS) \
	$(LLAMA_OBJS) \
	$(COMMON_OBJS) \
	$(GGUF_OBJS)

o/$(MODE)/llama.cpp/llama.cpp.a: $(LLAMA_CPP_OBJS)

# ==============================================================================
# MTMD Library (Multimodal - for server)
# ==============================================================================

MTMD_SRCS_CPP := \
	llama.cpp/tools/mtmd/clip.cpp \
	llama.cpp/tools/mtmd/mtmd.cpp \
	llama.cpp/tools/mtmd/mtmd-helper.cpp \
	llama.cpp/tools/mtmd/mtmd-audio.cpp \
	llama.cpp/tools/mtmd/mtmd-image.cpp \
	llama.cpp/tools/mtmd/models/cogvlm.cpp \
	llama.cpp/tools/mtmd/models/deepseekocr.cpp \
	llama.cpp/tools/mtmd/models/deepseekocr2.cpp \
	llama.cpp/tools/mtmd/models/conformer.cpp \
	llama.cpp/tools/mtmd/models/dotsocr.cpp \
	llama.cpp/tools/mtmd/models/exaone4_5.cpp \
	llama.cpp/tools/mtmd/models/gemma4a.cpp \
	llama.cpp/tools/mtmd/models/gemma4ua.cpp \
	llama.cpp/tools/mtmd/models/gemma4uv.cpp \
	llama.cpp/tools/mtmd/models/gemma4v.cpp \
	llama.cpp/tools/mtmd/models/glm4v.cpp \
	llama.cpp/tools/mtmd/models/granite-speech.cpp \
	llama.cpp/tools/mtmd/models/granite4-vision.cpp \
	llama.cpp/tools/mtmd/models/hunyuanvl.cpp \
	llama.cpp/tools/mtmd/models/internvl.cpp \
	llama.cpp/tools/mtmd/models/kimik25.cpp \
	llama.cpp/tools/mtmd/models/kimivl.cpp \
	llama.cpp/tools/mtmd/models/llama4.cpp \
	llama.cpp/tools/mtmd/models/llava.cpp \
	llama.cpp/tools/mtmd/models/mimovl.cpp \
	llama.cpp/tools/mtmd/models/minicpmv.cpp \
	llama.cpp/tools/mtmd/models/mobilenetv5.cpp \
	llama.cpp/tools/mtmd/models/nemotron-v2-vl.cpp \
	llama.cpp/tools/mtmd/models/paddleocr.cpp \
	llama.cpp/tools/mtmd/models/pixtral.cpp \
	llama.cpp/tools/mtmd/models/qwen2vl.cpp \
	llama.cpp/tools/mtmd/models/qwen3a.cpp \
	llama.cpp/tools/mtmd/models/qwen3vl.cpp \
	llama.cpp/tools/mtmd/models/siglip.cpp \
	llama.cpp/tools/mtmd/models/step3vl.cpp \
	llama.cpp/tools/mtmd/models/whisper-enc.cpp \
	llama.cpp/tools/mtmd/models/yasa2.cpp \
	llama.cpp/tools/mtmd/models/youtuvl.cpp

MTMD_OBJS := $(MTMD_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)

# ==============================================================================
# cpp-httplib (HTTP library for server)
# ==============================================================================

HTTPLIB_SRCS := llama.cpp/vendor/cpp-httplib/httplib.cpp
HTTPLIB_OBJS := $(HTTPLIB_SRCS:%.cpp=o/$(MODE)/%.cpp.o)

# ==============================================================================
# Web UI assets
# ==============================================================================
#
# Upstream switched from prebuilt bundles in tools/server/public/ to a
# Svelte/PWA project under tools/ui/, embedded at CMake time by
# tools/ui/embed.cpp into a generated ui.cpp + ui.h. cosmocc has no JS
# toolchain, so apply-patches.sh (run by `make setup`) downloads the
# prebuilt site tarball (dist.tar.gz) from the ggml-org/llama-ui Hugging
# Face bucket and extracts the whole static site into
# llama.cpp/tools/ui/dist/ (see fetch-ui-assets.sh). embed.cpp then
# recursively embeds every file under that directory, keyed by its
# relative path (e.g. "_app/immutable/bundle.HASH.js"). fetch-ui-assets.sh
# also builds a dist/_gzip/ mirror of gzip-compressed files; embed.cpp
# auto-detects it and emits gzip-encoded assets (keeping the embedded
# payload small), which server-http.cpp serves with Content-Encoding: gzip.
# With assets present, ui.h defines LLAMA_UI_HAS_ASSETS and server-http.cpp
# registers a route per asset (index.html at /); without them, embed.cpp
# emits a no-op llama_ui_find_asset and the UI routes stay unregistered.

UI_DIST       := llama.cpp/tools/ui/dist
UI_GEN_DIR    := o/$(MODE)/llama.cpp/tools/ui
UI_EMBED_SRC  := llama.cpp/tools/ui/embed.cpp
UI_EMBED_TOOL := $(UI_GEN_DIR)/llama-ui-embed
UI_CPP_GEN    := $(UI_GEN_DIR)/ui.cpp
UI_H_GEN      := $(UI_GEN_DIR)/ui.h

# index.html exists iff fetch-ui-assets.sh successfully populated dist/. It
# serves both as the "do we have a UI?" gate and as the rebuild trigger: it is
# rewritten on every fetch and references the hashed bundle names, so it
# changes whenever the embedded assets do. wildcard returns "" when absent,
# letting the build proceed UI-less (offline / asset build not yet published).
UI_ASSETS_INDEX_HTML := $(wildcard $(UI_DIST)/index.html)

# Build embed.cpp standalone (no llamafile flags, no llama.cpp includes).
# cosmoc++ produces an APE that runs on the build host, so we don't need
# a separate system compiler. Compiled with stock C++17 (embed.cpp uses
# <filesystem>) so the source isn't entangled with -DCOSMOCC or GGML defines.
$(UI_EMBED_TOOL): $(UI_EMBED_SRC) $(COSMOCC)
	@mkdir -p $(@D)
	$(CXX) -O2 -std=gnu++17 -o $@ $<

# Generate ui.cpp/ui.h. Re-run when the embed tool or the fetched UI changes.
# When dist/ has assets, pass the directory: embed.cpp recurses it (auto-using
# dist/_gzip when present). When dist/ is empty, pass no directory so embed
# emits its no-asset stub.
$(UI_CPP_GEN) $(UI_H_GEN) &: $(UI_EMBED_TOOL) $(UI_ASSETS_INDEX_HTML)
	@mkdir -p $(UI_GEN_DIR)
	$(UI_EMBED_TOOL) $(UI_CPP_GEN) $(UI_H_GEN) \
		$(if $(UI_ASSETS_INDEX_HTML),$(UI_DIST))

# ==============================================================================
# Tools (in tools/ directory)
# ==============================================================================

# Tool source files
TOOL_QUANTIZE_SRCS := llama.cpp/tools/quantize/quantize.cpp
TOOL_IMATRIX_SRCS := llama.cpp/tools/imatrix/imatrix.cpp
TOOL_PERPLEXITY_SRCS := llama.cpp/tools/perplexity/perplexity.cpp
TOOL_BENCH_SRCS := llama.cpp/tools/llama-bench/llama-bench.cpp

TOOL_SERVER_SRCS := \
	llama.cpp/tools/server/server.cpp \
	llama.cpp/tools/server/server-chat.cpp \
	llama.cpp/tools/server/server-common.cpp \
	llama.cpp/tools/server/server-context.cpp \
	llama.cpp/tools/server/server-http.cpp \
	llama.cpp/tools/server/server-models.cpp \
	llama.cpp/tools/server/server-queue.cpp \
	llama.cpp/tools/server/server-schema.cpp \
	llama.cpp/tools/server/server-stream.cpp \
	llama.cpp/tools/server/server-task.cpp \
	llama.cpp/tools/server/server-tools.cpp

# Tool object files
TOOL_QUANTIZE_OBJS := $(TOOL_QUANTIZE_SRCS:%.cpp=o/$(MODE)/%.cpp.o)
TOOL_IMATRIX_OBJS := $(TOOL_IMATRIX_SRCS:%.cpp=o/$(MODE)/%.cpp.o)
TOOL_PERPLEXITY_OBJS := $(TOOL_PERPLEXITY_SRCS:%.cpp=o/$(MODE)/%.cpp.o)
TOOL_BENCH_OBJS := $(TOOL_BENCH_SRCS:%.cpp=o/$(MODE)/%.cpp.o)
TOOL_SERVER_OBJS := $(TOOL_SERVER_SRCS:%.cpp=o/$(MODE)/%.cpp.o)
# ui.cpp is generated by embed; placed under o/$(MODE)/ and built via the
# generic %.cpp -> %.cpp.o rule.
UI_GEN_OBJ := $(UI_CPP_GEN:%.cpp=%.cpp.o)
# llamafile objects are used to add dynamic GPU support (Metal, CUDA, ROCm, Vulkan)
TOOL_LLAMAFILE_OBJS := \
	o/$(MODE)/llamafile/llamafile.o \
	o/$(MODE)/llamafile/gpu.a \
	o/$(MODE)/llamafile/sandbox.o \
	o/$(MODE)/llamafile/zip.o

# Server objects depend on the llamafile bridge header and on the
# generated ui.h (server-http.cpp #includes it directly).
$(TOOL_SERVER_OBJS): llamafile/llamafile.h $(UI_H_GEN)

# ==============================================================================
# Compiler flags
# ==============================================================================

# Include paths for new llama.cpp structure
$(LLAMA_CPP_OBJS) $(TOOL_QUANTIZE_OBJS) $(TOOL_IMATRIX_OBJS) \
$(TOOL_PERPLEXITY_OBJS) $(TOOL_BENCH_OBJS) $(TOOL_SERVER_OBJS) $(MTMD_OBJS): \
	private CPPFLAGS += \
		-iquote llama.cpp/common \
		-iquote llama.cpp/include \
		-iquote llama.cpp/ggml/include \
		-iquote llama.cpp/ggml/src \
		-iquote llama.cpp/ggml/src/ggml-cpu \
		-iquote llama.cpp/src \
		-iquote llama.cpp/tools/mtmd \
		-iquote $(UI_GEN_DIR) \
		-isystem llama.cpp/vendor

# HTTPS support: build cpp-httplib with its Mbed TLS backend against the
# mbedtls vendored in third_party/mbedtls (the same TLS stack llamafile
# <= 0.9.3 used); third_party/mbedtls/include maps the canonical
# <mbedtls/*.h> include paths onto it. The macro changes httplib class
# layouts, so every object that includes httplib.h (directly or via
# common/http.h, server-http.h, server-cors-proxy.h) must see it.
$(LLAMA_CPP_OBJS) $(TOOL_QUANTIZE_OBJS) $(TOOL_IMATRIX_OBJS) \
$(TOOL_PERPLEXITY_OBJS) $(TOOL_BENCH_OBJS) $(TOOL_SERVER_OBJS) $(MTMD_OBJS) \
$(HTTPLIB_OBJS): \
	private CPPFLAGS += \
		-DCPPHTTPLIB_MBEDTLS_SUPPORT \
		-isystem third_party/mbedtls/include

# Server needs llamafile headers for Metal support.
# The generated ui.h sits in $(UI_GEN_DIR); the -iquote above lets every
# server source resolve `#include "ui.h"`.
$(TOOL_SERVER_OBJS): private CPPFLAGS += -iquote llamafile

# Compile the generated ui.cpp with the same llama.cpp CPPFLAGS so
# stddef.h and friends resolve via cosmocc's libc.
$(UI_GEN_OBJ): private CPPFLAGS += -iquote $(UI_GEN_DIR)
$(UI_GEN_OBJ): $(UI_H_GEN)

# Version definitions
$(GGML_OBJS): private CCFLAGS += \
	-DGGML_VERSION=\"$(GGML_VERSION)\" \
	-DGGML_COMMIT=\"$(GGML_COMMIT)\"

$(LLAMA_OBJS): private CCFLAGS += \
	-DLLAMA_VERSION=\"$(LLAMA_VERSION)\" \
	-DLLAMA_COMMIT=\"$(LLAMA_COMMIT)\"

# Base flags for all objects
$(LLAMA_CPP_OBJS) $(TOOL_SERVER_OBJS): private CCFLAGS += \
	-DCOSMOCC=1 \
	-DGGML_MULTIPLATFORM \
	-DGGML_USE_LLAMAFILE \
	-DGGML_USE_CPU \
	-DGGML_USE_CPU_REPACK \
	-DGGML_USE_OPENMP \
	-DGGML_CPU_GENERIC \
	-DGGML_SCHED_MAX_COPIES=4 \
	-fopenmp

# Common library needs httplib support
$(COMMON_OBJS): private CCFLAGS += -DLLAMA_USE_HTTPLIB

# Optimization flags for specific components
$(LLAMA_OBJS) $(COMMON_OBJS): private CCFLAGS += -DNDEBUG

# Memory management and backend - use default -O2 (backend is in hot path)
o/$(MODE)/llama.cpp/ggml/src/ggml-alloc.c.o \
o/$(MODE)/llama.cpp/ggml/src/ggml-backend.cpp.o: \
	private CCFLAGS += -mgcc

# Backend registration and utilities - can optimize for size
o/$(MODE)/llama.cpp/ggml/src/ggml-backend-reg.cpp.o \
o/$(MODE)/llama.cpp/common/arg.cpp.o \
o/$(MODE)/llama.cpp/common/log.cpp.o: \
	private CCFLAGS += -Os

# Unicode data - use gcc for better compatibility
o/$(MODE)/llama.cpp/src/unicode-data.cpp.o: \
	private CCFLAGS += -mgcc

# Core GGML and vector operations - optimize for performance
o/$(MODE)/llama.cpp/ggml/src/ggml.c.o \
o/$(MODE)/llama.cpp/ggml/src/ggml-cpu/vec.cpp.o \
o/$(MODE)/llama.cpp/ggml/src/ggml-cpu/ops.cpp.o \
o/$(MODE)/llama.cpp/ggml/src/ggml-cpu/binary-ops.cpp.o \
o/$(MODE)/llama.cpp/ggml/src/ggml-cpu/unary-ops.cpp.o: \
	private CCFLAGS += -O3 -mgcc

# Quantization - optimize for performance (critical hot path)
o/$(MODE)/llama.cpp/ggml/src/ggml-quants.c.o \
o/$(MODE)/llama.cpp/ggml/src/ggml-cpu/quants.c.o: \
	private CCFLAGS += -O3 -mgcc

# ==============================================================================
# Tool executables
# ==============================================================================

# Enable secondary expansion for prerequisites that reference variables defined
# in other BUILD.mk files (e.g., TINYBLAS_CPU_OBJS from llamafile/BUILD.mk).
# Without this, $(TINYBLAS_CPU_OBJS) would expand to empty since llamafile/BUILD.mk
# is included after this file.
.SECONDEXPANSION:

# All llama.cpp tools need pthread and OpenMP for threading
o/$(MODE)/llama.cpp/quantize/quantize \
o/$(MODE)/llama.cpp/imatrix/imatrix \
o/$(MODE)/llama.cpp/perplexity/perplexity \
o/$(MODE)/llama.cpp/llama-bench/llama-bench \
o/$(MODE)/llama.cpp/server/llama-server: \
	private LDFLAGS += -fopenmp
o/$(MODE)/llama.cpp/quantize/quantize \
o/$(MODE)/llama.cpp/imatrix/imatrix \
o/$(MODE)/llama.cpp/perplexity/perplexity \
o/$(MODE)/llama.cpp/llama-bench/llama-bench \
o/$(MODE)/llama.cpp/server/llama-server: \
	private LDLIBS += -lpthread

o/$(MODE)/llama.cpp/quantize/quantize: \
	$(TOOL_QUANTIZE_OBJS) \
	$$(TINYBLAS_CPU_OBJS) \
	o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llama.cpp/imatrix/imatrix: \
	$(TOOL_IMATRIX_OBJS) \
	$$(TINYBLAS_CPU_OBJS) \
	o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llama.cpp/perplexity/perplexity: \
	$(TOOL_PERPLEXITY_OBJS) \
	$$(TINYBLAS_CPU_OBJS) \
	o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llama.cpp/llama-bench/llama-bench: \
	$(TOOL_BENCH_OBJS) \
	$$(TINYBLAS_CPU_OBJS) \
	o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llama.cpp/server/llama-server: \
	$(TOOL_SERVER_OBJS) \
	$(UI_GEN_OBJ) \
	$(MTMD_OBJS) \
	$(HTTPLIB_OBJS) \
	$(TOOL_LLAMAFILE_OBJS) \
	$$(TINYBLAS_CPU_OBJS) \
	o/$(MODE)/llama.cpp/llama.cpp.a \
	o/$(MODE)/third_party/mbedtls/mbedtls.a
	@mkdir -p $(dir $@)
	$(LINK.o) $(TOOL_SERVER_OBJS) $(UI_GEN_OBJ) $(MTMD_OBJS) $(HTTPLIB_OBJS) $(TOOL_LLAMAFILE_OBJS) $(TINYBLAS_CPU_OBJS) o/$(MODE)/llama.cpp/llama.cpp.a o/$(MODE)/third_party/mbedtls/mbedtls.a $(LOADLIBES) $(LDLIBS) -o $@

# ==============================================================================
# Dependencies
# ==============================================================================

$(LLAMA_CPP_OBJS): llama.cpp/BUILD.mk
$(TOOL_QUANTIZE_OBJS) $(TOOL_IMATRIX_OBJS) \
$(TOOL_PERPLEXITY_OBJS) $(TOOL_BENCH_OBJS) $(TOOL_SERVER_OBJS): llama.cpp/BUILD.mk

# ==============================================================================
# Main target
# ==============================================================================

.PHONY: o/$(MODE)/llama.cpp
o/$(MODE)/llama.cpp: \
	o/$(MODE)/llama.cpp/llama.cpp.a \
	o/$(MODE)/llama.cpp/server/llama-server \
	o/$(MODE)/llama.cpp/quantize/quantize \
	o/$(MODE)/llama.cpp/imatrix/imatrix \
	o/$(MODE)/llama.cpp/perplexity/perplexity \
	o/$(MODE)/llama.cpp/llama-bench/llama-bench
