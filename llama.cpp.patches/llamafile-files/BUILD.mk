#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMA_CPP

# ==============================================================================
# Version information
# ==============================================================================

GGML_VERSION := 0.9.4
GGML_COMMIT := $(shell cd llama.cpp/ggml 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
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
	llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp \
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
	llama.cpp/src/models/cohere2-iswa.cpp \
	llama.cpp/src/models/command-r.cpp \
	llama.cpp/src/models/dbrx.cpp \
	llama.cpp/src/models/deci.cpp \
	llama.cpp/src/models/deepseek.cpp \
	llama.cpp/src/models/deepseek2.cpp \
	llama.cpp/src/models/dots1.cpp \
	llama.cpp/src/models/dream.cpp \
	llama.cpp/src/models/ernie4-5-moe.cpp \
	llama.cpp/src/models/ernie4-5.cpp \
	llama.cpp/src/models/exaone.cpp \
	llama.cpp/src/models/exaone4.cpp \
	llama.cpp/src/models/falcon-h1.cpp \
	llama.cpp/src/models/falcon.cpp \
	llama.cpp/src/models/gemma-embedding.cpp \
	llama.cpp/src/models/gemma.cpp \
	llama.cpp/src/models/gemma2-iswa.cpp \
	llama.cpp/src/models/gemma3-iswa.cpp \
	llama.cpp/src/models/gemma3n-iswa.cpp \
	llama.cpp/src/models/glm4-moe.cpp \
	llama.cpp/src/models/glm4.cpp \
	llama.cpp/src/models/gpt2.cpp \
	llama.cpp/src/models/gptneox.cpp \
	llama.cpp/src/models/granite-hybrid.cpp \
	llama.cpp/src/models/granite.cpp \
	llama.cpp/src/models/graph-context-mamba.cpp \
	llama.cpp/src/models/grok.cpp \
	llama.cpp/src/models/grovemoe.cpp \
	llama.cpp/src/models/hunyuan-dense.cpp \
	llama.cpp/src/models/hunyuan-moe.cpp \
	llama.cpp/src/models/internlm2.cpp \
	llama.cpp/src/models/jais.cpp \
	llama.cpp/src/models/jamba.cpp \
	llama.cpp/src/models/lfm2.cpp \
	llama.cpp/src/models/llada-moe.cpp \
	llama.cpp/src/models/llada.cpp \
	llama.cpp/src/models/llama-iswa.cpp \
	llama.cpp/src/models/llama.cpp \
	llama.cpp/src/models/mamba.cpp \
	llama.cpp/src/models/minicpm3.cpp \
	llama.cpp/src/models/minimax-m2.cpp \
	llama.cpp/src/models/mistral3.cpp \
	llama.cpp/src/models/mpt.cpp \
	llama.cpp/src/models/nemotron-h.cpp \
	llama.cpp/src/models/nemotron.cpp \
	llama.cpp/src/models/neo-bert.cpp \
	llama.cpp/src/models/olmo.cpp \
	llama.cpp/src/models/olmo2.cpp \
	llama.cpp/src/models/olmoe.cpp \
	llama.cpp/src/models/openai-moe-iswa.cpp \
	llama.cpp/src/models/openelm.cpp \
	llama.cpp/src/models/orion.cpp \
	llama.cpp/src/models/pangu-embedded.cpp \
	llama.cpp/src/models/phi2.cpp \
	llama.cpp/src/models/phi3.cpp \
	llama.cpp/src/models/plamo.cpp \
	llama.cpp/src/models/plamo2.cpp \
	llama.cpp/src/models/plm.cpp \
	llama.cpp/src/models/qwen.cpp \
	llama.cpp/src/models/qwen2.cpp \
	llama.cpp/src/models/qwen2moe.cpp \
	llama.cpp/src/models/qwen2vl.cpp \
	llama.cpp/src/models/qwen3.cpp \
	llama.cpp/src/models/qwen3moe.cpp \
	llama.cpp/src/models/qwen3next.cpp \
	llama.cpp/src/models/qwen3vl-moe.cpp \
	llama.cpp/src/models/qwen3vl.cpp \
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
	llama.cpp/src/models/t5-dec.cpp \
	llama.cpp/src/models/t5-enc.cpp \
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
	llama.cpp/src/llama-kv-cache-iswa.cpp \
	llama.cpp/src/llama-kv-cache.cpp \
	llama.cpp/src/llama-memory-hybrid.cpp \
	llama.cpp/src/llama-memory-recurrent.cpp \
	llama.cpp/src/llama-memory.cpp \
	llama.cpp/src/llama-mmap.cpp \
	llama.cpp/src/llama-model-loader.cpp \
	llama.cpp/src/llama-model-saver.cpp \
	llama.cpp/src/llama-model.cpp \
	llama.cpp/src/llama-quant.cpp \
	llama.cpp/src/llama-sampling.cpp \
	llama.cpp/src/llama-vocab.cpp \
	llama.cpp/src/unicode-data.cpp \
	llama.cpp/src/unicode.cpp

LLAMA_OBJS := $(LLAMA_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)

# ==============================================================================
# Common Library (Utilities shared across tools)
# ==============================================================================

COMMON_SRCS_CPP := \
	llama.cpp/common/arg.cpp \
	llama.cpp/common/chat-parser-xml-toolcall.cpp \
	llama.cpp/common/chat-parser.cpp \
	llama.cpp/common/chat-peg-parser.cpp \
	llama.cpp/common/chat.cpp \
	llama.cpp/common/common.cpp \
	llama.cpp/common/console.cpp \
	llama.cpp/common/download.cpp \
	llama.cpp/common/json-partial.cpp \
	llama.cpp/common/json-schema-to-grammar.cpp \
	llama.cpp/common/llguidance.cpp \
	llama.cpp/common/log.cpp \
	llama.cpp/common/ngram-cache.cpp \
	llama.cpp/common/peg-parser.cpp \
	llama.cpp/common/regex-partial.cpp \
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
	llama.cpp/tools/mtmd/mtmd-audio.cpp

MTMD_OBJS := $(MTMD_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)

# ==============================================================================
# cpp-httplib (HTTP library for server)
# ==============================================================================

HTTPLIB_SRCS := llama.cpp/vendor/cpp-httplib/httplib.cpp
HTTPLIB_OBJS := $(HTTPLIB_SRCS:%.cpp=o/$(MODE)/%.cpp.o)

# ==============================================================================
# Server Assets (convert HTML to C++ headers)
# ==============================================================================

# Generate .hpp files from binary assets using xxd-like conversion
o/$(MODE)/llama.cpp/tools/server/%.hpp: llama.cpp/tools/server/public/%
	@mkdir -p $(dir $@)
	$(eval VARNAME := $(shell echo "$(notdir $*)" | sed 's/[.-]/_/g'))
	@echo 'unsigned char $(VARNAME)[] = {' > $@
	@od -An -tx1 -v $< | awk '{for(i=1;i<=NF;i++){if(NR>1||i>1)printf", "; printf"0x%s",$$i}}' >> $@
	@echo >> $@
	@echo '};' >> $@
	@echo 'unsigned int $(VARNAME)_len = sizeof($(VARNAME));' >> $@

SERVER_ASSETS := \
	o/$(MODE)/llama.cpp/tools/server/index.html.gz.hpp \
	o/$(MODE)/llama.cpp/tools/server/loading.html.hpp

# ==============================================================================
# Tools (in tools/ directory)
# ==============================================================================

# Tool source files
TOOL_MAIN_SRCS := llama.cpp/tools/main/main.cpp
TOOL_QUANTIZE_SRCS := llama.cpp/tools/quantize/quantize.cpp
TOOL_IMATRIX_SRCS := llama.cpp/tools/imatrix/imatrix.cpp
TOOL_PERPLEXITY_SRCS := llama.cpp/tools/perplexity/perplexity.cpp
TOOL_BENCH_SRCS := llama.cpp/tools/llama-bench/llama-bench.cpp

TOOL_SERVER_SRCS := \
	llama.cpp/tools/server/server.cpp \
	llama.cpp/tools/server/server-common.cpp \
	llama.cpp/tools/server/server-context.cpp \
	llama.cpp/tools/server/server-http.cpp \
	llama.cpp/tools/server/server-models.cpp \
	llama.cpp/tools/server/server-queue.cpp \
	llama.cpp/tools/server/server-task.cpp

# Tool object files
TOOL_MAIN_OBJS := $(TOOL_MAIN_SRCS:%.cpp=o/$(MODE)/%.cpp.o)
TOOL_QUANTIZE_OBJS := $(TOOL_QUANTIZE_SRCS:%.cpp=o/$(MODE)/%.cpp.o)
TOOL_IMATRIX_OBJS := $(TOOL_IMATRIX_SRCS:%.cpp=o/$(MODE)/%.cpp.o)
TOOL_PERPLEXITY_OBJS := $(TOOL_PERPLEXITY_SRCS:%.cpp=o/$(MODE)/%.cpp.o)
TOOL_BENCH_OBJS := $(TOOL_BENCH_SRCS:%.cpp=o/$(MODE)/%.cpp.o)
TOOL_SERVER_OBJS := $(TOOL_SERVER_SRCS:%.cpp=o/$(MODE)/%.cpp.o)
# llamafile objects are used to add dynamic metal support
TOOL_LLAMAFILE_OBJS := \
	o/$(MODE)/llamafile/llamafile.o \
	o/$(MODE)/llamafile/metal.o \
	o/$(MODE)/llamafile/zip.o

# Server objects depend on generated assets
$(TOOL_SERVER_OBJS): $(SERVER_ASSETS) llamafile/llamafile.h

# ==============================================================================
# Compiler flags
# ==============================================================================

# Include paths for new llama.cpp structure
$(LLAMA_CPP_OBJS) $(TOOL_MAIN_OBJS) $(TOOL_QUANTIZE_OBJS) $(TOOL_IMATRIX_OBJS) \
$(TOOL_PERPLEXITY_OBJS) $(TOOL_BENCH_OBJS) $(TOOL_SERVER_OBJS) $(MTMD_OBJS): \
	private CPPFLAGS += \
		-iquote llama.cpp/common \
		-iquote llama.cpp/include \
		-iquote llama.cpp/ggml/include \
		-iquote llama.cpp/ggml/src \
		-iquote llama.cpp/ggml/src/ggml-cpu \
		-iquote llama.cpp/src \
		-iquote llama.cpp/tools/mtmd \
		-iquote o/$(MODE)/llama.cpp/tools/server \
		-isystem llama.cpp/vendor

# Server needs llamafile headers for Metal support
$(TOOL_SERVER_OBJS): private CPPFLAGS += -iquote llamafile

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

# All llama.cpp tools need pthread and OpenMP for threading
o/$(MODE)/llama.cpp/main/main \
o/$(MODE)/llama.cpp/quantize/quantize \
o/$(MODE)/llama.cpp/imatrix/imatrix \
o/$(MODE)/llama.cpp/perplexity/perplexity \
o/$(MODE)/llama.cpp/llama-bench/llama-bench \
o/$(MODE)/llama.cpp/server/llama-server: \
	private LDFLAGS += -fopenmp
o/$(MODE)/llama.cpp/main/main \
o/$(MODE)/llama.cpp/quantize/quantize \
o/$(MODE)/llama.cpp/imatrix/imatrix \
o/$(MODE)/llama.cpp/perplexity/perplexity \
o/$(MODE)/llama.cpp/llama-bench/llama-bench \
o/$(MODE)/llama.cpp/server/llama-server: \
	private LDLIBS += -lpthread

o/$(MODE)/llama.cpp/main/main: \
	$(TOOL_MAIN_OBJS) \
	o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llama.cpp/quantize/quantize: \
	$(TOOL_QUANTIZE_OBJS) \
	o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llama.cpp/imatrix/imatrix: \
	$(TOOL_IMATRIX_OBJS) \
	o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llama.cpp/perplexity/perplexity: \
	$(TOOL_PERPLEXITY_OBJS) \
	o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llama.cpp/llama-bench/llama-bench: \
	$(TOOL_BENCH_OBJS) \
	o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/llama.cpp/server/llama-server: \
	$(TOOL_SERVER_OBJS) \
	$(MTMD_OBJS) \
	$(HTTPLIB_OBJS) \
	$(TOOL_LLAMAFILE_OBJS) \
	o/$(MODE)/llama.cpp/llama.cpp.a \
	$(SERVER_ASSETS)
	@mkdir -p $(dir $@)
	$(LINK.o) $(TOOL_SERVER_OBJS) $(MTMD_OBJS) $(HTTPLIB_OBJS) $(TOOL_LLAMAFILE_OBJS) o/$(MODE)/llama.cpp/llama.cpp.a $(LOADLIBES) $(LDLIBS) -o $@

# ==============================================================================
# Dependencies
# ==============================================================================

$(LLAMA_CPP_OBJS): llama.cpp/BUILD.mk
$(TOOL_MAIN_OBJS) $(TOOL_QUANTIZE_OBJS) $(TOOL_IMATRIX_OBJS) \
$(TOOL_PERPLEXITY_OBJS) $(TOOL_BENCH_OBJS) $(TOOL_SERVER_OBJS): llama.cpp/BUILD.mk

# ==============================================================================
# Main target
# ==============================================================================

.PHONY: o/$(MODE)/llama.cpp
o/$(MODE)/llama.cpp: \
	o/$(MODE)/llama.cpp/llama.cpp.a \
	o/$(MODE)/llama.cpp/main/main \
	o/$(MODE)/llama.cpp/server/llama-server \
	o/$(MODE)/llama.cpp/quantize/quantize \
	o/$(MODE)/llama.cpp/imatrix/imatrix \
	o/$(MODE)/llama.cpp/perplexity/perplexity \
	o/$(MODE)/llama.cpp/llama-bench/llama-bench
