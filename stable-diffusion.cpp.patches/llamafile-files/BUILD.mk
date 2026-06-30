#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘
#
# BUILD.mk for stable-diffusion.cpp (CPU-only build with cosmocc)
#
# This BUILD.mk is derived from stable-diffusion.cpp's CMake configuration.
# stable-diffusion.cpp uses llama.cpp's GGML (via the llamafile fork).
#

PKGS += STABLE_DIFFUSION_CPP

# ==============================================================================
# Version Information
# ==============================================================================

SD_VERSION := baf7eda
SD_GGML_VERSION := 0.9.5
SD_GGML_COMMIT := unknown

# ==============================================================================
# Include Paths (from CMakeLists.txt)
# ==============================================================================

SD_INCS := \
	-iquote stable-diffusion.cpp \
	-iquote stable-diffusion.cpp/src \
	-iquote stable-diffusion.cpp/include \
	-isystem stable-diffusion.cpp/thirdparty \
	-iquote stable-diffusion.cpp/examples \
	-iquote llama.cpp/ggml/include \
	-iquote llama.cpp/ggml/src \
	-iquote llama.cpp/include

# ==============================================================================
# Common Compiler Flags
# ==============================================================================

SD_CPPFLAGS := \
	-DGGML_MAX_NAME=128 \
	-DGGML_USE_CPU

# ==============================================================================
# GGML Base Library Sources (from llama.cpp/ggml)
# Note: We reuse $(GGML_OBJS) from llama.cpp/BUILD.mk instead of rebuilding
# ==============================================================================

# ==============================================================================
# SD Library Sources (from src/CMakeLists.txt)
# ==============================================================================

SD_CORE_SRCS_CPP := \
	stable-diffusion.cpp/src/stable-diffusion.cpp \
	stable-diffusion.cpp/src/ggml_extend_backend.cpp \
	stable-diffusion.cpp/src/ggml_graph_cut.cpp \
	stable-diffusion.cpp/src/guidance.cpp \
	stable-diffusion.cpp/src/model.cpp \
	stable-diffusion.cpp/src/name_conversion.cpp \
	stable-diffusion.cpp/src/sample-cache.cpp \
	stable-diffusion.cpp/src/upscaler.cpp \
	stable-diffusion.cpp/src/util.cpp \
	stable-diffusion.cpp/src/version.cpp \
	stable-diffusion.cpp/src/model_io/gguf_io.cpp \
	stable-diffusion.cpp/src/model_io/pickle_io.cpp \
	stable-diffusion.cpp/src/model_io/safetensors_io.cpp \
	stable-diffusion.cpp/src/model_io/torch_legacy_io.cpp \
	stable-diffusion.cpp/src/model_io/torch_zip_io.cpp \
	stable-diffusion.cpp/src/tokenizers/bpe_tokenizer.cpp \
	stable-diffusion.cpp/src/tokenizers/clip_tokenizer.cpp \
	stable-diffusion.cpp/src/tokenizers/gemma_tokenizer.cpp \
	stable-diffusion.cpp/src/tokenizers/mistral_tokenizer.cpp \
	stable-diffusion.cpp/src/tokenizers/qwen2_tokenizer.cpp \
	stable-diffusion.cpp/src/tokenizers/t5_unigram_tokenizer.cpp \
	stable-diffusion.cpp/src/tokenizers/tokenize_util.cpp \
	stable-diffusion.cpp/src/tokenizers/tokenizer.cpp \
	stable-diffusion.cpp/src/tokenizers/vocab/vocab.cpp

SD_CORE_OBJS := $(SD_CORE_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)

# ==============================================================================
# Common Library Sources (from examples/)
# ==============================================================================

SD_COMMON_SRCS_CPP := \
	stable-diffusion.cpp/examples/common/common.cpp \
	stable-diffusion.cpp/examples/common/log.cpp \
	stable-diffusion.cpp/examples/common/media_io.cpp

SD_COMMON_OBJS := $(SD_COMMON_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)

# ==============================================================================
# CLI Tool Sources
# ==============================================================================

SD_CLI_SRCS := stable-diffusion.cpp/examples/cli/main.cpp
SD_CLI_EXTRA_SRCS := \
	stable-diffusion.cpp/examples/cli/image_metadata.cpp \
	stable-diffusion.cpp/src/convert.cpp

SD_CLI_EXTRA_OBJS := $(SD_CLI_EXTRA_SRCS:%.cpp=o/$(MODE)/%.cpp.o)

# ==============================================================================
# Object Files
# ==============================================================================

# All library objects (excluding GGML — we link $(GGML_OBJS) instead)
STABLE_DIFFUSION_CPP_OBJS := \
	$(SD_CORE_OBJS) \
	$(SD_COMMON_OBJS) \
	$(SD_CLI_EXTRA_OBJS)

# ==============================================================================
# Static Library
# ==============================================================================

o/$(MODE)/stable-diffusion.cpp/stable-diffusion.cpp.a: $(STABLE_DIFFUSION_CPP_OBJS)

# ==============================================================================
# Compilation Rules - SD Core
# ==============================================================================

$(SD_CORE_OBJS): o/$(MODE)/%.cpp.o: %.cpp stable-diffusion.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(SD_INCS) $(SD_CPPFLAGS) -frtti -std=gnu++23 -o $@ $<

# ==============================================================================
# Compilation Rules - Common Library
# ==============================================================================

$(SD_COMMON_OBJS): o/$(MODE)/%.cpp.o: %.cpp stable-diffusion.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(SD_INCS) $(SD_CPPFLAGS) -frtti -std=gnu++23 -o $@ $<

# ==============================================================================
# Compilation Rules - CLI Tool
# ==============================================================================

$(SD_CLI_SRCS:%.cpp=o/$(MODE)/%.cpp.o): o/$(MODE)/%.cpp.o: %.cpp stable-diffusion.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(SD_INCS) $(SD_CPPFLAGS) -frtti -std=gnu++23 -o $@ $<

$(SD_CLI_EXTRA_SRCS:%.cpp=o/$(MODE)/%.cpp.o): o/$(MODE)/%.cpp.o: %.cpp stable-diffusion.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(SD_INCS) $(SD_CPPFLAGS) -frtti -std=gnu++23 -o $@ $<

# ==============================================================================
# cli.cpp for diffusionfile build (with DIFFUSIONFILE defined)
# ==============================================================================

o/$(MODE)/stable-diffusion.cpp/examples/cli/main.diffusionfile.cpp.o: \
		stable-diffusion.cpp/examples/cli/main.cpp stable-diffusion.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(SD_INCS) $(SD_CPPFLAGS) -DDIFFUSIONFILE -frtti -std=gnu++23 -o $@ $<

# ==============================================================================
# Math shim for cosmocc (llround missing from libc)
# ==============================================================================

o/$(MODE)/stable-diffusion.cpp.patches/sd_math_shim.o: stable-diffusion.cpp.patches/sd_math_shim.cpp stable-diffusion.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(SD_INCS) -frtti -std=gnu++23 -o $@ $<

# ==============================================================================
# Third-party zip (needed by torch_zip_io)
# ==============================================================================

o/$(MODE)/stable-diffusion.cpp/thirdparty/zip.c.o: \
		stable-diffusion.cpp/thirdparty/zip.c stable-diffusion.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.c) $(SD_INCS) -o $@ $<

# ==============================================================================
# Main Target
# ==============================================================================

.PHONY: o/$(MODE)/stable-diffusion.cpp
o/$(MODE)/stable-diffusion.cpp: \
	o/$(MODE)/stable-diffusion.cpp/stable-diffusion.cpp.a
