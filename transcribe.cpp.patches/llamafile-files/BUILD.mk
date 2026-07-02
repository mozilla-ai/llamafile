#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘
#
# BUILD.mk for transcribe.cpp (CPU-only build with cosmocc)
#
# transcribe.cpp ships its OWN vendored ggml (ggml-org/ggml @ a pinned
# SHA, see ggml/UPSTREAM) which differs from llama.cpp/ggml, so we build
# that ggml here rather than reusing llama.cpp's objects — the same
# approach whisper.cpp uses.
#
# Derived from transcribe.cpp's CMake configuration:
#   ggml/src/CMakeLists.txt, ggml/src/ggml-cpu/CMakeLists.txt, src/CMakeLists.txt
#
# The ggml CPU build mirrors llama.cpp's proven cross-arch recipe:
# -DGGML_CPU_GENERIC plus the top-level quants.c/repack.cpp only (no
# arch/x86 or arch/arm sources), so a single cosmocc fat binary works on
# both x86_64 and aarch64.

PKGS += TRANSCRIBE_CPP

# ==============================================================================
# Version Information
# ==============================================================================

TRANSCRIBE_GGML_VERSION := 0.15.2
TRANSCRIBE_GGML_COMMIT := $(shell cd transcribe.cpp/ggml 2>/dev/null && cat UPSTREAM 2>/dev/null | sed -n 's/^sha:[[:space:]]*//p' | cut -c1-7 || echo "unknown")

# ==============================================================================
# Include Paths
# ==============================================================================

TRANSCRIBE_GGML_INCS := \
	-iquote transcribe.cpp/ggml/include \
	-iquote transcribe.cpp/ggml/src \
	-iquote transcribe.cpp/ggml/src/ggml-cpu

TRANSCRIBE_LIB_INCS := \
	-iquote transcribe.cpp/include \
	-iquote transcribe.cpp/src \
	-iquote transcribe.cpp/ggml/include \
	-iquote transcribe.cpp/ggml/src \
	-isystem $(COSMOCC)/include/third_party/zlib

# ==============================================================================
# Common Compiler Flags
# ==============================================================================

# GGML_MULTIPLATFORM turns the GGML_CALL annotations (ported from the
# llama.cpp patch set) into __ms_abi__ on x86-64, keeping the backend
# interface structs ABI-identical to the GPU dylibs, which are built
# with the same define (a no-op on aarch64, where the attribute doesn't
# exist — hence the global -Wno-attributes). It is already inherited
# from the global CPPFLAGS_ in build/config.mk; it is repeated here, as
# in llama.cpp's BUILD.mk, so the cross-DSO ABI contract is explicit.
TRANSCRIBE_GGML_CPPFLAGS := \
	-DGGML_MULTIPLATFORM \
	-DGGML_USE_CPU \
	-DGGML_CPU_GENERIC \
	-DGGML_USE_CPU_REPACK \
	-DGGML_SCHED_MAX_COPIES=4 \
	-DGGML_VERSION=\"$(TRANSCRIBE_GGML_VERSION)\" \
	-DGGML_COMMIT=\"$(TRANSCRIBE_GGML_COMMIT)\"

# TRANSCRIBE_HAS_BLAS is intentionally NOT defined: the decoder falls
# back to a correct (if slower) scalar loop, avoiding a BLAS dependency
# under cosmocc.
#
# MINIZ_NO_* defines mirror upstream src/CMakeLists.txt: they trim the
# vendored miniz to the compress-only deflate path Whisper's
# compression-ratio metric needs, and avoid colliding with any system
# zlib downstream (NO_ZLIB_COMPATIBLE_NAMES — call sites use the
# mz_-prefixed API).
TRANSCRIBE_LIB_CPPFLAGS := \
	-DGGML_MULTIPLATFORM \
	-DTRANSCRIBE_BUILD \
	-DGGML_SCHED_MAX_COPIES=4 \
	-DMINIZ_NO_INFLATE_APIS \
	-DMINIZ_NO_STDIO \
	-DMINIZ_NO_TIME \
	-DMINIZ_NO_ZLIB_COMPATIBLE_NAMES

# ==============================================================================
# GGML Sources (transcribe.cpp/ggml/src — vendored ggml 0.9.8)
# ==============================================================================

TRANSCRIBE_GGML_SRCS_C := \
	transcribe.cpp/ggml/src/ggml-alloc.c \
	transcribe.cpp/ggml/src/ggml-quants.c \
	transcribe.cpp/ggml/src/ggml.c \
	transcribe.cpp/ggml/src/ggml-cpu/ggml-cpu.c \
	transcribe.cpp/ggml/src/ggml-cpu/quants.c

TRANSCRIBE_GGML_SRCS_CPP := \
	transcribe.cpp/ggml/src/ggml-backend-dl.cpp \
	transcribe.cpp/ggml/src/ggml-backend-meta.cpp \
	transcribe.cpp/ggml/src/ggml-backend-reg.cpp \
	transcribe.cpp/ggml/src/ggml-backend.cpp \
	transcribe.cpp/ggml/src/ggml-opt.cpp \
	transcribe.cpp/ggml/src/ggml-threading.cpp \
	transcribe.cpp/ggml/src/ggml.cpp \
	transcribe.cpp/ggml/src/gguf.cpp \
	transcribe.cpp/ggml/src/ggml-cpu/binary-ops.cpp \
	transcribe.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp \
	transcribe.cpp/ggml/src/ggml-cpu/hbm.cpp \
	transcribe.cpp/ggml/src/ggml-cpu/ops.cpp \
	transcribe.cpp/ggml/src/ggml-cpu/repack.cpp \
	transcribe.cpp/ggml/src/ggml-cpu/traits.cpp \
	transcribe.cpp/ggml/src/ggml-cpu/unary-ops.cpp \
	transcribe.cpp/ggml/src/ggml-cpu/vec.cpp \
	transcribe.cpp/ggml/src/ggml-cpu/amx/amx.cpp \
	transcribe.cpp/ggml/src/ggml-cpu/amx/mmq.cpp

# ==============================================================================
# Transcribe Library Sources (transcribe.cpp/src — from src/CMakeLists.txt)
# ==============================================================================

TRANSCRIBE_LIB_SRCS_CPP := \
	transcribe.cpp/src/transcribe.cpp \
	transcribe.cpp/src/transcribe-loader.cpp \
	transcribe.cpp/src/transcribe-arch.cpp \
	transcribe.cpp/src/transcribe-meta.cpp \
	transcribe.cpp/src/transcribe-kaldi-fbank.cpp \
	transcribe.cpp/src/transcribe-mel.cpp \
	transcribe.cpp/src/transcribe-model.cpp \
	transcribe.cpp/src/transcribe-tokenizer.cpp \
	transcribe.cpp/src/transcribe-unicode.cpp \
	transcribe.cpp/src/transcribe-unicode-data.cpp \
	transcribe.cpp/src/transcribe-debug.cpp \
	transcribe.cpp/src/transcribe-env.cpp \
	transcribe.cpp/src/transcribe-flash-policy.cpp \
	transcribe.cpp/src/transcribe-backend.cpp \
	transcribe.cpp/src/transcribe-load-common.cpp \
	transcribe.cpp/src/transcribe-weights-util.cpp \
	transcribe.cpp/src/transcribe-bin-loader.cpp \
	transcribe.cpp/src/transcribe-batch-util.cpp \
	transcribe.cpp/src/conformer/conformer.cpp \
	transcribe.cpp/src/sanm/sanm.cpp \
	transcribe.cpp/src/causal_lm/causal_lm.cpp \
	transcribe.cpp/src/granite_conformer/shaw_attn.cpp \
	transcribe.cpp/src/arch/parakeet/model.cpp \
	transcribe.cpp/src/arch/parakeet/capabilities.cpp \
	transcribe.cpp/src/arch/parakeet/weights.cpp \
	transcribe.cpp/src/arch/parakeet/encoder.cpp \
	transcribe.cpp/src/arch/parakeet/decoder.cpp \
	transcribe.cpp/src/arch/cohere/model.cpp \
	transcribe.cpp/src/arch/cohere/capabilities.cpp \
	transcribe.cpp/src/arch/cohere/weights.cpp \
	transcribe.cpp/src/arch/cohere/encoder.cpp \
	transcribe.cpp/src/arch/cohere/decoder.cpp \
	transcribe.cpp/src/arch/canary/model.cpp \
	transcribe.cpp/src/arch/canary/capabilities.cpp \
	transcribe.cpp/src/arch/canary/weights.cpp \
	transcribe.cpp/src/arch/canary/encoder.cpp \
	transcribe.cpp/src/arch/canary/decoder.cpp \
	transcribe.cpp/src/arch/qwen3_asr/model.cpp \
	transcribe.cpp/src/arch/qwen3_asr/capabilities.cpp \
	transcribe.cpp/src/arch/qwen3_asr/weights.cpp \
	transcribe.cpp/src/arch/qwen3_asr/encoder.cpp \
	transcribe.cpp/src/arch/qwen3_asr/decoder.cpp \
	transcribe.cpp/src/arch/canary_qwen/model.cpp \
	transcribe.cpp/src/arch/canary_qwen/capabilities.cpp \
	transcribe.cpp/src/arch/canary_qwen/weights.cpp \
	transcribe.cpp/src/arch/canary_qwen/encoder.cpp \
	transcribe.cpp/src/arch/canary_qwen/decoder.cpp \
	transcribe.cpp/src/arch/whisper/model.cpp \
	transcribe.cpp/src/arch/whisper/capabilities.cpp \
	transcribe.cpp/src/arch/whisper/weights.cpp \
	transcribe.cpp/src/arch/whisper/encoder.cpp \
	transcribe.cpp/src/arch/whisper/decoder.cpp \
	transcribe.cpp/src/arch/whisper/bin_load.cpp \
	transcribe.cpp/src/arch/whisper/public.cpp \
	transcribe.cpp/src/arch/moonshine/model.cpp \
	transcribe.cpp/src/arch/moonshine/capabilities.cpp \
	transcribe.cpp/src/arch/moonshine/weights.cpp \
	transcribe.cpp/src/arch/moonshine/encoder.cpp \
	transcribe.cpp/src/arch/moonshine/decoder.cpp \
	transcribe.cpp/src/arch/moonshine_streaming/model.cpp \
	transcribe.cpp/src/arch/moonshine_streaming/capabilities.cpp \
	transcribe.cpp/src/arch/moonshine_streaming/weights.cpp \
	transcribe.cpp/src/arch/moonshine_streaming/encoder.cpp \
	transcribe.cpp/src/arch/moonshine_streaming/decoder.cpp \
	transcribe.cpp/src/arch/sensevoice/model.cpp \
	transcribe.cpp/src/arch/sensevoice/capabilities.cpp \
	transcribe.cpp/src/arch/sensevoice/weights.cpp \
	transcribe.cpp/src/arch/sensevoice/encoder.cpp \
	transcribe.cpp/src/arch/funasr_nano/model.cpp \
	transcribe.cpp/src/arch/funasr_nano/capabilities.cpp \
	transcribe.cpp/src/arch/funasr_nano/weights.cpp \
	transcribe.cpp/src/arch/funasr_nano/encoder.cpp \
	transcribe.cpp/src/arch/funasr_nano/decoder.cpp \
	transcribe.cpp/src/arch/funasr_nano/adaptor.cpp \
	transcribe.cpp/src/arch/gigaam/model.cpp \
	transcribe.cpp/src/arch/gigaam/capabilities.cpp \
	transcribe.cpp/src/arch/gigaam/weights.cpp \
	transcribe.cpp/src/arch/gigaam/encoder.cpp \
	transcribe.cpp/src/arch/gigaam/decoder.cpp \
	transcribe.cpp/src/arch/gigaam/mel.cpp \
	transcribe.cpp/src/arch/granite/model.cpp \
	transcribe.cpp/src/arch/granite/capabilities.cpp \
	transcribe.cpp/src/arch/granite/weights.cpp \
	transcribe.cpp/src/arch/granite/encoder.cpp \
	transcribe.cpp/src/arch/granite/projector.cpp \
	transcribe.cpp/src/arch/granite/decoder.cpp \
	transcribe.cpp/src/arch/granite_nar/model.cpp \
	transcribe.cpp/src/arch/granite_nar/capabilities.cpp \
	transcribe.cpp/src/arch/granite_nar/weights.cpp \
	transcribe.cpp/src/arch/granite_nar/encoder.cpp \
	transcribe.cpp/src/arch/granite_nar/projector.cpp \
	transcribe.cpp/src/arch/granite_nar/decoder.cpp \
	transcribe.cpp/src/arch/medasr/model.cpp \
	transcribe.cpp/src/arch/medasr/capabilities.cpp \
	transcribe.cpp/src/arch/medasr/weights.cpp \
	transcribe.cpp/src/arch/medasr/encoder.cpp \
	transcribe.cpp/src/arch/voxtral/model.cpp \
	transcribe.cpp/src/arch/voxtral/capabilities.cpp \
	transcribe.cpp/src/arch/voxtral/weights.cpp \
	transcribe.cpp/src/arch/voxtral/encoder.cpp \
	transcribe.cpp/src/arch/voxtral/decoder.cpp \
	transcribe.cpp/src/arch/voxtral_realtime/model.cpp \
	transcribe.cpp/src/arch/voxtral_realtime/capabilities.cpp \
	transcribe.cpp/src/arch/voxtral_realtime/weights.cpp \
	transcribe.cpp/src/arch/voxtral_realtime/encoder.cpp \
	transcribe.cpp/src/arch/voxtral_realtime/decoder.cpp

# Vendored miniz (C) — only consumer is arch/whisper/model.cpp's
# compression-ratio heuristic. Built into the transcribe lib so the
# binary stays self-contained (no system zlib).
TRANSCRIBE_LIB_SRCS_C := \
	transcribe.cpp/src/third_party/miniz/miniz.c

# ==============================================================================
# Object Files
# ==============================================================================

TRANSCRIBE_GGML_OBJS := \
	$(TRANSCRIBE_GGML_SRCS_C:%.c=o/$(MODE)/%.c.o) \
	$(TRANSCRIBE_GGML_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)

TRANSCRIBE_LIB_OBJS := \
	$(TRANSCRIBE_LIB_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o) \
	$(TRANSCRIBE_LIB_SRCS_C:%.c=o/$(MODE)/%.c.o)

# All static-library objects.
TRANSCRIBE_CPP_OBJS := \
	$(TRANSCRIBE_GGML_OBJS) \
	$(TRANSCRIBE_LIB_OBJS)

# ==============================================================================
# Static Library
# ==============================================================================

o/$(MODE)/transcribe.cpp/transcribe.cpp.a: $(TRANSCRIBE_CPP_OBJS)

# ==============================================================================
# Compilation Rules — GGML
# ==============================================================================

$(TRANSCRIBE_GGML_SRCS_C:%.c=o/$(MODE)/%.c.o): \
		o/$(MODE)/%.c.o: %.c transcribe.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.c) $(TRANSCRIBE_GGML_INCS) $(TRANSCRIBE_GGML_CPPFLAGS) -o $@ $<

$(TRANSCRIBE_GGML_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o): \
		o/$(MODE)/%.cpp.o: %.cpp transcribe.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(TRANSCRIBE_GGML_INCS) $(TRANSCRIBE_GGML_CPPFLAGS) -o $@ $<

# ==============================================================================
# Compilation Rules — Transcribe Library
# ==============================================================================

$(TRANSCRIBE_LIB_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o): \
		o/$(MODE)/%.cpp.o: %.cpp transcribe.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(TRANSCRIBE_LIB_INCS) $(TRANSCRIBE_LIB_CPPFLAGS) -o $@ $<

# Vendored miniz.c uses third-party style that trips strict warnings;
# silence them for this one file (upstream does the same via -w).
$(TRANSCRIBE_LIB_SRCS_C:%.c=o/$(MODE)/%.c.o): \
		o/$(MODE)/%.c.o: %.c transcribe.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.c) $(TRANSCRIBE_LIB_INCS) $(TRANSCRIBE_LIB_CPPFLAGS) -w -o $@ $<

# ==============================================================================
# Compiler Flag Overrides (mirror llama.cpp's cosmocc recipe)
# ==============================================================================
# -mgcc is required for files that use errno in switch statements
# (cosmopolitan's errno is a dynamic value, not a compile-time constant).

# Memory allocation and backend.
o/$(MODE)/transcribe.cpp/ggml/src/ggml-alloc.c.o \
o/$(MODE)/transcribe.cpp/ggml/src/ggml-backend.cpp.o: \
	private CCFLAGS += -mgcc

# Backend registration — optimize for size.
o/$(MODE)/transcribe.cpp/ggml/src/ggml-backend-reg.cpp.o: \
	private CCFLAGS += -Os

# Core GGML and vector operations — optimize for performance.
o/$(MODE)/transcribe.cpp/ggml/src/ggml.c.o \
o/$(MODE)/transcribe.cpp/ggml/src/ggml-cpu/vec.cpp.o \
o/$(MODE)/transcribe.cpp/ggml/src/ggml-cpu/ops.cpp.o \
o/$(MODE)/transcribe.cpp/ggml/src/ggml-cpu/binary-ops.cpp.o \
o/$(MODE)/transcribe.cpp/ggml/src/ggml-cpu/unary-ops.cpp.o: \
	private CCFLAGS += -O3 -mgcc

# Quantization — critical hot path.
o/$(MODE)/transcribe.cpp/ggml/src/ggml-quants.c.o \
o/$(MODE)/transcribe.cpp/ggml/src/ggml-cpu/quants.c.o: \
	private CCFLAGS += -O3 -mgcc

# ==============================================================================
# Dependencies
# ==============================================================================

$(TRANSCRIBE_CPP_OBJS): transcribe.cpp/BUILD.mk

# ==============================================================================
# Main Target
# ==============================================================================

.PHONY: o/$(MODE)/transcribe.cpp
o/$(MODE)/transcribe.cpp: o/$(MODE)/transcribe.cpp/transcribe.cpp.a
