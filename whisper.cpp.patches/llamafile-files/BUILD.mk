#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘
#
# BUILD.mk for whisper.cpp v1.8.3 (CPU-only build with cosmocc)
#
# This BUILD.mk is derived from whisper.cpp's CMake configuration.
# whisper.cpp uses its own bundled GGML (the APIs have diverged from llama.cpp).
#

PKGS += WHISPER_CPP

# ==============================================================================
# Version Information
# ==============================================================================

WHISPER_VERSION := 1.8.3
WHISPER_GGML_VERSION := 0.9.5
WHISPER_GGML_COMMIT := unknown

# ==============================================================================
# Include Paths (from CMakeLists.txt)
# ==============================================================================

WHISPER_INCS := \
	-iquote whisper.cpp/include \
	-iquote whisper.cpp/src \
	-iquote whisper.cpp/examples \
	-iquote whisper.cpp/ggml/include \
	-iquote whisper.cpp/ggml/src \
	-iquote whisper.cpp/ggml/src/ggml-cpu

# ==============================================================================
# Common Compiler Flags
# ==============================================================================

WHISPER_CPPFLAGS := \
	-D_XOPEN_SOURCE=600 \
	-DGGML_SCHED_MAX_COPIES=4 \
	-DGGML_USE_CPU

# ==============================================================================
# GGML Base Library Sources (from ggml/src/CMakeLists.txt)
# ==============================================================================

WHISPER_GGML_BASE_SRCS_C := \
	whisper.cpp/ggml/src/ggml.c \
	whisper.cpp/ggml/src/ggml-alloc.c \
	whisper.cpp/ggml/src/ggml-quants.c

WHISPER_GGML_BASE_SRCS_CPP := \
	whisper.cpp/ggml/src/ggml.cpp \
	whisper.cpp/ggml/src/ggml-backend.cpp \
	whisper.cpp/ggml/src/ggml-backend-reg.cpp \
	whisper.cpp/ggml/src/ggml-opt.cpp \
	whisper.cpp/ggml/src/ggml-threading.cpp \
	whisper.cpp/ggml/src/gguf.cpp

# ==============================================================================
# GGML CPU Backend Sources (from ggml/src/ggml-cpu/CMakeLists.txt)
# ==============================================================================

WHISPER_GGML_CPU_SRCS_C := \
	whisper.cpp/ggml/src/ggml-cpu/ggml-cpu.c \
	whisper.cpp/ggml/src/ggml-cpu/quants.c

WHISPER_GGML_CPU_SRCS_CPP := \
	whisper.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp \
	whisper.cpp/ggml/src/ggml-cpu/repack.cpp \
	whisper.cpp/ggml/src/ggml-cpu/hbm.cpp \
	whisper.cpp/ggml/src/ggml-cpu/traits.cpp \
	whisper.cpp/ggml/src/ggml-cpu/binary-ops.cpp \
	whisper.cpp/ggml/src/ggml-cpu/unary-ops.cpp \
	whisper.cpp/ggml/src/ggml-cpu/vec.cpp \
	whisper.cpp/ggml/src/ggml-cpu/ops.cpp \
	whisper.cpp/ggml/src/ggml-cpu/amx/amx.cpp \
	whisper.cpp/ggml/src/ggml-cpu/amx/mmq.cpp

# Architecture-specific sources (x86 for now, aarch64 can be added later)
WHISPER_GGML_CPU_ARCH_SRCS_C := \
	whisper.cpp/ggml/src/ggml-cpu/arch/x86/quants.c

WHISPER_GGML_CPU_ARCH_SRCS_CPP := \
	whisper.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp

# ==============================================================================
# Whisper Library Sources (from src/CMakeLists.txt)
# ==============================================================================

WHISPER_CORE_SRCS_CPP := \
	whisper.cpp/src/whisper.cpp

# ==============================================================================
# Common Library Sources (from examples/CMakeLists.txt)
# ==============================================================================
# Note: Requires patches from whisper.cpp.patches/ to be applied:
# - examples_miniaudio.h.patch: fixes Windows detection for cosmopolitan
# - examples_common.cpp.patch: replaces std::partial_sort with nth_element+sort

WHISPER_COMMON_SRCS_CPP := \
	whisper.cpp/examples/common.cpp \
	whisper.cpp/examples/common-ggml.cpp \
	whisper.cpp/examples/common-whisper.cpp \
	whisper.cpp/examples/grammar-parser.cpp

# ==============================================================================
# Tool Sources
# ==============================================================================

WHISPER_CLI_SRCS := whisper.cpp/examples/cli/cli.cpp

# ==============================================================================
# Object Files
# ==============================================================================

WHISPER_GGML_BASE_OBJS := \
	$(WHISPER_GGML_BASE_SRCS_C:%.c=o/$(MODE)/%.c.o) \
	$(WHISPER_GGML_BASE_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)

WHISPER_GGML_CPU_OBJS := \
	$(WHISPER_GGML_CPU_SRCS_C:%.c=o/$(MODE)/%.c.o) \
	$(WHISPER_GGML_CPU_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o) \
	$(WHISPER_GGML_CPU_ARCH_SRCS_C:%.c=o/$(MODE)/%.c.o) \
	$(WHISPER_GGML_CPU_ARCH_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)

WHISPER_CORE_OBJS := $(WHISPER_CORE_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)
WHISPER_COMMON_OBJS := $(WHISPER_COMMON_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)
WHISPER_CLI_OBJS := $(WHISPER_CLI_SRCS:%.cpp=o/$(MODE)/%.cpp.o)

# All library objects
WHISPER_CPP_OBJS := \
	$(WHISPER_GGML_BASE_OBJS) \
	$(WHISPER_GGML_CPU_OBJS) \
	$(WHISPER_CORE_OBJS) \
	$(WHISPER_COMMON_OBJS)

# ==============================================================================
# Package Sources (NOT using deps.mk SRCS/HDRS mechanism)
# ==============================================================================
# Note: We don't define WHISPER_CPP_SRCS or WHISPER_CPP_HDRS because:
# 1. whisper.cpp uses relative includes like "ggml-cpu/ggml-cpu-impl.h"
# 2. mkdeps can't resolve these relative paths against full-path HDRS entries
# 3. This matches llama.cpp's approach which also doesn't define LLAMA_CPP_SRCS
# Dependencies are handled via explicit BUILD.mk dependencies in pattern rules.

# ==============================================================================
# Static Library
# ==============================================================================

o/$(MODE)/whisper.cpp/whisper.cpp.a: $(WHISPER_CPP_OBJS)

# ==============================================================================
# Compilation Rules - GGML Base
# ==============================================================================

$(WHISPER_GGML_BASE_SRCS_C:%.c=o/$(MODE)/%.c.o): \
		o/$(MODE)/%.c.o: %.c whisper.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.c) $(WHISPER_INCS) $(WHISPER_CPPFLAGS) \
		-DGGML_VERSION=\"$(WHISPER_GGML_VERSION)\" \
		-DGGML_COMMIT=\"$(WHISPER_GGML_COMMIT)\" \
		-o $@ $<

$(WHISPER_GGML_BASE_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o): \
		o/$(MODE)/%.cpp.o: %.cpp whisper.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(WHISPER_INCS) $(WHISPER_CPPFLAGS) \
		-DGGML_VERSION=\"$(WHISPER_GGML_VERSION)\" \
		-DGGML_COMMIT=\"$(WHISPER_GGML_COMMIT)\" \
		-frtti -o $@ $<

# ==============================================================================
# Compilation Rules - GGML CPU Backend
# ==============================================================================

$(WHISPER_GGML_CPU_SRCS_C:%.c=o/$(MODE)/%.c.o): \
		o/$(MODE)/%.c.o: %.c whisper.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.c) $(WHISPER_INCS) $(WHISPER_CPPFLAGS) -o $@ $<

$(WHISPER_GGML_CPU_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o): \
		o/$(MODE)/%.cpp.o: %.cpp whisper.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(WHISPER_INCS) $(WHISPER_CPPFLAGS) -frtti -o $@ $<

$(WHISPER_GGML_CPU_ARCH_SRCS_C:%.c=o/$(MODE)/%.c.o): \
		o/$(MODE)/%.c.o: %.c whisper.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.c) $(WHISPER_INCS) $(WHISPER_CPPFLAGS) -o $@ $<

$(WHISPER_GGML_CPU_ARCH_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o): \
		o/$(MODE)/%.cpp.o: %.cpp whisper.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(WHISPER_INCS) $(WHISPER_CPPFLAGS) -frtti -o $@ $<

# ==============================================================================
# Compilation Rules - Whisper Core
# ==============================================================================

$(WHISPER_CORE_OBJS): o/$(MODE)/%.cpp.o: %.cpp whisper.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(WHISPER_INCS) $(WHISPER_CPPFLAGS) \
		-DWHISPER_VERSION=\"$(WHISPER_VERSION)\" \
		-frtti -o $@ $<

# ==============================================================================
# Compilation Rules - Common Library
# ==============================================================================

$(WHISPER_COMMON_OBJS): o/$(MODE)/%.cpp.o: %.cpp whisper.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(WHISPER_INCS) $(WHISPER_CPPFLAGS) -frtti -o $@ $<

# ==============================================================================
# Compilation Rules - CLI Tool
# ==============================================================================

$(WHISPER_CLI_OBJS): o/$(MODE)/%.cpp.o: %.cpp whisper.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(WHISPER_INCS) $(WHISPER_CPPFLAGS) -frtti -o $@ $<

# ==============================================================================
# Compiler Flag Overrides
# ==============================================================================
# Note: -mgcc is required for files using errno in switch statements
# (cosmopolitan uses dynamic errno values, not constants)

# Core GGML - uses errno in switch statements, optimize for performance
o/$(MODE)/whisper.cpp/ggml/src/ggml.c.o: \
	private CCFLAGS += -O3 -mgcc

# Memory allocation and backend
o/$(MODE)/whisper.cpp/ggml/src/ggml-alloc.c.o \
o/$(MODE)/whisper.cpp/ggml/src/ggml-backend.cpp.o: \
	private CCFLAGS += -mgcc

# Vector operations - optimize for performance
o/$(MODE)/whisper.cpp/ggml/src/ggml-cpu/vec.cpp.o \
o/$(MODE)/whisper.cpp/ggml/src/ggml-cpu/ops.cpp.o \
o/$(MODE)/whisper.cpp/ggml/src/ggml-cpu/binary-ops.cpp.o \
o/$(MODE)/whisper.cpp/ggml/src/ggml-cpu/unary-ops.cpp.o: \
	private CCFLAGS += -O3 -mgcc

# Quantization - optimize for performance (critical hot path)
o/$(MODE)/whisper.cpp/ggml/src/ggml-quants.c.o \
o/$(MODE)/whisper.cpp/ggml/src/ggml-cpu/quants.c.o: \
	private CCFLAGS += -O3 -mgcc

# ==============================================================================
# Executable - whisper-cli (vanilla, no llamafile features)
# ==============================================================================

o/$(MODE)/whisper.cpp/whisper-cli: \
		$(WHISPER_CLI_OBJS) \
		o/$(MODE)/whisper.cpp/whisper.cpp.a
	@mkdir -p $(@D)
	$(LINK.o) $(WHISPER_CLI_OBJS) o/$(MODE)/whisper.cpp/whisper.cpp.a \
		$(LOADLIBES) $(LDLIBS) -o $@

# ==============================================================================
# cli.cpp for whisperfile build (with WHISPERFILE defined)
# ==============================================================================
# This object is used by whisperfile/BUILD.mk to build the whisperfile executable.
# The -DWHISPERFILE flag excludes cli.cpp's main() so whisperfile can provide its own.

o/$(MODE)/whisper.cpp/examples/cli/cli.whisperfile.cpp.o: \
		whisper.cpp/examples/cli/cli.cpp whisper.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(WHISPER_INCS) $(WHISPER_CPPFLAGS) -DWHISPERFILE -frtti -o $@ $<

# ==============================================================================
# server.cpp for whisper-server build (with WHISPERFILE defined)
# ==============================================================================
# This object is used by whisperfile/BUILD.mk to build the whisper-server executable.
# The -DWHISPERFILE flag excludes server.cpp's main() so whisper-server can provide its own.

WHISPER_SERVER_INCS := \
	$(WHISPER_INCS) \
	-iquote . \
	-iquote whisper.cpp/examples/server

o/$(MODE)/whisper.cpp/examples/server/server.whisperfile.cpp.o: \
		whisper.cpp/examples/server/server.cpp whisper.cpp/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(WHISPER_SERVER_INCS) $(WHISPER_CPPFLAGS) -DWHISPERFILE -frtti -o $@ $<

# ==============================================================================
# Main Target
# ==============================================================================

.PHONY: o/$(MODE)/whisper.cpp
o/$(MODE)/whisper.cpp: \
	o/$(MODE)/whisper.cpp/whisper.cpp.a \
	o/$(MODE)/whisper.cpp/whisper-cli
