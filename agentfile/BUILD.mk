#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘
#
# Copyright 2026 Mozilla.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# BUILD.mk for agentfile (agentic CLI binary)
#
# agentfile is a cosmocc-compiled APE binary that runs an agentic loop
# end-to-end from a single CLI invocation. It mirrors the relationship
# between llama.cpp <-> llamafile and whisper.cpp <-> whisperfile: this
# package wraps agent.cpp's Agent/Model primitives on top of llamafile's
# already-built llama.cpp.a and the TOOL_LLAMAFILE_OBJS that provide GPU
# dispatch + zip filesystem support.
#

PKGS += AGENTFILE

# ==============================================================================
# Source files
# ==============================================================================

AGENTFILE_SRCS_CPP := \
	agentfile/agentfile.cpp

AGENTFILE_OBJS := $(AGENTFILE_SRCS_CPP:%.cpp=o/$(MODE)/%.o)

# ==============================================================================
# Include paths
# ==============================================================================

AGENTFILE_INCLUDES := \
	-iquote . \
	-iquote agentfile \
	-iquote llamafile \
	-iquote agent.cpp/src \
	-iquote llama.cpp/common \
	-iquote llama.cpp/include \
	-iquote llama.cpp/ggml/include \
	-isystem llama.cpp/vendor

# ==============================================================================
# Compiler flags
# ==============================================================================

AGENTFILE_CPPFLAGS := $(AGENTFILE_INCLUDES) \
	-DLLAMAFILE_VERSION_STRING=\"$(LLAMAFILE_VERSION_STRING)\"

o/$(MODE)/agentfile/%.o: agentfile/%.cpp agentfile/BUILD.mk
	@mkdir -p $(@D)
	$(COMPILE.cc) $(AGENTFILE_CPPFLAGS) -frtti -fexceptions -o $@ $<

# ==============================================================================
# Executable
# ==============================================================================
# Links:
#   - agentfile sources (agentfile.cpp)
#   - agent.cpp static lib (Agent, Model, ModelWeights)
#   - llama.cpp static archive (ggml, llama, common)
#   - llamafile objects for runtime GPU dispatch + zip filesystem
#   - LLAMAFILE_METAL_SOURCES: embedded Metal kernel sources (macOS)
#   - TINYBLAS_CPU_OBJS for matmul kernels
#   - HTTPLIB_OBJS: cpp-httplib implementation for the http_fetch tool

o/$(MODE)/agentfile/agentfile: \
		$(AGENTFILE_OBJS) \
		o/$(MODE)/agent.cpp/agent.cpp.a \
		o/$(MODE)/llama.cpp/llama.cpp.a \
		$(TOOL_LLAMAFILE_OBJS) \
		$(LLAMAFILE_METAL_SOURCES) \
		$(TINYBLAS_CPU_OBJS) \
		$(HTTPLIB_OBJS)
	@mkdir -p $(@D)
	$(LINK.o) $(AGENTFILE_OBJS) o/$(MODE)/agent.cpp/agent.cpp.a $(TOOL_LLAMAFILE_OBJS) $(LLAMAFILE_METAL_SOURCES) $(TINYBLAS_CPU_OBJS) $(HTTPLIB_OBJS) o/$(MODE)/llama.cpp/llama.cpp.a $(LOADLIBES) $(LDLIBS) -fopenmp -lpthread -o $@

# ==============================================================================
# Dependencies
# ==============================================================================

$(AGENTFILE_OBJS): agentfile/BUILD.mk

# ==============================================================================
# Main target
# ==============================================================================

.PHONY: o/$(MODE)/agentfile
o/$(MODE)/agentfile: o/$(MODE)/agentfile/agentfile
