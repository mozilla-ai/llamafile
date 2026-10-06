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
# end-to-end from a single CLI invocation. It builds on pieces the tree
# already compiles: agent.cpp's Agent/Model (loop + llama.cpp wrapper),
# llama.cpp's server-tools objects (the tool implementations), llama.cpp.a,
# and the TOOL_LLAMAFILE_OBJS providing GPU dispatch + zip filesystem.
#

PKGS += AGENTFILE

# ==============================================================================
# Source files
# ==============================================================================

AGENTFILE_SRCS_CPP := \
	agentfile/agentfile.cpp

AGENTFILE_HDRS := $(wildcard agentfile/*.h) \
	$(wildcard agentfile/callbacks/*.h) \
	$(wildcard agentfile/tools/*.h)

# External headers whose ABI leaks into agentfile objects: agent.cpp's
# list (agent.cpp/src, llama.cpp common/include/ggml, nlohmann) plus the
# server, mtmd, vendored and llamafile headers agentfile.cpp also reaches.
# Submodule bumps rewrite these without touching agentfile sources; without
# this dependency the old agentfile.o gets linked against new objects and
# fails with undefined or mismatched symbols (seen 2026-09-10: httplib's
# Headers container type changed).
AGENTFILE_EXT_HDRS := \
	$(AGENT_CPP_DEP_HDRS) \
	$(wildcard llama.cpp/tools/server/server-*.h) \
	$(wildcard llama.cpp/tools/mtmd/*.h) \
	llama.cpp/vendor/cpp-httplib/httplib.h \
	llama.cpp/vendor/sheredom/subprocess.h \
	llamafile/llamafile.h

# llama.cpp server-tools objects the adapter needs (tool implementations
# plus the TUs providing their symbols; no server.cpp — that has main()).
AGENTFILE_SERVER_OBJS := \
	o/$(MODE)/llama.cpp/tools/server/server-tools.cpp.o \
	o/$(MODE)/llama.cpp/tools/server/server-common.cpp.o \
	o/$(MODE)/llama.cpp/tools/server/server-queue.cpp.o \
	o/$(MODE)/llama.cpp/tools/server/server-mcp.cpp.o \
	o/$(MODE)/llama.cpp/tools/server/server-http.cpp.o

AGENTFILE_OBJS := $(AGENTFILE_SRCS_CPP:%.cpp=o/$(MODE)/%.o)

# ==============================================================================
# Include paths
# ==============================================================================

AGENTFILE_INCLUDES := \
	-iquote agentfile \
	-iquote llamafile \
	-iquote agent.cpp/src \
	-iquote llama.cpp/common \
	-iquote llama.cpp/include \
	-iquote llama.cpp/ggml/include \
	-iquote llama.cpp/tools/server \
	-iquote llama.cpp/tools/mtmd \
	-isystem llama.cpp/vendor

# ==============================================================================
# Compiler flags
# ==============================================================================

# COSMOCC enables the cosmo_args("/zip/.args") call in agentfile.cpp, as in
# llamafile/BUILD.mk; cosmocc itself only predefines __COSMOCC__.
AGENTFILE_CPPFLAGS := $(AGENTFILE_INCLUDES) \
	-DLLAMAFILE_VERSION_STRING=\"$(LLAMAFILE_VERSION_STRING)\" \
	-DCOSMOCC=1

# cpp-httplib is built with its Mbed TLS backend (see the HTTPS section in
# llama.cpp.patches/llamafile-files/BUILD.mk). The macro changes httplib
# class layouts, so agentfile objects that include httplib.h (http_fetch,
# web_search) must define it too, and the executable must link mbedtls.a.
AGENTFILE_CPPFLAGS += \
	-DCPPHTTPLIB_MBEDTLS_SUPPORT \
	-isystem third_party/mbedtls/include

# common/subproc.h (reached through server-common.h) switches subprocess_s
# between its real and dummy definition on LLAMA_SUBPROCESS. The server and
# common objects linked below are built with it (see the subprocess section
# in llama.cpp.patches/llamafile-files/BUILD.mk), so agentfile must agree.
AGENTFILE_CPPFLAGS += -DLLAMA_SUBPROCESS

o/$(MODE)/agentfile/%.o: agentfile/%.cpp agentfile/BUILD.mk $(AGENTFILE_HDRS) $(AGENTFILE_EXT_HDRS)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(AGENTFILE_CPPFLAGS) -o $@ $<

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
#   - HTTPLIB_OBJS: cpp-httplib implementation (server tools, http_fetch,
#     web_search)

o/$(MODE)/agentfile/agentfile: \
		$(AGENTFILE_OBJS) \
		o/$(MODE)/agent.cpp/agent.cpp.a \
		o/$(MODE)/llama.cpp/llama.cpp.a \
		o/$(MODE)/third_party/mbedtls/mbedtls.a \
		$(AGENTFILE_SERVER_OBJS) \
		$(TOOL_LLAMAFILE_OBJS) \
		$(LLAMAFILE_METAL_SOURCES) \
		$(TINYBLAS_CPU_OBJS) \
		$(HTTPLIB_OBJS)
	@mkdir -p $(@D)
	$(LINK.o) $(AGENTFILE_OBJS) o/$(MODE)/agent.cpp/agent.cpp.a $(AGENTFILE_SERVER_OBJS) $(TOOL_LLAMAFILE_OBJS) $(LLAMAFILE_METAL_SOURCES) $(TINYBLAS_CPU_OBJS) $(HTTPLIB_OBJS) o/$(MODE)/llama.cpp/llama.cpp.a o/$(MODE)/third_party/mbedtls/mbedtls.a $(LOADLIBES) $(LDLIBS) -fopenmp -lpthread -o $@

# ==============================================================================
# Main target
# ==============================================================================

.PHONY: o/$(MODE)/agentfile
o/$(MODE)/agentfile: o/$(MODE)/agentfile/agentfile
