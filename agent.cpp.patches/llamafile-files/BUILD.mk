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
# BUILD.mk for agent.cpp library
#
# agent.cpp provides an agentic loop on top of llama.cpp:
#   - Agent: orchestrates LLM <-> tool conversation
#   - Model: thin wrapper around llama_context with sampler
#   - Tool/Callback/Error: extension points
#
# This BUILD.mk produces a single static library (agent.cpp.a) that
# combines model.cpp + agent.cpp. Links against llama.cpp.a's common
# library for chat templating and parsing.
#
# MCP, OAuth, and OpenTelemetry support are explicitly disabled in v0.
#

PKGS += AGENT_CPP

# ==============================================================================
# Source files
# ==============================================================================

AGENT_CPP_SRCS_CPP := \
	agent.cpp/src/model.cpp \
	agent.cpp/src/agent.cpp

AGENT_CPP_OBJS := $(AGENT_CPP_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)

# ==============================================================================
# Include paths
# ==============================================================================

AGENT_CPP_INCLUDES := \
	-iquote agent.cpp/src \
	-iquote llama.cpp/common \
	-iquote llama.cpp/include \
	-iquote llama.cpp/ggml/include \
	-isystem llama.cpp/vendor

# ==============================================================================
# Compiler flags
# ==============================================================================

$(AGENT_CPP_OBJS): private CPPFLAGS += $(AGENT_CPP_INCLUDES)
$(AGENT_CPP_OBJS): private CCFLAGS += -DNDEBUG

# C++17 + RTTI (agent.cpp uses dynamic_cast for tool dispatch, exceptions)
o/$(MODE)/agent.cpp/%.cpp.o: agent.cpp/%.cpp agent.cpp/BUILD.mk
	@mkdir -p $(@D)
	$(COMPILE.cc) -frtti -fexceptions -o $@ $<

# ==============================================================================
# Combined static library
# ==============================================================================

o/$(MODE)/agent.cpp/agent.cpp.a: $(AGENT_CPP_OBJS)

# ==============================================================================
# Dependencies
# ==============================================================================

# Headers whose layout is compiled into these objects: agent.cpp's own
# classes and the llama.cpp common types they embed (common_chat_msg,
# common_sampler, ...). AGENT_CPP declares no SRCS/HDRS, so mkdeps never
# scans these sources; without this list a patch to model.h or a llama.cpp
# bump leaves stale objects in agent.cpp.a (the failure AGENTFILE_EXT_HDRS
# in agentfile/BUILD.mk guards against for agentfile.o).
AGENT_CPP_DEP_HDRS := \
	$(wildcard agent.cpp/src/*.h) \
	$(wildcard llama.cpp/common/*.h) \
	$(wildcard llama.cpp/include/*.h)

$(AGENT_CPP_OBJS): agent.cpp/BUILD.mk $(AGENT_CPP_DEP_HDRS)

# ==============================================================================
# Main target
# ==============================================================================

.PHONY: o/$(MODE)/agent.cpp
o/$(MODE)/agent.cpp: o/$(MODE)/agent.cpp/agent.cpp.a
