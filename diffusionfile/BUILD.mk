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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# BUILD.mk for diffusionfile tools
#
# diffusionfile provides llamafile-integrated stable-diffusion tools:
#   - diffusionfile: CLI image generation with llamafile features
#

PKGS += DIFFUSIONFILE

# ==============================================================================
# Package Sources (NOT using deps.mk SRCS/HDRS mechanism)
# ==============================================================================
# Note: We don't define DIFFUSIONFILE_SRCS or DIFFUSIONFILE_HDRS because:
# 1. Our sources include stable-diffusion.cpp headers with relative paths
# 2. mkdeps can't resolve these relative paths against full-path HDRS entries
# This matches the pattern used by stable-diffusion.cpp/BUILD.mk

# ==============================================================================
# Source files
# ==============================================================================

DIFFUSIONFILE_CLI_SRCS := \
	diffusionfile/diffusionfile.cpp

# ==============================================================================
# Object files
# ==============================================================================

DIFFUSIONFILE_OBJS := $(DIFFUSIONFILE_CLI_SRCS:%.cpp=o/$(MODE)/%.o)

# ==============================================================================
# Include paths
# ==============================================================================
# Note: Using DIFFUSIONFILE_INCLUDES (not _INCS) to avoid being collected by
# deps.mk which expects _INCS to be a list of .inc files, not compiler flags.

DIFFUSIONFILE_INCLUDES := \
	-iquote . \
	-iquote diffusionfile \
	-iquote stable-diffusion.cpp \
	-iquote stable-diffusion.cpp/src \
	-iquote stable-diffusion.cpp/include \
	-iquote stable-diffusion.cpp/thirdparty \
	-iquote stable-diffusion.cpp/examples \
	-iquote llama.cpp/ggml/include \
	-iquote llama.cpp/include

# ==============================================================================
# Compiler flags
# ==============================================================================

DIFFUSIONFILE_CPPFLAGS := \
	$(DIFFUSIONFILE_INCLUDES) \
	-DLLAMAFILE_VERSION_STRING=\"$(LLAMAFILE_VERSION_STRING)\"

# ==============================================================================
# Dependencies - llamafile objects for GPU support
# ==============================================================================
# Same pattern as llama.cpp/BUILD.mk TOOL_LLAMAFILE_OBJS

DIFFUSIONFILE_LLAMAFILE_OBJS := \
	o/$(MODE)/llamafile/llamafile.o \
	o/$(MODE)/llamafile/metal.o \
	o/$(MODE)/llamafile/cuda.o \
	o/$(MODE)/llamafile/vulkan.o \
	o/$(MODE)/llamafile/zip.o \
	o/$(MODE)/llamafile/check_cpu.o

# ==============================================================================
# Compilation rules
# ==============================================================================

o/$(MODE)/diffusionfile/%.o: diffusionfile/%.cpp diffusionfile/BUILD.mk
	@mkdir -p $(@D)
	$(COMPILE.cc) $(DIFFUSIONFILE_CPPFLAGS) -frtti -o $@ $<

# ==============================================================================
# Executable - diffusionfile (llamafile-integrated CLI)
# ==============================================================================
# Links main.cpp compiled with -DDIFFUSIONFILE to exclude its main()

o/$(MODE)/diffusionfile/diffusionfile: \
		$(DIFFUSIONFILE_OBJS) \
		o/$(MODE)/stable-diffusion.cpp/examples/cli/main.diffusionfile.cpp.o \
		o/$(MODE)/stable-diffusion.cpp/stable-diffusion.cpp.a \
		o/$(MODE)/stable-diffusion.cpp/thirdparty/zip.c.o \
		o/$(MODE)/stable-diffusion.cpp.patches/sd_math_shim.o \
		$(DIFFUSIONFILE_LLAMAFILE_OBJS) \
		$(GGML_OBJS) \
		$(TINYBLAS_CPU_OBJS)
	@mkdir -p $(@D)
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -lm -o $@

# ==============================================================================
# Main target
# ==============================================================================

.PHONY: o/$(MODE)/diffusionfile
o/$(MODE)/diffusionfile: \
	o/$(MODE)/diffusionfile/diffusionfile
