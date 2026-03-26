#
# Copyright 2024 Mozilla Foundation
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

PKGS += LLAMAFILE

# ==============================================================================
# Header files (for mkdeps dependency tracking)
# ==============================================================================

# ==============================================================================
# Package Sources (NOT using full deps.mk SRCS/HDRS mechanism)
# ==============================================================================
# Note: We only list headers that:
# 1. Are needed by code scanned by mkdeps (like third_party sources)
# 2. Only include standard library headers (no llama.cpp dependencies)
# Headers like chatbot.h that include llama.cpp headers are excluded
# because mkdeps can't resolve those include paths.

LLAMAFILE_HDRS := \
	llamafile/llamafile.h \
	llamafile/sgemm.h

# ==============================================================================
# Include paths
# ==============================================================================

LLAMAFILE_INCLUDES := \
	-iquote llamafile \
	-iquote llama.cpp/common \
	-iquote llama.cpp/include \
	-iquote llama.cpp/ggml/include \
	-iquote llama.cpp/ggml/src \
	-iquote llama.cpp/ggml/src/ggml-cpu \
	-iquote llama.cpp/src \
	-iquote llama.cpp/tools/mtmd \
	-isystem llama.cpp/vendor \
	-isystem third_party

# ==============================================================================
# Compiler flags
# ==============================================================================
# When LLAMAFILE_TUI is defined, llama.cpp server's main() function is renamed
# to server_main() and called by llamafile's main.cpp. In the standalone build,
# this flag is off and a new main() function is compiled to call server_main
# (see llama.cpp/tools/server/server.cpp).

LLAMAFILE_CPPFLAGS := \
	$(LLAMAFILE_INCLUDES) \
	-DLLAMAFILE_TUI \
	-DCOSMOCC=1

# ==============================================================================
# Source files - Highlight library
# ==============================================================================

LLAMAFILE_HIGHLIGHT_SRCS := \
	llamafile/highlight/color_bleeder.cpp \
	llamafile/highlight/highlight.cpp \
	llamafile/highlight/highlight_ada.cpp \
	llamafile/highlight/highlight_asm.cpp \
	llamafile/highlight/highlight_basic.cpp \
	llamafile/highlight/highlight_bnf.cpp \
	llamafile/highlight/highlight_c.cpp \
	llamafile/highlight/highlight_cmake.cpp \
	llamafile/highlight/highlight_cobol.cpp \
	llamafile/highlight/highlight_csharp.cpp \
	llamafile/highlight/highlight_css.cpp \
	llamafile/highlight/highlight_d.cpp \
	llamafile/highlight/highlight_forth.cpp \
	llamafile/highlight/highlight_fortran.cpp \
	llamafile/highlight/highlight_go.cpp \
	llamafile/highlight/highlight_haskell.cpp \
	llamafile/highlight/highlight_html.cpp \
	llamafile/highlight/highlight_java.cpp \
	llamafile/highlight/highlight_js.cpp \
	llamafile/highlight/highlight_julia.cpp \
	llamafile/highlight/highlight_kotlin.cpp \
	llamafile/highlight/highlight_ld.cpp \
	llamafile/highlight/highlight_lisp.cpp \
	llamafile/highlight/highlight_lua.cpp \
	llamafile/highlight/highlight_m4.cpp \
	llamafile/highlight/highlight_make.cpp \
	llamafile/highlight/highlight_markdown.cpp \
	llamafile/highlight/highlight_matlab.cpp \
	llamafile/highlight/highlight_ocaml.cpp \
	llamafile/highlight/highlight_pascal.cpp \
	llamafile/highlight/highlight_perl.cpp \
	llamafile/highlight/highlight_php.cpp \
	llamafile/highlight/highlight_python.cpp \
	llamafile/highlight/highlight_r.cpp \
	llamafile/highlight/highlight_ruby.cpp \
	llamafile/highlight/highlight_rust.cpp \
	llamafile/highlight/highlight_scala.cpp \
	llamafile/highlight/highlight_shell.cpp \
	llamafile/highlight/highlight_sql.cpp \
	llamafile/highlight/highlight_swift.cpp \
	llamafile/highlight/highlight_tcl.cpp \
	llamafile/highlight/highlight_tex.cpp \
	llamafile/highlight/highlight_txt.cpp \
	llamafile/highlight/highlight_typescript.cpp \
	llamafile/highlight/highlight_zig.cpp \
	llamafile/highlight/util.cpp

# ==============================================================================
# Source files - Core TUI
# ==============================================================================

LLAMAFILE_SRCS_C := \
	llamafile/bestline.c \
	llamafile/cuda.c \
	llamafile/llamafile.c \
	llamafile/metal.c \
	llamafile/vulkan.c \
	llamafile/zip.c

LLAMAFILE_SRCS_CPP := \
	llamafile/args.cpp \
	llamafile/chatbot_api.cpp \
	llamafile/chatbot_cli.cpp \
	llamafile/chatbot_comm.cpp \
	llamafile/chatbot_comp.cpp \
	llamafile/chatbot_direct.cpp \
	llamafile/chatbot_eval.cpp \
	llamafile/chatbot_file.cpp \
	llamafile/chatbot_help.cpp \
	llamafile/chatbot_hint.cpp \
	llamafile/chatbot_hist.cpp \
	llamafile/chatbot_logo.cpp \
	llamafile/chatbot_main.cpp \
	llamafile/chatbot_repl.cpp \
	llamafile/compute.cpp \
	llamafile/datauri.cpp \
	llamafile/extract_data_uris.cpp \
	llamafile/image.cpp \
	llamafile/llama.cpp \
	llamafile/string.cpp \
	llamafile/xterm.cpp \
	$(LLAMAFILE_HIGHLIGHT_SRCS)

# ==============================================================================
# TinyBLAS CPU Optimized Kernels
# ==============================================================================
# These provide runtime CPU dispatch to architecture-specific SIMD implementations
# for matrix multiplication (sgemm) and mixture-of-experts (mixmul) operations.

TINYBLAS_CPU_SGEMM_SRCS := \
	llamafile/tinyblas_cpu_sgemm_amd_avx.cpp \
	llamafile/tinyblas_cpu_sgemm_amd_fma.cpp \
	llamafile/tinyblas_cpu_sgemm_amd_avx2.cpp \
	llamafile/tinyblas_cpu_sgemm_amd_avxvnni.cpp \
	llamafile/tinyblas_cpu_sgemm_amd_avx512f.cpp \
	llamafile/tinyblas_cpu_sgemm_amd_zen4.cpp \
	llamafile/tinyblas_cpu_sgemm_arm80.cpp \
	llamafile/tinyblas_cpu_sgemm_arm82.cpp \
	llamafile/tinyblas_cpu_unsupported.cpp

TINYBLAS_CPU_MIXMUL_SRCS := \
	llamafile/tinyblas_cpu_mixmul_amd_avx.cpp \
	llamafile/tinyblas_cpu_mixmul_amd_fma.cpp \
	llamafile/tinyblas_cpu_mixmul_amd_avx2.cpp \
	llamafile/tinyblas_cpu_mixmul_amd_avxvnni.cpp \
	llamafile/tinyblas_cpu_mixmul_amd_avx512f.cpp \
	llamafile/tinyblas_cpu_mixmul_amd_zen4.cpp \
	llamafile/tinyblas_cpu_mixmul_arm80.cpp \
	llamafile/tinyblas_cpu_mixmul_arm82.cpp

# IQK (Integer Quantized Kernels) for optimized k-quant/i-quant matmul
# Provides 150-400% speedup for Q4_K, Q5_K, Q6_K quantized models
TINYBLAS_CPU_IQK_SRCS := \
	llamafile/iqk_mul_mat_amd_avx2.cpp \
	llamafile/iqk_mul_mat_amd_zen4.cpp \
	llamafile/iqk_mul_mat_arm82.cpp

TINYBLAS_CPU_SRCS := \
	llamafile/sgemm.cpp \
	$(TINYBLAS_CPU_SGEMM_SRCS) \
	$(TINYBLAS_CPU_MIXMUL_SRCS) \
	$(TINYBLAS_CPU_IQK_SRCS)

TINYBLAS_CPU_OBJS := $(TINYBLAS_CPU_SRCS:%.cpp=o/$(MODE)/%.o)

# ==============================================================================
# Object files
# ==============================================================================

LLAMAFILE_OBJS := \
	$(LLAMAFILE_SRCS_C:%.c=o/$(MODE)/%.o) \
	$(LLAMAFILE_SRCS_CPP:%.cpp=o/$(MODE)/%.o)

# ==============================================================================
# Dependency libraries
# ==============================================================================

# Dependencies from llama.cpp/BUILD.mk:
#   GGML_OBJS   - Core tensor operations
#   LLAMA_OBJS  - LLM inference
#   COMMON_OBJS - Common utilities (arg parsing, sampling, chat templates)
#   MTMD_OBJS   - Multimodal support (vision models)
#   HTTPLIB_OBJS - HTTP client support for downloads
# Dependencies from llamafile/highlight/BUILD.mk:
#   We only need the gperf-generated keyword dictionary objects, not the
#   highlight cpp files (since we have our own copies in llamafile/highlight)

LLAMAFILE_HIGHLIGHT_GPERF_FILES := $(wildcard llamafile/highlight/*.gperf)
LLAMAFILE_HIGHLIGHT_KEYWORDS := $(LLAMAFILE_HIGHLIGHT_GPERF_FILES:%.gperf=o/$(MODE)/%.o)

# Server objects for llamafile
LLAMAFILE_SERVER_SUPPORT_OBJS := \
	o/$(MODE)/llama.cpp/tools/server/server-common.cpp.o \
	o/$(MODE)/llama.cpp/tools/server/server-context.cpp.o \
	o/$(MODE)/llama.cpp/tools/server/server-http.cpp.o \
	o/$(MODE)/llama.cpp/tools/server/server-models.cpp.o \
	o/$(MODE)/llama.cpp/tools/server/server-queue.cpp.o \
	o/$(MODE)/llama.cpp/tools/server/server-task.cpp.o

# Metal source files to embed in the executable (for runtime compilation on macOS)
# These are extracted at runtime and compiled into ggml-metal.dylib
LLAMAFILE_METAL_SOURCES := \
	o/$(MODE)/llama.cpp/ggml/src/ggml.c.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-alloc.c.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-backend.cpp.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-quants.c.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-threading.cpp.zip.o \
	o/$(MODE)/llama.cpp/ggml/include/ggml.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/include/gguf.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/include/ggml-cpu.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/include/ggml-alloc.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/include/ggml-backend.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/include/ggml-metal.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-impl.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-common.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-quants.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-threading.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-backend-impl.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-cpu/ggml-cpu-impl.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal.cpp.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-common.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-common.cpp.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.h.zip.o \
	o/$(MODE)/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp.zip.o

# Use deferred expansion (=) since this depends on variables from llama.cpp/BUILD.mk
LLAMAFILE_DEPS = \
	$(GGML_OBJS) \
	$(LLAMA_OBJS) \
	$(COMMON_OBJS) \
	$(MTMD_OBJS) \
	$(HTTPLIB_OBJS) \
	$(LLAMAFILE_SERVER_SUPPORT_OBJS) \
	$(LLAMAFILE_HIGHLIGHT_KEYWORDS) \
	$(LLAMAFILE_METAL_SOURCES) \
	$(TINYBLAS_CPU_OBJS) \
	o/$(MODE)/third_party/stb/stb_image_resize2.o

# ==============================================================================
# Server integration
# ==============================================================================

# Include paths needed for server compilation
LLAMAFILE_SERVER_INCS := \
	$(LLAMAFILE_INCLUDES) \
	-iquote llama.cpp/tools/server \
	-iquote o/$(MODE)/llama.cpp/tools/server

# Compile server.cpp
o/$(MODE)/llamafile/server.cpp.o: llama.cpp/tools/server/server.cpp $(SERVER_ASSETS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(LLAMAFILE_CPPFLAGS) $(LLAMAFILE_SERVER_INCS) -c -o $@ $<

# ==============================================================================
# Main executable
# ==============================================================================

# main.cpp: no special includes needed (combined mode uses server_main via forward decl)
o/$(MODE)/llamafile/main.o: llamafile/main.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(LLAMAFILE_CPPFLAGS) -c -o $@ $<

o/$(MODE)/llamafile/llamafile: \
		o/$(MODE)/llamafile/main.o \
		o/$(MODE)/llamafile/server.cpp.o \
		$(LLAMAFILE_OBJS) \
		$(LLAMAFILE_DEPS) \
		$(SERVER_ASSETS)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -o $@ $(filter %.o,$^) $(LDLIBS)

# ==============================================================================
# Pattern rules for llamafile sources
# ==============================================================================

# metal.c needs GGML_VERSION and GGML_COMMIT for runtime Metal compilation
# GGML_VERSION and GGML_COMMIT are inherited from build/config.mk
o/$(MODE)/llamafile/metal.o: llamafile/metal.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LLAMAFILE_CPPFLAGS) \
		-DGGML_VERSION=\"$(GGML_VERSION)\" \
		-DGGML_COMMIT=\"$(GGML_COMMIT)\" \
		-c -o $@ $<

o/$(MODE)/llamafile/%.o: llamafile/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LLAMAFILE_CPPFLAGS) -c -o $@ $<

o/$(MODE)/llamafile/%.o: llamafile/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(LLAMAFILE_CPPFLAGS) -c -o $@ $<

o/$(MODE)/llamafile/highlight/%.o: llamafile/highlight/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(LLAMAFILE_CPPFLAGS) -c -o $@ $<

# ==============================================================================
# TinyBLAS CPU Architecture-Specific Compilation Flags
# ==============================================================================
# Each variant is compiled with flags specific to its target CPU architecture.
# The -Xx86_64 and -Xaarch64 prefixes are cosmocc conventions for arch-specific flags.
# The -mgcc flag is critical for enabling GCC SIMD intrinsics with cosmocc.

# Static pattern rule for tinyblas CPU files
# This ensures these targets use the specialized recipe with SIMD flags
$(TINYBLAS_CPU_OBJS): o/$(MODE)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(CCFLAGS) $(TARGET_ARCH) -c -o $@ $<

# Base flags for all tinyblas CPU files
# -mgcc enables GCC intrinsics (__m128, __m256, etc.) with cosmocc
$(TINYBLAS_CPU_OBJS): private CCFLAGS += -O3 -fopenmp -mgcc
$(TINYBLAS_CPU_OBJS): private CPPFLAGS += $(LLAMAFILE_INCLUDES) -DCOSMOCC=1 -DGGML_USE_LLAMAFILE

# x86_64 AVX (Sandy Bridge, Ivy Bridge - 2010-2012)
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avx.o \
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avx.o: \
	private TARGET_ARCH += -Xx86_64-mtune=sandybridge -Xx86_64-mavx -Xx86_64-mf16c

# x86_64 FMA (AMD Piledriver - 2011-2014)
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_fma.o \
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_fma.o: \
	private TARGET_ARCH += -Xx86_64-mtune=bdver2 -Xx86_64-mavx -Xx86_64-mf16c -Xx86_64-mfma

# x86_64 AVX2 (Haswell, Broadwell, Skylake - 2013-2020)
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avx2.o \
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avx2.o: \
	private TARGET_ARCH += -Xx86_64-mtune=skylake -Xx86_64-mavx -Xx86_64-mf16c -Xx86_64-mfma -Xx86_64-mavx2

# x86_64 AVX-VNNI (Intel Alder Lake - 2021+)
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avxvnni.o \
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avxvnni.o: \
	private TARGET_ARCH += -Xx86_64-mtune=alderlake -Xx86_64-mavx -Xx86_64-mf16c -Xx86_64-mfma -Xx86_64-mavx2 -Xx86_64-mavxvnni

# x86_64 AVX-512F (Intel Skylake-X, Xeon - 2015+)
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avx512f.o \
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avx512f.o: \
	private TARGET_ARCH += -Xx86_64-mtune=cannonlake -Xx86_64-mavx -Xx86_64-mf16c -Xx86_64-mfma -Xx86_64-mavx2 -Xx86_64-mavx512f

# x86_64 Zen4 (AMD Zen 4 - 2023+, with AVX-512 BF16/VNNI)
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_zen4.o \
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_zen4.o: \
	private TARGET_ARCH += -Xx86_64-mtune=znver4 -Xx86_64-mavx -Xx86_64-mf16c -Xx86_64-mfma -Xx86_64-mavx2 -Xx86_64-mavx512f -Xx86_64-mavx512vl -Xx86_64-mavx512vnni -Xx86_64-mavx512bf16

# ARM64 v8.2-a (Apple M1/M2, Raspberry Pi 5 - with FP16 and dotprod)
o/$(MODE)/llamafile/tinyblas_cpu_sgemm_arm82.o \
o/$(MODE)/llamafile/tinyblas_cpu_mixmul_arm82.o: \
	private TARGET_ARCH += -Xaarch64-march=armv8.2-a+dotprod+fp16

# ARM64 v8.0-a baseline and unsupported have no special flags

# IQK (Integer Quantized Kernels) architecture-specific flags
# AVX2 variant (Haswell+)
o/$(MODE)/llamafile/iqk_mul_mat_amd_avx2.o: \
	private TARGET_ARCH += -Xx86_64-mtune=skylake -Xx86_64-mavx -Xx86_64-mavx2 -Xx86_64-mfma -Xx86_64-mf16c

# Zen4 variant (AMD Zen 4+ with AVX-512)
o/$(MODE)/llamafile/iqk_mul_mat_amd_zen4.o: \
	private TARGET_ARCH += -Xx86_64-mtune=skylake -Xx86_64-mavx -Xx86_64-mavx2 -Xx86_64-mfma -Xx86_64-mf16c -Xx86_64-mavx512f -Xx86_64-mavx512vl -Xx86_64-mavx512vnni -Xx86_64-mavx512bw -Xx86_64-mavx512dq

# ARM82 variant (Apple M1+, Raspberry Pi 5)
o/$(MODE)/llamafile/iqk_mul_mat_arm82.o: \
	private TARGET_ARCH += -Xaarch64-march=armv8.2-a+dotprod+fp16

# ==============================================================================
# Targets
# ==============================================================================

.PHONY: o/$(MODE)/llamafile
o/$(MODE)/llamafile: o/$(MODE)/llamafile/llamafile
