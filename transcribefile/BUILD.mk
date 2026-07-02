#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘
#
# BUILD.mk for transcribefile.
#
# Entry point is transcribefile/main.cpp, a thin wrapper that loads default
# arguments from the executable's own zip store (/zip/.args, via cosmo_args)
# and then calls transcribe.cpp's example CLI logic. The CLI lives in
# transcribe.cpp/examples/cli/main.cpp, compiled with -DTRANSCRIBEFILE so its
# main() is renamed to transcribe_cli_main() and our wrapper owns the entry
# point (the standalone CLI main() is dropped). The WAV loader comes from
# transcribe.cpp/examples/common/wav.cpp. Everything links against the
# cosmocc-built transcribe.cpp.a.

PKGS += TRANSCRIBEFILE

# Version string surfaced by `transcribefile --version`.
TRANSCRIBEFILE_VERSION_STRING := $(shell cd transcribe.cpp 2>/dev/null && git describe --tags --always 2>/dev/null || echo "0.0.0-dev")

# ==============================================================================
# Sources (our wrapper + upstream example CLI + WAV loader)
# ==============================================================================

TRANSCRIBEFILE_SRCS_CPP := \
	transcribefile/main.cpp \
	transcribe.cpp/examples/cli/main.cpp \
	transcribe.cpp/examples/common/wav.cpp

# Our own code (cosmocc libc compatibility shims, etc.).
TRANSCRIBEFILE_SRCS_C := \
	transcribefile/cosmo_compat.c

TRANSCRIBEFILE_CPP_OBJS := $(TRANSCRIBEFILE_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)
TRANSCRIBEFILE_C_OBJS   := $(TRANSCRIBEFILE_SRCS_C:%.c=o/$(MODE)/%.c.o)

TRANSCRIBEFILE_OBJS := \
	$(TRANSCRIBEFILE_CPP_OBJS) \
	$(TRANSCRIBEFILE_C_OBJS)

TRANSCRIBEFILE_INCLUDES := \
	-iquote transcribe.cpp/include \
	-iquote transcribe.cpp/examples/common

# -DTRANSCRIBEFILE renames the upstream CLI main() to transcribe_cli_main();
# the version string feeds `--version`. dr_wav.h trips a couple of warnings
# that are noise here.
TRANSCRIBEFILE_CPPFLAGS := \
	-DTRANSCRIBEFILE \
	-DTRANSCRIBEFILE_VERSION_STRING=\"$(TRANSCRIBEFILE_VERSION_STRING)\" \
	-Wno-sign-compare \
	-Wno-unused-function

# ==============================================================================
# Compilation Rules
# ==============================================================================

$(TRANSCRIBEFILE_CPP_OBJS): o/$(MODE)/%.cpp.o: %.cpp transcribefile/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(TRANSCRIBEFILE_INCLUDES) $(TRANSCRIBEFILE_CPPFLAGS) -o $@ $<

$(TRANSCRIBEFILE_C_OBJS): o/$(MODE)/%.c.o: %.c transcribefile/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.c) -o $@ $<

# ==============================================================================
# Dependencies - llamafile objects for GPU support
# ==============================================================================
# Same pattern as whisperfile/BUILD.mk: llamafile.o carries the FLAG_*
# globals and DSO/app-dir helpers, gpu.a the runtime GPU loaders. The
# GPU backends register into transcribe.cpp's vendored ggml registry
# (ggml_backend_register resolves from transcribe.cpp.a; the ggml trees
# are ABI-aligned at 0.15.2). $(LLAMAFILE_METAL_SOURCES) embeds the
# patched llama.cpp ggml/metal sources in the zip store, which metal.c
# extracts and compiles into ggml-metal.dylib at runtime on macOS.

TRANSCRIBEFILE_LLAMAFILE_OBJS := \
	o/$(MODE)/llamafile/llamafile.o \
	o/$(MODE)/llamafile/gpu.a \
	o/$(MODE)/llamafile/zip.o \
	o/$(MODE)/llamafile/check_cpu.o

# ==============================================================================
# Executable
# ==============================================================================

o/$(MODE)/transcribefile/transcribefile: \
		$(TRANSCRIBEFILE_OBJS) \
		o/$(MODE)/transcribe.cpp/transcribe.cpp.a \
		$(TRANSCRIBEFILE_LLAMAFILE_OBJS) \
		$(LLAMAFILE_METAL_SOURCES)
	@mkdir -p $(@D)
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@

# ==============================================================================
# Main Target
# ==============================================================================

.PHONY: o/$(MODE)/transcribefile
o/$(MODE)/transcribefile: o/$(MODE)/transcribefile/transcribefile
