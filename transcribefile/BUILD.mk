#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘
#
# BUILD.mk for transcribefile.
#
# For the CPU milestone the entry point is transcribe.cpp's own example
# CLI (examples/cli/main.cpp) plus its WAV loader (examples/common/wav.cpp),
# linked against the cosmocc-built transcribe.cpp.a. A llamafile-style
# wrapper (argument/zip machinery, packaged archives) can replace main.cpp
# in a later phase.

PKGS += TRANSCRIBEFILE

# ==============================================================================
# Sources (upstream example CLI + WAV loader, from the submodule)
# ==============================================================================

TRANSCRIBEFILE_SRCS_CPP := \
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

# dr_wav.h trips a few warnings that are noise here.
TRANSCRIBEFILE_CPPFLAGS := \
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
# Executable
# ==============================================================================

o/$(MODE)/transcribefile/transcribefile: \
		$(TRANSCRIBEFILE_OBJS) \
		o/$(MODE)/transcribe.cpp/transcribe.cpp.a
	@mkdir -p $(@D)
	$(LINK.o) $(TRANSCRIBEFILE_OBJS) \
		o/$(MODE)/transcribe.cpp/transcribe.cpp.a \
		$(LOADLIBES) $(LDLIBS) -o $@

# ==============================================================================
# Main Target
# ==============================================================================

.PHONY: o/$(MODE)/transcribefile
o/$(MODE)/transcribefile: o/$(MODE)/transcribefile/transcribefile
