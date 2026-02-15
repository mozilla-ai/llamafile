#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘
#
# BUILD.mk for whisperfile tools
#
# whisperfile provides llamafile-integrated whisper tools:
#   - whisperfile: CLI transcription with llamafile features
#   - whisper-server: HTTP API server for transcription
#   - stream: Real-time microphone transcription
#   - mic2txt: Record-then-transcribe
#   - mic2raw: Debug tool with raw token output
#

PKGS += WHISPERFILE

# ==============================================================================
# Source files
# ==============================================================================

WHISPERFILE_SRCS := \
	whisperfile/whisperfile.cpp \
	whisperfile/slurp.cpp \
	whisperfile/color.cpp

WHISPERFILE_STREAM_SRCS := \
	whisperfile/stream.cpp \
	whisperfile/color.cpp

WHISPERFILE_MIC2TXT_SRCS := \
	whisperfile/mic2txt.cpp \
	whisperfile/color.cpp

WHISPERFILE_MIC2RAW_SRCS := \
	whisperfile/mic2raw.cpp \
	whisperfile/color.cpp

WHISPERFILE_SERVER_SRCS := \
	whisperfile/whisper-server.cpp \
	whisperfile/slurp.cpp

# ==============================================================================
# Object files
# ==============================================================================

WHISPERFILE_OBJS := $(WHISPERFILE_SRCS:%.cpp=o/$(MODE)/%.o)
WHISPERFILE_STREAM_OBJS := $(WHISPERFILE_STREAM_SRCS:%.cpp=o/$(MODE)/%.o)
WHISPERFILE_MIC2TXT_OBJS := $(WHISPERFILE_MIC2TXT_SRCS:%.cpp=o/$(MODE)/%.o)
WHISPERFILE_MIC2RAW_OBJS := $(WHISPERFILE_MIC2RAW_SRCS:%.cpp=o/$(MODE)/%.o)
WHISPERFILE_SERVER_OBJS := $(WHISPERFILE_SERVER_SRCS:%.cpp=o/$(MODE)/%.o)

# ==============================================================================
# Include paths
# ==============================================================================

WHISPERFILE_INCS := \
	-iquote . \
	-iquote whisperfile \
	-iquote whisper.cpp/include \
	-iquote whisper.cpp/src \
	-iquote whisper.cpp/examples \
	-iquote whisper.cpp/ggml/include

# ==============================================================================
# Compiler flags
# ==============================================================================

WHISPERFILE_CPPFLAGS := \
	$(WHISPERFILE_INCS) \
	-DLLAMAFILE_VERSION_STRING=\"$(LLAMAFILE_VERSION_STRING)\"

# ==============================================================================
# Dependencies - llamafile objects for GPU support
# ==============================================================================
# Same pattern as llama.cpp/BUILD.mk TOOL_LLAMAFILE_OBJS

WHISPERFILE_LLAMAFILE_OBJS := \
	o/$(MODE)/llamafile/llamafile.o \
	o/$(MODE)/llamafile/metal.o \
	o/$(MODE)/llamafile/cuda.o \
	o/$(MODE)/llamafile/zip.o \
	o/$(MODE)/llamafile/check_cpu.o

# ==============================================================================
# Compilation rules
# ==============================================================================

o/$(MODE)/whisperfile/%.o: whisperfile/%.cpp whisperfile/BUILD.mk
	@mkdir -p $(@D)
	$(COMPILE.cc) $(WHISPERFILE_CPPFLAGS) -frtti -o $@ $<

# ==============================================================================
# Executable - whisperfile (llamafile-integrated CLI)
# ==============================================================================
# Links cli.cpp compiled with -DWHISPERFILE to exclude its main()

o/$(MODE)/whisperfile/whisperfile: \
		$(WHISPERFILE_OBJS) \
		o/$(MODE)/whisper.cpp/examples/cli/cli.whisperfile.cpp.o \
		o/$(MODE)/whisper.cpp/whisper.cpp.a \
		$(WHISPERFILE_LLAMAFILE_OBJS)
	@mkdir -p $(@D)
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@

# ==============================================================================
# Executable - stream (real-time microphone transcription)
# ==============================================================================

o/$(MODE)/whisperfile/stream: \
		$(WHISPERFILE_STREAM_OBJS) \
		o/$(MODE)/whisper.cpp/whisper.cpp.a \
		$(WHISPERFILE_LLAMAFILE_OBJS)
	@mkdir -p $(@D)
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@

# ==============================================================================
# Executable - mic2txt (record then transcribe)
# ==============================================================================

o/$(MODE)/whisperfile/mic2txt: \
		$(WHISPERFILE_MIC2TXT_OBJS) \
		o/$(MODE)/whisper.cpp/whisper.cpp.a \
		$(WHISPERFILE_LLAMAFILE_OBJS)
	@mkdir -p $(@D)
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@

# ==============================================================================
# Executable - mic2raw (debug token output)
# ==============================================================================

o/$(MODE)/whisperfile/mic2raw: \
		$(WHISPERFILE_MIC2RAW_OBJS) \
		o/$(MODE)/whisper.cpp/whisper.cpp.a \
		$(WHISPERFILE_LLAMAFILE_OBJS)
	@mkdir -p $(@D)
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@

# ==============================================================================
# Executable - whisper-server (HTTP API server)
# ==============================================================================

o/$(MODE)/whisperfile/whisper-server: \
		$(WHISPERFILE_SERVER_OBJS) \
		o/$(MODE)/whisper.cpp/examples/server/server.whisperfile.cpp.o \
		o/$(MODE)/whisper.cpp/whisper.cpp.a \
		$(WHISPERFILE_LLAMAFILE_OBJS)
	@mkdir -p $(@D)
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@

# ==============================================================================
# Main target
# ==============================================================================

.PHONY: o/$(MODE)/whisperfile
o/$(MODE)/whisperfile: \
	o/$(MODE)/whisperfile/whisperfile \
	o/$(MODE)/whisperfile/whisper-server \
	o/$(MODE)/whisperfile/stream \
	o/$(MODE)/whisperfile/mic2txt \
	o/$(MODE)/whisperfile/mic2raw
