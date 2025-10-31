#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += WHISPER_CPP

WHISPER_CPP_FILES := $(wildcard whisper.cpp.submodule/*.*)
WHISPER_CPP_INCS = $(filter %.inc,$(WHISPER_CPP_FILES))
WHISPER_CPP_SRCS_C = $(filter %.c,$(WHISPER_CPP_FILES))
WHISPER_CPP_SRCS_CPP = $(filter %.cpp,$(WHISPER_CPP_FILES))
WHISPER_CPP_SRCS = $(WHISPER_CPP_SRCS_C) $(WHISPER_CPP_SRCS_CPP)

WHISPER_CPP_HDRS =					\
	$(filter %.h,$(WHISPER_CPP_FILES))		\
	$(filter %.hpp,$(WHISPER_CPP_FILES))

WHISPER_CPP_OBJS =					\
	$(WHISPER_CPP_SRCS_C:%.c=o/$(MODE)/%.o)		\
	$(WHISPER_CPP_SRCS_CPP:%.cpp=o/$(MODE)/%.o)

o/$(MODE)/whisper.cpp.submodule/whisper.cpp.a: $(WHISPER_CPP_OBJS)

$(WHISPER_CPP_OBJS): private				\
		CCFLAGS +=				\
			-DGGML_MULTIPLATFORM

$(WHISPER_CPP_OBJS): private				\
		CXXFLAGS +=				\
			-frtti				\
			-Wno-deprecated-declarations

o/$(MODE)/whisper.cpp.submodule/main:				\
		o/$(MODE)/whisper.cpp.submodule/main.o		\
		o/$(MODE)/whisper.cpp.submodule/main.1.asc.zip.o	\
		o/$(MODE)/whisper.cpp.submodule/whisper.cpp.a	\
		o/$(MODE)/llama.cpp/llama.cpp.a		\
		o/$(MODE)/third_party/stb/stb.a		\

o/$(MODE)/whisper.cpp.submodule/stream:				\
		o/$(MODE)/whisper.cpp.submodule/whisper.cpp.a	\
		o/$(MODE)/llama.cpp/llama.cpp.a		\
		o/$(MODE)/third_party/stb/stb.a		\

o/$(MODE)/whisper.cpp.submodule/mic2txt:				\
		o/$(MODE)/whisper.cpp.submodule/whisper.cpp.a	\
		o/$(MODE)/llama.cpp/llama.cpp.a		\

o/$(MODE)/whisper.cpp.submodule/mic2raw:				\
		o/$(MODE)/whisper.cpp.submodule/whisper.cpp.a	\
		o/$(MODE)/llama.cpp/llama.cpp.a		\

o/$(MODE)/whisper.cpp.submodule/miniaudio.o: private COPTS += -O3

$(WHISPER_CPP_OBJS): whisper.cpp.submodule/BUILD.mk

.PHONY: o/$(MODE)/whisper.cpp.submodule
o/$(MODE)/whisper.cpp.submodule:					\
		o/$(MODE)/whisper.cpp.submodule/main		\
		o/$(MODE)/whisper.cpp.submodule/stream		\
		o/$(MODE)/whisper.cpp.submodule/mic2txt		\
		o/$(MODE)/whisper.cpp.submodule/mic2raw		\
