#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += WHISPER_CPP

WHISPER_CPP_FILES := $(wildcard whisper.cpp/*.*)
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

o/$(MODE)/whisper.cpp/whisper.cpp.a: $(WHISPER_CPP_OBJS)

$(WHISPER_CPP_OBJS): private				\
		CCFLAGS +=				\
			-DGGML_MULTIPLATFORM

$(WHISPER_CPP_OBJS): private				\
		CXXFLAGS +=				\
			-frtti				\
			-Wno-deprecated-declarations

o/$(MODE)/whisper.cpp/main:				\
		o/$(MODE)/whisper.cpp/main.o		\
		o/$(MODE)/whisper.cpp/main.1.asc.zip.o	\
		o/$(MODE)/whisper.cpp/whisper.cpp.a	\
		o/$(MODE)/llama.cpp/llama.cpp.a		\
		o/$(MODE)/third_party/stb/stb.a		\

o/$(MODE)/whisper.cpp/stream:				\
		o/$(MODE)/whisper.cpp/whisper.cpp.a	\
		o/$(MODE)/llama.cpp/llama.cpp.a		\
		o/$(MODE)/third_party/stb/stb.a		\

o/$(MODE)/whisper.cpp/mic2txt:				\
		o/$(MODE)/whisper.cpp/whisper.cpp.a	\
		o/$(MODE)/llama.cpp/llama.cpp.a		\

o/$(MODE)/whisper.cpp/mic2raw:				\
		o/$(MODE)/whisper.cpp/whisper.cpp.a	\
		o/$(MODE)/llama.cpp/llama.cpp.a		\

o/$(MODE)/whisper.cpp/miniaudio.o: private COPTS += -O3

$(WHISPER_CPP_OBJS): whisper.cpp/BUILD.mk

.PHONY: o/$(MODE)/whisper.cpp
o/$(MODE)/whisper.cpp:					\
		o/$(MODE)/whisper.cpp/main		\
		o/$(MODE)/whisper.cpp/stream		\
		o/$(MODE)/whisper.cpp/mic2txt		\
		o/$(MODE)/whisper.cpp/mic2raw		\
