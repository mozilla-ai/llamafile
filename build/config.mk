#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PREFIX = /usr/local
COSMOCC = .cosmocc/4.0.2
TOOLCHAIN = $(COSMOCC)/bin/cosmo

CC = $(TOOLCHAIN)cc
CXX = $(TOOLCHAIN)c++
AR = $(COSMOCC)/bin/ar.ape
ZIPOBJ = $(COSMOCC)/bin/zipobj
MKDEPS = $(COSMOCC)/bin/mkdeps
INSTALL = install

ARFLAGS = rcsD
CXXFLAGS = -frtti -std=gnu++23
CCFLAGS = -O2 -g -fexceptions -ffunction-sections -fdata-sections -mclang
CPPFLAGS_ = -iquote. -mcosmo -DGGML_MULTIPLATFORM -Wno-attributes -DLLAMAFILE_DEBUG
TARGET_ARCH = -Xx86_64-mtune=znver4

TMPDIR = o//tmp
IGNORE := $(shell mkdir -p $(TMPDIR))
ARCH := $(shell uname -m)

# apple still distributes a 17 year old version of gnu make
ifeq ($(MAKE_VERSION), 3.81)
$(error please use bin/make from cosmocc.zip rather than old xcode make)
endif

# let `make m=foo` be shorthand for `make MODE=foo`
ifneq ($(m),)
ifeq ($(MODE),)
MODE := $(m)
endif
endif

# make build more deterministic
LC_ALL = C.UTF-8
SOURCE_DATE_EPOCH = 0
export MODE
export TMPDIR
export LC_ALL
export SOURCE_DATE_EPOCH

# `make` runs `make all` by default
.PHONY: all
all: o/$(MODE)/

.PHONY: clean
clean:; rm -rf o

.PHONY: distclean
distclean:; rm -rf o .cosmocc

.cosmocc/3.9.7:
	build/download-cosmocc.sh $@ 3.9.7 3f559555d08ece35bab1a66293a2101f359ac9841d563419756efa9c79f7a150

.cosmocc/4.0.2:
	build/download-cosmocc.sh $@ 4.0.2 85b8c37a406d862e656ad4ec14be9f6ce474c1b436b9615e91a55208aced3f44
