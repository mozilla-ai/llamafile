#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

SHELL = /bin/sh
MAKEFLAGS += --no-builtin-rules

.SUFFIXES:
.DELETE_ON_ERROR:
.FEATURES: output-sync

# setup target needs to run before build/config.mk checks make version
ifneq ($(MAKECMDGOALS),setup)
include build/config.mk
include build/rules.mk

include third_party/BUILD.mk
include llamafile/BUILD.mk
include llama.cpp/BUILD.mk
include stable-diffusion.cpp/BUILD.mk
include whisper.cpp/BUILD.mk
include localscore/BUILD.mk
endif

# the root package is `o//` by default
# building a package also builds its sub-packages
.PHONY: o/$(MODE)/
o/$(MODE)/:	o/$(MODE)/llamafile					\
		o/$(MODE)/llama.cpp					\
		o/$(MODE)/stable-diffusion.cpp				\
		o/$(MODE)/whisper.cpp					\
		o/$(MODE)/localscore					\
		o/$(MODE)/third_party					\
		o/$(MODE)/depend.test

# for installing to `make PREFIX=/usr/local`
.PHONY: install
install:	llamafile/zipalign.1					\
		llamafile/server/main.1					\
		llama.cpp/main/main.1					\
		llama.cpp/imatrix/imatrix.1				\
		llama.cpp/quantize/quantize.1				\
		llama.cpp/perplexity/perplexity.1			\
		llama.cpp/llava/llava-quantize.1			\
		whisper.cpp/main.1					\
		o/$(MODE)/llamafile/zipalign				\
		o/$(MODE)/llamafile/tokenize				\
		o/$(MODE)/llama.cpp/main/main				\
		o/$(MODE)/llama.cpp/imatrix/imatrix			\
		o/$(MODE)/llama.cpp/quantize/quantize			\
		o/$(MODE)/llama.cpp/llama-bench/llama-bench		\
		o/$(MODE)/localscore/localscore		\
		o/$(MODE)/llama.cpp/perplexity/perplexity		\
		o/$(MODE)/llama.cpp/llava/llava-quantize		\
		o/$(MODE)/stable-diffusion.cpp/main			\
		o/$(MODE)/whisper.cpp/main				\
		o/$(MODE)/llamafile/server/main
	mkdir -p $(PREFIX)/bin
	$(INSTALL) o/$(MODE)/llamafile/zipalign $(PREFIX)/bin/zipalign
	$(INSTALL) o/$(MODE)/llamafile/tokenize $(PREFIX)/bin/llamafile-tokenize
	$(INSTALL) o/$(MODE)/llama.cpp/main/main $(PREFIX)/bin/llamafile
	$(INSTALL) o/$(MODE)/llama.cpp/imatrix/imatrix $(PREFIX)/bin/llamafile-imatrix
	$(INSTALL) o/$(MODE)/llama.cpp/quantize/quantize $(PREFIX)/bin/llamafile-quantize
	$(INSTALL) o/$(MODE)/llama.cpp/llama-bench/llama-bench $(PREFIX)/bin/llamafile-bench
	$(INSTALL) o/$(MODE)/localscore/localscore $(PREFIX)/bin/localscore
	$(INSTALL) build/llamafile-convert $(PREFIX)/bin/llamafile-convert
	$(INSTALL) build/llamafile-upgrade-engine $(PREFIX)/bin/llamafile-upgrade-engine
	$(INSTALL) o/$(MODE)/llama.cpp/perplexity/perplexity $(PREFIX)/bin/llamafile-perplexity
	$(INSTALL) o/$(MODE)/llama.cpp/llava/llava-quantize $(PREFIX)/bin/llava-quantize
	$(INSTALL) o/$(MODE)/llamafile/server/main $(PREFIX)/bin/llamafiler
	$(INSTALL) o/$(MODE)/stable-diffusion.cpp/main $(PREFIX)/bin/sdfile
	$(INSTALL) o/$(MODE)/whisper.cpp/main $(PREFIX)/bin/whisperfile
	mkdir -p $(PREFIX)/share/man/man1
	$(INSTALL) -m 0644 llamafile/zipalign.1 $(PREFIX)/share/man/man1/zipalign.1
	$(INSTALL) -m 0644 llamafile/server/main.1 $(PREFIX)/share/man/man1/llamafiler.1
	$(INSTALL) -m 0644 llama.cpp/main/main.1 $(PREFIX)/share/man/man1/llamafile.1
	$(INSTALL) -m 0644 llama.cpp/imatrix/imatrix.1 $(PREFIX)/share/man/man1/llamafile-imatrix.1
	$(INSTALL) -m 0644 llama.cpp/quantize/quantize.1 $(PREFIX)/share/man/man1/llamafile-quantize.1
	$(INSTALL) -m 0644 llama.cpp/perplexity/perplexity.1 $(PREFIX)/share/man/man1/llamafile-perplexity.1
	$(INSTALL) -m 0644 llama.cpp/llava/llava-quantize.1 $(PREFIX)/share/man/man1/llava-quantize.1
	$(INSTALL) -m 0644 whisper.cpp/main.1 $(PREFIX)/share/man/man1/whisperfile.1

.PHONY: check
check: o/$(MODE)/llamafile/check

.PHONY: cosmocc
cosmocc: $(COSMOCC) # cosmocc toolchain setup

.PHONY: cosmocc-ci
cosmocc-ci: $(COSMOCC) $(PREFIX)/bin/ape # cosmocc toolchain setup in ci context

.PHONY: setup
setup: # Initialize and configure all dependencies (submodules, patches, etc.)
	@echo "Setting up dependencies..."
	@mkdir -p o/tmp
	@if [ ! -f whisper.cpp/.git ]; then \
		echo "Initializing whisper.cpp submodule..."; \
		git submodule update --init whisper.cpp; \
	fi
	@echo "Applying whisper.cpp patches..."
	@export TMPDIR=$$(pwd)/o/tmp && ./whisper.cpp.patches/apply-patches.sh
	@if [ ! -f stable-diffusion.cpp/.git ]; then \
		echo "Initializing stable-diffusion.cpp submodule..."; \
		git submodule update --init stable-diffusion.cpp; \
	fi
	@echo "Applying stable-diffusion.cpp patches..."
	@export TMPDIR=$$(pwd)/o/tmp && ./stable-diffusion.cpp.patches/apply-patches.sh
	@echo "Setup complete!"


ifneq ($(MAKECMDGOALS),setup)
include build/deps.mk
include build/tags.mk
endif
