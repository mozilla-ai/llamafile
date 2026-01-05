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
include llama.cpp/BUILD.mk
include llamafile/BUILD.mk
endif

# the root package is `o//` by default
# building a package also builds its sub-packages
.PHONY: o/$(MODE)/
o/$(MODE)/:	o/$(MODE)/llamafile					\
		o/$(MODE)/llama.cpp

.PHONY: install
install: o/$(MODE)/llamafile/llamafile
	mkdir -p $(PREFIX)/bin
	$(INSTALL) o/$(MODE)/llamafile/llamafile $(PREFIX)/bin/llamafile

.PHONY: check
check:
	@echo "No tests defined yet for new llamafile build"

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

	@if [ ! -f llama.cpp/.git ]; then \
		echo "Initializing llama.cpp submodule..."; \
		git submodule update --init llama.cpp; \
	fi
	@echo "Initializing llama.cpp dependencies (nested submodules)..."
	@cd llama.cpp && git submodule update --init
	@echo "Applying llama.cpp patches..."
	@export TMPDIR=$$(pwd)/o/tmp && ./llama.cpp.patches/apply-patches.sh
	@echo "Setup complete!"

ifneq ($(MAKECMDGOALS),setup)
include build/deps.mk
include build/tags.mk
endif
