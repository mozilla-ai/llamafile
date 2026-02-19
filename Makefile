#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

SHELL = /bin/sh
MAKEFLAGS += --no-builtin-rules

.SUFFIXES:
.DELETE_ON_ERROR:
.FEATURES: output-sync

# setup and reset-repo targets need to run before build/config.mk checks make version
ifeq ($(filter $(MAKECMDGOALS),setup reset-repo claude),)
include build/config.mk
include build/rules.mk

include third_party/BUILD.mk
include llama.cpp/BUILD.mk
include llamafile/BUILD.mk
include tests/BUILD.mk
endif

# the root package is `o//` by default
# building a package also builds its sub-packages
.PHONY: o/$(MODE)/
o/$(MODE)/:	o/$(MODE)/llamafile	\
		o/$(MODE)/llama.cpp \
		o/$(MODE)/third_party/zipalign

.PHONY: install
install: o/$(MODE)/llamafile/llamafile
	mkdir -p $(PREFIX)/bin
	$(INSTALL) o/$(MODE)/llamafile/llamafile $(PREFIX)/bin/llamafile

.PHONY: check
check: o/$(MODE)/tests

# ==============================================================================
# GPU Backend Targets
# ==============================================================================
# These targets build GPU backend shared libraries that can be loaded at runtime.
# They pass GGML_VERSION and GGML_COMMIT from build/config.mk to the build scripts.

.PHONY: cuda
cuda: # Build CUDA backend with TinyBLAS (NVIDIA GPUs)
	GGML_VERSION=$(GGML_VERSION) GGML_COMMIT=$(GGML_COMMIT) llamafile/cuda.sh

.PHONY: cublas
cublas: # Build CUDA backend with cuBLAS (NVIDIA GPUs, requires cuBLAS at runtime)
	GGML_VERSION=$(GGML_VERSION) GGML_COMMIT=$(GGML_COMMIT) llamafile/cuda.sh --cublas

.PHONY: rocm
rocm: # Build ROCm backend with TinyBLAS (AMD GPUs)
	GGML_VERSION=$(GGML_VERSION) GGML_COMMIT=$(GGML_COMMIT) llamafile/rocm.sh

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

	@if [ ! -f third_party/zipalign/.git ]; then \
		echo "Initializing zipalign submodule..."; \
		git submodule update --init third_party/zipalign; \
	fi
	@echo "Setup complete!"

.PHONY: reset-repo
reset-repo: # Reset all submodules to their original state (removes patches or any other change)
	@echo "Resetting submodules to original state..."
	@for dir in llama.cpp whisper.cpp stable-diffusion.cpp third_party/zipalign; do \
		if [ -e "$$dir" ]; then \
			echo "Removing $$dir..."; \
			rm -rf "$$dir"; \
		fi; \
		echo "Restoring $$dir..."; \
		git checkout "$$dir"; \
	done
	@echo "Reset complete. Run 'make setup' to reinitialize and apply patches."

.PHONY: claude
claude: # Set up Claude Code plugin and CLAUDE.md symlink
	@echo "Setting up Claude Code llamafile plugin..."
	@mkdir -p .claude/plugins/llamafile
	@for dir in skills commands; do \
		if [ -e .claude/plugins/llamafile/$$dir ] && [ ! -L .claude/plugins/llamafile/$$dir ]; then \
			echo "Error: .claude/plugins/llamafile/$$dir exists and is not a symlink"; \
			exit 1; \
		fi; \
		rm -f .claude/plugins/llamafile/$$dir; \
		ln -s ../../../docs/$$dir .claude/plugins/llamafile/$$dir; \
	done
	@if [ ! -f .claude/plugins/llamafile/plugin.json ]; then \
		echo '{"name":"llamafile","version":"0.1.0","description":"Build guidance and commands for the llamafile project"}' > .claude/plugins/llamafile/plugin.json; \
	fi
	@if [ -e CLAUDE.md ] && [ ! -L CLAUDE.md ]; then \
		echo "Error: CLAUDE.md exists and is not a symlink"; \
		exit 1; \
	fi
	@rm -f CLAUDE.md
	@ln -s docs/AGENTS.md CLAUDE.md
	@echo ""
	@echo "Claude Code configured:"
	@echo "  CLAUDE.md -> docs/AGENTS.md"
	@echo "  .claude/plugins/llamafile/skills -> ../../../docs/skills"
	@echo "  .claude/plugins/llamafile/commands -> ../../../docs/commands"
	@echo ""
	@echo "To enable the plugin, run: tools/enable-claude-plugin.sh"
	@echo "Available commands after enabling: /build"

ifeq ($(filter $(MAKECMDGOALS),setup reset-repo claude),)
include build/deps.mk
include build/tags.mk
endif
