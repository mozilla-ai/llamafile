#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += LLAMAFILE

# ==============================================================================
# Version information
# ==============================================================================

LLAMAFILE_VERSION_MAJOR := 0
LLAMAFILE_VERSION_MINOR := 10
LLAMAFILE_VERSION_PATCH := 0
LLAMAFILE_VERSION_STRING := $(LLAMAFILE_VERSION_MAJOR).$(LLAMAFILE_VERSION_MINOR).$(LLAMAFILE_VERSION_PATCH)-dev

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
	-DLLAMAFILE_VERSION_MAJOR=$(LLAMAFILE_VERSION_MAJOR) \
	-DLLAMAFILE_VERSION_MINOR=$(LLAMAFILE_VERSION_MINOR) \
	-DLLAMAFILE_VERSION_PATCH=$(LLAMAFILE_VERSION_PATCH) \
	-DLLAMAFILE_VERSION_STRING=\"$(LLAMAFILE_VERSION_STRING)\" \
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
	llamafile/llamafile.c \
	llamafile/metal.c \
	llamafile/zip.c

LLAMAFILE_SRCS_CPP := \
	llamafile/chatbot_comm.cpp \
	llamafile/chatbot_comp.cpp \
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
	llamafile/image.cpp \
	llamafile/llama.cpp \
	llamafile/string.cpp \
	llamafile/xterm.cpp \
	$(LLAMAFILE_HIGHLIGHT_SRCS)

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

LLAMAFILE_DEPS := \
	$(GGML_OBJS) \
	$(LLAMA_OBJS) \
	$(COMMON_OBJS) \
	$(MTMD_OBJS) \
	$(HTTPLIB_OBJS) \
	$(LLAMAFILE_SERVER_SUPPORT_OBJS) \
	$(LLAMAFILE_HIGHLIGHT_KEYWORDS) \
	$(LLAMAFILE_METAL_SOURCES) \
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
# Targets
# ==============================================================================

.PHONY: o/$(MODE)/llamafile
o/$(MODE)/llamafile: o/$(MODE)/llamafile/llamafile
