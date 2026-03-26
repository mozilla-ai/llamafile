#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

PKGS += TESTS

include tests/sgemm/BUILD.mk
include tests/strsm/BUILD.mk

# ==============================================================================
# Include paths (reuse llamafile includes)
# ==============================================================================

TESTS_CPPFLAGS := $(LLAMAFILE_INCLUDES)

# ==============================================================================
# Test: extract_data_uris_test
# ==============================================================================

# Dependencies for extract_data_uris test:
#   - extract_data_uris.o: contains extract_data_uris function (isolated)
#   - datauri.o: DataUri class for parsing data URIs
#   - image.o: is_image function for validating images
#   - string.o: lf::startscasewith helper
#   - xterm.o: terminal utilities (required by image.o)
#   - stb.a: stb_image for image validation

EXTRACT_DATA_URIS_TEST_DEPS := \
	o/$(MODE)/llamafile/extract_data_uris.o \
	o/$(MODE)/llamafile/datauri.o \
	o/$(MODE)/llamafile/image.o \
	o/$(MODE)/llamafile/string.o \
	o/$(MODE)/llamafile/xterm.o \
	o/$(MODE)/third_party/stb/stb.a \
	o/$(MODE)/llama.cpp/common/build-info.cpp.o

o/$(MODE)/tests/extract_data_uris_test.o: tests/extract_data_uris_test.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(TESTS_CPPFLAGS) -c -o $@ $<

o/$(MODE)/tests/extract_data_uris_test: \
		o/$(MODE)/tests/extract_data_uris_test.o \
		$(EXTRACT_DATA_URIS_TEST_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# ==============================================================================
# Phony targets
# ==============================================================================

.PHONY: o/$(MODE)/tests
o/$(MODE)/tests: \
	o/$(MODE)/tests/extract_data_uris_test.runs
