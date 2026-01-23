#-*-mode:makefile-unix;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘
#
# SYNOPSIS
#
#   Minja Comment Parsing Tests
#
# OVERVIEW
#
#   This builds tests for minja comment parsing. The tests verify that
#   the manual string-scanning approach (used to fix regex stack overflow)
#   correctly parses Jinja comments.
#
#   Tests:
#   - minja_integration_test: Integration test using actual patched minja.hpp (runs during make check)
#
#   Examples (not run in CI):
#   - test_comment_parsing: Demo comparing regex vs manual approaches
#   - test_comment_parsing_native: Native build of demo (for comparison)
#
# USAGE
#
#   # Run CI tests
#   make check
#
#   # Build and run examples manually
#   make o/$(MODE)/tests/minja/test_comment_parsing
#   ./o/$(MODE)/tests/minja/test_comment_parsing

# ==============================================================================
# CI Test: minja_integration_test (uses actual patched minja.hpp)
# ==============================================================================
# This test includes the real minja.hpp to verify the patch works end-to-end.

MINJA_INTEGRATION_TEST_CPPFLAGS := \
	-isystem llama.cpp/vendor \
	-std=c++17

o/$(MODE)/tests/minja/minja_integration_test.o:				\
		tests/minja/minja_integration_test.cpp			\
		llama.cpp/vendor/minja/minja.hpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(MINJA_INTEGRATION_TEST_CPPFLAGS) -c -o $@ $<

o/$(MODE)/tests/minja/minja_integration_test:				\
		o/$(MODE)/tests/minja/minja_integration_test.o
	$(CXX) $(LDFLAGS) -o $@ $<

# ==============================================================================
# Example Programs (not in CI)
# ==============================================================================

# Cosmo build of demo showing regex vs manual parsing
o/$(MODE)/tests/minja/test_comment_parsing.o:				\
		tests/minja/test_comment_parsing.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -std=c++17 -c -o $@ $<

o/$(MODE)/tests/minja/test_comment_parsing:				\
		o/$(MODE)/tests/minja/test_comment_parsing.o
	$(CXX) $(LDFLAGS) -o $@ $<

# Native build of demo (for comparison with cosmo build)
o/tests/minja/test_comment_parsing_native:				\
		tests/minja/test_comment_parsing.cpp
	@mkdir -p $(@D)
	g++ -O2 -std=c++17 -o $@ $<

# ==============================================================================
# Phony targets
# ==============================================================================

.PHONY: tests/minja
tests/minja: o/$(MODE)/tests/minja/minja_integration_test.runs

.PHONY: tests/minja/examples
tests/minja/examples:							\
		o/$(MODE)/tests/minja/test_comment_parsing		\
		o/tests/minja/test_comment_parsing_native
