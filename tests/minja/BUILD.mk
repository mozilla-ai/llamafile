#-*-mode:makefile-unix;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘
#
# SYNOPSIS
#
#   Minja Regex Stack Overflow Test
#
# OVERVIEW
#
#   This builds test programs demonstrating a regex stack overflow bug
#   in Jinja comment parsing. The bug manifests when using Cosmopolitan
#   Libc's C++ standard library implementation with recursive regex engines.
#
#   The test builds two versions:
#   - Native: Uses system g++ (works correctly)
#   - Cosmo: Uses cosmopolitan compiler (crashes on regex approach)
#
# USAGE
#
#   # Build tests
#   make -j8 o/tests/minja/test_comment_parsing
#   make -j8 o/tests/minja/test_comment_parsing_native
#
#   # Run native version (should pass both tests)
#   ./o/tests/minja/test_comment_parsing_native
#
#   # Run cosmo version (crashes on regex test with default 15KB comment)
#   ./o/tests/minja/test_comment_parsing
#

TESTS_MINJA_SRCS = \
	tests/minja/test_comment_parsing.cpp

TESTS_MINJA_OBJS = \
	$(TESTS_MINJA_SRCS:%.cpp=o/$(MODE)/%.o)

# Cosmopolitan build (default)
o/$(MODE)/tests/minja/test_comment_parsing:				\
		o/$(MODE)/tests/minja/test_comment_parsing.o
	$(CXX) $(LDFLAGS) -o $@ $<

o/$(MODE)/tests/minja/test_comment_parsing.o:				\
		tests/minja/test_comment_parsing.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -std=c++17 -c -o $@ $<

# Native build (for comparison)
o/tests/minja/test_comment_parsing_native:				\
		tests/minja/test_comment_parsing.cpp
	@mkdir -p $(@D)
	g++ -O2 -std=c++17 -o $@ $<

# Convenience targets
.PHONY: tests/minja
tests/minja:								\
		o/$(MODE)/tests/minja/test_comment_parsing		\
		o/tests/minja/test_comment_parsing_native

.PHONY: tests/minja/native
tests/minja/native: o/tests/minja/test_comment_parsing_native

.PHONY: tests/minja/cosmo
tests/minja/cosmo: o/$(MODE)/tests/minja/test_comment_parsing
