#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

# ==============================================================================
# SGEMM Manual Tests (Benchmarks)
# ==============================================================================
# These are manual benchmark tests for validating sgemm kernel correctness
# and performance. They are NOT included in `make check` because:
#   - They take a long time to run (large matrices)
#   - They require manual inspection of results (ULP comparisons)
#   - They are primarily for performance benchmarking
#
# To build: make o/$(MODE)/tests/sgemm
# To run manually: ./o//tests/sgemm/sgemm_sss_test

SGEMM_TEST_CPPFLAGS := \
	$(LLAMAFILE_INCLUDES) \
	-iquote tests/sgemm

SGEMM_TEST_DEPS := \
	$(TINYBLAS_CPU_OBJS) \
	$(GGML_OBJS) \
	o/$(MODE)/llamafile/llamafile.o

# ==============================================================================
# Test: sgemm_sss_test (F32 x F32 -> F32)
# ==============================================================================

o/$(MODE)/tests/sgemm/sgemm_sss_test.o: tests/sgemm/sgemm_sss_test.cpp tests/sgemm/sgemm_test_utils.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(SGEMM_TEST_CPPFLAGS) -fopenmp -c -o $@ $<

o/$(MODE)/tests/sgemm/sgemm_sss_test: \
		o/$(MODE)/tests/sgemm/sgemm_sss_test.o \
		$(SGEMM_TEST_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -fopenmp -o $@ $^ $(LDLIBS)

# ==============================================================================
# Test: sgemm_matmul_test (various matrix sizes)
# ==============================================================================

o/$(MODE)/tests/sgemm/sgemm_matmul_test.o: tests/sgemm/sgemm_matmul_test.cpp tests/sgemm/sgemm_test_utils.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(SGEMM_TEST_CPPFLAGS) -fopenmp -c -o $@ $<

o/$(MODE)/tests/sgemm/sgemm_matmul_test: \
		o/$(MODE)/tests/sgemm/sgemm_matmul_test.o \
		$(SGEMM_TEST_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -fopenmp -o $@ $^ $(LDLIBS)

# ==============================================================================
# Test: sgemm_vecdot_test (vector dot product, n=1)
# ==============================================================================

o/$(MODE)/tests/sgemm/sgemm_vecdot_test.o: tests/sgemm/sgemm_vecdot_test.cpp tests/sgemm/sgemm_test_utils.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(SGEMM_TEST_CPPFLAGS) -fopenmp -c -o $@ $<

o/$(MODE)/tests/sgemm/sgemm_vecdot_test: \
		o/$(MODE)/tests/sgemm/sgemm_vecdot_test.o \
		$(SGEMM_TEST_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -fopenmp -o $@ $^ $(LDLIBS)

# ==============================================================================
# Test: iqk_test (Integer Quantized Kernels benchmark)
# ==============================================================================

o/$(MODE)/tests/sgemm/iqk_test.o: tests/sgemm/iqk_test.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(SGEMM_TEST_CPPFLAGS) -fopenmp -c -o $@ $<

o/$(MODE)/tests/sgemm/iqk_test: \
		o/$(MODE)/tests/sgemm/iqk_test.o \
		$(SGEMM_TEST_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -fopenmp -o $@ $^ $(LDLIBS)

# ==============================================================================
# Test: q8_0_layout_test (standalone diagnostic)
# ==============================================================================
# This is a standalone diagnostic that demonstrates why IQK code gathers
# scale values individually rather than casting to block_q8_0_x4.
# NOT included in the main test target - run manually when needed.
# See commit 474c8b6 for context.

o/$(MODE)/tests/sgemm/q8_0_layout_test.o: tests/sgemm/q8_0_layout_test.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(SGEMM_TEST_CPPFLAGS) -c -o $@ $<

o/$(MODE)/tests/sgemm/q8_0_layout_test: \
		o/$(MODE)/tests/sgemm/q8_0_layout_test.o
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# ==============================================================================
# Phony target to build all sgemm tests
# ==============================================================================

.PHONY: o/$(MODE)/tests/sgemm
o/$(MODE)/tests/sgemm: \
	o/$(MODE)/tests/sgemm/sgemm_sss_test \
	o/$(MODE)/tests/sgemm/sgemm_matmul_test \
	o/$(MODE)/tests/sgemm/sgemm_vecdot_test \
	o/$(MODE)/tests/sgemm/iqk_test
