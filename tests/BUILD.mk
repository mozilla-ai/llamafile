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
	o/$(MODE)/llama.cpp/common/build-info.cpp.o \
	o/$(MODE)/llama.cpp/common/jinja/caps.cpp.o \
	o/$(MODE)/llama.cpp/common/jinja/lexer.cpp.o \
	o/$(MODE)/llama.cpp/common/jinja/parser.cpp.o \
	o/$(MODE)/llama.cpp/common/jinja/runtime.cpp.o \
	o/$(MODE)/llama.cpp/common/jinja/string.cpp.o \
	o/$(MODE)/llama.cpp/common/jinja/value.cpp.o

o/$(MODE)/tests/extract_data_uris_test.o: tests/extract_data_uris_test.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(TESTS_CPPFLAGS) -c -o $@ $<

o/$(MODE)/tests/extract_data_uris_test: \
		o/$(MODE)/tests/extract_data_uris_test.o \
		$(EXTRACT_DATA_URIS_TEST_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# ==============================================================================
# Test: fa_helpers_test (issue #975 numerical equivalence)
# ==============================================================================
#
# Compares llamafile_fa_vec_dot_f16 / llamafile_fa_fp16_to_fp32_row
# against upstream ggml_vec_dot_f16 / ggml_fp16_to_fp32_row on the
# same random + edge-case inputs. Catches numerical regressions in
# the alternative AVX-512F implementations we ship via sgemm.cpp's
# dispatch. On CPUs without AVX-512F the helpers report unsupported
# and the corresponding assertions are skipped.

FA_HELPERS_TEST_DEPS := \
	o/$(MODE)/llamafile/sgemm.o \
	o/$(MODE)/llamafile/llamafile.o \
	o/$(MODE)/llamafile/fa_helpers_amd_avx512f.o \
	o/$(MODE)/llamafile/fa_helpers_unsupported.o \
	o/$(MODE)/llamafile/fa_simd_gemm_amd_avx512f.o \
	o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avx.o \
	o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_fma.o \
	o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avx2.o \
	o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avxvnni.o \
	o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_avx512f.o \
	o/$(MODE)/llamafile/tinyblas_cpu_sgemm_amd_zen4.o \
	o/$(MODE)/llamafile/tinyblas_cpu_sgemm_arm80.o \
	o/$(MODE)/llamafile/tinyblas_cpu_sgemm_arm82.o \
	o/$(MODE)/llamafile/tinyblas_cpu_unsupported.o \
	o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avx.o \
	o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_fma.o \
	o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avx2.o \
	o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avxvnni.o \
	o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_avx512f.o \
	o/$(MODE)/llamafile/tinyblas_cpu_mixmul_amd_zen4.o \
	o/$(MODE)/llamafile/tinyblas_cpu_mixmul_arm80.o \
	o/$(MODE)/llamafile/tinyblas_cpu_mixmul_arm82.o \
	o/$(MODE)/llamafile/iqk_mul_mat_amd_avx2.o \
	o/$(MODE)/llamafile/iqk_mul_mat_amd_zen4.o \
	o/$(MODE)/llamafile/iqk_mul_mat_arm82.o \
	o/$(MODE)/llama.cpp/llama.cpp.a

o/$(MODE)/tests/fa_helpers_test.o: tests/fa_helpers_test.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(TESTS_CPPFLAGS) -fopenmp -c -o $@ $<

o/$(MODE)/tests/fa_helpers_test: \
		o/$(MODE)/tests/fa_helpers_test.o \
		$(FA_HELPERS_TEST_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -fopenmp -o $@ $^ $(LDLIBS)

# ==============================================================================
# Test: gpu_backend_test (issue #988 device-count gate / fallback)
# ==============================================================================
#
# Exercises the shared GPU backend probe core. The test injects stub entry
# points into a GpuBackend and provides its own doubles for gpu_backend.c's
# externs, so it links against gpu_backend.o alone — no DSO, no GPU, no
# llamafile.o/llama.cpp.a.

GPU_BACKEND_TEST_DEPS := \
	o/$(MODE)/llamafile/gpu_backend.o

o/$(MODE)/tests/gpu_backend_test.o: tests/gpu_backend_test.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(TESTS_CPPFLAGS) -c -o $@ $<

o/$(MODE)/tests/gpu_backend_test: \
		o/$(MODE)/tests/gpu_backend_test.o \
		$(GPU_BACKEND_TEST_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# ==============================================================================
# Test: backend_ops_test (upstream ggml test-backend-ops under llamafile)
# ==============================================================================
#
# Compares every ggml op on each runtime-loaded GPU backend against the CPU
# reference (NMSE thresholds). Used to verify numerical consistency of the
# dlopen'd backend DSOs (e.g. CPU vs Vulkan matmuls, issue #938 follow-up).
# Not part of the default check suite: it needs a GPU and the backend DSO at
# runtime. Build with: make o/$(MODE)/tests/backend_ops_test

# upstream test uses #include <ggml.h>, so the ggml include dirs must be on
# the system search path (-iquote from LLAMAFILE_INCLUDES is not enough)
BACKEND_OPS_INCLUDES := \
	-isystem llama.cpp/ggml/include \
	-isystem llama.cpp/include

o/$(MODE)/tests/test_backend_ops_impl.o: llama.cpp/tests/test-backend-ops.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(TESTS_CPPFLAGS) $(BACKEND_OPS_INCLUDES) -Dmain=backend_ops_main -c -o $@ $<

o/$(MODE)/tests/backend_ops_harness.o: tests/backend_ops_harness.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(TESTS_CPPFLAGS) -c -o $@ $<

o/$(MODE)/tests/backend_ops_test: \
		o/$(MODE)/tests/backend_ops_harness.o \
		o/$(MODE)/tests/test_backend_ops_impl.o \
		o/$(MODE)/llamafile/llamafile.o \
		o/$(MODE)/llamafile/gpu.a \
		o/$(MODE)/llamafile/zip.o \
		o/$(MODE)/llamafile/check_cpu.o \
		$(GGML_OBJS) \
		$(TINYBLAS_CPU_OBJS)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# ==============================================================================
# Phony targets
# ==============================================================================

.PHONY: o/$(MODE)/tests
o/$(MODE)/tests: \
	o/$(MODE)/tests/extract_data_uris_test.runs \
	o/$(MODE)/tests/fa_helpers_test.runs \
	o/$(MODE)/tests/gpu_backend_test.runs
