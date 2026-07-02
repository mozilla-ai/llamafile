// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
// Unit tests for the shared GPU backend probe core (llamafile/gpu_backend.c).
//
// The device-count gate exercised here is the fix for issue #988: a backend
// DSO that loads but exposes no usable device must be rejected (and unlinked)
// so AUTO mode can fall through to the next backend and ultimately to CPU,
// instead of registering a dead backend and crashing / blocking fallback.
//
// Because GpuBackend stores its DSO entry points as plain pointers, we can
// drive gpu_backend_probe()/gpu_backend_register() with injected stub
// functions and never touch a real DSO or GPU. The handful of externs
// gpu_backend.c needs (FLAG_verbose, llamafile_info, llamafile_log_callback_null,
// ggml_backend_register) are provided here as test doubles so the test links
// against gpu_backend.o alone.
//

#include "gpu_backend.h"
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Test doubles for gpu_backend.c's external dependencies
// ---------------------------------------------------------------------------

extern "C" {

int FLAG_verbose;

void llamafile_info(const char *, const char *, ...) {
    // swallow log output during tests
}

void llamafile_log_callback_null(int, const char *, void *) {}

// Records the last registry handed to ggml so register tests can assert on it.
static ggml_backend_reg_t g_last_registered = (ggml_backend_reg_t)-1;
void ggml_backend_register(ggml_backend_reg_t reg) {
    g_last_registered = reg;
}

} // extern "C"

// ---------------------------------------------------------------------------
// Injected backend entry points (stubs)
// ---------------------------------------------------------------------------

static int dc_two(void) { return 2; }
static int dc_zero(void) { return 0; }
static int dc_negative(void) { return -1; }

// Stand-ins for a backend whose get_device_count faults during driver init:
// a SIGSEGV (the cosmo-translated form of the #988 C++ exception) and a
// SIGABRT (the GGML_ASSERT -> abort() path).
static int dc_segv(void) {
    raise(SIGSEGV);
    return 0;
}
static int dc_abort(void) {
    abort();
    return 0;
}

static int g_log_set_calls;
static void log_set_spy(llamafile_log_callback, void *) { ++g_log_set_calls; }

static ggml_backend_reg_t g_fake_reg = (ggml_backend_reg_t)0x1234;
static ggml_backend_reg_t reg_ok(void) { return g_fake_reg; }
static ggml_backend_reg_t reg_null(void) { return NULL; }

static const GpuBackendDesc TEST_DESC = {
    /*.tag*/ "test",
    /*.name*/ "Test",
    /*.init*/ "x_init",
    /*.reg*/ "x_reg",
    /*.get_device_count*/ "x_get_device_count",
    /*.get_device_description*/ "x_get_device_description",
};

// A non-NULL sentinel for symbol pointers, so we can assert unlink() cleared
// them. lib_handle stays NULL so unlink() doesn't call cosmo_dlclose() on it.
static void *const SENTINEL = (void *)0xABCD;

// Fills `b` in place (GpuBackend holds an atomic_uint, so it is not copyable
// in C++ and cannot be returned by value).
static void init_linked_backend(GpuBackend *b, void *get_device_count, void *log_set) {
    memset(b, 0, sizeof(*b));
    b->desc = &TEST_DESC;
    b->lib_handle = NULL; // pretend "loaded" without a real dlopen handle
    b->backend_init = SENTINEL;
    b->backend_reg = SENTINEL;
    b->get_device_count = get_device_count;
    b->get_device_description = SENTINEL;
    b->log_set = log_set;
}

static bool is_unlinked(const GpuBackend *b) {
    return b->lib_handle == NULL && b->backend_init == NULL && b->backend_reg == NULL &&
           b->get_device_count == NULL && b->get_device_description == NULL &&
           b->log_set == NULL;
}

// ---------------------------------------------------------------------------
// Tiny test harness
// ---------------------------------------------------------------------------

static int g_failures;
static int g_checks;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
        }                                                                        \
    } while (0)

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// A backend with >0 devices is accepted and left intact for registration.
static void test_probe_accepts_when_devices_present(void) {
    FLAG_verbose = 1; // skip suppression branch
    GpuBackend b;
    init_linked_backend(&b, (void *)dc_two, (void *)log_set_spy);
    CHECK(gpu_backend_probe(&b) == true);
    CHECK(b.backend_init == SENTINEL);       // still linked
    CHECK(b.get_device_count == (void *)dc_two);
}

// The #988 regression guard: a loaded-but-0-device backend is rejected AND
// unlinked, so the caller falls through to the next backend / CPU.
static void test_probe_rejects_and_unlinks_when_zero_devices(void) {
    FLAG_verbose = 1;
    GpuBackend b;
    init_linked_backend(&b, (void *)dc_zero, (void *)log_set_spy);
    CHECK(gpu_backend_probe(&b) == false);
    CHECK(is_unlinked(&b));
    CHECK(b.desc == &TEST_DESC); // identity is preserved across unlink
}

// A negative device count is treated the same as zero.
static void test_probe_rejects_on_negative_count(void) {
    FLAG_verbose = 1;
    GpuBackend b;
    init_linked_backend(&b, (void *)dc_negative, (void *)log_set_spy);
    CHECK(gpu_backend_probe(&b) == false);
    CHECK(is_unlinked(&b));
}

// A missing get_device_count symbol means we can't confirm any device, so the
// backend is rejected rather than optimistically registered.
static void test_probe_rejects_when_device_count_symbol_absent(void) {
    FLAG_verbose = 1;
    GpuBackend b;
    init_linked_backend(&b, /*get_device_count=*/NULL, (void *)log_set_spy);
    CHECK(gpu_backend_probe(&b) == false);
    CHECK(is_unlinked(&b));
}

// Without --verbose, the DSO's logging is suppressed before device enumeration;
// with --verbose it is left alone.
static void test_probe_log_suppression_respects_verbose(void) {
    g_log_set_calls = 0;
    FLAG_verbose = 0;
    GpuBackend quiet;
    init_linked_backend(&quiet, (void *)dc_two, (void *)log_set_spy);
    gpu_backend_probe(&quiet);
    CHECK(g_log_set_calls == 1); // suppression installed

    g_log_set_calls = 0;
    FLAG_verbose = 1;
    GpuBackend loud;
    init_linked_backend(&loud, (void *)dc_two, (void *)log_set_spy);
    gpu_backend_probe(&loud);
    CHECK(g_log_set_calls == 0); // left alone
}

// The crash boundary (issue #988): a get_device_count that faults (SIGSEGV)
// must be caught and turned into a clean rejection + unlink, so AUTO mode falls
// through to the next backend / CPU instead of the process dying.
static void test_probe_survives_segv_in_device_count(void) {
    FLAG_verbose = 1;
    GpuBackend b;
    init_linked_backend(&b, (void *)dc_segv, (void *)log_set_spy);
    CHECK(gpu_backend_probe(&b) == false);
    CHECK(is_unlinked(&b));
}

// Same for a GGML_ASSERT-style abort() (SIGABRT) during probing.
static void test_probe_survives_abort_in_device_count(void) {
    FLAG_verbose = 1;
    GpuBackend b;
    init_linked_backend(&b, (void *)dc_abort, (void *)log_set_spy);
    CHECK(gpu_backend_probe(&b) == false);
    CHECK(is_unlinked(&b));
}

// After a guarded fault, normal probing must still work (handlers restored, no
// lingering signal mask / disposition).
static void test_probe_works_after_guarded_fault(void) {
    FLAG_verbose = 1;
    GpuBackend crashed;
    init_linked_backend(&crashed, (void *)dc_segv, (void *)log_set_spy);
    gpu_backend_probe(&crashed); // faults, handled

    GpuBackend good;
    init_linked_backend(&good, (void *)dc_two, (void *)log_set_spy);
    CHECK(gpu_backend_probe(&good) == true);
}

// register() forwards a real registry to ggml, and skips a NULL one.
static void test_register_forwards_non_null_reg(void) {
    g_last_registered = (ggml_backend_reg_t)-1;
    GpuBackend b;
    init_linked_backend(&b, (void *)dc_two, (void *)log_set_spy);
    b.backend_reg = (void *)reg_ok;
    gpu_backend_register(&b);
    CHECK(g_last_registered == g_fake_reg);
}

static void test_register_skips_null_reg(void) {
    g_last_registered = (ggml_backend_reg_t)-1;
    GpuBackend b;
    init_linked_backend(&b, (void *)dc_two, (void *)log_set_spy);
    b.backend_reg = (void *)reg_null;
    gpu_backend_register(&b);
    CHECK(g_last_registered == (ggml_backend_reg_t)-1); // never called
}

// The ABI call helpers tolerate NULL pointers like the old per-backend wrappers.
static void test_call_helpers_tolerate_null(void) {
    char buf[32] = "untouched";
    CHECK(gpu_call_reg(NULL) == NULL);
    CHECK(gpu_call_device_count(NULL) == 0);
    gpu_call_get_description(NULL, 0, buf, sizeof(buf)); // must not crash/modify
    CHECK(strcmp(buf, "untouched") == 0);
    gpu_call_log_set(NULL, llamafile_log_callback_null, NULL); // must not crash
    CHECK(gpu_call_device_count((void *)dc_two) == 2);         // and dispatches when set
}

int main(void) {
    test_probe_accepts_when_devices_present();
    test_probe_rejects_and_unlinks_when_zero_devices();
    test_probe_rejects_on_negative_count();
    test_probe_rejects_when_device_count_symbol_absent();
    test_probe_log_suppression_respects_verbose();
    test_probe_survives_segv_in_device_count();
    test_probe_survives_abort_in_device_count();
    test_probe_works_after_guarded_fault();
    test_register_forwards_non_null_reg();
    test_register_skips_null_reg();
    test_call_helpers_tolerate_null();

    if (g_failures) {
        fprintf(stderr, "gpu_backend_test: %d/%d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("gpu_backend_test: all %d checks PASSED\n", g_checks);
    return 0;
}
