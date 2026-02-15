// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2023 Mozilla Foundation
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

#include <cosmo.h>
#include <errno.h>
#include <stdlib.h>

/**
 * Dies if CPU doesn't have mandatory features.
 *
 * This check is based on which `$(TARGET_ARCH)` microarchitecture
 * features were used globally. Object files that are specifically
 * written to use runtime dispatching should be configured so that
 * microarchitecture flags only get passed to that specific object
 */
void llamafile_check_cpu(void) {
    // side effect: the constructor below has now been linked
}

static int on_missing_feature(const char *name) {
    tinyprint(2, GetProgramExecutableName(), ": fatal error: the cpu feature ", name,
              " was required at build time but isn't available on this system\n", NULL);
#if defined(__AVX2__) && !defined(__AVX512F__)
    tinyprint(2,
              "note: amd microprocessors made after 2017 usually work\n"
              "note: intel microprocessors made after 2013 usually work\n",
              NULL);
#endif
    tinyprint(2, "exiting process.\n", NULL);
    _Exit(1);
}

// We need to perform this early in the initialization process, before
// C++ codes built with -mavx has a chance to start allocating dynamic
// memory that would otherwise crash the cpu before this could be done
__attribute__((__constructor__(101))) static void llamafile_actually_check_cpu(void) {
    if (X86_NEED(SSE3) && !X86_CHECK(SSE3)) {
        on_missing_feature("SSE3");
    }
    if (X86_NEED(SSSE3) && !X86_CHECK(SSSE3)) {
        on_missing_feature("SSSE3");
    }
    if (X86_NEED(AVX) && !X86_CHECK(AVX)) {
        on_missing_feature("AVX");
    }
    if (X86_NEED(AVX2) && !X86_CHECK(AVX2)) {
        on_missing_feature("AVX2");
    }
    if (X86_NEED(FMA) && !X86_CHECK(FMA)) {
        on_missing_feature("FMA");
    }
    if (X86_NEED(F16C) && !X86_CHECK(F16C)) {
        on_missing_feature("F16C");
    }
    if (X86_NEED(AVX512F) && !X86_CHECK(AVX512F)) {
        on_missing_feature("AVX512F");
    }
    if (X86_NEED(AVX512VBMI) && !X86_CHECK(AVX512VBMI)) {
        on_missing_feature("AVX512VBMI");
    }
    if (X86_NEED(AVX512_VNNI) && !X86_CHECK(AVX512_VNNI)) {
        on_missing_feature("AVX512_VNNI");
    }
}
