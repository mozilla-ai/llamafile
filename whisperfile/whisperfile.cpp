// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
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

#include <cosmo.h>
#include <stdio.h>

#include "llamafile/llamafile.h"

// LLAMAFILE_VERSION_STRING is defined by BUILD.mk
#ifndef LLAMAFILE_VERSION_STRING
#define LLAMAFILE_VERSION_STRING "0.0.0-dev"
#endif

// Forward declaration - defined in whisper.cpp/examples/cli/cli.cpp
// When compiled with -DWHISPERFILE, cli.cpp renames main() to whisper_cli_main()
int whisper_cli_main(int argc, char ** argv);

int main(int argc, char ** argv) {
    // CPU feature check and crash reports
    llamafile_check_cpu();
    ShowCrashReports();

    // Handle --version before anything else
    if (llamafile_has(argv, "--version")) {
        puts("whisperfile v" LLAMAFILE_VERSION_STRING);
        return 0;
    }

    // Load default arguments from embedded .args file (for packaged whisperfiles)
    argc = cosmo_args("/zip/.args", &argv);

    // Future: GPU initialization
    // When GPU support is added, uncomment:
    // llamafile_has_gpu();

    return whisper_cli_main(argc, argv);
}
