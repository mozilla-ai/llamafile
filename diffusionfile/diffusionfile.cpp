// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
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

#include <cosmo.h>
#include <stdio.h>

#include "llamafile/llamafile.h"

// LLAMAFILE_VERSION_STRING is defined by BUILD.mk
#ifndef LLAMAFILE_VERSION_STRING
#define LLAMAFILE_VERSION_STRING "0.0.0-dev"
#endif

// Forward declaration - defined in stable-diffusion.cpp/examples/cli/main.cpp
// When compiled with -DDIFFUSIONFILE, main.cpp renames main() to diffusion_cli_main()
int diffusion_cli_main(int argc, const char ** argv);

int main(int argc, char ** argv) {
    // CPU feature check and crash reports
    llamafile_check_cpu();
    ShowCrashReports();

    // Handle --version before anything else
    if (llamafile_has(argv, "--version")) {
        puts("diffusionfile v" LLAMAFILE_VERSION_STRING);
        return 0;
    }

    // Load default arguments from embedded .args file (for packaged diffusionfiles)
    argc = cosmo_args("/zip/.args", &argv);

    return diffusion_cli_main(argc, (const char **)argv);
}
