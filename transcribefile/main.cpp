// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// transcribefile entry point.
//
// Wraps transcribe.cpp's example CLI (transcribe_cli_main, exposed when
// examples/cli/main.cpp is built with -DTRANSCRIBEFILE) with the
// llamafile-style conveniences: a crash handler and, crucially, loading
// default arguments from the executable's own zip store at /zip/.args.
//
// That last bit is what makes a self-contained `foo.transcribefile`
// possible: zipalign a GGUF plus a .args file (e.g. containing
// `-m\n/zip/model.gguf`) into the executable and it runs with no
// command-line arguments. Models can still be loaded from disk, or from
// the zip explicitly via `-m /zip/<name>.gguf`.

#include <cosmo.h>
#include <stdio.h>
#include <string.h>

#include "llamafile/llamafile.h"

// Defined in transcribe.cpp/examples/cli/main.cpp, compiled with
// -DTRANSCRIBEFILE (which renames its main() to transcribe_cli_main() and
// drops the standalone main()). C++ linkage — both sides are C++.
int transcribe_cli_main(int argc, char ** argv);

#ifndef TRANSCRIBEFILE_VERSION_STRING
#define TRANSCRIBEFILE_VERSION_STRING "0.0.0-dev"
#endif

static bool has_flag(char ** argv, const char * flag) {
    for (char ** p = argv + 1; p && *p; ++p) {
        if (!strcmp(*p, flag)) {
            return true;
        }
    }
    return false;
}

// Register GPU backends with ggml before the CLI enumerates devices.
// Only Metal is wired up for now (Vulkan/CUDA are follow-up work), via
// llamafile's runtime loader: on Apple Silicon it compiles the bundled
// ggml Metal sources into ggml-metal.dylib (cached under ~/.llamafile),
// loads it with cosmo_dlopen, and registers it with transcribe.cpp's
// ggml. The CLI owns --backend semantics; this only controls which
// backends get a chance to register:
//   - cpu / cpu_accel: don't touch the GPU machinery at all
//   - auto / metal:    try Metal; on failure or non-mac hardware nothing
//                      registers and selection proceeds CPU-only
// FLAG_gpu stays AUTO (not APPLE) even for --backend metal so a failed
// load degrades silently here; transcribe.cpp reports the missing
// backend itself, with wording that matches its own CLI.
static void load_gpu_backends(char ** argv) {
    const char * backend = "auto";
    for (char ** p = argv + 1; p && *p; ++p) {
        if (!strcmp(*p, "--backend") && p[1]) {
            backend = p[1];
        }
    }
    if (strcmp(backend, "auto") != 0 && strcmp(backend, "metal") != 0) {
        FLAG_gpu = LLAMAFILE_GPU_DISABLE;
        return;
    }
    FLAG_gpu = LLAMAFILE_GPU_AUTO;
    llamafile_has_metal();
}

int main(int argc, char ** argv) {
    // Symbolized backtraces on crash (cosmopolitan).
    ShowCrashReports();

    // Answer --version before touching args or the zip store.
    if (has_flag(argv, "--version")) {
        puts("transcribefile v" TRANSCRIBEFILE_VERSION_STRING);
        return 0;
    }

    // Merge default arguments embedded at /zip/.args (if present) with the
    // user's argv. No-op for a bare executable with no bundled .args.
    argc = cosmo_args("/zip/.args", &argv);

    load_gpu_backends(argv);

    return transcribe_cli_main(argc, argv);
}
