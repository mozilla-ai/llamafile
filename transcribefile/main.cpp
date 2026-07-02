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
#include <stdlib.h>
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

// Remove `flag` from argv (first occurrence) and report whether it was
// there. For flags the wrapper owns, like --verbose: upstream's CLI
// errors out on options it doesn't know, so these must not be forwarded.
static bool consume_flag(char ** argv, const char * flag) {
    for (char ** p = argv + 1; p && *p; ++p) {
        if (!strcmp(*p, flag)) {
            do {
                p[0] = p[1];
            } while (*p++);
            return true;
        }
    }
    return false;
}

// Appended to upstream's usage text (same stream: stderr) so the flags
// and behaviors added by this wrapper are discoverable from --help.
static void print_wrapper_help(void) {
    fprintf(stderr,
            "\n"
            "transcribefile options:\n"
            "  --version             print transcribefile version and exit\n"
            "  --verbose             show GPU loader and backend logging\n"
            "                        (suppressed by default)\n"
            "transcribefile notes:\n"
            "  GPU backends wired up in this build: metal (macOS on Apple\n"
            "  Silicon; used by --backend auto/metal, first run compiles\n"
            "  ggml-metal.dylib and caches it under ~/.transcribefile).\n"
            "  vulkan and cuda are not available yet.\n"
            "  Default arguments are read from .args in the executable's zip\n"
            "  store (one per line), so models can be bundled with zipalign.\n");
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
    // The Metal dylib logs through ggml's default stderr callback: device
    // init banners plus one "compiling pipeline" DEBUG line per kernel on
    // EVERY run (pipeline-state objects live in an in-memory cache only,
    // so each process recreates them; the on-disk dylib cache is a
    // different layer). Route the dylib's logging to the null sink unless
    // --verbose was given, exactly like llamafile: FLAG_verbose also
    // unlocks the loader's own llamafile_info diagnostics.
    if (!FLAG_verbose) {
        llamafile_metal_log_set(llamafile_log_callback_null, nullptr);
    }
    FLAG_gpu = LLAMAFILE_GPU_AUTO;
    llamafile_has_metal();
}

int main(int argc, char ** argv) {
    // Symbolized backtraces on crash (cosmopolitan).
    ShowCrashReports();

    // Cached artifacts (the runtime-compiled Metal dylib and the ggml
    // sources it is built from) go under ~/.transcribefile, not
    // ~/.llamafile: the two products' ggml trees are aligned today but
    // may diverge, and neither should ever overwrite the other's cache.
    llamafile_set_app_name("transcribefile");

    // Answer --version before touching args or the zip store. The
    // version string is transcribe.cpp's `git describe` output, which
    // already carries its own "v" prefix (e.g. v0.0.11-7-gdf1a4ad).
    if (has_flag(argv, "--version")) {
        puts("transcribefile " TRANSCRIBEFILE_VERSION_STRING);
        return 0;
    }

    // Merge default arguments embedded at /zip/.args (if present) with the
    // user's argv. No-op for a bare executable with no bundled .args.
    argc = cosmo_args("/zip/.args", &argv);

    // Wrapper-owned flag, same meaning as llamafile's --verbose: GPU
    // loader diagnostics and dylib logging. Consumed here because the
    // upstream CLI rejects options it doesn't know.
    if (consume_flag(argv, "--verbose")) {
        FLAG_verbose = 1;
        --argc;
    }

    // --help never needs a compute device: printing usage should not pay
    // the Metal dylib build/load latency or emit device-init logs, so the
    // GPU load below is skipped. The CLI's help path ends in std::exit(0)
    // (and its print_usage has internal linkage), so the wrapper's help
    // section is appended via atexit: it runs right after upstream's
    // usage text, whether the CLI exits or returns.
    if (has_flag(argv, "-h") || has_flag(argv, "--help")) {
        atexit(print_wrapper_help);
        return transcribe_cli_main(argc, argv);
    }

    load_gpu_backends(argv);

    return transcribe_cli_main(argc, argv);
}
