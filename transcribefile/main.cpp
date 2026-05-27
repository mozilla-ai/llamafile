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

    return transcribe_cli_main(argc, argv);
}
