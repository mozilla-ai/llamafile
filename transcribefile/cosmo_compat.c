// Cosmopolitan libc compatibility shims for transcribefile.
//
// cosmocc's <math.h> declares the llround family but the toolchain ships
// only the lround family (lround/lroundf/lroundl are in libcosmo.a;
// llround/llroundf/llroundl are not). transcribe.cpp uses std::llround in
// a few timestamp computations, which libc++ forwards to ::llround.
//
// On every cosmopolitan target (x86-64 and aarch64) the ABI is LP64, so
// `long` and `long long` are both 64-bit and the llround family is exactly
// equivalent to the lround family. Define the missing symbols in terms of
// the ones that exist.

#include <math.h>

long long llround(double x)       { return lround(x);  }
long long llroundf(float x)       { return lroundf(x); }
long long llroundl(long double x) { return lroundl(x); }
