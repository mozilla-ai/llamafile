#ifndef __GGML_EXTEND_HPP__
#define __GGML_EXTEND_HPP__

#include "llama.cpp/ggml-alloc.h"
#include "llama.cpp/ggml-backend.h"
#include "llama.cpp/ggml.h"

#ifdef SD_USE_CUBLAS
#include "llama.cpp/ggml-cuda.h"
#endif

#ifdef SD_USE_METAL
#include "llama.cpp/ggml-metal.h"
#endif

#endif  // __GGML_EXTEND__HPP__
