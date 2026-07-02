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
//
// sd_math_shim.cpp — provides llround/llroundf/llroundl for cosmocc
// which declares but may not implement these functions.

#include <cmath>

extern "C" {
    long long llround(double x) {
        return round(x);
    }
    long long llroundf(float x) {
        return roundf(x);
    }
    long long llroundl(long double x) {
        return roundl(x);
    }
}
