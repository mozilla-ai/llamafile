// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
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

#pragma once

#define LLAMAFILE_MAJOR 0
#define LLAMAFILE_MINOR 10
#define LLAMAFILE_PATCH 0
#define LLAMAFILE_VERSION \
    (100000000 * LLAMAFILE_MAJOR + 1000000 * LLAMAFILE_MINOR + LLAMAFILE_PATCH)

#define MKVERSION__(x, y, z) #x "." #y "." #z
#define MKVERSION_(x, y, z) MKVERSION__(x, y, z)
#define LLAMAFILE_VERSION_STRING MKVERSION_(LLAMAFILE_MAJOR, LLAMAFILE_MINOR, LLAMAFILE_PATCH)
