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

#include "zip.h"
#include <stdint.h>

int64_t get_zip_cfile_uncompressed_size(const uint8_t *z) {
    if (ZIP_CFILE_UNCOMPRESSEDSIZE(z) != 0xFFFFFFFFu)
        return ZIP_CFILE_UNCOMPRESSEDSIZE(z);
    const uint8_t *p = ZIP_CFILE_EXTRA(z);
    const uint8_t *pe = p + ZIP_CFILE_EXTRASIZE(z);
    for (; p + kZipExtraHdrSize <= pe && p + ZIP_EXTRA_SIZE(p) <= pe; p += ZIP_EXTRA_SIZE(p))
        if (ZIP_EXTRA_HEADERID(p) == kZipExtraZip64)
            if (8 <= ZIP_EXTRA_CONTENTSIZE(p))
                return ZIP_READ64(ZIP_EXTRA_CONTENT(p));
    return -1;
}

int64_t get_zip_cfile_compressed_size(const uint8_t *z) {
    if (ZIP_CFILE_COMPRESSEDSIZE(z) != 0xFFFFFFFFu)
        return ZIP_CFILE_COMPRESSEDSIZE(z);
    const uint8_t *p = ZIP_CFILE_EXTRA(z);
    const uint8_t *pe = p + ZIP_CFILE_EXTRASIZE(z);
    for (; p + kZipExtraHdrSize <= pe && p + ZIP_EXTRA_SIZE(p) <= pe; p += ZIP_EXTRA_SIZE(p))
        if (ZIP_EXTRA_HEADERID(p) == kZipExtraZip64) {
            int offset = 0;
            if (ZIP_CFILE_UNCOMPRESSEDSIZE(z) == 0xFFFFFFFFu)
                offset += 8;
            if (offset + 8 <= ZIP_EXTRA_CONTENTSIZE(p))
                return ZIP_READ64(ZIP_EXTRA_CONTENT(p) + offset);
        }
    return -1;
}

int64_t get_zip_cfile_offset(const uint8_t *z) {
    if (ZIP_CFILE_OFFSET(z) != 0xFFFFFFFFu)
        return ZIP_CFILE_OFFSET(z);
    const uint8_t *p = ZIP_CFILE_EXTRA(z);
    const uint8_t *pe = p + ZIP_CFILE_EXTRASIZE(z);
    for (; p + kZipExtraHdrSize <= pe && p + ZIP_EXTRA_SIZE(p) <= pe; p += ZIP_EXTRA_SIZE(p))
        if (ZIP_EXTRA_HEADERID(p) == kZipExtraZip64) {
            int offset = 0;
            if (ZIP_CFILE_UNCOMPRESSEDSIZE(z) == 0xFFFFFFFFu)
                offset += 8;
            if (ZIP_CFILE_COMPRESSEDSIZE(z) == 0xFFFFFFFFu)
                offset += 8;
            if (offset + 8 <= ZIP_EXTRA_CONTENTSIZE(p))
                return ZIP_READ64(ZIP_EXTRA_CONTENT(p) + offset);
        }
    return -1;
}
