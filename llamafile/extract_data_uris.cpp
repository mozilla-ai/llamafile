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

#include "chatbot.h"
#include "base64.hpp"
#include "datauri.h"
#include "image.h"
#include "string.h"

namespace lf {
namespace chatbot {

// Extract data URIs from text, replacing them with a marker.
// Returns the modified text and decoded image data.
// This function is independent of model state and can be unit tested.
DataUriExtraction extract_data_uris(std::string_view s, const char *marker) {
    DataUriExtraction result;
    result.marker = marker;
    size_t i = 0;
    size_t last_pos = 0;

    while (i < s.size()) {
        // look for the offset where a data URI might start
        size_t pos = s.find("data:", i);
        if (pos == std::string_view::npos)
            break;

        // if "data:" is present, try to parse the following as a URI
        i = pos + 5;
        DataUri uri;
        size_t end = uri.parse(s.substr(pos + 5));
        if (end == std::string_view::npos)
            // not a valid URI
            continue;
        if (!lf::startscasewith(uri.mime, "image/"))
            // not an image mime type
            continue;

        std::string image;
        try {
            image = uri.decode();
        } catch (const base64_error &e) {
            // could not decode the base64 data
            continue;
        }
        if (!is_image(image))
            // not a valid image
            continue;

        // Append text before this data URI
        result.modified_text += s.substr(last_pos, pos - last_pos);

        // Store decoded image data
        result.images.push_back(std::move(image));

        // Add marker where image should go
        result.modified_text += marker;

        // Move past this data URI
        last_pos = i + end;
        i = last_pos;
    }

    // Append any remaining text after the last processed position
    // (handles case where loop exits normally due to invalid trailing data: URIs)
    if (last_pos < s.size())
        result.modified_text += s.substr(last_pos);

    return result;
}

} // namespace chatbot
} // namespace lf
