// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla Foundation
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
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace lf::chatbot;

static int test_count = 0;
static int fail_count = 0;

#define TEST(name) \
    static void test_##name(); \
    static struct TestRegister_##name { \
        TestRegister_##name() { test_##name(); } \
    } test_register_##name; \
    static void test_##name()

#define ASSERT_EQ(expected, actual, msg) \
    do { \
        test_count++; \
        if ((expected) != (actual)) { \
            fprintf(stderr, "FAIL: %s\n  expected: %s\n  actual: %s\n", \
                    msg, #expected, #actual); \
            fail_count++; \
        } \
    } while (0)

#define ASSERT_STR_EQ(expected, actual, msg) \
    do { \
        test_count++; \
        if ((expected) != (actual)) { \
            fprintf(stderr, "FAIL: %s\n  expected: \"%s\"\n  actual: \"%s\"\n", \
                    msg, (expected).c_str(), (actual).c_str()); \
            fail_count++; \
        } \
    } while (0)

// Minimal valid 1x1 red PNG (67 bytes)
static const char PNG_BASE64[] =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

// Minimal valid 1x1 GIF (35 bytes)
static const char GIF_BASE64[] =
    "R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==";

TEST(plain_text_no_data_uri) {
    const char *marker = "[IMG]";
    // plain text
    auto result = extract_data_uris("Hello, world!", marker);

    ASSERT_STR_EQ(std::string("Hello, world!"), result.modified_text,
                  "plain text should be unchanged");
    ASSERT_EQ(0u, result.images.size(), "no images should be extracted");
}

TEST(text_with_invalid_data_uri) {
    const char *marker = "[IMG]";
    // data: without valid content
    auto result = extract_data_uris("Hello data:invalid world", marker);

    ASSERT_STR_EQ(std::string("Hello data:invalid world"), result.modified_text,
                  "invalid data URI should be left as-is");
    ASSERT_EQ(0u, result.images.size(), "no images should be extracted");
}

TEST(text_with_non_image_data_uri) {
    const char *marker = "[IMG]";
    // valid data URI but not an image
    auto result = extract_data_uris("Hello data:text/plain,hello world", marker);

    ASSERT_STR_EQ(std::string("Hello data:text/plain,hello world"), result.modified_text,
                  "non-image data URI should be left as-is");
    ASSERT_EQ(0u, result.images.size(), "no images should be extracted");
}

TEST(trailing_invalid_data_uri) {
    const char *marker = "[IMG]";
    // trailing "data:" that doesn't parse
    auto result = extract_data_uris("Hello world data:", marker);

    ASSERT_STR_EQ(std::string("Hello world data:"), result.modified_text,
                  "trailing invalid data: should be preserved");
    ASSERT_EQ(0u, result.images.size(), "no images should be extracted");
}

TEST(valid_png_data_uri) {
    const char *marker = "[IMG]";
    // valid image within text
    std::string input = "Look at this: data:image/png;base64,";
    input += PNG_BASE64;
    input += " nice!";

    auto result = extract_data_uris(input, marker);

    ASSERT_STR_EQ(std::string("Look at this: [IMG] nice!"), result.modified_text,
                  "valid PNG data URI should be replaced with marker");
    ASSERT_EQ(1u, result.images.size(), "one image should be extracted");
}

TEST(valid_gif_data_uri) {
    const char *marker = "[IMG]";
    // valid image alone
    std::string input = "data:image/gif;base64,";
    input += GIF_BASE64;

    auto result = extract_data_uris(input, marker);

    ASSERT_STR_EQ(std::string("[IMG]"), result.modified_text,
                  "valid GIF data URI should be replaced with marker");
    ASSERT_EQ(1u, result.images.size(), "one image should be extracted");
}

TEST(multiple_images) {
    const char *marker = "[IMG]";
    // two images mixed with text
    std::string input = "First: data:image/png;base64,";
    input += PNG_BASE64;
    input += " Second: data:image/gif;base64,";
    input += GIF_BASE64;
    input += " End.";

    auto result = extract_data_uris(input, marker);

    ASSERT_STR_EQ(std::string("First: [IMG] Second: [IMG] End."), result.modified_text,
                  "multiple images should be replaced with markers");
    ASSERT_EQ(2u, result.images.size(), "two images should be extracted");
}

TEST(valid_image_followed_by_trailing_data) {
    const char *marker = "[IMG]";
    // valid image then invalid trailing "data:"
    std::string input = "data:image/png;base64,";
    input += PNG_BASE64;
    input += " trailing data:";

    auto result = extract_data_uris(input, marker);

    ASSERT_STR_EQ(std::string("[IMG] trailing data:"), result.modified_text,
                  "valid image followed by trailing 'data:' should preserve both");
    ASSERT_EQ(1u, result.images.size(), "one image should be extracted");
}

TEST(empty_string) {
    const char *marker = "[IMG]";
    auto result = extract_data_uris("", marker);

    ASSERT_STR_EQ(std::string(""), result.modified_text,
                  "empty string should remain empty");
    ASSERT_EQ(0u, result.images.size(), "no images should be extracted");
}

TEST(marker_is_stored) {
    const char *marker = "<image>";
    auto result = extract_data_uris("test", marker);

    ASSERT_EQ(marker, result.marker, "marker should be stored in result");
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    fprintf(stderr, "Running extract_data_uris tests...\n");

    if (fail_count > 0) {
        fprintf(stderr, "\n%d/%d tests FAILED\n", fail_count, test_count);
        return 1;
    }

    fprintf(stderr, "All %d tests PASSED\n", test_count);
    return 0;
}
