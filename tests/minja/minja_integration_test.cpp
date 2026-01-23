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

// Integration test for minja comment parsing fix.
//
// This test uses the actual patched minja.hpp to parse templates with large
// comments, ensuring end-to-end verification of the fix.
//
// The fix replaces regex-based comment parsing with manual string scanning to
// avoid stack overflow on large comments (like the ~16KB comments found in
// gpt-oss-20b chat templates).

#include <minja/minja.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

static int test_count = 0;
static int fail_count = 0;

#define ASSERT_TRUE(cond, msg) \
    do { \
        test_count++; \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s\n  condition: %s\n", msg, #cond); \
            fail_count++; \
        } \
    } while (0)

#define ASSERT_NO_THROW(expr, msg) \
    do { \
        test_count++; \
        try { \
            expr; \
        } catch (const std::exception& e) { \
            fprintf(stderr, "FAIL: %s\n  threw: %s\n", msg, e.what()); \
            fail_count++; \
        } catch (...) { \
            fprintf(stderr, "FAIL: %s\n  threw unknown exception\n", msg); \
            fail_count++; \
        } \
    } while (0)

// Test parsing a template with a large comment (similar to gpt-oss-20b).
// This is the key test - the old regex-based approach would stack overflow here.
static void test_large_comment_parsing() {
    // Build ~20KB of comment content (similar to gpt-oss-20b chat template)
    std::string large_content;
    for (int i = 0; i < 400; i++) {
        large_content += "  This is documentation line ";
        large_content += std::to_string(i);
        large_content += " inside a Jinja comment.\n";
    }

    std::string template_str = "{#-\n" + large_content + "-#}Hello, World!";

    minja::Options options = {false, false, false};

    ASSERT_NO_THROW(
        minja::Parser::parse(template_str, options),
        "parsing template with large comment should not throw"
    );

    auto result = minja::Parser::parse(template_str, options);
    ASSERT_TRUE(result != nullptr, "parsed template should not be null");
}

// Test parsing a template with a very large comment (stress test).
static void test_very_large_comment_parsing() {
    // Build ~100KB comment
    std::string content(100000, 'x');
    std::string template_str = "{#" + content + "#}Output";

    minja::Options options = {false, false, false};

    ASSERT_NO_THROW(
        minja::Parser::parse(template_str, options),
        "parsing template with very large comment should not throw"
    );

    auto result = minja::Parser::parse(template_str, options);
    ASSERT_TRUE(result != nullptr, "parsed template with very large comment should not be null");
}

// Test parsing a template with comment flags (- and ~).
static void test_comment_with_flags() {
    std::string template_str = "Before{#- comment content -#}After";

    minja::Options options = {false, false, false};

    ASSERT_NO_THROW(
        minja::Parser::parse(template_str, options),
        "parsing template with comment flags should not throw"
    );

    auto result = minja::Parser::parse(template_str, options);
    ASSERT_TRUE(result != nullptr, "parsed template with comment flags should not be null");
}

// Test parsing a template with multiple comments.
static void test_multiple_comments() {
    std::string template_str = "{# first #}Middle{# second #}End";

    minja::Options options = {false, false, false};

    ASSERT_NO_THROW(
        minja::Parser::parse(template_str, options),
        "parsing template with multiple comments should not throw"
    );

    auto result = minja::Parser::parse(template_str, options);
    ASSERT_TRUE(result != nullptr, "parsed template with multiple comments should not be null");
}

// Test parsing a template with comment containing special characters.
static void test_comment_special_characters() {
    std::string template_str = "{# {{ }} {% %} # -> <> & | #}Output";

    minja::Options options = {false, false, false};

    ASSERT_NO_THROW(
        minja::Parser::parse(template_str, options),
        "parsing template with special characters in comment should not throw"
    );

    auto result = minja::Parser::parse(template_str, options);
    ASSERT_TRUE(result != nullptr, "parsed template with special characters should not be null");
}

// Test parsing a template mixing comments with other Jinja constructs.
static void test_mixed_template() {
    std::string template_str =
        "{# This is a comment #}\n"
        "{% if true %}\n"
        "  {# Another comment #}\n"
        "  {{ \"Hello\" }}\n"
        "{% endif %}\n";

    minja::Options options = {false, false, false};

    ASSERT_NO_THROW(
        minja::Parser::parse(template_str, options),
        "parsing mixed template should not throw"
    );

    auto result = minja::Parser::parse(template_str, options);
    ASSERT_TRUE(result != nullptr, "parsed mixed template should not be null");
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    fprintf(stderr, "Running minja integration tests (using actual minja.hpp)...\n");

    test_large_comment_parsing();
    test_very_large_comment_parsing();
    test_comment_with_flags();
    test_multiple_comments();
    test_comment_special_characters();
    test_mixed_template();

    if (fail_count > 0) {
        fprintf(stderr, "\n%d/%d tests FAILED\n", fail_count, test_count);
        return 1;
    }

    fprintf(stderr, "All %d tests PASSED\n", test_count);
    return 0;
}
