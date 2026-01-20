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

// Unit tests for minja comment parsing fix.
//
// This tests the manual string-scanning approach used to parse Jinja comments,
// which replaces the regex-based approach that caused stack overflow on large
// comments (like the ~16KB comments found in gpt-oss-20b chat templates).
//
// The manual parsing is implemented in the minja.hpp patch and avoids recursive
// backtracking issues present in some regex implementations (including cosmocc).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

#define ASSERT_FALSE(cond, msg) \
    do { \
        test_count++; \
        if ((cond)) { \
            fprintf(stderr, "FAIL: %s\n  expected false: %s\n", msg, #cond); \
            fail_count++; \
        } \
    } while (0)

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

//=============================================================================
// Manual comment parsing implementation (mirrors minja.hpp patch)
//=============================================================================

struct CommentParseResult {
    bool found = false;
    std::string pre_flag;   // - or ~ (pre-space control)
    std::string content;    // Comment body
    std::string post_flag;  // - or ~ (post-space control)
    size_t end_pos = 0;     // Position after closing #}
};

// Parse a Jinja comment starting at position `pos` in `str`.
// Returns result with found=true if a complete comment was parsed.
// This implementation avoids regex to prevent stack overflow on large comments.
CommentParseResult parseComment(const std::string& str, size_t pos) {
    CommentParseResult result;
    size_t start_pos = pos;

    // Need at least 4 chars for minimal comment {##}
    if (pos + 4 > str.size()) return result;
    if (str[pos] != '{' || str[pos + 1] != '#') return result;

    pos += 2;

    // Parse optional pre-space flag (- or ~)
    if (pos < str.size() && (str[pos] == '-' || str[pos] == '~')) {
        result.pre_flag = std::string(1, str[pos]);
        pos++;
    }

    size_t content_start = pos;

    // Find closing #}
    while (pos < str.size()) {
        if (str[pos] == '#' && pos + 1 < str.size() && str[pos + 1] == '}') {
            // Check for post-space flag (character before #)
            if (pos > content_start) {
                char prev = str[pos - 1];
                if (prev == '-' || prev == '~') {
                    result.post_flag = std::string(1, prev);
                    result.content = str.substr(content_start, pos - content_start - 1);
                } else {
                    result.content = str.substr(content_start, pos - content_start);
                }
            } else {
                result.content = "";
            }
            result.found = true;
            result.end_pos = pos + 2;
            return result;
        }
        pos++;
    }

    // No closing #} found
    return result;
}

//=============================================================================
// Tests
//=============================================================================

static void test_empty_comment() {
    auto r = parseComment("{##}", 0);
    ASSERT_TRUE(r.found, "empty comment should be found");
    ASSERT_STR_EQ(std::string(""), r.content, "empty comment content");
    ASSERT_STR_EQ(std::string(""), r.pre_flag, "no pre flag");
    ASSERT_STR_EQ(std::string(""), r.post_flag, "no post flag");
    ASSERT_EQ(4u, r.end_pos, "end position after {##}");
}

static void test_simple_comment() {
    auto r = parseComment("{# hello world #}", 0);
    ASSERT_TRUE(r.found, "simple comment should be found");
    ASSERT_STR_EQ(std::string(" hello world "), r.content, "simple comment content");
    ASSERT_STR_EQ(std::string(""), r.pre_flag, "no pre flag");
    ASSERT_STR_EQ(std::string(""), r.post_flag, "no post flag");
}

static void test_pre_flag_dash() {
    auto r = parseComment("{#- trimmed #}", 0);
    ASSERT_TRUE(r.found, "comment with pre-flag should be found");
    ASSERT_STR_EQ(std::string(" trimmed "), r.content, "content after pre-flag");
    ASSERT_STR_EQ(std::string("-"), r.pre_flag, "dash pre flag");
    ASSERT_STR_EQ(std::string(""), r.post_flag, "no post flag");
}

static void test_pre_flag_tilde() {
    auto r = parseComment("{#~ trimmed #}", 0);
    ASSERT_TRUE(r.found, "comment with tilde pre-flag");
    ASSERT_STR_EQ(std::string(" trimmed "), r.content, "content after tilde pre-flag");
    ASSERT_STR_EQ(std::string("~"), r.pre_flag, "tilde pre flag");
}

static void test_post_flag_dash() {
    auto r = parseComment("{# trimmed -#}", 0);
    ASSERT_TRUE(r.found, "comment with post-flag should be found");
    ASSERT_STR_EQ(std::string(" trimmed "), r.content, "content before post-flag");
    ASSERT_STR_EQ(std::string(""), r.pre_flag, "no pre flag");
    ASSERT_STR_EQ(std::string("-"), r.post_flag, "dash post flag");
}

static void test_post_flag_tilde() {
    auto r = parseComment("{# trimmed ~#}", 0);
    ASSERT_TRUE(r.found, "comment with tilde post-flag");
    ASSERT_STR_EQ(std::string(" trimmed "), r.content, "content before tilde post-flag");
    ASSERT_STR_EQ(std::string("~"), r.post_flag, "tilde post flag");
}

static void test_both_flags() {
    auto r = parseComment("{#- both flags -#}", 0);
    ASSERT_TRUE(r.found, "comment with both flags");
    ASSERT_STR_EQ(std::string(" both flags "), r.content, "content with both flags");
    ASSERT_STR_EQ(std::string("-"), r.pre_flag, "dash pre flag");
    ASSERT_STR_EQ(std::string("-"), r.post_flag, "dash post flag");
}

static void test_multiline_comment() {
    const char* input = "{#\nLine 1\nLine 2\nLine 3\n#}";
    auto r = parseComment(input, 0);
    ASSERT_TRUE(r.found, "multiline comment should be found");
    ASSERT_STR_EQ(std::string("\nLine 1\nLine 2\nLine 3\n"), r.content, "multiline content");
}

static void test_comment_with_hash() {
    // Comment containing # characters that aren't followed by }
    auto r = parseComment("{# foo # bar # baz #}", 0);
    ASSERT_TRUE(r.found, "comment with internal hashes");
    ASSERT_STR_EQ(std::string(" foo # bar # baz "), r.content, "content with hashes");
}

static void test_comment_at_offset() {
    const char* input = "prefix {# comment #} suffix";
    auto r = parseComment(input, 7);  // Start at {#
    ASSERT_TRUE(r.found, "comment at offset");
    ASSERT_STR_EQ(std::string(" comment "), r.content, "content at offset");
    ASSERT_EQ(20u, r.end_pos, "end position at offset");
}

static void test_no_opening() {
    auto r = parseComment("not a comment", 0);
    ASSERT_FALSE(r.found, "no {# opening");
}

static void test_no_closing() {
    auto r = parseComment("{# unclosed comment", 0);
    ASSERT_FALSE(r.found, "no #} closing");
}

static void test_partial_opening() {
    auto r = parseComment("{ #not comment#}", 0);
    ASSERT_FALSE(r.found, "space between { and # is not a comment");
}

static void test_too_short() {
    auto r = parseComment("{#}", 0);
    ASSERT_FALSE(r.found, "too short for valid comment");
}

static void test_empty_string() {
    auto r = parseComment("", 0);
    ASSERT_FALSE(r.found, "empty string");
}

static void test_position_past_end() {
    auto r = parseComment("{##}", 10);
    ASSERT_FALSE(r.found, "position past string end");
}

// Test with large comment content (similar to gpt-oss-20b chat template)
// This is the key test - regex would stack overflow here
static void test_large_comment() {
    std::string large_content;
    // Build ~20KB of content
    for (int i = 0; i < 400; i++) {
        large_content += "  This is documentation line ";
        large_content += std::to_string(i);
        large_content += " inside a Jinja comment.\n";
    }

    std::string input = "{#-\n" + large_content + "-#}";
    auto r = parseComment(input, 0);

    ASSERT_TRUE(r.found, "large comment should be found");
    ASSERT_STR_EQ(std::string("-"), r.pre_flag, "large comment pre flag");
    ASSERT_STR_EQ(std::string("-"), r.post_flag, "large comment post flag");
    ASSERT_TRUE(r.content.size() > 15000, "large comment content size > 15KB");
    ASSERT_EQ(input.size(), r.end_pos, "large comment end position");
}

// Test with very large comment (stress test)
static void test_very_large_comment() {
    // Build ~100KB comment
    std::string content(100000, 'x');
    std::string input = "{#" + content + "#}";

    auto r = parseComment(input, 0);

    ASSERT_TRUE(r.found, "very large comment should be found");
    ASSERT_EQ(100000u, r.content.size(), "very large comment content size");
}

// Test that special characters don't break parsing
static void test_special_characters() {
    auto r = parseComment("{# \t\r\n\v\f #}", 0);
    ASSERT_TRUE(r.found, "whitespace characters");

    r = parseComment("{# <>\"'&;$`|\\/ #}", 0);
    ASSERT_TRUE(r.found, "special shell characters");

    r = parseComment("{# {{ }} {% %} #}", 0);
    ASSERT_TRUE(r.found, "jinja delimiters in comment");
}

// Test consecutive comments
static void test_consecutive_comments() {
    const char* input = "{# first #}{# second #}";

    auto r1 = parseComment(input, 0);
    ASSERT_TRUE(r1.found, "first consecutive comment");
    ASSERT_STR_EQ(std::string(" first "), r1.content, "first content");

    auto r2 = parseComment(input, r1.end_pos);
    ASSERT_TRUE(r2.found, "second consecutive comment");
    ASSERT_STR_EQ(std::string(" second "), r2.content, "second content");
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    fprintf(stderr, "Running minja comment parsing tests...\n");

    test_empty_comment();
    test_simple_comment();
    test_pre_flag_dash();
    test_pre_flag_tilde();
    test_post_flag_dash();
    test_post_flag_tilde();
    test_both_flags();
    test_multiline_comment();
    test_comment_with_hash();
    test_comment_at_offset();
    test_no_opening();
    test_no_closing();
    test_partial_opening();
    test_too_short();
    test_empty_string();
    test_position_past_end();
    test_large_comment();
    test_very_large_comment();
    test_special_characters();
    test_consecutive_comments();

    if (fail_count > 0) {
        fprintf(stderr, "\n%d/%d tests FAILED\n", fail_count, test_count);
        return 1;
    }

    fprintf(stderr, "All %d tests PASSED\n", test_count);
    return 0;
}
