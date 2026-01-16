// Test case demonstrating regex stack overflow when parsing large Jinja comments
//
// This test shows that the regex pattern `\{#([-~]?)([\s\S]*?)([-~]?)#\}` used
// to match Jinja comments can cause stack overflow in some C++ standard library
// implementations due to recursive backtracking.
//
// The issue manifests when:
// - Using Cosmopolitan Libc (cosmocc) - crashes even on ~200 byte inputs
// - Possibly other C++ stdlib implementations with recursive regex engines
//
// The fix is to use manual string scanning instead of regex for comment parsing.
//
// Usage:
//   ./test_comment_parsing [comment_size_in_bytes]
//
// Related issue: Chat templates from models like gpt-oss-20b contain ~16KB
// comments at the start, triggering this crash.

#include <iostream>
#include <string>
#include <regex>
#include <chrono>

//=============================================================================
// ORIGINAL APPROACH (from minja.hpp) - causes stack overflow
//=============================================================================

static std::regex comment_tok_regex(R"(\{#([-~]?)([\s\S]*?)([-~]?)#\})");

struct RegexResult {
    bool found = false;
    std::string pre_flag;
    std::string content;
    std::string post_flag;
};

RegexResult parseCommentWithRegex(const std::string& str, size_t pos) {
    RegexResult result;

    std::smatch match;
    auto it = str.cbegin() + pos;
    auto end = str.cend();

    if (std::regex_search(it, end, match, comment_tok_regex) && match.position() == 0) {
        result.found = true;
        result.pre_flag = match[1].str();
        result.content = match[2].str();
        result.post_flag = match[3].str();
    }

    return result;
}

//=============================================================================
// FIXED APPROACH - manual string scanning, no stack overflow
//=============================================================================

struct ManualResult {
    bool found = false;
    std::string pre_flag;
    std::string content;
    std::string post_flag;
    size_t end_pos = 0;  // Position after the closing #}
};

ManualResult parseCommentManually(const std::string& str, size_t pos) {
    ManualResult result;

    if (pos + 4 > str.size()) return result;  // Need at least {##}
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

    return result;  // No closing #} found
}

//=============================================================================
// Test driver
//=============================================================================

std::string createTestTemplate(size_t comment_content_size) {
    std::string result = "{#-\n";

    // Add content to reach target size
    while (result.size() < comment_content_size + 4) {  // +4 for "{#-\n"
        result += "  This is documentation text inside a Jinja comment.\n";
        result += "  Such comments are common in chat templates for LLMs.\n\n";
    }

    result += "-#}";  // Closing with post-space flag

    // Add some actual template content after the comment
    result += "\n{% for msg in messages %}{{ msg.content }}{% endfor %}\n";

    return result;
}

int main(int argc, char* argv[]) {
    size_t target_size = 15000;  // Similar to gpt-oss-20b chat template
    if (argc > 1) {
        target_size = std::stoul(argv[1]);
    }

    std::cout << "=== Jinja Comment Parsing Test ===" << std::endl;
    std::cout << "Target comment size: " << target_size << " bytes" << std::endl;
    std::cout << std::endl;

    auto template_str = createTestTemplate(target_size);
    std::cout << "Total template size: " << template_str.size() << " bytes" << std::endl;
    std::cout << std::endl;

    // Test manual approach (should always work)
    {
        std::cout << "Testing MANUAL approach..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        auto result = parseCommentManually(template_str, 0);

        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

        if (result.found) {
            std::cout << "  SUCCESS: Found comment" << std::endl;
            std::cout << "  Content size: " << result.content.size() << " bytes" << std::endl;
            std::cout << "  Pre-flag: '" << result.pre_flag << "'" << std::endl;
            std::cout << "  Post-flag: '" << result.post_flag << "'" << std::endl;
        } else {
            std::cout << "  FAILED: No comment found" << std::endl;
        }
        std::cout << "  Time: " << us << " us" << std::endl;
    }

    std::cout << std::endl;

    // Test regex approach (may crash on some platforms)
    {
        std::cout << "Testing REGEX approach..." << std::endl;
        std::cout << "  (May crash with stack overflow on some C++ stdlib implementations)" << std::endl;
        std::cout.flush();

        auto start = std::chrono::high_resolution_clock::now();

        auto result = parseCommentWithRegex(template_str, 0);

        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

        if (result.found) {
            std::cout << "  SUCCESS: Found comment" << std::endl;
            std::cout << "  Content size: " << result.content.size() << " bytes" << std::endl;
            std::cout << "  Pre-flag: '" << result.pre_flag << "'" << std::endl;
            std::cout << "  Post-flag: '" << result.post_flag << "'" << std::endl;
        } else {
            std::cout << "  FAILED: No comment found" << std::endl;
        }
        std::cout << "  Time: " << us << " us" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== Test completed ===" << std::endl;

    return 0;
}
