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
#include "chatbot_backend.h"

#include <cctype>
#include <csignal>
#include <cstdio>
#include <string_view>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "bestline.h"
#include "color.h"
#include "highlight/highlight.h"
#include "llama.h"  // llamafile wrapper

namespace lf {
namespace chatbot {

bool g_has_ephemeral;
bool g_said_something;
char g_last_printed_char;
volatile sig_atomic_t g_got_sigint;
ChatBackend *g_backend = nullptr;

// Replace RESET (\e[0m) with RESET+FAINT (\e[0m\e[2m) to maintain dim styling
// when markdown highlighting resets attributes inside reasoning content.
static std::string maintain_faint_styling(const std::string &s) {
    std::string result;
    result.reserve(s.size() + 32);
    size_t pos = 0;
    while (pos < s.size()) {
        // Look for \e[0m (RESET)
        if (pos + 3 < s.size() && s[pos] == '\e' && s[pos+1] == '[' && s[pos+2] == '0' && s[pos+3] == 'm') {
            // Replace with \e[0m\e[2m (RESET + FAINT)
            result += "\e[0m\e[2m";
            pos += 4;
        } else {
            result += s[pos++];
        }
    }
    return result;
}

// Helper to apply chat template with enable_thinking support for Qwen3.5-style models.
// common_chat_format_single() doesn't support enable_thinking, so we need this wrapper.
// Only used by DirectBackend path (API backend lets the server handle templates).
std::string apply_chat_template_with_thinking(
    const std::vector<common_chat_msg> &past_msgs,
    const common_chat_msg &new_msg,
    bool add_generation_prompt) {

    if (!g_chat_templates)
        return "";

    // Check if template supports thinking mode
    bool supports_thinking = common_chat_templates_support_enable_thinking(g_chat_templates.get());

    common_chat_templates_inputs inputs;
    inputs.messages = past_msgs;
    inputs.messages.push_back(new_msg);
    inputs.use_jinja = true;
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.enable_thinking = supports_thinking;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

    auto chat_params = common_chat_templates_apply(g_chat_templates.get(), inputs);
    return chat_params.prompt;
}

void on_sigint(int sig) {
    g_got_sigint = 1;
}

// Flag to track if we're exiting due to interrupt (skip cleanup)
bool g_interrupted_exit = false;

bool is_empty(const char *s) {
    int c;
    while ((c = *s++))
        if (!isspace(c))
            return false;
    return true;
}

void print(const std::string_view &s) {
    for (char c : s) {
        g_last_printed_char = c;
        fputc(c, stdout);
        if (c == '\n')
            g_has_ephemeral = false;
    }
}

void ensure_newline() {
    if (g_last_printed_char != '\n')
        print("\n");
}

void err(const char *fmt, ...) {
    va_list ap;
    clear_ephemeral();
    ensure_newline();
    va_start(ap, fmt);
    fputs(BRIGHT_RED, stderr);
    vfprintf(stderr, fmt, ap);
    fputs(RESET "\n", stderr);
    va_end(ap);
}

void print_ephemeral(const std::string_view &description) {
    fprintf(stderr, " " BRIGHT_BLACK "%.*s" UNFOREGROUND "\r", (int)description.size(),
            description.data());
    g_has_ephemeral = true;
}

void clear_ephemeral(void) {
    if (g_has_ephemeral) {
        fprintf(stderr, CLEAR_FORWARD);
        g_has_ephemeral = false;
    }
}

bool out_of_context(int extra) {
    err("error: ran out of context window at %d tokens\n"
        "consider passing `-c %d` at startup for the maximum\n"
        "you can free up more space using /forget or /clear",
        g_backend->context_used() + extra,
        g_backend->context_max());
    return false;
}

void repl(ChatBackend &backend) {

    // setup system prompt for message history
    // (Direct backend handles BOS token and system prompt eval in chatbot_main.cpp
    //  before calling repl(); API backend just needs the message history)
    if (!g_params->prompt.empty()) {
        if (!is_base_model()) {
            // Chat models: add system prompt to messages array
            common_chat_msg sys_msg;
            sys_msg.role = "system";
            sys_msg.content = g_params->prompt;
            g_messages.push_back(sys_msg);
        }
        // Display system prompt at startup
        if (g_params->display_prompt)
            printf("%s\n", g_params->prompt.c_str());
    }

    // perform important setup
    HighlightTxt txt;
    HighlightMarkdown markdown;
    ColorBleeder bleeder(is_base_model() ? (Highlight *)&txt : (Highlight *)&markdown);

    // Save old signal handler and install ours
    // NOTE: In combined mode, this overrides the server's SIGINT handler.
    // Only install if we're NOT in API mode (no local model = API mode).
    struct sigaction sa, old_sa;
    if (g_model) {
        // Direct mode: install our own handler
        sa.sa_handler = on_sigint;
        sa.sa_flags = 0;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGINT, &sa, &old_sa);
    }

    // run chatbot
    for (;;) {
        record_undo();
        bestlineLlamaMode(true);
        bestlineSetHintsCallback(on_hint);
        bestlineSetFreeHintsCallback(free);
        bestlineSetCompletionCallback(on_completion);
        write(1, get_role_color(g_role), strlen(get_role_color(g_role)));
        char *line = bestlineWithHistory(">>> ", "llamafile");
        write(1, RESET, strlen(RESET));
        g_last_printed_char = '\n';
        if (!line) {
            if (g_got_sigint) {
                ensure_newline();
            }
            // Skip cleanup to avoid Metal crash (see chatbot_main)
            // Setting g_interrupted_exit here covers both CTRL+C
            // (sigint) and CTRL+D (newline)
            g_interrupted_exit = true;
            break;
        }
        if (!is_base_model() && is_empty(line)) {
            if (g_manual_mode) {
                g_role = cycle_role(g_role);
                write(1, "\033[F", 3);
            }
            free(line);
            continue;
        }
        g_said_something = true;
        if (handle_command(line)) {
            free(line);
            continue;
        }

        // Manual mode: only available with direct backend
        if (g_manual_mode && !backend.supports_manual_mode()) {
            err("manual mode not available in this mode — use --chat for direct model access");
            free(line);
            continue;
        }

        bool add_assi = !g_manual_mode;
        int tokens_before = backend.context_used();

        // Combine any pending file content with user's message
        std::string user_content;
        if (!g_pending_file_content.empty()) {
            user_content = g_pending_file_content;
            user_content += "\n\n";
            user_content += line;
            g_pending_file_content.clear();
        } else {
            user_content = line;
        }

        // Build the message
        common_chat_msg user_msg;
        user_msg.role = get_role_name(g_role);
        user_msg.content = user_content;

        // Direct backend: format and eval the prompt ourselves
        if (backend.supports_manual_mode()) {
            std::string msg;
            if (is_base_model()) {
                msg = user_content;
            } else {
                msg = apply_chat_template_with_thinking(g_messages, user_msg, add_assi);
            }
            if (!eval_string(msg, DONT_ADD_SPECIAL, PARSE_SPECIAL)) {
                rewind(tokens_before);
                free(line);
                continue;
            }
        }

        // Track message in history
        if (!is_base_model()) {
            g_messages.push_back(user_msg);
        }

        if (g_manual_mode) {
            g_role = get_next_role(g_role);
            free(line);
            continue;
        }

        // Generate response via backend
        bool in_reasoning = false;
        std::string assistant_content = backend.complete(g_messages,
            [&](const std::string &content, const std::string &reasoning) -> bool {
                if (!reasoning.empty()) {
                    if (!in_reasoning) {
                        print(FAINT);
                        in_reasoning = true;
                    }
                    std::string s;
                    bleeder.feed(&s, reasoning);
                    print(maintain_faint_styling(s));
                }
                if (!content.empty()) {
                    if (in_reasoning) {
                        print(UNBOLD);
                        print("\n\n");
                        in_reasoning = false;
                    }
                    std::string s;
                    bleeder.feed(&s, content);
                    print(s);
                }
                fflush(stdout);
                return !g_got_sigint;
            });

        // End reasoning mode if still active
        if (in_reasoning) {
            print(UNBOLD);
        }

        // Track assistant response in message history
        if (!is_base_model() && !assistant_content.empty()) {
            common_chat_msg asst_msg;
            asst_msg.role = "assistant";
            asst_msg.content = assistant_content;
            g_messages.push_back(asst_msg);
        }

        g_got_sigint = 0;
        free(line);
        std::string s;
        bleeder.flush(&s);
        print(s);
        ensure_newline();
    }

    // Restore original signal handler before cleanup
    if (g_model) {
        sigaction(SIGINT, &old_sa, nullptr);
    }
}

} // namespace chatbot
} // namespace lf
