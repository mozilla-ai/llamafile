// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
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
// CLI mode: single prompt → response, then exit
//
// This mode is designed for programmatic use:
// - No logo, no streaming decorations
// - Uses chat completions (applies chat template)
// - Clean output suitable for piping
// - Exits after response completes
//
// Usage: llamafile -m model.gguf --cli -p "Your prompt here"
//        llamafile -m model.gguf --cli --nothink -p "Your prompt here"
//

#include "chatbot.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits.h>
#include <signal.h>
#include <string>
#include <vector>

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include "llamafile.h"

namespace lf {
namespace chatbot {

// Forward declarations from chatbot_repl.cpp
extern void on_sigint(int sig);

// Result of applying chat template - includes prompt and parser params for output parsing
struct cli_chat_template_result {
    std::string prompt;
    common_chat_parser_params parser_params;
};

// Helper to apply chat template with full control over inputs
static cli_chat_template_result cli_apply_chat_template_full(llama_model *model,
                                                              common_chat_templates *templates,
                                                              const common_params &params,
                                                              const std::vector<common_chat_msg> &messages,
                                                              bool add_assistant,
                                                              bool enable_thinking) {
    cli_chat_template_result result;

    if (templates) {
        common_chat_templates_inputs inputs;
        inputs.messages = messages;
        inputs.use_jinja = true;
        inputs.add_generation_prompt = add_assistant;
        inputs.enable_thinking = enable_thinking;
        // Set reasoning_format so the PEG parser includes reasoning extraction
        inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

        auto chat_params = common_chat_templates_apply(templates, inputs);
        result.prompt = chat_params.prompt;

        // Initialize parser params from chat_params
        result.parser_params.format = chat_params.format;
        result.parser_params.thinking_forced_open = chat_params.thinking_forced_open;
        if (!chat_params.parser.empty()) {
            result.parser_params.parser.load(chat_params.parser);
        }
        // Enable reasoning parsing for thinking models
        result.parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        result.parser_params.reasoning_in_content = false;
        return result;
    }

    // Fallback to heuristic-based template (doesn't support enable_thinking)
    const char *tmpl = params.chat_template.empty()
                       ? llama_model_chat_template(model, nullptr)
                       : params.chat_template.c_str();

    // Build llama_chat_message array from messages
    std::vector<llama_chat_message> chat;
    for (const auto &msg : messages) {
        chat.push_back({msg.role.c_str(), msg.content.c_str()});
    }

    int len = llama_chat_apply_template(tmpl, chat.data(), chat.size(), add_assistant, nullptr, 0);
    if (len < 0) {
        return result;
    }

    result.prompt.resize(len);
    llama_chat_apply_template(tmpl, chat.data(), chat.size(), add_assistant, &result.prompt[0], result.prompt.size());
    // For fallback, parser_params will be default (COMMON_CHAT_FORMAT_CONTENT_ONLY)
    return result;
}

int cli_main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);

    // Parse flags quietly (no logo, no ephemeral messages)
    common_params params;
    params.sampling.n_prev = 64;
    params.n_batch = 256;
    params.sampling.temp = 0;  // deterministic by default

    // Note: FLAG_nothink, FLAG_verbose, FLAG_nologo are set by main.cpp
    // before calling cli_main(). GPU is also initialized there;

    // Fully disable common_log system BEFORE common_init() to prevent build info log
    // This pauses the log worker thread so LOG_INF calls become no-ops
    common_log_pause(common_log_main());

    // Set llama log callback to null
    llama_log_set((ggml_log_callback)llamafile_log_callback_null, NULL);

    // Initialize backend and common
    llama_backend_init();
    common_init();

    // Parse arguments (argv is already filtered by parse_llamafile_args in args.cpp)
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
        fprintf(stderr, "error: failed to parse arguments\n");
        return 1;
    }

    // Check that a prompt was provided
    if (params.prompt.empty()) {
        fprintf(stderr, "error: --cli mode requires -p \"prompt\"\n");
        return 1;
    }

    // GPU layers default
    if (llamafile_has_metal() && params.n_gpu_layers < 0) {
        params.n_gpu_layers = INT_MAX;
    }

    // Load model
    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model *model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "error: failed to load model: %s\n", params.model.path.c_str());
        return 2;
    }

    // Adjust context size
    if (params.n_ctx <= 0 || params.n_ctx > (int)llama_model_n_ctx_train(model))
        params.n_ctx = llama_model_n_ctx_train(model);
    if (params.n_ctx < params.n_batch)
        params.n_batch = params.n_ctx;

    // Create context
    llama_context_params ctx_params = common_context_params_to_llama(params);
    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model);
        return 3;
    }

    // Initialize sampler
    common_sampler *sampler = common_sampler_init(model, params.sampling);
    if (!sampler) {
        fprintf(stderr, "error: failed to initialize sampler\n");
        llama_free(ctx);
        llama_model_free(model);
        return 4;
    }

    // Initialize chat templates
    common_chat_templates_ptr chat_templates;
    bool is_chat_model = llama_model_meta_val_str(model, "tokenizer.chat_template", 0, 0) != -1
                         || !params.chat_template.empty();

    if (is_chat_model) {
        chat_templates = common_chat_templates_init(model, params.chat_template);
    }

    // Build the prompt
    std::string formatted_prompt;
    common_chat_parser_params parser_params;  // For parsing output
    bool enable_thinking = false;
    const llama_vocab *vocab = llama_model_get_vocab(model);

    if (is_chat_model) {
        // Build message list
        std::vector<common_chat_msg> messages;

        if (!params.system_prompt.empty()) {
            common_chat_msg sys_msg;
            sys_msg.role = "system";
            sys_msg.content = params.system_prompt;
            messages.push_back(sys_msg);
        }

        common_chat_msg user_msg;
        user_msg.role = "user";
        user_msg.content = params.prompt;
        messages.push_back(user_msg);

        // Apply chat template with enable_thinking based on --nothink flag
        // When --nothink is set, we tell the template to disable thinking mode
        // so the model won't produce <think>...</think> output at all
        enable_thinking = !FLAG_nothink;
        auto template_result = cli_apply_chat_template_full(model, chat_templates.get(), params,
                                                            messages, true, enable_thinking);
        formatted_prompt = template_result.prompt;
        parser_params = template_result.parser_params;
    } else {
        // Base model: use prompt as-is
        formatted_prompt = params.prompt;
    }

    // Tokenize
    std::vector<llama_token> tokens = llamafile_tokenize(model, formatted_prompt, false, true);

    // Add BOS if needed
    if (llama_vocab_get_add_bos(vocab)) {
        tokens.insert(tokens.begin(), llama_vocab_bos(vocab));
    }

    // Check context
    if ((int)tokens.size() > params.n_ctx) {
        fprintf(stderr, "error: prompt too long (%zu tokens, context is %d)\n",
                tokens.size(), params.n_ctx);
        common_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        return 5;
    }

    // Evaluate prompt
    for (int i = 0; i < (int)tokens.size(); i += params.n_batch) {
        int n_eval = std::min(params.n_batch, (int)tokens.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(&tokens[i], n_eval))) {
            fprintf(stderr, "error: failed to evaluate prompt\n");
            common_sampler_free(sampler);
            llama_free(ctx);
            llama_model_free(model);
            return 6;
        }
    }

    // Install signal handler for graceful interrupt
    struct sigaction sa, old_sa;
    sa.sa_handler = on_sigint;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, &old_sa);

    // Generate response
    // When thinking is enabled, we parse the output to show <think>...</think> and content.
    int n_cur = tokens.size();
    const bool use_chat_parser = enable_thinking &&
                                 parser_params.format != COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::string raw_output;           // Accumulates raw token output for parsing
    common_chat_msg prev_msg;         // Previous parse result for diff computation
    bool think_tag_opened = false;    // Track if we've printed <think>
    bool think_tag_closed = false;    // Track if we've printed </think>

    while (n_cur < params.n_ctx) {
        if (g_got_sigint) {
            g_got_sigint = 0;
            break;
        }

        llama_token id = common_sampler_sample(sampler, ctx, -1);
        common_sampler_accept(sampler, id, true);

        // Check for end of generation
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }

        if (use_chat_parser) {
            // Accumulate tokens and parse to extract content
            std::string token_str = llamafile_token_to_piece(ctx, id, true);
            raw_output += token_str;

            // Parse incrementally
            auto msg = common_chat_parse(raw_output, /*is_partial=*/true, parser_params);

            // Compute diffs to find new content
            auto diffs = common_chat_msg_diff::compute_diffs(prev_msg, msg);

            for (const auto &diff : diffs) {
                // Output reasoning content wrapped in <think> tags
                if (!diff.reasoning_content_delta.empty()) {
                    if (!think_tag_opened) {
                        fputs("<think>", stdout);
                        think_tag_opened = true;
                    }
                    fputs(diff.reasoning_content_delta.c_str(), stdout);
                    fflush(stdout);
                }
                // Output final content (close think tag first if needed)
                if (!diff.content_delta.empty()) {
                    if (think_tag_opened && !think_tag_closed) {
                        fputs("</think>\n", stdout);
                        think_tag_closed = true;
                    }
                    fputs(diff.content_delta.c_str(), stdout);
                    fflush(stdout);
                }
            }

            prev_msg = msg;
        } else {
            // No parsing needed - output token directly
            std::string piece = llamafile_token_to_piece(ctx, id, false);
            fputs(piece.c_str(), stdout);
            fflush(stdout);
        }

        // Evaluate token
        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) {
            break;
        }
        n_cur++;
    }

    // Ensure output ends with newline
    printf("\n");

    // Restore signal handler
    sigaction(SIGINT, &old_sa, nullptr);

    // Cleanup
    common_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}

} // namespace chatbot
} // namespace lf
