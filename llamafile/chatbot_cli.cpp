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

// Helper to apply chat template
static std::string cli_apply_chat_template(llama_model *model,
                                           common_chat_templates *templates,
                                           const common_params &params,
                                           const char *role,
                                           const char *content,
                                           bool add_assistant) {
    if (templates) {
        common_chat_msg msg;
        msg.role = role;
        msg.content = content;
        std::vector<common_chat_msg> past_msg;
        return common_chat_format_single(templates, past_msg, msg, add_assistant, /*use_jinja=*/true);
    }

    // Fallback to heuristic-based template
    const char *tmpl = params.chat_template.empty()
                       ? llama_model_chat_template(model, nullptr)
                       : params.chat_template.c_str();

    llama_chat_message chat[] = {{role, content}};
    int len = llama_chat_apply_template(tmpl, chat, 1, add_assistant, nullptr, 0);
    if (len < 0) {
        return "";
    }

    std::string result(len, '\0');
    llama_chat_apply_template(tmpl, chat, 1, add_assistant, &result[0], result.size());
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

    // Initialize chat parser for --nothink mode
    common_chat_parser_params chat_syntax;
    if (FLAG_nothink && is_chat_model && chat_templates) {
        // Set up chat parser to detect and filter reasoning content
        common_chat_msg dummy_msg;
        dummy_msg.role = "user";
        dummy_msg.content = "test";

        common_chat_templates_inputs inputs;
        inputs.messages = {dummy_msg};
        inputs.use_jinja = true;

        try {
            auto chat_params = common_chat_templates_apply(chat_templates.get(), inputs);
            chat_syntax.format = chat_params.format;
            chat_syntax.thinking_forced_open = chat_params.thinking_forced_open;

            if (!chat_params.parser.empty()) {
                chat_syntax.parser.load(chat_params.parser);
            }

            // Enable reasoning extraction
            chat_syntax.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            chat_syntax.reasoning_in_content = false;
        } catch (...) {
            // Fall back to no parsing if template fails
            FLAG_nothink = false;
        }
    } else if (FLAG_nothink && !is_chat_model) {
        // Can't filter thinking on base models
        FLAG_nothink = false;
    }

    // Build the prompt
    std::string formatted_prompt;
    const llama_vocab *vocab = llama_model_get_vocab(model);

    if (is_chat_model) {
        // Apply chat template: system prompt (if any) + user message
        if (!params.system_prompt.empty()) {
            formatted_prompt = cli_apply_chat_template(model, chat_templates.get(), params,
                                                       "system", params.system_prompt.c_str(), false);
        }
        formatted_prompt += cli_apply_chat_template(model, chat_templates.get(), params,
                                                    "user", params.prompt.c_str(), true);
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
    int n_cur = tokens.size();
    std::string raw_output;           // For --nothink: accumulates raw output for parsing
    common_chat_msg prev_msg;         // For --nothink: previous parse state
    bool use_chat_parser = FLAG_nothink && (chat_syntax.format != COMMON_CHAT_FORMAT_CONTENT_ONLY);

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
            // --nothink mode: parse output to filter reasoning content
            std::string token_str = llamafile_token_to_piece(ctx, id, true);
            raw_output += token_str;

            // Parse incrementally
            auto msg = common_chat_parse(raw_output, /*is_partial=*/true, chat_syntax);
            auto diffs = common_chat_msg_diff::compute_diffs(prev_msg, msg);

            // Only output content, skip reasoning
            for (const auto &diff : diffs) {
                if (!diff.content_delta.empty()) {
                    fputs(diff.content_delta.c_str(), stdout);
                    fflush(stdout);
                }
            }
            prev_msg = msg;
        } else {
            // Normal mode: output tokens directly
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
