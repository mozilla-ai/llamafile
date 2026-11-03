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

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "llamafile.h"
#include "sampling.h"

namespace lf {
namespace chatbot {

// DirectBackend: wraps llama_decode for --chat mode (standalone TUI).
// Uses the existing global state (g_ctx, g_model, g_sampler, etc.)
// for inference and KV cache management.
class DirectBackend : public ChatBackend {
public:
    std::string complete(
        const std::vector<common_chat_msg> &messages,
        TokenCallback on_token) override
    {
        std::string assistant_content;
        const llama_vocab *vocab = llama_model_get_vocab(g_model);

        // Check if we should use chat parsing (for think mode models)
        const bool use_chat_parser =
            g_chat_syntax.format != COMMON_CHAT_FORMAT_CONTENT_ONLY;
        std::string raw_output;
        common_chat_msg prev_msg;

        for (;;) {
            if (g_got_sigint) {
                eval_token(llamafile_token_eot(g_model));
                break;
            }
            llama_token id = common_sampler_sample(g_sampler, g_ctx, -1);
            common_sampler_accept(g_sampler, id, true);
            if (!eval_token(id))
                break;
            if (llama_vocab_is_eog(vocab, id))
                break;

            if (use_chat_parser) {
                std::string token_str = token_to_piece(g_ctx, id, true);
                raw_output += token_str;
                auto msg = common_chat_parse(raw_output, true, g_chat_syntax);
                auto diffs = common_chat_msg_diff::compute_diffs(prev_msg, msg);

                for (const auto &diff : diffs) {
                    std::string content_delta = diff.content_delta;
                    std::string reasoning_delta = diff.reasoning_content_delta;
                    if (!content_delta.empty())
                        assistant_content += content_delta;
                    if (!content_delta.empty() || !reasoning_delta.empty()) {
                        if (!on_token(content_delta, reasoning_delta))
                            goto done;
                    }
                }
                prev_msg = msg;
            } else {
                std::string token_str = token_to_piece(g_ctx, id, g_params->special);
                assistant_content += token_str;
                if (!on_token(token_str, ""))
                    goto done;
            }
        }
    done:
        return assistant_content;
    }

    int context_used() override {
        return tokens_used();
    }

    int context_max() override {
        return llama_n_ctx(g_ctx);
    }

    void print_stats() override {
        // Format and print directly rather than going through llama.cpp's
        // common_perf_print(). This keeps /stats consistent across chat and
        // combined mode and off the logging path entirely.
        const llama_perf_context_data ctx_perf = llama_perf_context(g_ctx);
        const llama_perf_sampler_data smpl_perf =
            llama_perf_sampler(common_sampler_get(g_sampler));
        const double total_ms = 1e-3 * ggml_time_us() - ctx_perf.t_start_ms;

        // llama_perf_context() clamps both token counts with std::max(1, ...)
        // (llama-context.cpp, perf_get_data) so that its own printer can divide
        // by them. Before anything is decoded the real counters are 0 and that
        // guard would show up here as a phantom token on each row. The
        // durations are not clamped, so use them to tell the cases apart.
        const int prompt_n = ctx_perf.t_p_eval_ms > 0 ? ctx_perf.n_p_eval : 0;
        const int gen_n = ctx_perf.t_eval_ms > 0 ? ctx_perf.n_eval : 0;

        // n_eval and t_eval_us accumulate together, one per decoded token, so
        // here the rate basis is simply the token count
        if (!print_stats_core(prompt_n, ctx_perf.t_p_eval_ms, gen_n,
                              ctx_perf.t_eval_ms, gen_n, total_ms))
            return; // nothing decoded yet; load time reappears after the first turn

        // --chat only: the server backend has no equivalent for these
        printf("  %-10s %17.2f ms\n", "load", ctx_perf.t_load_ms);
        if (smpl_perf.n_sample > 0)
            printf("  %-10s %17.2f ms\n", "samplers", smpl_perf.t_sample_ms);
        printf("  %-10s %17d\n", "graphs", ctx_perf.n_reused);
    }

    void on_clear() override {
        lf::chatbot::on_clear({});
    }

    void on_push() override {
        lf::chatbot::on_push({});
    }

    void on_pop() override {
        lf::chatbot::on_pop({});
    }

    void on_undo() override {
        lf::chatbot::on_undo({});
    }

    void on_forget(int n) override {
        lf::chatbot::on_forget({});
    }

    bool supports_dump() override { return true; }

    void on_dump(int fd) override {
        std::vector<std::string> args = {"dump"};
        lf::chatbot::on_dump(args);
    }

    bool supports_manual_mode() override { return true; }
};

// Factory function
std::unique_ptr<ChatBackend> create_direct_backend() {
    return std::make_unique<DirectBackend>();
}

} // namespace chatbot
} // namespace lf
