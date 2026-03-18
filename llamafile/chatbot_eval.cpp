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
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "datauri.h"
#include "image.h"
#include "llama.h"  // llamafile wrapper
#include "string.h"
#include <cassert>
#include <string>
#include <vector>

namespace lf {
namespace chatbot {

bool eval_tokens(std::vector<llama_token> tokens) {
    int N = (int)tokens.size();
    if (tokens_used() + N > llama_n_ctx(g_ctx))
        return out_of_context(N);
    for (int i = 0; i < N; i += g_params->n_batch) {
        if (g_got_sigint) {
            g_got_sigint = false;
            clear_ephemeral();
            return false;
        }
        if (N > g_params->n_batch)
            print_ephemeral(format("loading prompt %d%%...", (int)((double)i / N * 100)));
        int n_eval = (int)tokens.size() - i;
        if (n_eval > g_params->n_batch)
            n_eval = g_params->n_batch;
        if (llama_decode(g_ctx, llama_batch_get_one(&tokens[i], n_eval)))
            return out_of_context(n_eval);
        g_history.insert(g_history.end(), tokens.begin() + i, tokens.begin() + i + n_eval);
    }
    clear_ephemeral();
    // this function is what computes /stats. we need to call it now
    // since llama_decode() kicks the can down the road to functions
    // like common_sampler_sample(). that is bad because the chatbot
    // returns control to the repl rather than sampling when loading
    // system and image prompts.
    llama_synchronize(g_ctx);
    return true;
}

bool eval_token(int id) {
    return eval_tokens({id});
}

bool eval_plain_text(const std::string &str, bool add_special, bool parse_special) {
    return eval_tokens(llamafile_tokenize(g_model, str, add_special, parse_special));
}

// Helper to evaluate chunks from mtmd_tokenize and update g_history.
// Uses mtmd_helper_eval_chunk_single() for consistency with llama.cpp server.
// Tracks n_past explicitly to handle M-RoPE models where n_pos != n_tokens.
static bool eval_mtmd_chunks(mtmd_input_chunks *chunks) {
    size_t n_chunks = mtmd_input_chunks_size(chunks);

    // Check context using n_pos (not n_tokens) for M-RoPE compatibility
    llama_pos total_pos = mtmd_helper_get_n_pos(chunks);
    if (tokens_used() + total_pos > llama_n_ctx(g_ctx))
        return out_of_context(total_pos);

    // Track position explicitly across chunks (like llama.cpp server)
    llama_pos n_past = tokens_used();

    // Evaluate each chunk using the same helper as llama.cpp server
    for (size_t i = 0; i < n_chunks; i++) {
        if (g_got_sigint) {
            g_got_sigint = false;
            clear_ephemeral();
            return false;
        }

        const mtmd_input_chunk *chunk = mtmd_input_chunks_get(chunks, i);
        auto chunk_type = mtmd_input_chunk_get_type(chunk);

        // Show progress for large prompts or image processing
        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
            if ((int)n_tokens > g_params->n_batch)
                print_ephemeral("loading prompt...");
        } else {
            print_ephemeral("processing image...");
        }

        // Use the same helper function as llama.cpp server
        llama_pos new_n_past = n_past;
        int32_t ret = mtmd_helper_eval_chunk_single(
            g_mtmd, g_ctx, chunk,
            n_past,
            0,  // seq_id
            g_params->n_batch,
            true,  // logits_last
            &new_n_past);

        if (ret != 0) {
            if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT)
                err("failed to evaluate text chunk");
            else
                err("failed to evaluate image chunk");
            return false;
        }

        // Update history for context tracking
        // Use n_pos (not n_tokens) for M-RoPE model compatibility
        llama_pos n_pos = mtmd_input_chunk_get_n_pos(chunk);
        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            // Add actual tokens to history
            size_t n_text_tokens;
            const llama_token *tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_text_tokens);
            g_history.insert(g_history.end(), tokens, tokens + n_text_tokens);
        } else {
            // Add placeholder tokens for image/audio (use n_pos for M-RoPE)
            for (llama_pos j = 0; j < n_pos; j++) {
                g_history.push_back(IMAGE_PLACEHOLDER_TOKEN);
            }
        }

        // Update position for next chunk
        n_past = new_n_past;
    }

    clear_ephemeral();
    llama_synchronize(g_ctx);
    return true;
}

// Evaluate a string that may contain embedded data URIs for images.
// Images are processed using the mtmd API which requires tokenizing
// text and images together.
bool eval_string(std::string_view s, bool add_special, bool parse_special) {
    // Extract data URIs from the input
    DataUriExtraction extraction = extract_data_uris(s, mtmd_default_marker());

    // If no images found, just evaluate as plain text
    if (extraction.images.empty()) {
        return eval_plain_text(std::string(s), add_special, parse_special);
    }

    // We have images - check if we have multimodal support
    if (!g_mtmd) {
        err("multimodal model not loaded (use --mmproj to specify vision model)");
        return false;
    }

    // Create bitmaps from decoded image data
    mtmd::bitmaps bitmaps;
    for (const auto &image : extraction.images) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(
            g_mtmd, (const unsigned char *)image.data(), image.size()));
        if (!bmp.ptr) {
            err("failed to load image");
            return false;
        }
        bitmaps.entries.push_back(std::move(bmp));
    }

    // Use mtmd_tokenize to process text with images
    mtmd_input_text text;
    text.text = extraction.modified_text.c_str();
    text.add_special = add_special;
    text.parse_special = parse_special;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(g_mtmd, chunks.ptr.get(), &text,
                                bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    if (res != 0) {
        if (res == 1)
            err("number of images doesn't match number of markers in prompt");
        else if (res == 2)
            err("image preprocessing error");
        else
            err("failed to tokenize prompt with images (error %d)", res);
        return false;
    }

    // Evaluate the chunks
    return eval_mtmd_chunks(chunks.ptr.get());
}

} // namespace chatbot
} // namespace lf
