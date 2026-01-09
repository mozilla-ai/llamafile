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

#include "llama.h"   // llamafile wrapper
#include "common.h"  // llama.cpp common (pulls in llama.h types)
#include <cassert>
#include <string>
#include <vector>

int llamafile_token_eot(llama_model *model) {
    const llama_vocab *vocab = llama_model_get_vocab(model);
    llama_token eot = llama_vocab_eot(vocab);
    if (eot != LLAMA_TOKEN_NULL)
        return eot;
    return llama_vocab_eos(vocab);
}

std::string llamafile_token_to_piece(const llama_context *ctx, llama_token token, bool special) {
    const llama_vocab *vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::string piece;
    piece.resize(piece.capacity());
    const int n_chars =
        llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check =
            llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        assert(check == -n_chars);
    } else {
        piece.resize(n_chars);
    }
    return piece;
}

std::vector<llama_token> llamafile_tokenize(const struct llama_model *model,
                                            const std::string_view &text, bool add_special,
                                            bool parse_special) {
    const llama_vocab *vocab = llama_model_get_vocab(model);
    int n_tokens = text.size() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.size(), result.data(), result.size(),
                              add_special, parse_special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.size(), result.data(), result.size(),
                                   add_special, parse_special);
        assert(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}
