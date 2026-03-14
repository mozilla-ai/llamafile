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

#pragma once
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <signal.h>

#include "chat.h"
#include "chatbot_backend.h"

#define DEFAULT_SYSTEM_PROMPT \
    "A chat between a curious human and an artificial intelligence assistant. " \
    "The assistant gives helpful, detailed, and polite answers to the " \
    "human's questions."

struct bestlineCompletions;
struct common_params;
struct common_sampler;
struct llama_context;
struct llama_model;
struct mtmd_context;

namespace lf {
namespace chatbot {

enum Role {
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
};

enum SpecialToken {
    IMAGE_PLACEHOLDER_TOKEN = -31337,
};

// Result of extracting data URIs from text
struct DataUriExtraction {
    std::string modified_text;           // text with data URIs replaced by marker
    std::vector<std::string> images;     // decoded image data
    const char *marker;                  // marker string used for replacement
};

extern bool g_manual_mode;
extern bool g_said_something;
extern char g_last_printed_char;
extern mtmd_context *g_mtmd;          // multimodal context (replaces g_clip)
extern enum Role g_role;
extern common_params *g_params;       // pointer to params (replaces gpt_params)
extern common_sampler *g_sampler;     // sampler context (new)
extern std::vector<common_chat_msg> g_messages;  // chat message history
extern llama_context *g_ctx;
extern llama_model *g_model;
extern std::vector<int> g_history;
extern volatile sig_atomic_t g_got_sigint;
extern bool g_interrupted_exit;
extern common_chat_templates_ptr g_chat_templates;
extern common_chat_parser_params g_chat_syntax;
extern std::string g_pending_file_content;  // accumulated /upload content awaiting user message
extern ChatBackend *g_backend;               // active inference backend

// Original entry point: loads its own model (--chat mode)
int main(int argc, char **argv);

// API client entry point for combined mode (HTTP client to local server)
int api_main(const std::string &server_url, const std::string &system_prompt,
             const std::string &model_path, std::function<void()> shutdown_fn);

// CLI mode: single prompt -> response, then exit
int cli_main(int argc, char **argv);

// Backend factories
std::unique_ptr<ChatBackend> create_direct_backend();
std::unique_ptr<ChatBackend> create_api_backend(const std::string &server_url);

bool eval_string(std::string_view, bool, bool);
DataUriExtraction extract_data_uris(std::string_view, const char *marker);
bool eval_token(int);
bool eval_tokens(std::vector<int>);
bool handle_command(const char *);
bool is_base_model();
bool out_of_context(int);
char *on_hint(const char *, const char **, const char **);
const char *get_role_color(enum Role);
const char *get_role_name(enum Role);
enum Role cycle_role(enum Role);
enum Role get_next_role(enum Role);
int tokens_used(void);
std::string token_to_piece(const llama_context *, int, bool);
void adjust_stacks(int, int);
void clear_ephemeral(void);
void ensure_newline();
void err(const char *, ...);
void fix_stacks(void);
void logo(char **);
void on_clear(const std::vector<std::string> &);
void on_completion(const char *, int, bestlineCompletions *);
void on_context(const std::vector<std::string> &);
void on_dump(const std::vector<std::string> &);
void on_forget(const std::vector<std::string> &);
void on_help(const std::vector<std::string> &);
void on_manual(const std::vector<std::string> &);
void on_pop(const std::vector<std::string> &);
void on_push(const std::vector<std::string> &);
void on_stack(const std::vector<std::string> &);
void on_undo(const std::vector<std::string> &);
void on_upload(const std::vector<std::string> &);
void print(const std::string_view &);
void print_ephemeral(const std::string_view &);
void record_undo(void);
void repl(ChatBackend &backend);
void rewind(int);

} // namespace chatbot
} // namespace lf
