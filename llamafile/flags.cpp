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

#include "flags.h"
#include "debug.h"
#include "llamafile.h"
#include "trust.h"

#include <cosmo.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>

#include "llama.cpp/cores.h"
#include "llama.cpp/llama.h"
#include "llamafile/macros.h"

bool FLAGS_READY = false;
bool FLAG_ascii = false;
bool FLAG_completion_mode = false;
bool FLAG_fast = false;
bool FLAG_iq = false;
bool FLAG_log_disable = false;
bool FLAG_mlock = false;
bool FLAG_mmap = true;
bool FLAG_no_display_prompt = false;
bool FLAG_nocompile = false;
bool FLAG_nologo = false;
bool FLAG_precise = false;
bool FLAG_recompile = false;
bool FLAG_tinyblas = false;
bool FLAG_trace = false;
bool FLAG_unsecure = false;
bool FLAG_v2 = false;
bool FLAG_lora_init_without_apply = false;
const char *FLAG_chat_template = "";
const char *FLAG_db = nullptr;
const char *FLAG_db_startup_sql = "PRAGMA journal_mode=WAL;"
                                  "PRAGMA synchronous=NORMAL;";
const char *FLAG_file = nullptr;
const char *FLAG_ip_header = nullptr;
const char *FLAG_listen = "127.0.0.1:8080";
const char *FLAG_mmproj = nullptr;
const char *FLAG_model = nullptr;
const char *FLAG_prompt = nullptr;
const char *FLAG_url_prefix = "";
const char *FLAG_www_root = "/zip/www";
const char *FLAG_lora = nullptr;

// Multiple LoRA adapters support
struct llamafile_lora_adapter_info FLAG_lora_adapters[MAX_LORA_ADAPTERS] = {0};
int FLAG_lora_adapters_count = 0;
double FLAG_token_rate = 1;
float FLAG_decay_growth = .01;
float FLAG_frequency_penalty = 0;
float FLAG_presence_penalty = 0;
float FLAG_reserve_tokens = .15;
float FLAG_temperature = .8;
float FLAG_top_p = .95;
int FLAG_batch = 256;
int FLAG_ctx_size = 8192;
int FLAG_decay_delay = 60 * 5;
int FLAG_flash_attn = false;
int FLAG_gpu = 0;
int FLAG_http_ibuf_size = 5 * 1024 * 1024;
int FLAG_http_obuf_size = 1024 * 1024;
int FLAG_keepalive = 5;
int FLAG_main_gpu = 0;
int FLAG_n_gpu_layers = -1;
int FLAG_slots = 1;
int FLAG_split_mode = LLAMA_SPLIT_MODE_LAYER;
int FLAG_threads = MIN(cpu_get_num_math(), 20);
int FLAG_threads_batch = cpu_get_num_math();
int FLAG_token_burst = 100;
int FLAG_token_cidr = 24;
int FLAG_ubatch = 512;
int FLAG_verbose = 0;
int FLAG_warmup = true;
int FLAG_workers;
unsigned FLAG_seed = LLAMA_DEFAULT_SEED;

std::vector<std::string> FLAG_headers;

static wontreturn void usage(int rc, int fd) {
    tinyprint(fd, "usage: ", program_invocation_name, " -m MODEL -l [HOST:]PORT\n", NULL);
    exit(rc);
}

static wontreturn void error(const char *message) {
    tinyprint(2, program_invocation_name, ": ", message, "\n", NULL);
    exit(1);
}

static wontreturn void bad(const char *flag) {
    tinyprint(2, program_invocation_name, ": bad value for ", flag, "\n", NULL);
    exit(1);
}

static wontreturn void nogpu(const char *flag) {
    tinyprint(2, program_invocation_name, ": ", flag, " was passed but ",
              program_invocation_short_name, " doesn't support GPU mode yet.\n", NULL);
    exit(1);
}

static wontreturn void missing(const char *flag) {
    tinyprint(2, program_invocation_name, ": ", flag, " missing argument\n", NULL);
    exit(1);
}

static wontreturn void required(const char *flag) {
    tinyprint(2, program_invocation_name, ": ", flag, " is required\n", NULL);
    exit(1);
}

static wontreturn void unknown(const char *flag) {
    tinyprint(2, program_invocation_name, ": ", flag, " unknown argument\n", NULL);
    exit(1);
}

static bool is_valid_chat_template(const char *tmpl) {
    llama_chat_message chat[] = {{"user", "test"}};
    return llama_chat_apply_template(nullptr, tmpl, chat, 1, true, nullptr, 0) >= 0;
}

void llamafile_get_flags(int argc, char **argv) {
    bool program_supports_gpu = FLAG_gpu != LLAMAFILE_GPU_DISABLE;
    for (int i = 1; i < argc;) {
        const char *flag = argv[i++];

        if (*flag != '-')
            break;

        //////////////////////////////////////////////////////////////////////
        // logging flags

        if (!strcmp(flag, "--log-disable")) {
            FLAG_log_disable = true;
            continue;
        }

        if (!strcmp(flag, "-v") || !strcmp(flag, "--verbose")) {
            FLAG_verbose++;
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // chatbot flags

        if (!strcmp(flag, "--v2")) {
            FLAG_v2 = true;
            continue;
        }

        if (!strcmp(flag, "--ascii")) {
            FLAG_ascii = true;
            continue;
        }

        if (!strcmp(flag, "--nologo")) {
            FLAG_nologo = true;
            continue;
        }

        if (!strcmp(flag, "--chatbot-mode")) {
            FLAG_completion_mode = false;
            continue;
        }

        if (!strcmp(flag, "--completion-mode")) {
            FLAG_completion_mode = true;
            continue;
        }

        if (!strcmp(flag, "--no-display-prompt") || //
            !strcmp(flag, "--silent-prompt")) {
            FLAG_no_display_prompt = true;
            continue;
        }

        if (!strcmp(flag, "--display-prompt")) {
            FLAG_no_display_prompt = false;
            continue;
        }

        if (!strcmp(flag, "--prompt") || !strcmp(flag, "--system-prompt")) {
            if (i == argc)
                missing("--prompt");
            FLAG_prompt = argv[i++];
            continue;
        }

        if (!strcmp(flag, "--db")) {
            if (i == argc)
                missing("--db");
            FLAG_db = argv[i++];
            continue;
        }

        if (!strcmp(flag, "--db-startup-sql")) {
            if (i == argc)
                missing("--db-startup-sql");
            FLAG_db_startup_sql = argv[i++];
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // server flags

        if (!strcmp(flag, "--server"))
            continue;

        if (!strcmp(flag, "-l") || !strcmp(flag, "--listen")) {
            if (i == argc)
                missing("--listen");
            FLAG_listen = argv[i++];
            continue;
        }

        if (!strcmp(flag, "-k") || !strcmp(flag, "--keepalive")) {
            if (i == argc)
                missing("--keepalive");
            FLAG_keepalive = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "-w") || !strcmp(flag, "--workers")) {
            if (i == argc)
                missing("--workers");
            FLAG_workers = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "--ip-header")) {
            if (i == argc)
                missing("--ip-header");
            cidr ip;
            FLAG_ip_header = argv[i++];
            continue;
        }

        if (!strcmp(flag, "--trust")) {
            if (i == argc)
                missing("--trust");
            cidr ip;
            if (!parse_cidr(argv[i++], &ip))
                error("invalid --trust CIDR (expect like 192.168.0.0/24)");
            FLAG_trust.push_back(ip);
            continue;
        }

        if (!strcmp(flag, "--token-rate")) {
            if (i == argc)
                missing("--token-rate");
            double rate = atof(argv[i++]);
            if (!rate)
                error("--token-rate can't be zero");
            double micros = 1e6 / rate;
            if (isnan(micros) || isinf(micros))
                error("--token-rate invalid");
            if (micros < 10)
                error("--token-rate too frequent (can't be above 100000 a.k.a. 1 per 10µs)");
            if (micros > 60 * 60 * 1e6)
                error("--token-rate too infrequent (can't be below 1/(60*60) a.k.a. 1 per hour)");
            FLAG_token_rate = rate;
            continue;
        }

        if (!strcmp(flag, "--token-burst")) {
            if (i == argc)
                missing("--token-burst");
            int burst = atoi(argv[i++]);
            if (!(1 <= burst && burst <= 127))
                error("--token-burst must be 1..127");
            FLAG_token_burst = burst;
            continue;
        }

        if (!strcmp(flag, "--token-cidr")) {
            if (i == argc)
                missing("--token-cidr");
            int cidr = atoi(argv[i++]);
            if (!(3 <= cidr && cidr <= 32))
                error("--token-cidr must be 3..32");
            FLAG_token_cidr = cidr;
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // http server flags

        if (!strcmp(flag, "--www-root")) {
            if (i == argc)
                missing("--www-root");
            FLAG_www_root = argv[i++];
            continue;
        }

        if (!strcmp(flag, "--url-prefix")) {
            if (i == argc)
                missing("--url-prefix");
            FLAG_url_prefix = argv[i++];
            if (!IsAcceptablePath(FLAG_url_prefix, -1)) {
                tinyprint(2, "error: --url-prefix must not have // or /. or /./ or /../\n", NULL);
                exit(1);
            }
            if (endswith(FLAG_url_prefix, "/")) {
                tinyprint(2, "error: --url-prefix must not be slash or end with slash\n", NULL);
                exit(1);
            }
            continue;
        }

        if (!strcmp(flag, "-H") || !strcmp(flag, "--header")) {
            if (i == argc)
                missing("--header");
            FLAG_headers.push_back(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "--http-ibuf-size")) {
            if (i == argc)
                missing("--http-ibuf-size");
            FLAG_http_ibuf_size = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "--http-obuf-size")) {
            if (i == argc)
                missing("--http-obuf-size");
            FLAG_http_obuf_size = atoi(argv[i++]);
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // sampling flags

        if (!strcmp(flag, "--seed")) {
            if (i == argc)
                missing("--seed");
            FLAG_seed = strtol(argv[i++], 0, 0);
            continue;
        }

        if (!strcmp(flag, "--temp") || //
            !strcmp(flag, "--temperature")) {
            if (i == argc)
                missing("--temp");
            FLAG_temperature = atof(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "--top-p")) {
            if (i == argc)
                missing("--top-p");
            FLAG_top_p = atof(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "--frequency-penalty")) {
            if (i == argc)
                missing("--frequency-penalty");
            FLAG_frequency_penalty = atof(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "--presence-penalty")) {
            if (i == argc)
                missing("--presence-penalty");
            FLAG_presence_penalty = atof(argv[i++]);
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // LoRA flags

        if (!strcmp(flag, "--lora")) {
            if (i == argc)
                missing("--lora");
            if (FLAG_lora_adapters_count >= MAX_LORA_ADAPTERS) {
                error("too many LoRA adapters (max 8)");
            }
            FLAG_lora_adapters[FLAG_lora_adapters_count].path = argv[i++];
            FLAG_lora_adapters[FLAG_lora_adapters_count].scale = 1.0f;
            FLAG_lora_adapters_count++;
            
            // Keep FLAG_lora for backward compatibility
            if (!FLAG_lora) {
                FLAG_lora = FLAG_lora_adapters[0].path;
            }
            continue;
        }

        if (!strcmp(flag, "--lora-scaled")) {
            if (i == argc)
                missing("--lora-scaled");
            const char* lora_adapter = argv[i++];
            if (i == argc)
                missing("--lora-scaled scale value");
            if (FLAG_lora_adapters_count >= MAX_LORA_ADAPTERS) {
                error("too many LoRA adapters (max 8)");
            }
            FLAG_lora_adapters[FLAG_lora_adapters_count].path = lora_adapter;
            FLAG_lora_adapters[FLAG_lora_adapters_count].scale = atof(argv[i++]);
            FLAG_lora_adapters_count++;
            
            // Keep FLAG_lora for backward compatibility
            if (!FLAG_lora) {
                FLAG_lora = FLAG_lora_adapters[0].path;
            }
            continue;
        }

        if (!strcmp(flag, "--lora-init-without-apply")) {
            FLAG_lora_init_without_apply = true;
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // model flags

        if (!strcmp(flag, "-c") || !strcmp(flag, "--ctx-size")) {
            char *ep;
            if (i == argc)
                missing("--ctx-size");
            FLAG_ctx_size = strtol(argv[i++], &ep, 10);
            if (*ep == 'k')
                FLAG_ctx_size *= 1024;
            continue;
        }

        if (!strcmp(flag, "--chat-template")) {
            if (i == argc)
                missing("--chat-template");
            if (!is_valid_chat_template(argv[i]))
                bad("--chat-template");
            FLAG_chat_template = argv[i++];
            continue;
        }

        if (!strcmp(flag, "-m") || !strcmp(flag, "--model")) {
            if (i == argc)
                missing("--model");
            FLAG_model = argv[i++];
            continue;
        }

        if (!strcmp(flag, "-mm") || !strcmp(flag, "--mmproj")) {
            if (i == argc)
                missing("--mmproj");
            FLAG_mmproj = argv[i++];
            continue;
        }

        if (!strcmp(flag, "-f") || !strcmp(flag, "--file")) {
            if (i == argc)
                missing("--file");
            FLAG_file = argv[i++];
            continue;
        }

        if (!strcmp(flag, "-t") || !strcmp(flag, "--threads")) {
            if (i == argc)
                missing("--threads");
            FLAG_threads = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "-tb") || !strcmp(flag, "--threads-batch")) {
            if (i == argc)
                missing("--threads-batch");
            FLAG_threads_batch = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "-b") || !strcmp(flag, "--batch-size")) {
            if (i == argc)
                missing("--batch-size");
            FLAG_batch = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "-ub") || !strcmp(flag, "--ubatch-size")) {
            if (i == argc)
                missing("--ubatch-size");
            FLAG_ubatch = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "-fa") || !strcmp(flag, "--flash-attn")) {
            if (i == argc)
                missing("--flash-attn");
            FLAG_flash_attn = true;
            continue;
        }

        if (!strcmp(flag, "--no-warmup")) {
            FLAG_warmup = false;
            continue;
        }

        if (!strcmp(flag, "--reserve-tokens")) {
            if (i == argc)
                missing("--reserve-tokens");
            const char *s = argv[i++];
            if (strchr(s, '.')) {
                float f = atof(s);
                if (!(0 < f && f <= 1))
                    error("--reserve-tokens FLOAT must be on the interval (0,1]");
                FLAG_reserve_tokens = f;
            } else {
                int n = atoi(s);
                if (!(1 <= n && n <= 100))
                    error("--reserve-tokens INT must be between 1 and 100");
                FLAG_reserve_tokens = n / 100.;
            }
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // resource management flags

        if (!strcmp(flag, "-s") || !strcmp(flag, "--slots")) {
            if (i == argc)
                missing("--slots");
            FLAG_slots = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "--decay-delay")) {
            if (i == argc)
                missing("--decay-delay");
            int n = atoi(argv[i++]);
            if (!(0 <= n && n <= 31536000))
                error("--decay-delay INT must be between 1 and 31536000");
            FLAG_decay_delay = n;
            continue;
        }

        if (!strcmp(flag, "--decay-growth")) {
            if (i == argc)
                missing("--decay-growth");
            float n = atof(argv[i++]);
            if (!(isnormal(n) && n > 0))
                error("--decay-growth FLOAT must be greater than 0");
            FLAG_decay_growth = n;
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // cpu flags

        if (!strcmp(flag, "--iq")) {
            FLAG_iq = true;
            continue;
        }

        if (!strcmp(flag, "--fast")) {
            FLAG_fast = true;
            continue;
        }

        if (!strcmp(flag, "--trace")) {
            FLAG_trace = true;
            FLAG_unsecure = true;
            continue;
        }

        if (!strcmp(flag, "--precise")) {
            FLAG_precise = true;
            continue;
        }

        if (!strcmp(flag, "--trap")) {
            FLAG_trap = true;
            FLAG_unsecure = true;
            llamafile_trapping_enabled(+1);
            continue;
        }

        if (!strcmp(flag, "--mlock")) {
            FLAG_mlock = true;
            continue;
        }

        if (!strcmp(flag, "--no-mmap")) {
            FLAG_mmap = false;
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // gpu flags

        if (!strcmp(flag, "--tinyblas")) {
            if (!program_supports_gpu)
                nogpu("--tinyblas");
            FLAG_tinyblas = true;
            continue;
        }

        if (!strcmp(flag, "--nocompile")) {
            if (!program_supports_gpu)
                nogpu("--nocompile");
            FLAG_nocompile = true;
            continue;
        }

        if (!strcmp(flag, "--recompile")) {
            if (!program_supports_gpu)
                nogpu("--recompile");
            FLAG_recompile = true;
            continue;
        }

        if (!strcmp(flag, "--gpu")) {
            if (i == argc)
                missing("--gpu");
            FLAG_gpu = llamafile_gpu_parse(argv[i++]);
            if (FLAG_gpu == LLAMAFILE_GPU_ERROR)
                bad("--gpu");
            continue;
        }

        if (!strcmp(flag, "-ngl") || //
            !strcmp(flag, "--gpu-layers") || //
            !strcmp(flag, "--n-gpu-layers")) {
            if (!program_supports_gpu)
                nogpu("--n-gpu-layers");
            if (i == argc)
                missing("--n-gpu-layers");
            FLAG_n_gpu_layers = atoi(argv[i++]);
            if (FLAG_n_gpu_layers <= 0)
                FLAG_gpu = LLAMAFILE_GPU_DISABLE;
            continue;
        }

        if (!strcmp(flag, "-mg") || !strcmp(flag, "--main-gpu")) {
            if (!program_supports_gpu)
                nogpu("--main-gpu");
            if (i == argc)
                missing("--main-gpu");
            FLAG_main_gpu = atoi(argv[i++]);
            continue;
        }

        if (!strcmp(flag, "-sm") || !strcmp(flag, "--split-mode")) {
            if (!program_supports_gpu)
                nogpu("--split-mode");
            if (i == argc)
                missing("--split-mode");
            const char *value = argv[i];
            if (!strcmp(value, "none"))
                FLAG_split_mode = LLAMA_SPLIT_MODE_NONE;
            else if (!strcmp(value, "layer"))
                FLAG_split_mode = LLAMA_SPLIT_MODE_LAYER;
            else if (!strcmp(value, "row"))
                FLAG_split_mode = LLAMA_SPLIT_MODE_ROW;
            else
                bad("--split-mode");
            continue;
        }

        //////////////////////////////////////////////////////////////////////
        // security flags

        if (!strcmp(flag, "--unsecure")) {
            FLAG_unsecure = true;
            continue;
        }

        unknown(flag);
    }

    if (!FLAG_model)
        required("--model");

    FLAGS_READY = true;
    FLAG_n_gpu_layers = llamafile_gpu_layers(FLAG_n_gpu_layers);
}
