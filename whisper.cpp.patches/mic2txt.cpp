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

#include "color.h"
#include "whisper.h"

#include <math.h>
#include <cosmo.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>
#include <ctl/min.h>
#include <ctl/max.h>
#include <sys/stat.h>
#include <ctl/vector.h>
#include <cosmoaudio.h>

#define FRAMES_PER_SECOND 30
#define CHUNK_FRAMES (WHISPER_SAMPLE_RATE / FRAMES_PER_SECOND)

const char *g_model;
volatile sig_atomic_t g_done;
struct whisper_context *g_ctx;
struct whisper_context_params g_cparams;

static void onsig(int sig) {
    g_done = 1;
}

static void *load_model(void *arg) {
    g_ctx = whisper_init_from_file_with_params(g_model, g_cparams);
    if (!g_ctx) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        exit(2);
    }
    return 0;
}

int main(int argc, char *argv[]) {
    FLAG_gpu = LLAMAFILE_GPU_DISABLE;
    FLAG_log_disable = true;
    llamafile_check_cpu();
    ShowCrashReports();

    // get argument
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return 1;
    }
    struct stat st;
    g_model = argv[1];
    if (stat(g_model, &st)) {
        perror(g_model);
        return 1;
    }

    // detect teletypewriters
    bool should_print_color = isatty(1) && isatty(2);

    // connect to microphone
    int status;
    struct CosmoAudio *mic;
    struct CosmoAudioOpenOptions cao = {};
    cao.sizeofThis = sizeof(struct CosmoAudioOpenOptions);
    cao.deviceType = kCosmoAudioDeviceTypeCapture;
    cao.sampleRate = WHISPER_SAMPLE_RATE;
    cao.bufferFrames = CHUNK_FRAMES * 2;
    cao.channels = 1;
    if ((status = cosmoaudio_open(&mic, &cao)) != COSMOAUDIO_SUCCESS) {
        fprintf(stderr, "error: failed to open microphone: %d\n", status);
        return 1;
    }

    // load model
    pthread_t model_loader;
    g_cparams = whisper_context_default_params();
    unassert(!pthread_create(&model_loader, 0, load_model, 0));

    // setup signals
    struct sigaction sa;
    sa.sa_flags = 0;
    sa.sa_handler = onsig;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, 0);

    // record audio until ctrl-c is pressed
    ctl::vector<float> samples;
    while (!g_done) {
        size_t n = samples.size();
        samples.resize(n + CHUNK_FRAMES);
        cosmoaudio_poll(mic, (int[]){CHUNK_FRAMES}, 0);
        cosmoaudio_read(mic, &samples[n], CHUNK_FRAMES);
        fprintf(stderr, "\rcaptured %f seconds of audio... (press ctrl-c when done)",
                (double)samples.size() / WHISPER_SAMPLE_RATE);
        fflush(stderr);
    }
    fprintf(stderr, "\n");
    cosmoaudio_close(mic);

    // transcribe audio
    unassert(!pthread_join(model_loader, 0));
    whisper_full_params wparams =
            whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
    wparams.no_timestamps = true;
    wparams.suppress_non_speech_tokens = true;
    wparams.greedy.best_of = 8;
    wparams.beam_search.beam_size = 8;
    if ((status = whisper_full(g_ctx, wparams, samples.data(), samples.size()))) {
        fprintf(stderr, "error: whisper failed with %d\n", status);
        return 3;
    }
    int n_segments = whisper_full_n_segments(g_ctx);
    for (int i = 0; i < n_segments; ++i) {
        int n_tokens = whisper_full_n_tokens(g_ctx, i);
        for (int j = 0; j < n_tokens; ++j) {
            const whisper_token id = whisper_full_get_token_id(g_ctx, i, j);
            if (id >= whisper_token_eot(g_ctx))
                continue;
            const char *text = whisper_full_get_token_text(g_ctx, i, j);
            if (should_print_color) {
                float confidence = whisper_full_get_token_p(g_ctx, i, j);
                int colorcount = kRedToGreenXterm256.size();
                int colorindex = powf(confidence, 3) * colorcount;
                colorindex = ctl::max(0, ctl::min(colorcount - 1, colorindex));
                fprintf(stderr, "%s", kRedToGreenXterm256[colorindex].c_str());
                fflush(stderr);
            }
            printf("%s", text);
            fflush(stdout);
        }
    }
    if (should_print_color)
        fprintf(stderr, "\033[0m");
    printf("\n");
    whisper_free(g_ctx);
}
