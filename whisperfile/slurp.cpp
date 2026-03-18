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

#include "slurp.h"

// Use miniaudio declarations only - implementation is in whisper.cpp library
// (common-whisper.cpp defines MINIAUDIO_IMPLEMENTATION)
#include "miniaudio.h"

#include "llamafile/llamafile.h"
#include <math.h>
#include <stdio.h>

// Conditional logging macro - respects FLAG_log_disable
#define tinylogf(...) (void)(!FLAG_log_disable && (fprintf(stderr, __VA_ARGS__), 0))

static int get_audio_file_channels(const char *fname) {
    ma_decoder decoder;
    ma_result rc = ma_decoder_init_file(fname, NULL, &decoder);
    if (rc != MA_SUCCESS) {
        tinylogf("%s: failed to open audio file: %s (we support .wav, .mp3, .flac, and .ogg)\n",
                 fname, ma_result_description(rc));
        return -1;
    }
    int channels = decoder.outputChannels;
    ma_decoder_uninit(&decoder);
    return channels;
}

/**
 * Reads entire pulse-code modulation content of audio file into memory.
 *
 * This function reads raw audio data from an MP3/WAV/OGG/FLAC file into
 * `pcmf32` at 16kHz sample rate. Resampling, channel mixing, and
 * data type conversions will be performed as necessary.
 *
 * If `stereo` is true, then `pcmf32s` will also be populated with two
 * vectors, holding the left and right audio channels, and `pcmf32` will
 * receive their mixture. If the audio file does not have two or more
 * channels, then an error is returned.
 *
 * The output vectors are not cleared. Therefore this function may be
 * called multiple times to append audio files.
 */
bool slurp_audio_file(const char *fname,
                      std::vector<float> &pcmf32,
                      std::vector<std::vector<float>> &pcmf32s,
                      bool stereo) {

    // validate stereo is stereo
    if (stereo) {
        int channels = get_audio_file_channels(fname);
        if (channels == -1)
            return false;
        if (channels < 2) {
            tinylogf("%s: audio file is mono when stereo is required\n", fname);
            return false;
        }
    }

    // create decoder - output at 16kHz for Whisper
    ma_decoder_config decoderConfig =
            ma_decoder_config_init(ma_format_f32, 1 + stereo, 16000);
    decoderConfig.resampling.algorithm = ma_resample_algorithm_linear;
    decoderConfig.resampling.linear.lpfOrder = 8;

    // open input file
    ma_decoder decoder;
    ma_result rc = ma_decoder_init_file(fname, &decoderConfig, &decoder);
    if (rc != MA_SUCCESS) {
        tinylogf("%s: failed to open audio file: %s (we support .wav, .mp3, .flac, and .ogg)\n",
                 fname, ma_result_description(rc));
        return false;
    }

    // load pulse-code modulation samples
    if (!stereo) {
        ma_uint64 total = pcmf32.size();
        ma_uint64 want = 512;
        ma_uint64 got;
        do {
            pcmf32.resize(total + want);
            rc = ma_decoder_read_pcm_frames(&decoder, &pcmf32[total], want, &got);
            if (rc != MA_SUCCESS && rc != MA_AT_END) {
                ma_decoder_uninit(&decoder);
                tinylogf("%s: failed to read pcm frames from audio file: %s\n",
                         fname, ma_result_description(rc));
                return false;
            }
            pcmf32.resize((total += got));
        } while (got == want && rc != MA_AT_END);
    } else {
        float frames[512];
        ma_uint64 want = sizeof(frames) / sizeof(*frames) / 2;
        ma_uint64 got;
        pcmf32s.resize(2);
        do {
            rc = ma_decoder_read_pcm_frames(&decoder, frames, want, &got);
            if (rc != MA_SUCCESS && rc != MA_AT_END) {
                ma_decoder_uninit(&decoder);
                tinylogf("%s: failed to read pcm frames from audio file: %s\n",
                         fname, ma_result_description(rc));
                return false;
            }
            for (ma_uint64 i = 0; i < got; ++i) {
                float left = frames[i*2+0];
                float right = frames[i*2+1];
                // Mix stereo to mono using RMS for better audio quality
                pcmf32.push_back(sqrtf((left*left + right*right) / 2));
                pcmf32s[0].push_back(left);
                pcmf32s[1].push_back(right);
            }
        } while (got == want && rc != MA_AT_END);
    }

    // we're done
    ma_decoder_uninit(&decoder);
    return true;
}
