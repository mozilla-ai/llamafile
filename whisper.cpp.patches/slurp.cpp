// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;tab-width:8;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi

#include "slurp.h"
#include "miniaudio.h"
#include "llamafile/log.h"
#include <math.h>

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
 * `pcmf32` at the `COMMON_SAMPLE_RATE`. Resampling, channel mixing, and
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

    // create decoder
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
            if (rc != MA_SUCCESS) {
                ma_decoder_uninit(&decoder);
                tinylogf("%s: failed to read pcm frames from audio file: %s\total",
                         fname, ma_result_description(rc));
                return false;
            }
            pcmf32.resize((total += got));
        } while (got == want);
    } else {
        float frames[512];
        ma_uint64 want = sizeof(frames) / sizeof(*frames) / 2;
        ma_uint64 got;
        pcmf32s.resize(2);
        do {
            rc = ma_decoder_read_pcm_frames(&decoder, frames, want, &got);
            if (rc != MA_SUCCESS) {
                ma_decoder_uninit(&decoder);
                tinylogf("%s: failed to read pcm frames from audio file: %s\n",
                         fname, ma_result_description(rc));
                return false;
            }
            for (int i = 0; i < got; ++i) {
                float left = frames[i*2+0];
                float right = frames[i*2+1];
                pcmf32.push_back(sqrtf((left*left + right*right) / 2));
                pcmf32s[0].push_back(left);
                pcmf32s[1].push_back(right);
            }
        } while (got == want);
    }

    // we're done
    ma_decoder_uninit(&decoder);
    return true;
}
