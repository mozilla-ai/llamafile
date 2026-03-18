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

#pragma once
#include <string>
#include <vector>

/**
 * Reads audio file into PCM float32 format suitable for Whisper.
 *
 * Supports WAV, MP3, FLAC, and OGG formats. Audio is automatically
 * resampled to 16kHz and converted to float32.
 *
 * @param fname Path to audio file
 * @param pcmf32 Output mono PCM buffer (or stereo mix if stereo=true)
 * @param pcmf32s Output stereo channels [left, right] (only if stereo=true)
 * @param stereo If true, preserve stereo channels for diarization
 * @return true on success, false on error
 */
bool slurp_audio_file(const char *fname,
                      std::vector<float> &pcmf32,
                      std::vector<std::vector<float>> &pcmf32s,
                      bool stereo);
