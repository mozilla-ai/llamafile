#pragma once
#include <string>
#include <vector>

bool slurp_audio_file(const char *fname,
                      std::vector<float> &pcmf32,
                      std::vector<std::vector<float>> &pcmf32s,
                      bool stereo);
