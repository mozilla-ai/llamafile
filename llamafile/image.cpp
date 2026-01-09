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

#include "image.h"

#include <cosmo.h>
#include <stdint.h>
#include <termios.h>
#include <unistd.h>

#include "base64.hpp"
#include "macros.h"
#include "xterm.h"

// STB image libraries (from third_party via -isystem)
#include "stb/stb_image.h"
#include "stb/stb_image_resize2.h"

namespace lf {

/**
 * Returns true if binary is an image format we can use.
 *
 * This function performs extensive validation.
 */
bool is_image(const std::string_view &image) {
    if (!get_image_type(image))
        return false;
    int width, height, channels;
    unsigned char *data = stbi_load_from_memory((const unsigned char *)image.data(), image.size(),
                                                &width, &height, &channels, 0);
    if (!data)
        return false;
    stbi_image_free(data);
    return true;
}

/**
 * Returns true if binary is an image format we can use.
 *
 * This function performs extensive validation.
 */
void convert_image_to_uri(std::string *r, const std::string_view &image) {
    *r += "data:";
    *r += get_image_mime(get_image_type(image));
    *r += ";base64,";
    size_t size = r->size();
    size_t need = base64::required_encode_size(image.size());
    r->resize(size + need);
    r->resize(base64::encode(image.begin(), image.end(), r->begin() + size) - r->begin());
}

/**
 * Determines image file type from binary image content.
 *
 * Please be warned that some file formats have very broad magic numbers
 * that might overlap with unrelated files. For example the PGM magic is
 * "P5" which is a perfectly innocent ASCII prologue. To be certain that
 * binary is in fact an image, it should be loaded by a library like STB
 * that can do a more thorough validation.
 */
ImageType get_image_type(const std::string_view &binary) {
    if (binary.size() < 16)
        return IMAGE_TYPE_UNKNOWN;
    const char *p = binary.data();

    // check magic
    if (READ32LE(p) == READ32LE("\x1A\x45\xDF\xA3"))
        return IMAGE_TYPE_WEBM;
    if (READ32LE(p) == READ32LE("\x49\x49\x2A\x00") || //
        READ32LE(p) == READ32LE("\x4D\x4D\x00\x2A"))
        return IMAGE_TYPE_TIFF;
    if (READ32LE(p) == READ32LE("DDS "))
        return IMAGE_TYPE_DDS;
    if (READ32LE(p) == READ32LE("8BPS"))
        return IMAGE_TYPE_PSD;
    if (READ32LE(p) == READ32LE("IIBC"))
        return IMAGE_TYPE_JXR;
    if (READ32LE(p) == READ32LE("v/1\1"))
        return IMAGE_TYPE_EXR;
    if (READ32LE(p) == READ32LE("PIB "))
        return IMAGE_TYPE_PIC; // Softimage
    if (READ32LE(p) == READ32LE("\0\0\1\0"))
        return IMAGE_TYPE_ICO;
    if (READ32LE(p) == READ32LE("\0\0\2\0"))
        return IMAGE_TYPE_CUR;
    if ((READ32LE(p) & 0xffff) == READ32LE("BM\0"))
        return IMAGE_TYPE_BMP;
    if ((READ32LE(p) & 0xffffff) == READ32LE("\xFF\xD8\xFF"))
        return IMAGE_TYPE_JPG;
    if (READ64LE(p) == READ64LE("\x89\x50\x4E\x47\x0D\x0A\x1A\x0A"))
        return IMAGE_TYPE_PNG;
    if (READ64LE(p) == READ64LE("\x67\x69\x6D\x70\x20\x78\x63\x66"))
        return IMAGE_TYPE_XCF; // GIMP
    if ((READ64LE(p) & 0xffffffffffff) == READ64LE("GIF87a\0") ||
        (READ64LE(p) & 0xffffffffffff) == READ64LE("GIF89a\0"))
        return IMAGE_TYPE_GIF;
    if (READ32LE(p + 0) == READ32LE("RIFF") && //
        READ32LE(p + 8) == READ32LE("WEBP"))
        return IMAGE_TYPE_WEBP;
    if (READ32LE(p + 4) == READ32LE("ftyp")) {
        if (READ32LE(p + 8) == READ32LE("avif") || //
            READ32LE(p + 8) == READ32LE("avis"))
            return IMAGE_TYPE_AVIF;
        if (READ32LE(p + 8) == READ32LE("heic") || //
            READ32LE(p + 8) == READ32LE("heix"))
            return IMAGE_TYPE_HEIC;
        if (READ32LE(p + 8) == READ32LE("hevc") || //
            READ32LE(p + 8) == READ32LE("hevx"))
            return IMAGE_TYPE_HEIF;
    }

    // TGA: Check for valid image type and color map type
    if (binary.size() >= 18) {
        unsigned char imageType = binary[2];
        unsigned char colorMapType = binary[1];
        bool validImageType = (imageType == 1 || imageType == 2 || imageType == 3 ||
                               imageType == 9 || imageType == 10 || imageType == 11);
        bool validColorMapType = (colorMapType == 0 || colorMapType == 1);
        if (validImageType && validColorMapType)
            return IMAGE_TYPE_TGA;
    }

    // PCX: 0A followed by 0, 1, 2, 3, 4, or 5
    if (binary[0] == 0x0A && //
        (binary[1] >= 0x00 && binary[1] <= 0x05))
        return IMAGE_TYPE_PCX;

    // HDR: #?RADIANCE or #?RGBE
    if (!binary.substr(0, 10).find("#?RADIANCE") || //
        !binary.substr(0, 10).find("#?RGBE"))
        return IMAGE_TYPE_HDR;

    // PGM/PPM: P1-P6
    if (binary[0] == 'P') {
        char type = binary[1];
        if (type >= '1' && type <= '6') {
            if (type == '2' || type == '5') {
                return IMAGE_TYPE_PGM;
            } else if (type == '3' || type == '6') {
                return IMAGE_TYPE_PPM;
            }
        }
    }

    return IMAGE_TYPE_UNKNOWN;
}

const char *get_image_mime(ImageType type) {
    switch (type) {
    case IMAGE_TYPE_PNG:
        return "image/png";
    case IMAGE_TYPE_JPG:
        return "image/jpeg";
    case IMAGE_TYPE_GIF:
        return "image/gif";
    case IMAGE_TYPE_BMP:
        return "image/bmp";
    case IMAGE_TYPE_TGA:
        return "image/x-targa";
    case IMAGE_TYPE_HDR:
        return "image/vnd.radiance";
    case IMAGE_TYPE_PGM:
        return "image/x-portable-graymap";
    case IMAGE_TYPE_PPM:
        return "image/x-portable-pixmap";
    case IMAGE_TYPE_PIC:
        return "image/x-softimage";
    case IMAGE_TYPE_PSD:
        return "image/vnd.adobe.photoshop";
    case IMAGE_TYPE_WEBM:
        return "video/webm";
    case IMAGE_TYPE_WEBP:
        return "image/webp";
    case IMAGE_TYPE_ICO:
        return "image/x-icon";
    case IMAGE_TYPE_CUR:
        return "image/x-win-bitmap";
    case IMAGE_TYPE_TIFF:
        return "image/tiff";
    case IMAGE_TYPE_AVIF:
        return "image/avif";
    case IMAGE_TYPE_HEIF:
        return "image/heif";
    case IMAGE_TYPE_HEIC:
        return "image/heic";
    case IMAGE_TYPE_DDS:
        return "image/vnd-ms.dds";
    case IMAGE_TYPE_JXR:
        return "image/jxr";
    case IMAGE_TYPE_EXR:
        return "image/x-exr";
    case IMAGE_TYPE_PCX:
        return "image/x-pcx";
    case IMAGE_TYPE_XCF:
        return "image/x-xcf";
    default:
        return "application/octet-stream";
    }
}

/**
 * Prints image to terminal.
 */
int print_image(int fd, const std::string_view &image, int max_width) {

    // load image
    int width, height, channels;
    unsigned char *img = stbi_load_from_memory((const unsigned char *)image.data(), image.size(),
                                               &width, &height, &channels, 3);
    if (!img)
        return -1;

    // get terminal info
    bool use_rgb = is_rgb_terminal();
    struct winsize ws = {24, 80};
    tcgetwinsize(fd, &ws);

    // calculate new dimensions preserving aspect ratio
    int xn = MIN(max_width, ws.ws_col);
    int yn = (height * xn) / width; // *2 because we use half blocks
    yn = (yn + 1) & -2; // round up to even number

    // resize image
    unsigned char *resized = new unsigned char[xn * yn * 3];
    stbir_resize_uint8_srgb(img, width, height, 0, //
                            resized, xn, yn, 0, //
                            STBIR_RGB);
    stbi_image_free(img);

    // convert image to string using half blocks
    std::string s;
    for (int y = 0; y < yn; y += 2) {
        for (int x = 0; x < xn; ++x) {
            int upr = resized[((y + 0) * xn + x) * 3 + 0];
            int upg = resized[((y + 0) * xn + x) * 3 + 1];
            int upb = resized[((y + 0) * xn + x) * 3 + 2];
            int lor = resized[((y + 1) * xn + x) * 3 + 0];
            int log = resized[((y + 1) * xn + x) * 3 + 1];
            int lob = resized[((y + 1) * xn + x) * 3 + 2];
            char buf[48];
            if (use_rgb) {
                s.append(buf, snprintf(buf, sizeof(buf), "\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm▀",
                                       upr, upg, upb, lor, log, lob));
            } else {
                int upx = rgb2xterm256((upr << 16) | (upg << 8) | upb);
                int lox = rgb2xterm256((lor << 16) | (log << 8) | lob);
                s.append(buf, snprintf(buf, sizeof(buf), "\033[38;5;%dm\033[48;5;%dm▀", upx, lox));
            }
        }
        s += "\033[0m\n";
    }

    // write image to terminal
    int rc = write(fd, s.data(), s.size());
    delete[] resized;
    return rc;
}

} // namespace lf
