// Stub ui.h for llamafile builds. Upstream generates this file from
// embedded Svelte bundles via tools/ui/embed.cpp, but cosmocc builds do
// not run the JS toolchain. Without LLAMA_UI_HAS_ASSETS the server skips
// asset registration in tools/server/server-http.cpp.

#pragma once

#include <stddef.h>

struct llama_ui_asset {
    const char *          name;
    const unsigned char * data;
    size_t                size;
};

const llama_ui_asset * llama_ui_find_asset(const char * name);
