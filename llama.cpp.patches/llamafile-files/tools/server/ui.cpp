// Stub ui.cpp for llamafile builds. Returns no embedded assets so the
// server runs in API-only mode without the Svelte web UI.

#include "ui.h"

const llama_ui_asset * llama_ui_find_asset(const char *) {
    return nullptr;
}
