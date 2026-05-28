#!/bin/bash
# Fetch the prebuilt llama.cpp web UI assets from Hugging Face.
#
# Upstream's tools/ui/scripts/ui-assets.cmake pulls the Svelte build outputs
# from the ggml-org/llama-ui HF bucket. We do the same here so the cosmocc
# build never has to run a JS toolchain. If the fetch fails (no network,
# version not yet published, HF down) we leave tools/ui/dist empty; BUILD.mk's
# embed step then generates a no-asset ui.cpp and the server still works, just
# without the web UI.
#
# Run by apply-patches.sh (i.e. `make setup`); also safe to run standalone to
# re-fetch the UI without re-applying patches.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$SCRIPT_DIR/../llama.cpp"

HF_BUCKET="${LLAMAFILE_UI_HF_BUCKET:-llama-ui}"
HF_BASE="https://huggingface.co/buckets/ggml-org/${HF_BUCKET}/resolve"
HF_TREE_API="https://huggingface.co/api/buckets/ggml-org/${HF_BUCKET}/tree"
UI_DIST="$LLAMA_DIR/tools/ui/dist"
UI_ASSETS=(bundle.css bundle.js index.html loading.html)

# Echo the highest bNNNN tag in the bucket whose build number is <= $1, or
# nothing. The tree API returns directories in ascending order, 100 per page,
# with a `Link: <...>; rel="next"` header for the following page.
pick_ui_tag() {
    local cur="$1"
    local url="${HF_TREE_API}?limit=100&recursive=false"
    local hdrs best="" n saw_newer guard=0 body
    hdrs="$(mktemp)"
    while [ -n "$url" ] && [ "$guard" -lt 100 ]; do
        guard=$((guard + 1))
        body="$(curl -fsSL --max-time 30 -D "$hdrs" "$url" 2>/dev/null)" || break
        saw_newer=0
        for n in $(printf '%s' "$body" | grep -oE '"path":"b[0-9]+"' \
                | grep -oE '[0-9]+'); do
            if [ "$n" -le "$cur" ]; then
                if [ -z "$best" ] || [ "$n" -gt "$best" ]; then
                    best="$n"
                fi
            else
                saw_newer=1
            fi
        done
        # Once a page holds a tag newer than us, all later pages are newer too,
        # so there is nothing better to find.
        if [ "$saw_newer" -eq 1 ]; then
            break
        fi
        url="$(grep -i '^link:' "$hdrs" \
            | grep -oE '<[^>]+>; *rel="next"' | grep -oE 'https?://[^>]+' | head -1)"
    done
    rm -f "$hdrs"
    [ -n "$best" ] && printf 'b%s' "$best"
}

echo ""
echo "Fetching prebuilt web UI assets from Hugging Face..."

# Pick the version to download. Upstream's CMake just tries the exact build
# number and then "latest", but for llamafile that's fragile: the exact tag for
# our pinned commit is often not published yet, and "latest" can be built
# against a newer backend than the llama.cpp we've pinned. Instead we enumerate
# the tags actually present in the bucket and pick the newest one that is <= our
# build number. "latest" is kept only as a last-resort fallback (enumeration
# failed, e.g. offline / API change, or our commit predates every published
# tag) so the build can still get *some* UI rather than none.
UI_CUR_BUILD="$(cd "$LLAMA_DIR" && git describe --tags --always 2>/dev/null \
    | grep -oE '^b[0-9]+' | grep -oE '[0-9]+' || true)"

UI_CANDIDATES=()
if [ -n "$UI_CUR_BUILD" ]; then
    echo "  resolving newest UI tag <= b$UI_CUR_BUILD ..."
    UI_BEST_TAG="$(pick_ui_tag "$UI_CUR_BUILD")"
    if [ -n "$UI_BEST_TAG" ]; then
        echo "  selected UI tag $UI_BEST_TAG"
        UI_CANDIDATES+=("$UI_BEST_TAG")
    else
        echo "  no UI tag <= b$UI_CUR_BUILD found in bucket; will try 'latest'"
    fi
fi
UI_CANDIDATES+=("latest")

mkdir -p "$UI_DIST"
ui_ok=false
for v in "${UI_CANDIDATES[@]}"; do
    echo "  trying $HF_BASE/$v ..."
    fail=false
    for asset in "${UI_ASSETS[@]}" checksums.txt; do
        if ! curl -fsSL --max-time 60 -o "$UI_DIST/$asset" \
                "$HF_BASE/$v/$asset?download=true"; then
            fail=true
            break
        fi
    done
    if $fail; then
        continue
    fi

    # Best-effort sha256 verification against checksums.txt (one "<hash>  <name>"
    # line per asset). Skip if shasum/sha256sum isn't around.
    if command -v shasum >/dev/null 2>&1; then
        sha_cmd="shasum -a 256"
    elif command -v sha256sum >/dev/null 2>&1; then
        sha_cmd="sha256sum"
    else
        sha_cmd=""
    fi
    if [ -n "$sha_cmd" ] && [ -f "$UI_DIST/checksums.txt" ]; then
        bad=false
        for asset in "${UI_ASSETS[@]}"; do
            want=$(awk -v a="$asset" '$2 == a { print $1 }' "$UI_DIST/checksums.txt")
            got=$($sha_cmd "$UI_DIST/$asset" | awk '{print $1}')
            if [ -z "$want" ] || [ "$want" != "$got" ]; then
                echo "  checksum mismatch for $asset (want=$want got=$got)"
                bad=true
                break
            fi
        done
        if $bad; then
            continue
        fi
    fi

    echo "  fetched UI assets from $v"
    ui_ok=true
    break
done

if ! $ui_ok; then
    echo "  warning: could not download UI assets; server will build without the web UI"
    rm -f "$UI_DIST"/*
fi
