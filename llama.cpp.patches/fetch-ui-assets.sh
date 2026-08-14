#!/bin/bash
# Fetch the prebuilt llama.cpp web UI assets from Hugging Face.
#
# Upstream ships the Svelte/PWA web UI as a `dist.tar.gz` tarball (plus a
# matching `dist.tar.gz.sha256`) in the ggml-org/llama-ui HF bucket, one
# directory per bNNNN build. We download the tarball for the build matching our
# pinned llama.cpp, verify it, and extract the full static site into
# llama.cpp/tools/ui/dist/. BUILD.mk then runs tools/ui/embed.cpp over that
# directory to bake every file into ui.cpp/ui.h.
#
# To keep the embedded payload (and the final binaries) small, we also build a
# dist/_gzip/ mirror with every file gzip-compressed under its original name;
# embed.cpp auto-detects _gzip and emits gzip-encoded assets, which
# server-http.cpp serves with Content-Encoding: gzip.
#
# If anything fails (no network, version not yet published, HF down) we leave
# tools/ui/dist empty; BUILD.mk's embed step then generates a no-asset ui.cpp
# and the server still works, just without the web UI.
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

# Echo the highest bNNNN tag in the bucket whose build number is <= $1, or
# nothing. The tree API returns directories in ascending order, 100 per page,
# with a `Link: <...>; rel="next"` header for the following page.
pick_ui_tag() {
    local cur="$1"
    local url="${HF_TREE_API}?limit=1000&recursive=false"
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

# Build the dist/_gzip/ mirror: every regular file under dist/, gzip-compressed
# under its original relative path. embed.cpp prefers _gzip when present.
build_gzip_mirror() {
    ( cd "$UI_DIST" && find . -type f ! -path './_gzip/*' -print0 \
        | while IFS= read -r -d '' f; do
            mkdir -p "_gzip/$(dirname "$f")"
            gzip -9 -c "$f" > "_gzip/$f"
        done )
}

# Echo (one per line) any asset embed.cpp requires that is absent from the tree.
#
# !!! KEEP IN SYNC WITH UPSTREAM tools/ui/embed.cpp !!!
# The list below mirrors the required_check[] table in llama.cpp/tools/ui/embed.cpp.
# Once dist/ is non-empty, embed.cpp hard-fails the build (return 1) if any of
# those assets is missing. If we only checked index.html here, a partial/drifted
# tarball would pass our check but then abort the *entire* build at the embed
# step instead of falling back to a UI-less build. So we validate the same set
# embed.cpp does and, when it's incomplete, clear dist/ to take the UI-less path.
#
# embed.cpp is an UPSTREAM file, so it can change on a llama.cpp bump. When it
# does (a new required asset, a renamed one), this list must be updated to match
# -- otherwise the two checks disagree again and the build-failure bug returns.
ui_missing_assets() {
    local root="$1" f b
    local -a bases=()
    while IFS= read -r f; do
        bases+=("$(basename "$f")")
    done < <(find "$root" -type f ! -path '*/_gzip/*' 2>/dev/null)

    _have() {  # _have <case-glob> : true if any basename matches
        for b in "${bases[@]}"; do
            case "$b" in $1) return 0 ;; esac
        done
        return 1
    }
    _have 'index.html'           || echo 'index.html'
    _have 'manifest.webmanifest' || echo 'manifest.webmanifest'
    _have 'sw.js'                || echo 'sw.js'
    _have 'build.json'           || echo 'build.json'
    _have 'version.json'         || echo 'version.json'
    _have 'bundle*.js'           || echo 'bundle[hash].js'
    _have 'bundle*.css'          || echo 'bundle[hash].css'
    _have 'workbox*.js'          || echo 'workbox[hash].js'
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

# Pick a sha256 tool (verification is preferred but best-effort).
if command -v shasum >/dev/null 2>&1; then
    sha_cmd="shasum -a 256"
elif command -v sha256sum >/dev/null 2>&1; then
    sha_cmd="sha256sum"
else
    sha_cmd=""
fi

mkdir -p "$UI_DIST"
ui_ok=false
for v in "${UI_CANDIDATES[@]}"; do
    echo "  trying $HF_BASE/$v/dist.tar.gz ..."
    tmp="$(mktemp -d)"

    if ! curl -fsSL --max-time 120 -o "$tmp/dist.tar.gz" \
            "$HF_BASE/$v/dist.tar.gz?download=true"; then
        rm -rf "$tmp"
        continue
    fi

    # Verify the tarball against its published checksum when we can.
    if [ -n "$sha_cmd" ] && curl -fsSL --max-time 30 -o "$tmp/dist.tar.gz.sha256" \
            "$HF_BASE/$v/dist.tar.gz.sha256?download=true"; then
        want=$(awk '{print $1}' "$tmp/dist.tar.gz.sha256")
        got=$($sha_cmd "$tmp/dist.tar.gz" | awk '{print $1}')
        if [ -n "$want" ] && [ "$want" != "$got" ]; then
            echo "  checksum mismatch for dist.tar.gz (want=$want got=$got)"
            rm -rf "$tmp"
            continue
        fi
    fi

    # Replace dist/ atomically-ish: clear it, then extract the new tree. Use
    # -m so extracted files get the current mtime instead of the archive's
    # (otherwise an incremental build sees the assets as older than a stale
    # generated ui.cpp and skips regenerating it).
    rm -rf "$UI_DIST"
    mkdir -p "$UI_DIST"
    if ! tar xzmf "$tmp/dist.tar.gz" -C "$UI_DIST" 2>/dev/null; then
        echo "  failed to extract dist.tar.gz"
        rm -rf "$tmp" "$UI_DIST"
        mkdir -p "$UI_DIST"
        continue
    fi
    rm -rf "$tmp"

    # Reject a partial/drifted tree: embed.cpp requires the full asset set once
    # dist/ is non-empty and would otherwise abort the entire build. Clearing
    # dist/ here lets the embed step generate the no-asset stub instead, so the
    # server still builds (just without the web UI).
    missing="$(ui_missing_assets "$UI_DIST")"
    if [ -n "$missing" ]; then
        echo "  extracted tree is missing required asset(s); ignoring:"
        printf '    %s\n' $missing
        rm -rf "$UI_DIST"
        mkdir -p "$UI_DIST"
        continue
    fi

    build_gzip_mirror

    nfiles=$(find "$UI_DIST" -type f ! -path '*/_gzip/*' | wc -l | tr -d ' ')
    echo "  fetched + extracted UI assets from $v ($nfiles files, gzip-embedded)"
    ui_ok=true
    break
done

if ! $ui_ok; then
    echo "  warning: could not download UI assets; server will build without the web UI"
    rm -rf "$UI_DIST"
    mkdir -p "$UI_DIST"
fi
