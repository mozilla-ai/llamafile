#!/usr/bin/env bash
#
# Generate ui.cpp / ui.h embedding the web UI assets as C arrays.
#
# Usage:
#   ui-embed.sh <out_cpp> <out_h> [<asset_dir>]
#
# This is a translation of emit_files() in llama.cpp/scripts/ui-assets.cmake,
# the same way llamafile-files/BUILD.mk is a translation of llama.cpp's CMake
# build: upstream generates ui.{cpp,h} with configure_file() at CMake time, and
# the cosmocc build has no CMake step.
#
# It renders *upstream's own* templates, tools/ui/ui.{cpp,h}.in, so the
# generated interface (llama_ui_find_asset / llama_ui_get_assets /
# llama_ui_use_gzip and LLAMA_UI_HAS_ASSETS, all consumed by the unpatched
# tools/server/server-http.cpp) tracks upstream automatically and cannot drift.
# Only the template *substitutions* live here:
#
#   @ASSET_ARRAYS@   one `static const unsigned char asset_N[] = {...};` per asset
#   @ASSET_TABLE@    one `{ name, asset_N, sizeof(asset_N), etag, mime },` per asset
#   @N_ASSETS@       asset count
#   @USE_GZIP@       whether the embedded bytes are gzip-compressed
#   #cmakedefine LLAMA_UI_HAS_ASSETS
#
# Assets come from <asset_dir> (fetch-ui-assets.sh populates it). When it holds
# a _gzip/ mirror the compressed bytes are embedded instead, and server-http.cpp
# serves them with Content-Encoding: gzip. Without <asset_dir> — or with no
# index.html in it — this emits the no-asset stub and the server builds UI-less.

set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "usage: $0 <out_cpp> <out_h> [<asset_dir>]" >&2
    exit 1
fi

OUT_CPP="$1"
OUT_H="$2"
ASSET_DIR="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/../llama.cpp/tools/ui"
TPL_CPP="$TEMPLATE_DIR/ui.cpp.in"
TPL_H="$TEMPLATE_DIR/ui.h.in"

for t in "$TPL_CPP" "$TPL_H"; do
    if [ ! -f "$t" ]; then
        echo "ui-embed: missing upstream template $t" >&2
        exit 1
    fi
done

# ETag source. Upstream uses SHA-256; cksum (POSIX, always present) is a
# last-resort fallback — an ETag only has to be stable and change with content.
if command -v shasum >/dev/null 2>&1; then
    hash_file() { shasum -a 256 "$1" | cut -d' ' -f1; }
elif command -v sha256sum >/dev/null 2>&1; then
    hash_file() { sha256sum "$1" | cut -d' ' -f1; }
else
    hash_file() { cksum "$1" | cut -d' ' -f1,2 | tr ' ' '-'; }
fi

# Mirrors mime_from_ext() in scripts/ui-assets.cmake. An extension missing here
# is served as application/octet-stream rather than breaking the build.
mime_from_ext() {
    case "${1##*.}" in
        html)        echo "text/html; charset=utf-8" ;;
        css)         echo "text/css" ;;
        js)          echo "application/javascript" ;;
        json)        echo "application/json" ;;
        webmanifest) echo "application/manifest+json" ;;
        svg)         echo "image/svg+xml" ;;
        png)         echo "image/png" ;;
        jpg|jpeg)    echo "image/jpeg" ;;
        ico)         echo "image/x-icon" ;;
        woff)        echo "font/woff" ;;
        woff2)       echo "font/woff2" ;;
        *)           echo "application/octet-stream" ;;
    esac
}

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
arrays="$work/arrays"
table="$work/table"
: > "$arrays"
: > "$table"

n_assets=0
use_gzip=false

if [ -n "$ASSET_DIR" ] && [ -f "$ASSET_DIR/index.html" ]; then
    # Asset names are paths relative to the dist root; the gzip mirror repeats
    # that layout under _gzip/, so names come from dist/ and bytes from whichever
    # tree we embed.
    embed_dir="$ASSET_DIR"
    if [ -d "$ASSET_DIR/_gzip" ]; then
        embed_dir="$ASSET_DIR/_gzip"
        use_gzip=true
    fi

    # sort for reproducible output; directory order is unspecified
    assets=()
    while IFS= read -r f; do
        assets+=("${f#./}")
    done < <(cd "$ASSET_DIR" && find . -type f ! -path './_gzip/*' | LC_ALL=C sort)

    idx=0
    for name in "${assets[@]}"; do
        src="$embed_dir/$name"
        if [ ! -s "$src" ]; then
            # A zero-length asset would emit an invalid empty C array. Upstream
            # hard-errors; drop the whole UI instead so a corrupt tarball cannot
            # abort a build, matching fetch-ui-assets.sh's fallback.
            echo "ui-embed: empty asset '$name' - building without an embedded UI" >&2
            : > "$arrays"
            : > "$table"
            idx=0
            break
        fi

        # od is POSIX; 16 bytes per line keeps the generated source wrapped.
        {
            printf 'static const unsigned char asset_%d[] = {\n' "$idx"
            od -An -v -tx1 "$src" \
                | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
                      -e 's/\([0-9a-f][0-9a-f]\)/0x\1,/g' -e 's/[[:space:]]//g'
            printf '};\n'
        } >> "$arrays"

        printf '    { "%s", asset_%d, sizeof(asset_%d), "\\"%s\\"", "%s" },\n' \
            "$name" "$idx" "$idx" "$(hash_file "$src")" "$(mime_from_ext "$name")" \
            >> "$table"

        idx=$((idx + 1))
    done
    n_assets=$idx
fi

if [ "$n_assets" -eq 0 ]; then
    use_gzip=false
fi

# Render a template: stream the bulk substitutions in from their own files
# (they reach tens of MB, which no sed expression should carry) and expand the
# scalar ones inline.
render() {
    local tpl="$1" out="$2"
    awk -v n_assets="$n_assets" \
        -v use_gzip="$use_gzip" \
        -v has_assets="$([ "$n_assets" -gt 0 ] && echo 1 || echo 0)" \
        -v arrays="$arrays" \
        -v table="$table" '
        /@ASSET_ARRAYS@/ { fflush(); system("cat " arrays); next }
        /@ASSET_TABLE@/  { fflush(); system("cat " table);  next }
        {
            if ($0 ~ /^#cmakedefine LLAMA_UI_HAS_ASSETS/) {
                print (has_assets ? "#define LLAMA_UI_HAS_ASSETS 1" \
                                  : "/* #undef LLAMA_UI_HAS_ASSETS */")
                next
            }
            gsub(/@N_ASSETS@/, n_assets)
            gsub(/@USE_GZIP@/, use_gzip)
            print
        }
    ' "$tpl" > "$out.tmp"
    # Only touch the output when it changes, so a re-fetch of identical assets
    # does not force every dependent object to rebuild.
    if [ -f "$out" ] && cmp -s "$out.tmp" "$out"; then
        rm -f "$out.tmp"
    else
        mv -f "$out.tmp" "$out"
    fi
}

render "$TPL_H"   "$OUT_H"
render "$TPL_CPP" "$OUT_CPP"

if [ "$n_assets" -gt 0 ]; then
    echo "ui-embed: embedded $n_assets assets$([ "$use_gzip" = true ] && echo ' (gzip)')"
else
    echo "ui-embed: no assets - building without an embedded UI"
fi
