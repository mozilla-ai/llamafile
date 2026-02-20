PATCHES_DIR="llama.cpp.patches/patches"

for patch_file in "$PATCHES_DIR"/*.patch; do
    if [ -f "$patch_file" ]; then
        echo "Applying $(basename "$patch_file")..."
        git apply --check "$patch_file"
        #patch -p1 --dry-run < "$patch_file"
    fi
done
