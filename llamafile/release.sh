# update LLAMAFILE_VERSION for a new release
LLAMAFILE_VERSION="0.10.4"

# this is where the precompiled GPU libraries are stored
GPU_LIBS_DIR="/home/mala/gpulibs/${LLAMAFILE_VERSION}"

# this is where all the release files are gonna be saved
RELEASE_DIR="/home/mala/releases/${LLAMAFILE_VERSION}"

# this is where the zip package contents are being collected
ZIP_DIR="${RELEASE_DIR}/llamafile-${LLAMAFILE_VERSION}"

# run make install to store binaries and man pages in ZIP_DIR
make install PREFIX="${ZIP_DIR}"

# Check if make install actually created ZIP_DIR
if [ ! -d "${ZIP_DIR}" ]; then
  echo "Error: Source directory does not exist: ${ZIP_DIR}"
  exit 1
fi

# Copy the README into the release
cp README.md "${ZIP_DIR}/README.md"

# Make a copy of llamafile while it's thin
cp "${ZIP_DIR}/bin/llamafile" "${ZIP_DIR}/bin/llamafile-thin"

# Bundle llamafile binary with GPU libs
${ZIP_DIR}/bin/zipalign -j0 "${ZIP_DIR}/bin/llamafile" \
	"${GPU_LIBS_DIR}/ggml-cuda.so"                 \
	"${GPU_LIBS_DIR}/ggml-cuda.dll"                \
	"${GPU_LIBS_DIR}/ggml-vulkan.so"               \
	"${GPU_LIBS_DIR}/ggml-vulkan.dll" 

DEST_DIR="${RELEASE_DIR}/release"
mkdir "${DEST_DIR}"

# list of binaries to copy and rename
BINARIES="llamafile zipalign whisperfile diffusionfile transcribefile"

for binary in $BINARIES; do
  if [ -f "${ZIP_DIR}/bin/${binary}" ]; then
    cp "${ZIP_DIR}/bin/${binary}" "${DEST_DIR}/${binary}-${LLAMAFILE_VERSION}"
    echo "Copied ${binary} to ${DEST_DIR}/${binary}-${LLAMAFILE_VERSION}"
  else
    echo "Warning: ${ZIP_DIR}/bin/${binary} not found"
  fi
done

# move the thin llamafile in the release dir (we are not packing both in the zip)
mv "${ZIP_DIR}/bin/llamafile-thin" "${DEST_DIR}/llamafile-${LLAMAFILE_VERSION}-thin"

ZIP_FILE="${DEST_DIR}/llamafile-${LLAMAFILE_VERSION}.zip"

echo "Zipping ${ZIP_DIR} into ${ZIP_FILE}"

# now zip the release directory
cd "${RELEASE_DIR}"
zip -r "${ZIP_FILE}" "llamafile-${LLAMAFILE_VERSION}"

if [ -f "${ZIP_FILE}" ]; then
  echo "${ZIP_FILE} ready."
else
  echo "Error creating ${ZIP_FILE}"
fi
