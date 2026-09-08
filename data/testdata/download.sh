#!/usr/bin/env bash
set -euo pipefail

# Determine script and target directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/pigeons"

mkdir -p "${TARGET_DIR}"

USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

declare -a IMAGES=(
    "pigeon_000.jpg https://upload.wikimedia.org/wikipedia/commons/1/13/Columba_livia_-_01.jpg"
    "pigeon_001.jpg https://upload.wikimedia.org/wikipedia/commons/e/e5/Columba_livia_Luc_Viatour.jpg"
    "pigeon_002.jpg https://upload.wikimedia.org/wikipedia/commons/2/2b/Rock_dove_-_natures_pics.jpg"
    "pigeon_003.jpg https://upload.wikimedia.org/wikipedia/commons/0/03/Feral_pigeon_%28Columba_livia_domestica%29%2C_2017-05-27.jpg"
    "pigeon_004.jpg https://upload.wikimedia.org/wikipedia/commons/3/3c/Common_pigeon_at_Waterlow_Park%2C_London_01.jpg"
)

echo "Downloading ${#IMAGES[@]} pigeon test images into '${TARGET_DIR}'..."
echo ""

SUCCESS_COUNT=0
TOTAL_COUNT=${#IMAGES[@]}

for item in "${IMAGES[@]}"; do
    FILENAME=$(echo "${item}" | awk '{print $1}')
    URL=$(echo "${item}" | awk '{print $2}')
    DEST_PATH="${TARGET_DIR}/${FILENAME}"

    echo "Downloading ${FILENAME}..."
    if command -v curl >/dev/null 2>&1; then
        if curl -fSL -A "${USER_AGENT}" "${URL}" -o "${DEST_PATH}" --retry 3; then
            if [ -s "${DEST_PATH}" ]; then
                echo " -> Saved ${FILENAME} ($(du -h "${DEST_PATH}" | cut -f1))"
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            else
                echo " -> ERROR: Downloaded file ${FILENAME} is empty" >&2
                rm -f "${DEST_PATH}"
            fi
        else
            echo " -> ERROR: Failed to download ${FILENAME} from ${URL}" >&2
        fi
    elif command -v wget >/dev/null 2>&1; then
        if wget --user-agent="${USER_AGENT}" -q --show-progress -O "${DEST_PATH}" "${URL}"; then
            if [ -s "${DEST_PATH}" ]; then
                echo " -> Saved ${FILENAME} ($(du -h "${DEST_PATH}" | cut -f1))"
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            else
                echo " -> ERROR: Downloaded file ${FILENAME} is empty" >&2
                rm -f "${DEST_PATH}"
            fi
        else
            echo " -> ERROR: Failed to download ${FILENAME} from ${URL}" >&2
        fi
    else
        echo "ERROR: Neither curl nor wget is available." >&2
        exit 1
    fi
done

echo ""
echo "Download complete: ${SUCCESS_COUNT}/${TOTAL_COUNT} images downloaded successfully."

if [ "${SUCCESS_COUNT}" -eq "${TOTAL_COUNT}" ]; then
    exit 0
else
    exit 1
fi
