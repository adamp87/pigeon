#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}"

mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

echo "Downloading Coral EdgeTPU and LiteRT models into ${TARGET_DIR}..."

download_file() {
    local url="$1"
    local filename="$2"
    if [ -f "${filename}" ]; then
        echo " -> ${filename} already exists, skipping."
        return 0
    fi
    echo " -> Downloading ${filename}..."
    if command -v curl >/dev/null 2>&1; then
        curl -fSL -o "${filename}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "${filename}" "${url}"
    else
        echo "Error: curl or wget is required." >&2
        exit 1
    fi
}

# 1. Labels
download_file "https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt" "coco_labels.txt"
download_file "https://raw.githubusercontent.com/google-coral/test_data/master/inat_bird_labels.txt" "inat_bird_labels.txt"

# 2. Coral EdgeTPU Models (RPi4)
download_file "https://raw.githubusercontent.com/google-coral/test_data/master/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite" "ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite"
download_file "https://raw.githubusercontent.com/google-coral/test_data/master/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite" "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"

# 3. LiteRT / CPU Models (RPi5 / PC)
download_file "https://raw.githubusercontent.com/google-coral/test_data/master/ssdlite_mobiledet_coco_qat_postprocess.tflite" "ssdlite_mobiledet_coco_qat_postprocess.tflite"
download_file "https://raw.githubusercontent.com/google-coral/test_data/master/mobilenet_v2_1.0_224_inat_bird_quant.tflite" "mobilenet_v2_1.0_224_inat_bird_quant.tflite"
# YOLO26 / Custom YOLO models (Requires HuggingFace authentication or custom host)
# download_file "https://huggingface.co/EdgeFirst/yolo26-det/resolve/main/yolo26n-det-int8.tflite" "yolo26n-det-int8.tflite"

echo "Model downloads complete."
