#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Script to download and preprocess the PASCAL VOC 2012 dataset.
#
# Usage:
#   bash ./download_and_convert_voc2012.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - build_voc2012_data.py
#     - download_and_convert_voc2012.sh
#     - remove_gt_colormap.py
#     + pascal_voc_seg
#       + VOCdevkit
#         + VOC2012
#           + JPEGImages
#           + SegmentationClass
#

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./sar_ice"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"
cd "${CURRENT_DIR}"
# Root path for PASCAL VOC 2012 dataset.
SAR_ICE_ROOT="${WORK_DIR}" # /sar_ice"

# Remove the colormap in the ground truth annotations.
SEG_FOLDER="${SAR_ICE_ROOT}/dataset_${1}/SegmentationClass"
SEMANTIC_SEG_FOLDER="${SAR_ICE_ROOT}/dataset_${1}/SegmentationClassRaw"

#echo "Removing the color map in ground truth annotations..."
#python ./remove_gt_colormap.py \
#  --original_gt_folder="${SEG_FOLDER}" \
#  --output_dir="${SEMANTIC_SEG_FOLDER}"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord_${1}"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${SAR_ICE_ROOT}/dataset_${1}/JPEGImages"
LIST_FOLDER="${SAR_ICE_ROOT}/dataset_${1}/ImageSets" #/Segmentation"

echo "Converting sar_ice dataset..."
python ./build_sar_ice_2.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="png" \
  --output_dir="${OUTPUT_DIR}"
