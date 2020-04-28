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
# This script is used to run test on Cityscapes. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./run.sh
#
#

module load python_gpu/2.7.14

# Exit immediately if a command exits with a non-zero status.
set -e

# PARAMETERS:
# * Dataset
# * Learning rate
# * N_STEPS
# * Atrours factor 1-> 1,2,3. 2-> 2,4,6
DATASET_NAME=${1}
EXP_ID=${2}
LR=${3}
N_STEPS=${4}
AR=${5}
BS=${6}
PS=${7}
NETWORK=${8}
echo "Dataset name: ${DATASET_NAME}"
echo "Learning rate: ${LR}"
echo "N steps: ${N_STEPS}"
echo "Atrous rate: ${AR}"

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
# python "${WORK_DIR}"/model_test.py -v


DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
#sh download_and_convert_sar_ice.sh

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
SAR_ICE_FOLDER="sar_ice"
EXP_FOLDER="exp_${EXP_ID}/train_on_train_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${SAR_ICE_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${SAR_ICE_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${SAR_ICE_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${SAR_ICE_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${SAR_ICE_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

cd "${CURRENT_DIR}"

if [ ${NETWORK} = ${NETWORK} ]
        then
                echo "Mobilenetv2"
                CKPT_NAME="deeplabv3_mnv2_pascal_train_aug/model.ckpt-30000"
        else
                echo "Xception"
                CKPT_NAME="deeplabv3_cityscapes_train/model.ckpt"
fi

SAR_ICE_DATASET="${WORK_DIR}/${DATASET_DIR}/${SAR_ICE_FOLDER}/tfrecord_${DATASET_NAME}"
AR1=$(( ${AR} * 1 ))
AR2=$(( ${AR} * 2 ))
AR3=$(( ${AR} * 3 ))
: '
python deeplab/train.py \
  --logtostderr \
  --training_number_of_steps=${N_STEPS} \
  --train_split="train" \
  --model_variant="${NETWORK}" \
  --atrous_rates=${AR1} \
  --atrous_rates=${AR2} \
  --atrous_rates=${AR3} \
  --output_stride=2 \
  --decoder_output_stride=2 \
  --train_crop_size="${PS}" \
  --train_crop_size="${PS}" \
  --train_batch_size="${BS}" \
  --base_learning_rate="0.${LR}" \
  --dataset=${DATASET_NAME} \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${SAR_ICE_DATASET}" \
  --fine_tune_batch_norm=false \
  --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}" \ 

# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.
python deeplab/eval_pascal.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="${NETWORK}" \
  --atrous_rates=${AR1} \
  --atrous_rates=${AR2} \
  --atrous_rates=${AR3} \
  --output_stride=2 \
  --decoder_output_stride=2 \
  --eval_crop_size=882 \
  --eval_crop_size=530 \
  --dataset=${DATASET_NAME} \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --dataset_dir="${SAR_ICE_DATASET}" \
  --eval_logdir="${EVAL_LOGDIR}" \
# Visualize the results.
python deeplab/vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="${NETWORK}" \
  --atrous_rates=${AR1} \
  --atrous_rates=${AR2} \
  --atrous_rates=${AR3} \
  --output_stride=2 \
  --decoder_output_stride=2 \
  --vis_crop_size=882 \
  --vis_crop_size=530 \
  --dataset=${DATASET_NAME} \
  --colormap_type="sar_ice" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${SAR_ICE_DATASET}" \
  --max_number_of_iterations=1 \
# Export the trained checkpoint.

CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${N_STEPS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph_manu.pb" #/frozen_inference_graph_manu.pb"

python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="${NETWORK}" \
  --atrous_rates=${AR1} \
  --atrous_rates=${AR2} \
  --atrous_rates=${AR3} \
  --output_stride=2 \
  --decoder_output_stride=2 \
  --num_classes=3 \
  --inference_scales=1.0
'
cd ${WORK_DIR}
module load python_gpu/2.7.14
python patch_rebuildner.py ${DATASET_NAME} ${EXP_ID}
python quantify.py ${DATASET_NAME} ${EXP_ID}
