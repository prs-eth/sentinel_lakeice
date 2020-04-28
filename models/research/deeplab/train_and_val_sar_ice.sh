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

## Activate when working on leonhard
# module load python_gpu/2.7.14

# Exit immediately if a command exits with a non-zero status.
set -e

# PARAMETERS:
DATASET_NAME=${1}
EXP_ID=${2}
NETWORK=${3}
TASKS=${4} # T-> train, V -> val
echo "Dataset name: ${DATASET_NAME}"

if [[ "${TASKS}" == *"T"* ]]; then
LR=${5} # learning rate
N_STEPS=${6} 
AR=${7} # atrous rate
BS=${8} # patch size
PS=${9} # batch size
echo "Learning rate: ${LR}"
echo "N steps: ${N_STEPS}"
echo "Atrous rate: ${AR}"
AR1=$(( ${AR} * 1 ))
AR2=$(( ${AR} * 2 ))
AR3=$(( ${AR} * 3 ))
fi

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
python "${WORK_DIR}"/model_test.py -v


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

if [[ "${TASKS}" == *"T"* ]]; then
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
  --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}" \ ; 
fi
if [[ "${TASKS}" == *"V"* ]]; then
# Visualize the results.
python deeplab/vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="${NETWORK}" \
  --output_stride=2 \
  --decoder_output_stride=2 \
  --vis_crop_size=557 \
  --vis_crop_size=493 \
  --dataset=${DATASET_NAME} \
  --colormap_type="sar_ice" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${SAR_ICE_DATASET}" \
  --max_number_of_iterations=1 \
  --exp="${EXP_ID}" \ ;

echo datasets/sar_ice/exp_${EXP_ID}/output
cd deeplab
rm -rf datasets/sar_ice/exp_${EXP_ID}/output
python patch_rebuildner.py ${DATASET_NAME} ${EXP_ID}
python quantify.py ${DATASET_NAME} ${EXP_ID};
fi
