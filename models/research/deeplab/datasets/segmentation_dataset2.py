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

"""Provides data from semantic segmentation datasets.

The SegmentationDataset class provides both images and annotations (semantic
segmentation and/or instance segmentation) for TensorFlow. Currently, we
support the following datasets:

1. PASCAL VOC 2012 (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

PASCAL VOC 2012 semantic segmentation dataset annotates 20 foreground objects
(e.g., bike, person, and so on) and leaves all the other semantic classes as
one background class. The dataset contains 1464, 1449, and 1456 annotated
images for the training, validation and test respectively.

2. Cityscapes dataset (https://www.cityscapes-dataset.com)

The Cityscapes dataset contains 19 semantic labels (such as road, person, car,
and so on) for urban street scenes.

3. ADE20K dataset (http://groups.csail.mit.edu/vision/datasets/ADE20K)

The ADE20K dataset contains 150 semantic labels both urban street scenes and
indoor scenes.

References:
  M. Everingham, S. M. A. Eslami, L. V. Gool, C. K. I. Williams, J. Winn,
  and A. Zisserman, The pascal visual object classes challenge a retrospective.
  IJCV, 2014.

  M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson,
  U. Franke, S. Roth, and B. Schiele, "The cityscapes dataset for semantic urban
  scene understanding," In Proc. of CVPR, 2016.

  B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, A. Torralba, "Scene Parsing
  through ADE20K dataset", In Proc. of CVPR, 2017.
"""
import collections
import os.path
import tensorflow as tf

slim = tf.contrib.slim

dataset = slim.dataset

tfexample_decoder = slim.tfexample_decoder


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    ['splits_to_sizes',   # Splits of the dataset into training, val, and test.
     'num_classes',   # Number of semantic classes, including the background
                      # class (if exists). For example, there are 20
                      # foreground classes + 1 background class in the PASCAL
                      # VOC 2012 dataset. Thus, we set num_classes=21.
     'ignore_label',  # Ignore label value.
    ]
)

_CITYSCAPES_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 2975,
        'val': 500,
    },
    num_classes=19,
    ignore_label=255,
)

_PASCAL_VOC_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 1464,
        'train_aug': 10582,
        'trainval': 2913,
        'val': 1449,
    },
    num_classes=21,
    ignore_label=255,
)


_SAR_ICE_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 227, # number of file in the train folder
        'trainval': 336,# must be changed to sar_ice project
        'val': 109,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)


_SAR_ICE_2_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 229, # number of file in the train folder
        'trainval': 351,# must be changed to sar_ice project
        'val': 112,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)


_SAR_ICE_3_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 227, # number of file in the train folder
        'trainval': 336,# must be changed to sar_ice project
        'val': 109,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_4_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 227, # number of file in the train folder
        'trainval': 336,# must be changed to sar_ice project
        'val': 109,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_5_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 227, # number of file in the train folder
        'trainval': 336,# must be changed to sar_ice project
        'val': 109,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_6_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 227, # number of file in the train folder
        'trainval': 336,# must be changed to sar_ice project
        'val': 109,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_7_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 108, # number of file in the train folder
        'trainval': 165,# must be changed to sar_ice project
        'val': 57,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_8_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 227, # number of file in the train folder
        'trainval': 336,# must be changed to sar_ice project
        'val': 109,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_9_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 115, # number of file in the train folder
        'trainval': 170,# must be changed to sar_ice project
        'val': 55,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_10_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 114, # number of file in the train folder
        'trainval': 166,# must be changed to sar_ice project
        'val': 52,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_11_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 456, # number of file in the train folder
        'trainval': 672,# must be changed to sar_ice project
        'val': 216,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_12_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 495, # number of file in the train folder
        'trainval': 675,# must be changed to sar_ice project
        'val': 180,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_13_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 541, # number of file in the train folder
        'trainval': 672,# must be changed to sar_ice project
        'val': 131,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_14_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 380, # number of file in the train folder
        'trainval': 628,# must be changed to sar_ice project
        'val': 248,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_15_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 506, # number of file in the train folder
        'trainval': 628,# must be changed to sar_ice project
        'val': 122,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_16_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 491, # number of file in the train folder
        'trainval': 625,# must be changed to sar_ice project
        'val': 134,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_17_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 641, # number of file in the train folder
        'trainval': 724,# must be changed to sar_ice project
        'val': 83,
    },
    num_classes=4, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_18_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 388, # number of file in the train folder
        'trainval': 625,# must be changed to sar_ice project
        'val': 237,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_40_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 245, # number of file in the train folder
        'trainval': 758,# must be changed to sar_ice project
        'val': 513,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_41_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 456, # number of file in the train folder
        'trainval': 749,# must be changed to sar_ice project
        'val': 293,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_42_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 456, # number of file in the train folder
        'trainval': 701,# must be changed to sar_ice project
        'val': 245,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_43_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 245, # number of file in the train folder
        'trainval': 701,# must be changed to sar_ice project
        'val': 456,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_44_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 478, # number of file in the train folder
        'trainval': 701,# must be changed to sar_ice project
        'val': 223,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_45_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 464, # number of file in the train folder
        'trainval': 701,# must be changed to sar_ice project
        'val': 237,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_46_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 460, # number of file in the train folder
        'trainval': 701,# must be changed to sar_ice project
        'val': 241,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_50_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 451, # number of file in the train folder
        'trainval': 736,# must be changed to sar_ice project
        'val': 285,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)


_SAR_ICE_51_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 230, # number of file in the train folder
        'trainval': 765,# must be changed to sar_ice project
        'val': 535,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_53_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 227, # number of file in the train folder
        'trainval': 374,# must be changed to sar_ice project
        'val': 147,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)


_SAR_ICE_54_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 712, # number of file in the train folder
        'trainval': 791,# must be changed to sar_ice project
        'val': 79,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_55_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 749, # number of file in the train folder
        'trainval': 783,# must be changed to sar_ice project
        'val': 30,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_56_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 460, # number of file in the train folder
        'trainval': 746,# must be changed to sar_ice project
        'val': 286,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_57_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 454, # number of file in the train folder
        'trainval': 735,# must be changed to sar_ice project
        'val': 281,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_58_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 448, # number of file in the train folder
        'trainval': 701,# must be changed to sar_ice project
        'val': 253,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_60_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 519, # number of file in the train folder
        'trainval': 838,# must be changed to sar_ice project
        'val': 319,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_61_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 261, # number of file in the train folder
        'trainval': 875,# must be changed to sar_ice project
        'val': 614,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_62_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 681, # number of file in the train folder
        'trainval': 794,# must be changed to sar_ice project
        'val': 113,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_63_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 559, # number of file in the train folder
        'trainval': 845,# must be changed to sar_ice project
        'val': 286,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_64_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 533, # number of file in the train folder
        'trainval': 834,# must be changed to sar_ice project
        'val': 281,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_65_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 547, # number of file in the train folder
        'trainval': 800,# must be changed to sar_ice project
        'val': 253,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_70_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 451, # number of file in the train folder
        'trainval': 758,# must be changed to sar_ice project
        'val': 307,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_71_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 264, # number of file in the train folder
        'trainval': 799,# must be changed to sar_ice project
        'val': 535,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_80_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 0, # number of file in the train folder
        'trainval': 267,# must be changed to sar_ice project
        'val': 267,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_81_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 0, # number of file in the train folder
        'trainval': 281,# must be changed to sar_ice project
        'val': 281,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

_SAR_ICE_82_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 0, # number of file in the train folder
        'trainval': 221,# must be changed to sar_ice project
        'val': 221,
    },
    num_classes=3, # number of classes in your dataset
    ignore_label=255, # white edges that will be ignored to be class
)

# These number (i.e., 'train'/'test') seems to have to be hard coded
# These number (i.e., 'train'/'test') seems to have to be hard coded
# You are required to figure it out for your training/testing example.
_ADE20K_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 20210,  # num of samples in images/training
        'val': 2000,  # num of samples in images/validation
    },
    num_classes=151,
    ignore_label=0,
)


_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    '2016_17_2017_18_vv_66_168_no_tr':_SAR_ICE_SEG_INFORMATION,
    '2016_17_2017_18_vh_66_168_no_tr' : _SAR_ICE_2_SEG_INFORMATION,
    '2016_17_2017_18_vv_66_168_no_tr_3cl_augmented_ice' : _SAR_ICE_3_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_no_tr_3ch' : _SAR_ICE_4_SEG_INFORMATION,
    '2016_17_2017_18_vv_66_168_no_tr_temp16':_SAR_ICE_5_SEG_INFORMATION,
    '2016_17_2017_18_vv_66_168_no_tr_temp128':_SAR_ICE_6_SEG_INFORMATION,
    '2016_17_2017_18_vv_66_no_tr':_SAR_ICE_7_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_vh-vv_66_168_no_tr': _SAR_ICE_8_SEG_INFORMATION,
    '2016_17_2017_18_vv_15_no_tr' : _SAR_ICE_9_SEG_INFORMATION,
    '2016_17_2017_18_vv_117_no_tr' : _SAR_ICE_10_SEG_INFORMATION,
    '2016_17_2017_18_vv_66_168_15_117_no_tr' : _SAR_ICE_11_SEG_INFORMATION,
    '2016_17_2017_18_vv_66_168_15_117_no_tr_medfilt' : _SAR_ICE_11_SEG_INFORMATION,
    '2016_17_2017_18_vv_66_168_15_117_mostly_wf' : _SAR_ICE_12_SEG_INFORMATION,
    '2016_17_2017_18_vv_66_168_15_117_no_tr_randomvaltrain' : _SAR_ICE_13_SEG_INFORMATION,
    '2016_17_2017_18_vv_66_168_15_117_no_tr_10log' : _SAR_ICE_11_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_no_trlog10' : _SAR_ICE_14_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_no_trlog10_randomTrainVal' : _SAR_ICE_15_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_no_tr_2cl_log10_randomTrainVal' : _SAR_ICE_16_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_no_tr_2cl_log10_loo20_sihl' : _SAR_ICE_17_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_no_tr_2cl_log10_loo_silvaplana' : _SAR_ICE_18_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_tran_18' : _SAR_ICE_40_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_tran_16' : _SAR_ICE_41_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_no_tr_16' : _SAR_ICE_42_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_no_tr_18' : _SAR_ICE_43_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_no_tr_stmoritz' : _SAR_ICE_44_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_no_tr_silvaplana' : _SAR_ICE_45_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_no_tr_sils' : _SAR_ICE_46_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_tran_label_2016_17_smoothed' : _SAR_ICE_50_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_tran_2017_18_smoothed' : _SAR_ICE_51_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_15_117_2cl_tran_label_2016_17_smoothed' : _SAR_ICE_53_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_test_2017_18_sihl' : _SAR_ICE_54_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_test_2016_17_sihl' : _SAR_ICE_55_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_test_sils' : _SAR_ICE_56_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_test_silvaplana' : _SAR_ICE_57_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_test_stmoritz' : _SAR_ICE_58_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_testsihl_2016_17' : _SAR_ICE_60_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_testsihl_2017_18' : _SAR_ICE_61_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_testsihl_sihl' : _SAR_ICE_62_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_testsihl_sils' : _SAR_ICE_63_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_testsihl_silvaplana' : _SAR_ICE_64_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_testsihl_stmoritz' : _SAR_ICE_65_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl2016_17mimw_smoothed' : _SAR_ICE_70_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl2017_18mimw_smoothed' : _SAR_ICE_71_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2clstmoritz_testOnly' : _SAR_ICE_80_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_no_tr_silvaplana_testOnly' : _SAR_ICE_81_SEG_INFORMATION,
    '2016_17_2017_18_vv_vh_66_168_15_117_2cl_no_tr_sils_testOnly' : _SAR_ICE_82_SEG_INFORMATION,
}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'


def get_cityscapes_dataset_name():
  return 'cityscapes'


def get_dataset(dataset_name, split_name, dataset_dir):
  """Gets an instance of slim Dataset.

  Args:
    dataset_name: Dataset name.
    split_name: A train/val Split name.
    dataset_dir: The directory of the dataset sources.

  Returns:
    An instance of slim Dataset.

  Raises:
    ValueError: if the dataset_name or split_name is not recognized.
  """
  if dataset_name not in _DATASETS_INFORMATION:
    raise ValueError('The specified dataset is not supported yet.')

  splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes

  if split_name not in splits_to_sizes:
    raise ValueError('data split name %s not recognized' % split_name)

  # Prepare the variables for different datasets.
  num_classes = _DATASETS_INFORMATION[dataset_name].num_classes
  ignore_label = _DATASETS_INFORMATION[dataset_name].ignore_label

  file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Specify how the TF-Examples are decoded.
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/filename': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='jpeg'),
      'image/height': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/width': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/segmentation/class/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/segmentation/class/format': tf.FixedLenFeature(
          (), tf.string, default_value='png'),
  }
  items_to_handlers = {
      'image': tfexample_decoder.Image(
          image_key='image/encoded',
          format_key='image/format',
          channels=3),
      'image_name': tfexample_decoder.Tensor('image/filename'),
      'height': tfexample_decoder.Tensor('image/height'),
      'width': tfexample_decoder.Tensor('image/width'),
      'labels_class': tfexample_decoder.Image(
          image_key='image/segmentation/class/encoded',
          format_key='image/segmentation/class/format',
          channels=1),
  }

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=splits_to_sizes[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      ignore_label=ignore_label,
      num_classes=num_classes,
      name=dataset_name,
      multi_label=True)
