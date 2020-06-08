# Lake Ice Detection from Sentinel-1 SAR with Deep Learning

This repository contains the source code, dataset and pre-trained models corresponding to the paper [Lake Ice Detection from Sentinel-1 SAR with Deep Learning](https://arxiv.org/pdf/2002.07040.pdf) (accepted for ISPRS Congress, 2020, Nice, France)

![segmentation_sar](figures/qual_tran_sils.jpg)

This work is part of the [Lake Ice Project (Phase 2)](https://prs.igp.ethz.ch/research/current_projects/integrated-lake-ice-monitoring-and-generation-of-sustainable--re.html). Here is the link to [Phase 1](https://prs.igp.ethz.ch/research/completed_projects/integrated-monitoring-of-ice-in-selected-swiss-lakes.html) of the same project.

* Our Sentinel-1 SAR semantic segmentation system is based on [Deeplab v3+](https://arxiv.org/abs/1706.05587).
* Rasters corresponding to Sentinel-1 SAR 
were downloaded from [Google Earth Engine (GEE)](https://earthengine.google.com/)

## Dependencies

The required libraries can be installed by running
> pip install -r requirements.txt

## To reproduce our results of leave one winter out experiment (trained on winter 2017-18 data and tested on winter 2016-17):

> * Download the [zip file](https://polybox.ethz.ch/remote.php/webdav/lakeice_sentinel/sar_ice.zip) containing pre-trained weights, tfrecords, and dataset, and extract it in models/research/deeplab/datasets
- Folder structure:
    - Input weights shall be placed in *train* subfolder
    - Output visualization images will be stored in *vis* subfolder
> * cd models/research/deeplab
> * bash train_and_val_sar_ice.sh <dataset_name> <experiment_name> <network_type> <validation_mode>

#### More details on <dataset_name>:
A sample dataset 2016_17_2017_18_vv_vh_sils_silvaplana_stmoritz_2016_17 is provided in models/research/deeplab/datasets/sar_ice. Its name follows this convention:
\<winter1\>\_\<winter2\>\_\<band1\>\_\<band2\>\<lake1\>\_\<lake2\>\_\<lanen\>\_\<validationset\>

It was generated with the script data_scripting/preprocessing_sar.ipynb and in split in the following folders:

- dataset_2016_17_2017_18_vv_vh_sils_silvaplana_stmoritz_2016_17
    - ImageSets: training and validation file list
    - JPEG images: RGB composite with SAR images, VV -> red, VH -> green
    - SegmentationClass: pixel-wise ground truth. non-frozen -> red, frozen -> blue, background -> white
- tfrecord_2016_17_2017_18_vv_vh_sils_silvaplana_stmoritz_2016_17

#### Options for <network_type>:
-mobilenet_v2
-Xception-65

#### Options for <validation_mode>:
-V (Testing only)
-TV (Training and validation)

## To reproduce our results without training:
example:  bash train_and_val_sar_ice.sh 2016_17_2017_18_vv_vh_sils_silvaplana_stmoritz_2016_17 2016_17_regionsils mobilenet_v2 V

Takes ~30 min on GPU Quadro M1200

Results are stored in models/research/deeplab/datasets/sar_ice/exp_2016_17_regionsils


## To train a model with an existing dataset (tested on NVIDIA GeForceGTX1080Ti):

> bash train_and_val_sar_ice.sh 2016_17_2017_18_vv_vh_sils_silvaplana_stmoritz_2016_17 2016_17_regionsils mobilenet_v2 TV 0.001 40000 1 8 129
-----
#####  Training parameters:
- Dataset: 2016_17_2017_18_vv_vh_sils_silvaplana_stmoritz_2016_17
- Output folder (same as experiment name): 2016_17_regionsils
- Network: mobilenet_v2
- TV: validation mode
- Initial learning rate: 0.001
- Number of steps: 40000
    - This hyperparameter needs to be tuned when adding a new lake
- Atrous rate: 1 -> it generates a [1,2,3] value
- Batch size: 8
- Patch size: 129 -> it generates a 129x129 pixels window

Takes ~8 hours on GeForceGTX1080Ti

## To generate a new sub-dataset for a differnt experiment:
> * Download images from https://polybox.ethz.ch/index.php/s/AjRHiOQhvf0vrku&nbsp;
> * Place sentinel1 folder into data/rasters/&nbsp;
> * Open data_scripting/preprocessing_sar.ipynb&nbsp;
> * Configure winter and lakes for training and validation set&nbsp;
> * Update models/research/deeplab/datasets/segmentation_dataset.py&nbsp;

## Downloading from Google Earth Engine

An example code to download the images of lake Sils from winter 2017-18 (with VV) is available on: https://code.earthengine.google.com/?scriptPath=users%2Frobertoaguilar%2Flakeice%3Asentinel1. You will need to create a GEE account.

* To download several images, run the script: data_scripting/gee_browser.js on the browser once GEE tasks are ready (after running the downloading script).

### Citation

Kindly cite our paper, if you use this repo:

@inproceedings{tom_aguilar_2020:isprs, author={Tom, M. and Aguilar, R. and Imhof, P. and Leinss, S. and Baltsavias, E. and Schindler, K.}, booktitle={arXiv preprint: arXiv:2002.07040v2}, title={Lake Ice Detection from Sentinel-1 SAR with Deep Learning}, year={2020}, }
