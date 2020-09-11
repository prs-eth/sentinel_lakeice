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

## General syntax for running our model
> * cd models/research/deeplab
> * bash train_and_val_sar_ice.sh <dataset_name> <experiment_name> <network_type> <validation_mode>

#### More details on <dataset_name>:
A sample dataset "2016_17_2017_18_vv_vh_sils_silvaplana_stmoritz_2016_17" is provided in:
>models/research/deeplab/datasets/sar_ice

Its name follows this convention:
>\<winter1\>\_\<winter2\>\_\<band1\>\_\<band2\>\<lake1\>\_\<lake2\>\_\<lakeN\>\_\<validationset\>

It was generated with the script 
>data_scripting/preprocessing_sar.ipynb

and is split in the following folders:

- dataset_2016_17_2017_18_vv_vh_sils_silvaplana_stmoritz_2016_17
    - ImageSets: training and validation file list
    - JPEG images: RGB composite with SAR images, VV -> red, VH -> green
    - SegmentationClass: pixel-wise ground truth. non-frozen -> red, frozen -> blue, background -> white
- tfrecord_2016_17_2017_18_vv_vh_sils_silvaplana_stmoritz_2016_17

#### Options for <network_type>:
>mobilenet_v2, 
>Xception-65

#### Options for <validation_mode>:
>V (Testing only), 
>TV (Training and testing)

## To reproduce our results of leave one winter out experiment (train on w2017-18 and test on w2016-17) without training the network:
- * Download the dataset from GEE platform (we cannot provide the dataset due to copyright reasons) and extract it in: models/research/deeplab/datasets
(Note that, we processed all the available data from the beiginning of September till end of May in each winter)
- [Download](https://share.phys.ethz.ch/~pf/tommdata/Sentinel-1_SAR/pre-trained-model.zip) the Deeplab v3+ model (~105MB) pre-trained on our SAR dataset. 

Run:

> bash train_and_val_sar_ice.sh 2016_17_2017_18_vv_vh_sils_silvaplana_stmoritz_2016_17 2016_17_regionsils mobilenet_v2 V

Folder structure:
    - Input weights shall be placed in *train* subfolder
    - Output visualization images will be stored in *vis* subfolder

Takes ~30 min on GPU Quadro M1200

Results are stored in models/research/deeplab/datasets/sar_ice/exp_2016_17_regionsils

**Note:**  An entry with the name and size of 2016_17_2017_18_vv_vh_sils_silvaplana_stmoritz_2016_17 dataset has been already added to the models/research/deeplab/datasets/segmentation_dataset.py file. In case of adding a new dataset, an entry needs to be included in this file.

## To train a model

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

## Downloading from Google Earth Engine

An example code to download the images of lake Sils from winter 2017-18 (with VV) is available on: https://code.earthengine.google.com/?scriptPath=users%2Frobertoaguilar%2Flakeice%3Asentinel1. You will need to create a GEE account.

* To download several images, run the script: data_scripting/gee_browser.js on the browser once GEE tasks are ready (after running the downloading script).

### Citation

Kindly cite our paper, if you use this repo:

@inproceedings{tom_aguilar_2020:isprs, author={Tom, M. and Aguilar, R. and Imhof, P. and Leinss, S. and Baltsavias, E. and Schindler, K.}, booktitle={arXiv preprint: arXiv:2002.07040v2}, title={Lake Ice Detection from Sentinel-1 SAR with Deep Learning}, year={2020}, }
