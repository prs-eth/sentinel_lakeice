# Lake Ice Detection from Sentinel-1 SAR with Deep Learning

This repository contains the source code (tensorflow) and pre-trained model corresponding to the paper:<br>

>[Lake Ice Detection from Sentinel-1 SAR with Deep Learning](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-3-2020/409/2020/) (presented at the ISPRS Congress, 2020, virtual conference)<br>
> by Manu Tom, Roberto Aguilar, Pascal Imhof, Silvan Leinss, Emmanuel Baltsavias and Konrad Schindler

![segmentation_sar](figures/qual_tran_sils.jpg)

This work is part of the [Lake Ice Project (Phase 2)](https://prs.igp.ethz.ch/research/current_projects/integrated-lake-ice-monitoring-and-generation-of-sustainable--re.html) funded by MeteoSwiss in the GCOS Switzerland framework. Here is the link to [Phase 1](https://prs.igp.ethz.ch/research/completed_projects/integrated-monitoring-of-ice-in-selected-swiss-lakes.html) of the same project.

Note that:<br>
* Our Sentinel-1 SAR semantic segmentation system is based on [Deeplab v3+](https://arxiv.org/pdf/1802.02611.pdf).
* Rasters corresponding to Sentinel-1 SAR 
were downloaded from [Google Earth Engine (GEE)](https://earthengine.google.com/)

## Pre-trained model
[Download](https://share.phys.ethz.ch/~pf/tommdata/Sentinel-1_SAR/pre-trained-model.zip) the Deeplab v3+ network weights (~105MB) pre-trained on our SAR dataset.

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
- Download the dataset from GEE platform (we cannot provide the dataset due to copyright reasons) and extract it in: models/research/deeplab/datasets
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

An example code to download the images of lake Sils from winter 2017-18 (with VV polarisation) is available on: https://code.earthengine.google.com/?scriptPath=users%2Frobertoaguilar%2Flakeice%3Asentinel1. You will need to create a GEE account.

* To download several images, run the script: data_scripting/gee_browser.js on the browser once GEE tasks are ready (after running the downloading script).

## Citation

Kindly cite our paper, if you use this project in your research:

> @article{tom_aguilar_2020:isprs,<br>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;author    = {Manu Tom and Roberto Aguilar and Pascal Imhof and Silvan Leinss and Emmanuel Baltsavias and Konrad Schindler},<br>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;title     = {Lake Ice Detection from Sentinel-1 SAR with Deep Learning},<br>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;journal   = {ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci.},<br>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;year      = {2020},<br>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;volume    = {V-3-2020},<br>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pages     = {409--416},<br>
}

## Useful links
1. Multi-sensor lake ice monitoring with machine (deep) learning. [Project github page](https://github.com/czarmanu/lake-ice-ml).
2. Lake ice monitoring with webcams and crowd-sourced images with Deep-U-Lab. [Github repo (tensorflow code, pre-trained model)](https://github.com/czarmanu/deeplab-lakeice-webcams). 
3. Tom, M., Suetterlin, M., Bouffard, D., Rothermel, M., Wunderle, S., Baltsavias, E., 2019. [Integrated monitoring of ice
in selected Swiss lakes](https://arxiv.org/abs/2008.00512). Final Project Report

## Licence

MIT License

Copyright (c) 2020 ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Author: Manu Tom
