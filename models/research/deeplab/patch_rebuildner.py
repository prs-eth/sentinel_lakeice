#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Pascal
#
# Created:     20.04.2019
# Copyright:   (c) Pascal 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os, sys
from PIL import Image

main_path = os.path.abspath(os.path.join(os.getcwd(), "datasets", "sar_ice"))
dataset = "2016_17_2017_18_vv_66_no_tr"

dataset_folder = "dataset_"+sys.argv[1]
input_folder = os.path.join(main_path, dataset_folder)
exp_folder = "exp_"+sys.argv[2]
output_folder = os.path.join(main_path, exp_folder, "output")
os.makedirs(output_folder)
print(input_folder)

pred_path = os.path.join(main_path, exp_folder, "train_on_train_set", "vis", "segmentation_results")

f_name = os.path.join(input_folder, "ImageSets", "val.txt")
print(f_name)
f= open(f_name,"r")
i = 0
predictions = [ pred for pred in sorted(os.listdir(pred_path)) ]#if "prediction" in pred]
print(len(predictions))
while True:
    line = f.readline().rstrip()
    print(line)
    if line is "" or len(line) < 10:
        break
    print("{} {}".format(i, predictions[i]))
    original = os.path.join(main_path, dataset_folder, "JPEGImages", line+".png")
    img = Image.open(original)
    img.save(output_folder+"/original"+line+".png")
    print(output_folder+"/original"+line+".png")
    groundtruth = os.path.join(main_path, dataset_folder, "SegmentationClass", line+".png")
    img = Image.open(groundtruth)
    img.save(output_folder+"/groundtruth"+line+".png")
    prediction = os.path.join(pred_path, line+".png")#pred_path+"/"+predictions[i]
    img = Image.open(prediction)
    img.save(output_folder+"/prediction"+line+".png")
    i = i + 1
