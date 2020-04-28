#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Pascal
#
# Created:     18.04.2019
# Copyright:   (c) Pascal 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

# runns only with pyhton 3


import os
import sys
from PIL import Image
import numpy as np


def recall(nbr):
    zaehler = c[nbr][nbr]
    nenner = c[nbr][0]+c[nbr][1]
    if nenner == 0:
        return("nominator is equal to zero, it isn't possible to calculate the recall")
    else:
        return(zaehler/float(nenner))


def precision(nbr):
    zaehler = c[nbr][nbr]
    nenner = c[0][nbr]+c[1][nbr]
    if nenner == 0:
        return("nominator is equal to zero, it isn't possible to calculate the precision")
    else:
        return(zaehler/float(nenner))
        
def iou(nbr):
    zaehler = c[nbr][nbr]
    nenner = c[0][nbr]+c[1][nbr]+c[nbr][0]+c[nbr][1]-c[nbr][nbr]
    if nenner == 0:
        return("nominator is equal to zero, it isn't possible to calculate the IoU")
    else:
        return(zaehler/float(nenner))
        

dataset_nbr= "dataset_"+sys.argv[1]
exp = "exp_"+sys.argv[2]

i_path = "datasets/sar_ice/"+exp+"/output/"
output_file = "datasets/sar_ice/"+exp+"/log_file.txt"


""" read in pixel values of the prediction and the groundtruth"""
ground = []
pre = []
for filename in os.listdir(i_path):
    if filename.endswith(".png"):
        if filename[0] == "g":
            ground.append(filename)
        elif filename[0] == "p":
            pre.append(filename)
ground = sorted(ground)
pre = sorted(pre)      


""" calculate the confusion matrix"""
b = (255,255,255)
w = (255,0,0)
f = (0,0,255)
c =[[0,0,0],[0,0,0],[0,0,0]]
tot_pix = 0

for i in range(len(ground)):
    try:
        g_img = Image.open(i_path+ground[i])
        p_img = Image.open(i_path+pre[i])
        g_pix = list(g_img.getdata())
        p_pix = list(p_img.getdata())
    except:
        continue

    frozen_pixels = 0.0
    total_pixels = 0.0
    frozen_gt_pixels = 0.0
    tran_found = False
    prob_frozen = 0.0

    tot_pix = tot_pix+len(g_pix)
    counter = 0
    for j in range(len(g_pix)):
        if p_pix[j] == f:
            prob_frozen = prob_frozen + 1
        if g_pix[j] == w and p_pix[j] == w:
            c[0][0]=c[0][0]+1
            total_pixels= total_pixels + 1
        elif g_pix[j] == w and p_pix[j] == f:
            c[0][1]=c[0][1]+1
            frozen_pixels = frozen_pixels + 1
            total_pixels= total_pixels + 1
        elif g_pix[j] == w and p_pix[j] == b:
            c[0][2]=c[0][2]+1
            total_pixels= total_pixels + 1
        elif g_pix[j] == f and p_pix[j] == w:
            c[1][0]= c[1][0]+1
            total_pixels= total_pixels + 1
            frozen_gt_pixels = frozen_gt_pixels + 1
        elif g_pix[j] == f and p_pix[j] == f:
            c[1][1]=c[1][1]+1
            frozen_pixels = frozen_pixels + 1
            total_pixels= total_pixels + 1
            frozen_gt_pixels = frozen_gt_pixels + 1
        elif g_pix[j] == f and p_pix[j] == b:
            c[1][2]=c[1][2]+1
        elif g_pix[j] == b and p_pix[j] == w:
            c[2][0]=c[2][0]+1
        elif g_pix[j] == b and p_pix[j] == f:
            c[2][1]=c[2][1]+1
        elif g_pix[j] == b and p_pix[j] == b:
            c[2][2]=c[2][2]+1

    total_pixels = total_pixels + 1
    tot_pix = tot_pix-counter


"""" calculation recalls 0 = water, 1 = snow and ice, 2 = background """
rw = recall(0)
rsi = recall(1)

print("recall water: ",rw)
print("recall snow and ice: ",rsi)

"""" calculation precision 0 = water, 1 = snow and ice, 2 = background"""
pw = precision(0)
psi = precision(1)
print("precision water: ",pw)
print("precision snow and ice: ", psi)



"""calculation IoU 0 = water, 1 = snow and ice, 2 = background"""
i_w = iou(0)
i_si = iou(1)
print("IoU water: ",i_w)
print("IoU snow and ice: ", i_si)

"""calculation mean IoU water vs. snow/ice"""
mean_w_si = 0
if type(i_w) == str or type(i_si) == str:
    print("mean_iou: it isn't possible to calculate the mean IoU")
else:
    mean_w_si = (i_w+i_si)/2
    print("mean_iou_water_snow_ice: ", mean_w_si)    


"""water, snow/ice accurancy"""
tp_w_si = c[0][0]+c[1][1]
w_si_pix = c[0][0]+c[1][1]+c[0][1]+c[1][0]
oa_w_si = tp_w_si/float(w_si_pix)
print("accurancy water, snow/ice: ", oa_w_si)


""" write log file """
f = open(output_file,"a+")
f.write(dataset_nbr+"."+exp)
f.write("\n")
f.write("confusion matrix")
f.write("\n")
f.write(str(c[0]))
f.write("\n")
f.write(str(c[1]))
f.write("\n")

f.write("\n")
f.write("\n")
f.write("recall")
f.write("\n")
f.write("recall water: "+str(rw))
f.write("\n")
f.write("recall snow and ice: "+str(rsi))
f.write("\n")
f.write("\n")
f.write("precission:")
f.write("\n")
f.write("precision water: "+str(pw))
f.write("\n")
f.write("precision snow and ice: "+str(psi))

f.write("\n")
f.write("\n")
f.write("IoU:")
f.write("\n")
f.write("IoU water: "+str(i_w))
f.write("\n")
f.write("IoU snow and ice: "+str( i_si))
f.write("\n")
f.write("\n")
f.write("mean IoU water, snow/ice: ")
f.write(str(mean_w_si))
f.write("\n")
f.write("\n")
f.write("\n")
f.write("accurancy water, snow/ice: ")
f.write(str(oa_w_si))
f.write("\n")
f.write("\n")
f.close()
