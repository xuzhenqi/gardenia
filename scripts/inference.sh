#!/usr/bin/env bash

PROTOTXT=../proto/FCN_big_1_v6_deploy.prototxt
MODEL=../model/FCN_big_1_v6_iter_24400.caffemodel
LAYERNAME=conv4_7
ROOT=../data/crop_gt_1.2/
#FILELISTS=../data/crop_gt_1.2/test.txt
FILELISTS=../output/temp.txt
OUTPATH=../output/temp_pred.txt
MEANFILE=../data/train_mean.blob
MEANSHAPE=../model/align_mean_shape.txt
W=../model/W.txt
T=43
METHOD=mean_pca

python inference.py --prototxt=$PROTOTXT --model=$MODEL \
  --layername=$LAYERNAME --root=$ROOT --filelists=$FILELISTS \
  --outpath=$OUTPATH --mean_file=$MEANFILE --W=$W --t=$T \
  --mean_shape=$MEANSHAPE --method=$METHOD 
