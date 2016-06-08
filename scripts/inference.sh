#!/usr/bin/env bash

MEANFILE=../data/train_mean.blob
MEANSHAPE=../model/align_mean_shape.txt
W=../model/W.txt
T=43

PROTOTXT=../proto/FCN_big_1_v7_deploy.prototxt
MODEL=../model/FCN_big_1_v7_iter_120000.caffemodel
OUTPATH=../output/FCN_big_1_v7_pred
LAYERNAME=conv4_7
ROOT=../data/crop_gt_1.2/
FILELISTS=../data/crop_gt_1.2/test.txt
#FILELISTS=../output/temp.txt
METHOD=all

python inference.py --prototxt=$PROTOTXT --model=$MODEL \
  --layername=$LAYERNAME --root=$ROOT --filelists=$FILELISTS \
  --outpath=$OUTPATH --mean_file=$MEANFILE --W=$W --t=$T \
  --mean_shape=$MEANSHAPE --method=$METHOD

if [ "$METHOD" == "all" ]
then
  METHODS="max max_pca mean mean_pca"
  for m in $METHODS
  do
    echo "evaluating $m:"
    python evaluation.py $FILELISTS ${OUTPATH}_$m ${OUTPATH}_${m}_err
  done
else
  echo "evaluating:"
  python evaluation.py $FILELISTS $OUTPATH ${OUTPATH}_err
fi
