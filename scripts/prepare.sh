#!/usr/bin/env bash

set -e

matlab -nodisplay -nojvm -nosplash -r "augmentation;crop;quit"

# TODO: rewriting 

CAFFE_ROOT=../caffe
echo "computing mean ..."
TRAIN_ROOT=../data/
TRAIN_FILE=../data/train.txt
TRAIN_DB=../data/train_lmdb
MEAN_FILE=../data/train_mean.blob
$CAFFE_ROOT/build/tools/convert_imageset --check_size=true \
  $TRAIN_ROOT $TRAIN_FILE $TRAIN_DB
$CAFFE_ROOT/build/tools/compute_image_mean $TRAIN_DB $MEAN_FILE
