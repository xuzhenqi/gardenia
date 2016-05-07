#!/bin/bash
# TODO: 
#   - train 11 layer network from scratch
#   - dataset setting available
#   - caffe input and loss layer
#   - model initialization
#   - lost file from this machine

#CAFFE_ROOT=~/research/caffe
#PROJECT_ROOT=/home/xuzhenqi/research/face_alignment
#
#if [ $1 == "first" ] #1: 11 layer network
#then
#  SOLVER=../proto/FCN_big_1_v3_solver.prototxt
#  WEIGHT=../snapshot/FCN_big_1_v2_iter_33200.caffemodel
#  LOG=../log/fcn_big_1_v3.log
#elif [ $1 == "second" ] #2: 16 layer network 
#then
#  SOLVER=../proto/FCN_big_1_v4_1_solver.prototxt
#  WEIGHT=../snapshot/preserve/FCN_big_1_v4_1_begin.caffemodel
#  LOG=../log/fcn_big_1_v4_1.log
#elif [ $1 == "third" ] #3: 26 layer network
#  SOLVER=../proto/FCN_big_1_v5_solver.prototxt
#  WEIGHT=../snapshot/preserve/FCN_big_1_v5_begin.caffemodel
#  LOG=../log/fcn_big_1_v5.log
#elif [ $1 == "fourth" ] #4: 36 layer network
#  SOLVER=../proto/FCN_big_1_v6_solver.prototxt
#  WEIGHT=../snapshot/preserve/FCN_big_1_v6_begin.caffemodel
#  LOG=../log/fcn_big_1_v6.log
#fi
#
#$CAFFE_ROOT/build/tools/caffe train --solver=$SOLVER --weights=$WEIGHT 2> $LOG

CAFFE_ROOT=../caffe
NETNAME=FCN_big_1_v6
SOLVER=../proto/${NETNAME}_solver.prototxt
LOG=../log/${NETNAME}.log
$CAFFE_ROOT/build/tools/caffe train --solver=$SOLVER 2> $LOG
