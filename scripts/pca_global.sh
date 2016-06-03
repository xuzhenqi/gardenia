#!/usr/bin/env bash

set -e

pred="../output/test_pred_v2_gt_1.2_err_mean.txt"
path="../output/test_pred_v2_gt_1.2_err_mean_pca.txt"
err_path="../output/test_pred_v2_gt_1.2_mean_pca_err.txt"
W_file="../model/W.txt"
mean_shape_file="../model/align_mean_shape.txt"
t=43

matlab -nodisplay -nojvm -nosplash -r "pca_global_infer('$pred', '$W_file',\
  '$mean_shape_file', $t, '$path');exit;"

python evaluation.py ../data/crop_gt_1.2/test.txt $path $err_path 
