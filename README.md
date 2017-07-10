# gardenia
Gardenia is a face alignment project using Convolutional Neural Network. The project illstrate the training and testing process on 300W dataset.

## Installation
- Download and install [caffe from xuzhenqi] (https://github.com/xuzhenqi/caffe/tree/eye_detection). We use both the c++ and python interfaces.
- Matlab

## Tutorial
We need first to prepare the data, suppose the original data root path are `data/`. Run
```
scripts/prepare.sh
```

To crop the testset and augment the trainset, and we will get the `data/augment/train.txt`, `data/train_mean.blob` and `data/crop/test.txt`. 
After preparing the data, we can run 
```
scripts/train.sh
```
to train our model. The prototxt are set done on `proto` folder.

After training a model, suppose the model path to be `$MODEL`, run 
```
scripts/inference.sh # You may need to set some options in inference.sh to predict labels 
scripts/evalution.sh # report mean err on 300W dataset and CED curve.
```
The `inference.sh` will use trainded model to predict shapes, the pretrainded can be downloaded from [here] (http://pan.baidu.com/s/1cBOph8). Feel free to put an issue if you encounter problems.

## Speed
Tested on Ubuntu 16.04, Nvidia TITAN X, without cudnn, without image load and data processing.

| proto | num of layers | average forward speed | average backward speed |
|--|--|--|--|
|v3| 11 | 17.5707 ms / 1 sample | 38.4078 ms / 1 sample |
|v4| 16 | 20.783 ms / 1 sample | 41.3371 ms / 1 sample |
|v5| 26 | 22.3556 ms / 1 sample | 54.5015 ms / 1 sample |
|v6| 36 | 25.0671 ms / 1 sample | 54.6937 ms / 1 sample |
|v7| 41 | 26.0211 ms / 1 sample | 55.7797 ms / 1 sample |

## Citation
```
@inproceedings{xu2016learning,
  title={Learning Facial Point Response for Alignment by Purely Convolutional Network},
  author={Xu, Zhenqi and Deng, Weihong and Hu, Jiani},
  booktitle={Asian Conference on Computer Vision},
  pages={248--263},
  year={2016},
  organization={Springer}
}
```
You can get a pdf copy from https://pan.baidu.com/s/1qXKo85Q
