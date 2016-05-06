# gardenia
Gardenia is a face alignment project using Convolutional Neural Network. The project illstrate the training and testing process on 300W dataset.

## Installation
- Download and install [caffe from xuzhenqi] (https://github.com/xuzhenqi/caffe/tree/eye_detection). We use both the c++ and python interfaces.
- Matlab

## Train tutorial
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
python inference.py $prototxt $model $layername $root $filelists $outpath 
scripts/evalution.sh # report mean err on 300W dataset and CED curve.
```

