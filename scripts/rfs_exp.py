import numpy as np
import cv2
import matplotlib.pyplot as plt
from util import get_index, get_index_mean, softmax
from inference import get_preds_single 
caffe_root = '../caffe/'
import sys
import random
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_cpu()

mean_file = '../data/train_mean.blob'

protos = ['../proto/FCN_big_1_v3_deploy.prototxt',
    '../proto/FCN_big_1_v4_deploy.prototxt',
    '../proto/FCN_big_1_v5_deploy.prototxt',
    '../proto/FCN_big_1_v6_deploy.prototxt']

models = ['../model/FCN_big_1_v3_iter_12000.caffemodel',
    '../model/FCN_big_1_v4_iter_21200.caffemodel',
    '../model/FCN_big_1_v5_iter_16800.caffemodel',
    '../model/FCN_big_1_v6_iter_24400.caffemodel']

layernames = ['conv3_3', 'conv4_3', 'conv4_3', 'conv4_7']

def get_filenames(filelists):
  f = open(filelists)
  filenames = []
  for l in f.readlines():
    l = l.strip().split()
    filenames.append(l[0])
  f.close()
  return filenames

if __name__ == '__main__':
  # usage: python rfs_exp.py root filelists
  root = sys.argv[1]
  filelists = sys.argv[2]
  outRoot = sys.argv[3]
  nets = []
  for i in range(len(protos)):
    nets.append(caffe.Net(protos[i], models[i], caffe.TEST))
  filenames = get_filenames(filelists)
  filenames = filenames[555:]
  random.shuffle(filenames)

  # save
  f = 'ibug/image_051_1.jpg'
  img = caffe.io.load_image(root + f)
  for i in range(len(nets)):
    pred = get_preds_single(nets[i], layernames[i], img)
    response_map = pred[0, 0]
    shape = response_map.shape
    response_map = softmax(response_map.reshape((1, shape[0]*shape[1])))
    response_map = np.reshape(response_map, shape)
    plt.imsave(outRoot + str(i) + "_" + f[5:], response_map, cmap='gray', vmin=response_map.min(), vmax=response_map.max())
  exit(0)
  for f in filenames:
    print f
    img = caffe.io.load_image(root + f)
    for i in range(len(nets)):
      pred = get_preds_single(nets[i], layernames[i], img)
      response_map = pred[0, (0, 36, 30, 57)]
      shape = response_map.shape
      response_map = softmax(response_map.reshape((shape[0], shape[1]*shape[2])))
      response_map = response_map.reshape(shape)
      for j in range(shape[0]):
        plt.subplot(4, 4, i*4 + j + 1)
        plt.imshow(response_map[j], cmap='gray', vmin=0, vmax=0.2)
    #plt.savefig(outRoot + f[5:])
    plt.show()
