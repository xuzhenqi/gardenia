import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from util import get_index, get_index_mean, softmax
from inference import get_preds_multiple 
caffe_root = '../caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()

mean_file = '../data/train_mean.blob'

def get_filenames_bbx(filelists):
    f = open(filelists)
    filenames = []
    bbxs = []
    for l in f.readlines():
        l = l.strip().split()
        filenames.append(l[0])
        bbxs.append([float(l[i]) for i in range(1, len(l))])
    return (filenames, bbxs)

def bbx_expand(bbx, ratio, v, h):
    print 'bbx_expand(): input: ', bbx, ratio, v, h
    lr = ratio + h
    rr = ratio - h
    ur = ratio + v
    dr = ratio - v
    left = bbx[0] - (bbx[2] - bbx[0]) * lr
    right = bbx[2] + (bbx[2] - bbx[0]) * rr
    up = bbx[1] - (bbx[3] - bbx[1]) * ur
    down = bbx[3] + (bbx[3] - bbx[1]) * dr
    return [round(left), round(up), round(right), round(down)]


def crop(img_src, bbx, size):
    #import ipdb; ipdb.set_trace()
    print "crop() input: ", bbx, size
    img_crop = np.zeros((bbx[3] + 1 - bbx[1], bbx[2] + 1 - bbx[0], 3))
    shape = img_src.shape
    if bbx[2] > shape[1] - 1:
        right = shape[1] - 1 -  bbx[0]
        bbx[2] = shape[1] - 1
    else:
        right = bbx[2] - bbx[0]
    if bbx[3] > shape[0] - 1:
        down = shape[0] - 1 - bbx[1]
        bbx[3] = shape[0] - 1
    else:
        down =  bbx[3] - bbx[1]
    if bbx[0] < 0:
        left = -bbx[0]
        bbx[0] = 0
    else:
        left = 0
    if bbx[1] < 0:
        up = -bbx[1]
        bbx[1] = 0
    else:
        up = 0
    if down - up != bbx[3] - bbx[1]:
        print "[error] up:", up, "down:", down, "bbx3:", bbx[3], "bbx1:", bbx[1]
        exit(0)
    if right - left != bbx[2] - bbx[0]:
        print "[error] right:", right, "left:", left, "bbx2:", bbx[2], "bbx0:", bbx[0]
        exit(0)
    print down - up, bbx[3] - bbx[1], right - left, bbx[2] - bbx[0]
    #import ipdb; ipdb.set_trace()
    img_crop[up:(down + 1), left:(right+1)] = img_src[bbx[1]:(bbx[3]+1), bbx[0]:(bbx[2]+1)]
    return cv2.resize(img_crop, size) 

def shift_exp(root, filename, bbx, outRoot):
    img_src = caffe.io.load_image(root + filename)
    shape = img_src.shape
    if len(shape) != 3 and shape[2] != 3:
        print "[error] only support RGB images!"
        exit(0)
    vshift = [-0.15, 0, 0.15]
    hshift = [-0.15, 0, 0.15]
    #size = (int(round(224/1.1*1.4)), int(round(224/1.1*1.4)))
    size = (224, 224)
    img_crops = [] 
    i = 0
    for v in vshift:
        for h in hshift:
            bbx_e = bbx_expand(bbx, 0.2, v, h)
            img_crops.append(crop(img_src, bbx_e, size))
    return img_crops


if __name__ == '__main__':
    # usage: python shift_exp.py prototxt model layername root 
    # filelists outRoot
    prototxt = sys.argv[1]
    model = sys.argv[2]
    layername = sys.argv[3]
    root = sys.argv[4]
    filelists = sys.argv[5]
    outRoot = sys.argv[6]

    net = caffe.Net(prototxt, model, caffe.TEST)
    (filenames, bbxs) = get_filenames_bbx(filelists)
    index = range(len(filenames))
    random.shuffle(index)
    for i in index:
        print i, filenames[i], bbxs[i][0], bbxs[i][1], bbxs[i][2], bbxs[i][3]
        img_crops = shift_exp(root, filenames[i], bbxs[i], outRoot)
        preds = get_preds_multiple(net, layername, img_crops)
        preds_shape = preds.shape
        preds = softmax(np.reshape(preds, (preds_shape[0]*preds_shape[1], \
            preds_shape[2]*preds_shape[3])))
        preds = np.reshape(preds, preds_shape)
        (hp, wp) = get_index(preds)
        hp = hp * 4
        wp = wp * 4
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.imshow(img_crops[i])
            plt.plot(wp[i], hp[i],'.g', hold=True)
        plt.show()
