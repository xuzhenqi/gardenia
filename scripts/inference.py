# TODO: encapsule a class that receive params, and 
# give preds or shapes for single or multiple images.
import cv2
import numpy as np
import time
from util import get_index, get_index_mean, softmax
caffe_root = '../caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
# TODO: make CPU/GPU a choice
caffe.set_mode_cpu()

mean_file = '../data/train_mean.blob'

def get_mean():
    f = open(mean_file, 'r')
    mean_str = f.read()
    f.close()
    mean_blob = caffe.io.caffe_pb2.BlobProto()
    mean_blob.ParseFromString(mean_str)
    mean = caffe.io.blobproto_to_array(mean_blob)
    return mean

def get_filenames(filelists):
    f = open(filelists)
    filenames = []
    for l in f.readlines():
        l = l.strip().split()
        filenames.append(l[0])
    f.close()
    return filenames

def get_preds(net, layername, root, filenames):
    #TODO: using batch to accelerate
    mean = get_mean()
    preds = np.ndarray((0, 68, 56, 56))
    compute_time = 0
    load_time = 0
    for filename in filenames:
        start = time.clock()
        img_src = caffe.io.load_image(root + filename)
        load_time += time.clock() - start
        start = time.clock()
        preds_single = get_preds_single(net, layername, img_src)
        preds = np.concatenate((preds, preds_single), axis=0)
        compute_time += time.clock() - start
        print filename, compute_time, load_time
    print 'processing ', len(filenames), ' images, using ', compute_time + load_time, ' seconds.'
    return preds


def get_preds_single(net, layername, img_src):
    mean = get_mean()
    #shape = img_src.shape
    #mean = cv2.resize(mean, (shape[0], shape[1]))
    img = img_src.transpose(2, 0, 1)
    img = img[(2, 1, 0), :, :]
    img *= 255
    img -= mean.reshape(3, 224, 224)
    img *= 0.01
    net.blobs['data'].reshape(1, 3, 224, 224)
    net.blobs['data'].data[:] = img
    out = net.forward()
    return np.reshape(out[layername], (1, 68, 56, 56))

def get_preds_multiple(net, layername, imgs):
    # imgs: list of img
    preds = np.zeros((len(imgs), 68, 56, 56))
    i = 0
    for img in imgs:
        preds[i] = get_preds_single(net, layername, img)
        i += 1
    return preds


def dump(hp, wp, filename, filenames):
    sz = hp.shape
    f = open(filename, 'w')
    for i in range(sz[0]):
        f.write(filenames[i])
        for j in range(sz[1]):
            f.write(" %f %f" % (wp[i,j], hp[i,j]))
        f.write("\n")
    f.close()


def shape_map(preds, outfile_prefix, filenames):
    # max
    (hp, wp) = get_index(preds)
    hp = hp * 4
    wp = wp * 4
    dump(hp, wp, outfile_prefix + "_max.txt", filenames)
    (hp, wp) = get_index_mean(preds)
    hp = hp * 4
    wp = wp * 4
    dump(hp, wp, outfile_prefix + "_mean.txt", filenames)

if __name__ == '__main__':
    #usage: python inference.py prototxt model layername root filelists outpath
    prototxt = sys.argv[1]
    model = sys.argv[2]
    layername = sys.argv[3]
    root = sys.argv[4]
    filelists = sys.argv[5]
    outpath = sys.argv[6]

    net = caffe.Net(prototxt, model, caffe.TEST)
    #print dir(net)
    filenames = get_filenames(filelists)
    preds = get_preds(net, layername, root, filenames)
    preds_shape = preds.shape
    preds = softmax(np.reshape(preds, (preds_shape[0]*preds_shape[1], \
        preds_shape[2]*preds_shape[3])))
    preds = np.reshape(preds, preds_shape)
    shape_map(preds, outpath, filenames)
    del net
