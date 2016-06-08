import cv2
import numpy as np
import time
from util import get_index, get_index_mean, softmax, \
    align_shape, transform_reverse, get_filenames, dump, show_predict, \
    vectorize
caffe_root = '../caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from optparse import OptionParser
import matplotlib.pyplot as plt
import copy


class Alignment(object):
    def __init__(self, proto, model, layername, mean_file, W, t, mean_shape):
        self.net = caffe.Net(proto, model, caffe.TEST)
        self.layername = layername
        self.mean = self.get_mean(mean_file)
        (self.W, self.mean_shape) = self.get_pca_model(W, t, mean_shape)

    def get_mean(self, mean_file):
        f = open(mean_file, 'r')
        mean_str = f.read()
        f.close()
        mean_blob = caffe.io.caffe_pb2.BlobProto()
        mean_blob.ParseFromString(mean_str)
        mean = caffe.io.blobproto_to_array(mean_blob)
        return mean

    def get_pca_model(self, W, t, mean_shape):
        W_data = np.loadtxt(W)
        W_data = W_data[:, 0:t]
        mean_shape_data = np.loadtxt(mean_shape)
        return (W_data, mean_shape_data)

    def get_pred(self, img):
        '''img: face image, typically the cropped face based on bbx'''
        img = cv2.resize(img, (224, 224))
        self.img = copy.deepcopy(img)
        img = img.transpose(2, 0, 1)
        img = img[(2, 1, 0), :, :]  # TODO
        img *= 255
        img -= self.mean.reshape(3, 224, 224)
        img *= 0.01
        self.net.blobs['data'].reshape(1, 3, 224, 224)
        self.net.blobs['data'].data[:] = img
        out = self.net.forward()
        pred = np.reshape(out[self.layername], (68, 56 * 56))
        pred = softmax(pred)
        return np.reshape(pred, (1, 68, 56, 56))

    def shape_map(self, pred, method):
        '''method can be one of ['max', 'mean', 'max_pca', 'mean_pca']'''
        if method == 'max':
            shape = vectorize(get_index(pred))
            shape = shape * 4
        elif method == 'mean':
            shape = vectorize(get_index_mean(pred))
            shape = shape * 4
        elif method == 'max_pca':
            shape = vectorize(get_index(pred))
            shape = shape * 4
            shape = self.pca_noise_reduction(shape)
        elif method == 'mean_pca':
            shape = vectorize(get_index_mean(pred))
            shape = shape * 4
            shape_pca = self.pca_noise_reduction(shape)
            if p.debug:
                show_predict(self.img, shape)
                show_predict(self.img, shape_pca)
        else:
            print "Unknown method: ", method
            exit(0)
        return shape

    def shape_map_all(self, pred):
        shapes = []
        shape_max = vectorize(get_index(pred))
        shape_max = shape_max * 4
        shape_max_pca = self.pca_noise_reduction(shape_max)
        shape_mean = vectorize(get_index_mean(pred))
        shape_mean = shape_mean * 4
        shape_mean_pca = self.pca_noise_reduction(shape_mean)
        shapes.append(shape_max)
        shapes.append(shape_max_pca)
        shapes.append(shape_mean)
        shapes.append(shape_mean_pca)
        if p.debug:
            show_predict(self.img, shape_max)
            show_predict(self.img, shape_max_pca)
            show_predict(self.img, shape_mean)
            show_predict(self.img, shape_mean_pca)
        return shapes

    def pca_noise_reduction(self, shape):
        shape = shape.reshape((68, 2))
        shape_mean = shape.mean(axis=0)
        (shape_a, a, b) = align_shape(shape - shape_mean, self.mean_shape)
        shape_r = np.dot(shape_a - self.mean_shape.reshape(shape_a.shape),
                         self.W)
        shape_r = np.dot(shape_r, np.transpose(self.W)) + \
            self.mean_shape.reshape(shape_a.shape)
        return transform_reverse(shape_r, a, b, shape_mean)

    def process(self, img, method):
        pred = self.get_pred(img)
        shape = self.shape_map(pred, method)
        return shape

    def process_all(self, img):
        pred = self.get_pred(img)
        shapes = self.shape_map_all(pred)
        return shapes


def init():
    parser = OptionParser()
    parser.add_option('--prototxt', type='string')
    parser.add_option('--model', type='string')
    parser.add_option('--layername', type='string')
    parser.add_option('--root', type='string')
    parser.add_option('--filelists', type='string')
    parser.add_option('--outpath', type='string')
    parser.add_option('--mean_file', type='string')
    parser.add_option('--W', type='string', help='file to store W')
    parser.add_option('--t', type='int', help='the number of eigenvalues')
    parser.add_option('--mean_shape', type='string')
    parser.add_option('--debug', action="store_true", default=False)
    parser.add_option('--method', type='choice', default='mean_pca',
                      choices=['max', 'mean', 'max_pca', 'mean_pca', 'all'])
    (options, args) = parser.parse_args()
    return options

p = init()
if __name__ == '__main__':
    caffe.set_mode_gpu()
    m = Alignment(p.prototxt, p.model, p.layername, p.mean_file, p.W, p.t,
                  p.mean_shape)
    filenames = get_filenames(p.filelists)
    if p.method == 'all':
        shapes_max = []
        shapes_max_pca = []
        shapes_mean = []
        shapes_mean_pca = []
        for filename in filenames:
            print filename
            img = caffe.io.load_image(p.root + filename)
            shapes = m.process_all(img)
            shapes_max.append(shapes[0])
            shapes_max_pca.append(shapes[1])
            shapes_mean.append(shapes[2])
            shapes_mean_pca.append(shapes[3])
        dump(np.array(shapes_max), filenames, p.outpath + '_max')
        dump(np.array(shapes_max_pca), filenames, p.outpath + '_max_pca')
        dump(np.array(shapes_mean), filenames, p.outpath + '_mean')
        dump(np.array(shapes_mean_pca), filenames, p.outpath + '_mean_pca')
    else:
        shapes = []
        for filename in filenames:
            print filename
            img = caffe.io.load_image(p.root + filename)
            shape = m.process(img, p.method)
            shapes.append(shape)
        shapes = np.array(shapes)
        dump(shapes, filenames, p.outpath)
    print "Process finished!"
