# TODO:
#   - v4 vs. v4_1

import numpy as np
caffe_root = '/home/xuzhenqi/research/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_cpu()
from math import sqrt

def copy(blobvec1, blobvec2):
    for i in range(len(blobvec1)):
        blobvec1[i].data[:] = blobvec2[i].data[:]

def first2second(net1, net2):
    '''net1: 11 layer network model
       net2: 16 layer network model
    '''
    copy(net2.params['conv1_1'], net1.params['conv1_1'])
    copy(net2.params['conv1_2'], net1.params['conv1_2'])
    copy(net2.params['conv_pool_1'], net1.params['conv_pool_1'])
    copy(net2.params['conv2_0'], net1.params['conv2_0'])
    copy(net2.params['conv2_1'], net1.params['conv2_1'])
    net2.params['conv2_2'][0].data[:,:,:,:] = net1.params['conv2_2'][0].data[::2]
    net2.params['conv2_2'][1].data[:] = net1.params['conv2_2'][1].data[::2]
    net2.params['conv2_3'][0].data[:,:,:,:] = net1.params['conv2_2'][0].data[1::2]
    net2.params['conv2_3'][1].data[:] = net1.params['conv2_2'][1].data[1::2]
    net2.params['conv_pool_2'][0].data[:,:,:,:] = (net1.params['conv_pool_2'][0].data[::2] + net1.params['conv_pool_2'][0].data[1::2]) / 2
    net2.params['conv3_0'][0].data[:,:,:,:] = net1.params['conv3_0'][0].data[:,::2] + net1.params['conv3_0'][0].data[:,1::2]
    net2.params['conv3_0'][1].data[:] = net1.params['conv3_0'][1].data[:]
    copy(net2.params['conv3_1'], net1.params['conv3_0'])
    copy(net2.params['conv3_2'], net1.params['conv3_0'])
    copy(net2.params['conv3_3'], net1.params['conv3_0'])
    copy(net2.params['conv4_0'], net1.params['conv3_1'])
    copy(net2.params['conv4_1'], net1.params['conv3_2'])
    copy(net2.params['conv4_2'], net1.params['conv3_2'])
    copy(net2.params['conv4_3'], net1.params['conv3_3'])
    return net2

def second2third(net1, net2):
    net2.params['conv1_0'][0].data[:] = (net1.params['conv1_1'][0].data[::2] + net1.params['conv1_1'][0].data[1::2]) / 2
    net2.params['conv1_0'][1].data[:] = (net1.params['conv1_1'][1].data[::2] + net1.params['conv1_1'][1].data[1::2][1]) / 2
    net2.params['conv1_1'][0].data[:] = net1.params['conv1_2'][0].data[::2, ::2]
    net2.params['conv1_1'][1].data[:] = net1.params['conv1_2'][1].data[::2]
    net2.params['conv1_2'][0].data[:] = net1.params['conv1_2'][0].data[1::2, ::2]
    net2.params['conv1_2'][1].data[:] = net1.params['conv1_2'][1].data[1::2]
    net2.params['conv1_3'][0].data[:] = net1.params['conv1_2'][0].data[:, 1::2]
    net2.params['conv1_3'][1].data[:] = net1.params['conv1_2'][1].data[:]
    copy(net2.params['conv_pool_1'], net1.params['conv_pool_1'])
    net2.params['conv2_0'][0].data[:] = (net1.params['conv2_0'][0].data[::2] + net1.params['conv2_0'][0].data[1::2]) / 2
    net2.params['conv2_0'][1].data[:] = (net1.params['conv2_0'][1].data[::2] + net1.params['conv2_0'][1].data[1::2]) / 2
    net2.params['conv2_1'][0].data[:] = net1.params['conv2_1'][0].data[::2,::2]
    net2.params['conv2_1'][1].data[:] = net1.params['conv2_1'][1].data[::2]
    net2.params['conv2_2'][0].data[:] = net1.params['conv2_1'][0].data[::2,1::2]
    net2.params['conv2_2'][1].data[:] = net1.params['conv2_1'][1].data[::2]
    net2.params['conv2_3'][0].data[:] = net1.params['conv2_1'][0].data[1::2,::2]
    net2.params['conv2_3'][1].data[:] = net1.params['conv2_1'][1].data[1::2]
    net2.params['conv2_4'][0].data[:] = net1.params['conv2_1'][0].data[1::2,1::2]
    net2.params['conv2_4'][1].data[:] = net1.params['conv2_1'][1].data[1::2]
    net2.params['conv2_5'][0].data[:] = (net1.params['conv2_2'][0].data[::2, ::2] + net1.params['conv2_2'][0].data[1::2, ::2]) / 2
    net2.params['conv2_5'][1].data[:] = (net1.params['conv2_2'][1].data[::2] + net1.params['conv2_2'][1].data[1::2]) / 2
    net2.params['conv2_6'][0].data[:] = net1.params['conv2_2'][0].data[:, 1::2]
    net2.params['conv2_6'][1].data[:] = net1.params['conv2_2'][1].data[:]
    copy(net2.params['conv2_7'], net1.params['conv2_3'])
    copy(net2.params['conv_pool_2'], net1.params['conv_pool_2'])
    net2.params['conv3_0'][0].data[:] = (net1.params['conv3_0'][0].data[::2] + net1.params['conv3_0'][0].data[1::2]) / 2
    net2.params['conv3_0'][1].data[:] = (net1.params['conv3_0'][1].data[::2] + net1.params['conv3_0'][1].data[1::2]) / 2
    net2.params['conv3_1'][0].data[:] = net1.params['conv3_1'][0].data[::2,::2]
    net2.params['conv3_1'][1].data[:] = net1.params['conv3_1'][1].data[::2]
    net2.params['conv3_2'][0].data[:] = net1.params['conv3_1'][0].data[::2,1::2]
    net2.params['conv3_2'][1].data[:] = net1.params['conv3_1'][1].data[::2]
    net2.params['conv3_3'][0].data[:] = net1.params['conv3_1'][0].data[1::2,::2]
    net2.params['conv3_3'][1].data[:] = net1.params['conv3_1'][1].data[1::2]
    net2.params['conv3_4'][0].data[:] = net1.params['conv3_1'][0].data[1::2,1::2]
    net2.params['conv3_4'][1].data[:] = net1.params['conv3_1'][1].data[1::2]
    net2.params['conv3_5'][0].data[:] = (net1.params['conv3_2'][0].data[::2, ::2] + net1.params['conv3_2'][0].data[1::2, ::2]) / 2
    net2.params['conv3_5'][1].data[:] = (net1.params['conv3_2'][1].data[::2] + net1.params['conv3_2'][1].data[1::2]) / 2
    net2.params['conv3_6'][0].data[:] = net1.params['conv3_2'][0].data[:, 1::2]
    net2.params['conv3_6'][1].data[:] = net1.params['conv3_2'][1].data[:]
    copy(net2.params['conv3_7'], net1.params['conv3_3'])
    copy(net2.params['conv4_0'], net1.params['conv4_0'])
    copy(net2.params['conv4_1'], net1.params['conv4_1'])
    copy(net2.params['conv4_2'], net1.params['conv4_2'])
    copy(net2.params['conv4_3'], net1.params['conv4_3'])
    return net2

def third2fourth(net1, net2):
    copy(net2.params['conv1_0'], net1.params['conv1_0'])
    copy(net2.params['conv1_1'], net1.params['conv1_1'])
    copy(net2.params['conv1_2'], net1.params['conv1_2'])
    copy(net2.params['conv1_3'], net1.params['conv1_3'])
    copy(net2.params['conv_pool_1'], net1.params['conv_pool_1'])
    copy(net2.params['conv2_0'], net1.params['conv2_0'])
    copy(net2.params['conv2_1'], net1.params['conv2_1'])
    copy(net2.params['conv2_2'], net1.params['conv2_2'])
    copy(net2.params['conv2_3'], net1.params['conv2_3'])
    copy(net2.params['conv2_4'], net1.params['conv2_4'])
    copy(net2.params['conv2_5'], net1.params['conv2_5'])
    net2.params['conv2_6'][0].data[:] = net1.params['conv2_6'][0].data[::2]
    net2.params['conv2_6'][1].data[:] = net1.params['conv2_6'][1].data[::2]
    net2.params['conv2_7'][0].data[:] = net1.params['conv2_6'][0].data[1::2]
    net2.params['conv2_7'][1].data[:] = net1.params['conv2_6'][1].data[1::2]
    net2.params['conv2_8'][0].data[:] = net1.params['conv2_7'][0].data[::2, ::2] * sqrt(2)
    net2.params['conv2_8'][1].data[:] = net1.params['conv2_7'][1].data[::2]
    net2.params['conv2_9'][0].data[:] = net1.params['conv2_7'][0].data[1::2, ::2] * sqrt(2)
    net2.params['conv2_9'][1].data[:] = net1.params['conv2_7'][1].data[1::2]
    net2.params['conv2_10'][0].data[:] = net1.params['conv2_7'][0].data[:, 1::2] * sqrt(2)
    net2.params['conv2_10'][1].data[:] = net1.params['conv2_7'][1].data[:]
    copy(net2.params['conv_pool_2'], net1.params['conv_pool_2'])
    copy(net2.params['conv3_0'], net1.params['conv3_0'])
    copy(net2.params['conv3_1'], net1.params['conv3_1'])
    copy(net2.params['conv3_2'], net1.params['conv3_2'])
    copy(net2.params['conv3_3'], net1.params['conv3_3'])
    copy(net2.params['conv3_4'], net1.params['conv3_4'])
    copy(net2.params['conv3_5'], net1.params['conv3_5'])
    net2.params['conv3_6'][0].data[:] = net1.params['conv3_6'][0].data[::2]
    net2.params['conv3_6'][1].data[:] = net1.params['conv3_6'][1].data[::2]
    net2.params['conv3_7'][0].data[:] = net1.params['conv3_6'][0].data[1::2]
    net2.params['conv3_7'][1].data[:] = net1.params['conv3_6'][1].data[1::2]
    net2.params['conv3_8'][0].data[:] = net1.params['conv3_7'][0].data[::2, ::2] * sqrt(2)
    net2.params['conv3_8'][1].data[:] = net1.params['conv3_7'][1].data[::2]
    net2.params['conv3_9'][0].data[:] = net1.params['conv3_7'][0].data[1::2, ::2] * sqrt(2)
    net2.params['conv3_9'][1].data[:] = net1.params['conv3_7'][1].data[1::2]
    net2.params['conv3_10'][0].data[:] = net1.params['conv3_7'][0].data[:, 1::2] * sqrt(2)
    net2.params['conv3_10'][1].data[:] = net1.params['conv3_7'][1].data[:]
    net2.params['conv4_0'][0].data[:] = net1.params['conv4_0'][0].data[0:256]
    net2.params['conv4_0'][1].data[:] = net1.params['conv4_0'][1].data[0:256]
    net2.params['conv4_1'][0].data[:] = net1.params['conv4_0'][0].data[100:356]
    net2.params['conv4_1'][1].data[:] = net1.params['conv4_0'][1].data[100:356]
    net2.params['conv4_2'][0].data[:] = net1.params['conv4_0'][0].data[200:456]
    net2.params['conv4_2'][1].data[:] = net1.params['conv4_0'][1].data[200:456]
    net2.params['conv4_3'][0].data[:] = net1.params['conv4_0'][0].data[300:556]
    net2.params['conv4_3'][1].data[:] = net1.params['conv4_0'][1].data[300:556]
    net2.params['conv4_4'][0].data[:] = net1.params['conv4_0'][0].data[424:680]
    net2.params['conv4_4'][1].data[:] = net1.params['conv4_0'][1].data[424:680]
    net2.params['conv4_5'][0].data[:] = np.reshape(np.tile(net1.params['conv4_1'][0].data[0:512], (1, 13, 1, 1)), (256, 260, 3, 3))[:,0:256] / sqrt(25.6)
    net2.params['conv4_5'][1].data[:] = (net1.params['conv4_1'][1].data[0:256] + net1.params['conv4_1'][1].data[256:512]) / 2 
    net2.params['conv4_6'][0].data[:] = np.reshape(np.tile(net1.params['conv4_2'][0].data[:], (1, 13, 1, 1)), (340, 260, 3, 3))[:,0:256] / sqrt(25.6)
    net2.params['conv4_6'][1].data[:] = (net1.params['conv4_2'][1].data[0:340] + net1.params['conv4_2'][1].data[340:]) / 2
    net2.params['conv4_7'][0].data[:] = np.tile(net1.params['conv4_3'][0].data[:], (1, 34, 1, 1)) / sqrt(34)
    net2.params['conv4_7'][1].data[:] = net1.params['conv4_3'][1].data[:]


def six2seven(net1, net2):
    preserve = [
            'conv1_0', 'conv1_1', 'conv1_2', 'conv1_3', 'conv_pool_1',
            'conv2_0', 'conv2_1', 'conv2_2', 'conv2_3', 'conv2_4', 'conv2_5',
            'conv2_6', 'conv2_7', 'conv2_8', 'conv2_9', 'conv2_10', 
            'conv_pool_2',
            'conv3_0', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv3_5',
            'conv3_6', 'conv3_7', 'conv3_8', 'conv3_9', 'conv3_10',
            'conv4_0']
    for layerName in preserve:
        copy(net2.params[layerName], net1.params[layerName])
    for i in range(6):
        net2.params['conv4_' + str(i) +'_'+ str(i+1)][0].data[:] = net1.params['conv4_' + str(i+1)][0].data[128:256]
        net2.params['conv4_' + str(i) +'_'+ str(i+1)][1].data[:] = net1.params['conv4_' + str(i+1)][1].data[0:128]
        net2.params['conv4_' + str(i+1)][0].data[:] = np.swapaxes(net1.params['conv4_' + str(i+1)][0].data[128:256], 0, 1) * sqrt(2.)
        net2.params['conv4_' + str(i+1)][1].data[:] = net1.params['conv4_'+str(i+1)][1].data[0:256]
    net2.params['conv4_7'][0].data[:] = net1.params['conv4_7'][0].data[:,0:256] * sqrt(340./256)
    net2.params['conv4_7'][1].data[:] = net1.params['conv4_7'][1].data[:]
    return net2


if __name__ == '__main__':
    #usage: python model_change.py [funName] [modelPath] [newModelPath]
    if sys.argv[1] == "first2second":
        net1 = caffe.Net('../proto/FCN_big_1_v3.prototxt', sys.argv[2], caffe.TEST)
        net2 = caffe.Net('../proto/FCN_big_1_v4_1.prototxt', caffe.TEST)
        net2 = first2second(net1, net2)
    elif sys.argv[1] == "second2third":
        net1 = caffe.Net('../proto/FCN_big_1_v4_1.prototxt', sys.argv[2], caffe.TEST)
        net2 = caffe.Net('../proto/FCN_big_1_v5.prototxt', caffe.TEST)
        net2 = second2third(net1, net2)
    elif sys.argv[1] == "third2fourth":
        net1 = caffe.Net('../proto/FCN_big_1_v5.prototxt', sys.argv[2], caffe.TEST)
        net2 = caffe.Net('../proto/FCN_big_1_v6.prototxt', caffe.TEST)
        net2 = third2fourth(net1, net2)
    elif sys.argv[1] == "six2seven":
        net1 = caffe.Net('../proto/FCN_big_1_v6.prototxt', sys.argv[2], caffe.TEST)
        for k, v in net1.params.items():
            print k, 
            for i in v:
                print i.data.shape, 
            print
        net2 = caffe.Net('../proto/FCN_big_1_v7.prototxt', caffe.TEST)
        for k, v in net2.params.items():
            print k, 
            for i in v:
                print i.data.shape, 
            print
        net2 = six2seven(net1, net2)
    net2.save(sys.argv[3])
