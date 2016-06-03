import numpy as np
from util import errrate_v3
import sys
import os

def get_shape(shape_file, filename=True):
    f = open(shape_file, 'r')
    filenames = []
    shapes = []
    for l in f.readlines():
        l = l.split()
        if filename:
            filenames.append(l[0])
            shapes.append([float(i) for i in l[1:]])
        else:
            shapes.append([float(i) for i in l])
    f.close()
    shapes = np.array(shapes)
    return (filenames, shapes)

def errrate(preds, label):
    '''
        preds: size of N*(2P)
        label: size of N*(2P)
        return (average error, error for each sample)
    '''
    shape = preds.shape
    preds = preds.reshape(shape[0], shape[1]/2, 2)
    label = label.reshape(shape[0], shape[1]/2, 2)
    wp = preds[:,:,0]
    hp = preds[:,:,1]
    wl = label[:,:,0]
    hl = label[:,:,1]
    eye = np.sqrt(np.square(hl[:, 36:42].mean(axis=1) - hl[:, 42:48].mean(axis=1)) + np.square(wl[:,36:42].mean(axis=1) - wl[:,42:48].mean(axis=1)))
    err = np.sqrt((hp - hl) * (hp - hl) + (wp - wl) * (wp - wl)) / np.reshape(eye, (shape[0], 1))
    err = err.mean(axis=1)
    return (err.mean(), err)

def dumperr(err, path, filenames):
    f = open(path, 'w')
    for i in range(len(filenames)):
        f.write("%s %f\n" % (filenames[i], err[i]))
    f.close()

if __name__ == '__main__':
    #usage: python evaluation.py label_file prediction_file out_prefix
    label_file = sys.argv[1]
    prediction_file = sys.argv[2]
    out_prefix = sys.argv[3]

    (filenames, preds) = get_shape(prediction_file)
    (filenames, label) = get_shape(label_file)
    
    (aerr, err) = errrate(preds, label)
    dumperr(err, out_prefix, filenames)
    print "Full set:", aerr
    (aerr, err) = errrate(preds[554:689], label[554:689])
    print "Challenging set:", aerr 
    (aerr, err) = errrate(preds[0:554], label[0:554])
    print "Common set:", aerr
