import numpy as np
import math
import matplotlib.pyplot as plt


def dump(shapes, filenames, outfile):
    sz = shapes.shape
    f = open(outfile, 'w')
    for i in range(sz[0]):
        f.write(filenames[i])
        for j in range(sz[1]):
            f.write(" %f" % shapes[i, j])
        f.write("\n")
    f.close()


def get_filenames(filelists):
    f = open(filelists)
    filenames = []
    for l in f.readlines():
        l = l.strip().split()
        filenames.append(l[0])
    f.close()
    return filenames


def align_shape(shape, mean_shape):
    ''' can only process one image'''
    N = len(mean_shape) / 2
    shape = np.reshape(shape, (N, 2))
    mean_shape = np.reshape(mean_shape, (N, 2))
    xlen = np.dot(shape.reshape((N*2, )), shape.reshape((N*2, )))
    a = np.dot(shape.reshape((N*2, )), mean_shape.reshape((N*2, ))) / xlen
    b = (np.dot(shape[:, 0], mean_shape[:, 1]) -
         np.dot(shape[:, 1], mean_shape[:, 0])) / xlen
    new_shape = np.dot(shape, np.array([(a, b), (-b, a)]))
    return (vectorize(new_shape), a, b)


def transform_reverse(shape, a, b, shape_mean):
    N = shape.size / 2
    shape = np.reshape(shape, (N, 2))
    trans = np.ndarray((2, 2))
    trans[0, 0] = a
    trans[0, 1] = -b
    trans[1, 0] = b
    trans[1, 1] = a
    trans = trans / (a*a + b*b)
    ori_shape = np.dot(shape, trans) + shape_mean
    return ori_shape.reshape((N*2, ))


def show(img, show_max=False):
    '''show response map'''
    img = img - img.min()
    img = img / img.max()
    plt.imshow(img, cmap='gray')
    shape = img.shape
    idx = np.argmax(img)
    hi = idx / shape[1]
    wi = idx % shape[1]
    print hi, wi
    if show_max:
        plt.hold(True)
        plt.plot(wi, hi, 'r.', markersize=12)
        plt.axis('off')
        plt.axis('image')
    plt.show()


def gauss_map(label, in_size, std, scale):
    x = np.linspace(0, in_size, in_size, False)
    y = np.linspace(0, in_size, in_size, False)
    xv, yv = np.meshgrid(x, y)
    points = len(label) / 2
    maps = np.ndarray([points, in_size, in_size], dtype=float)
    for i in range(points):
        maps[i] = np.exp((-(xv-label[2*i])*(xv-label[2*i])-(yv-label[2*i+1])*(
            yv-label[2*i+1]))/std/std)
    ans = maps[:, ::scale, ::scale]
    return ans


def errrate(pre, label):
    '''TODO: recompute the eye distance'''
    shape = pre.shape
    idx = np.argmax(np.reshape(label, (shape[0], shape[1], shape[2]*shape[3])),
                    axis=2)
    hl = idx / shape[3]
    wl = idx % shape[3]
    idx = np.argmax(np.reshape(pre, (shape[0], shape[1], shape[2]*shape[3])),
                    axis=2)
    hp = idx / shape[3]
    wp = idx % shape[3]
    eye = np.sqrt((hl[:, 36] - hl[:, 45])*(hl[:, 36] - hl[:, 45])
                  + (wl[:, 36] - wl[:, 45])*(wl[:, 36] - wl[:, 45]))
    err = np.sqrt((hp - hl) * (hp - hl) + (wp - wl) * (wp - wl)) \
        / np.reshape(eye, (shape[0], 1))
    return np.sum(err) / shape[0] / shape[1]


def errrate_v2(pre, label):
    '''TODO: recompute the eye distance'''
    shape = pre.shape
    wl = label[:, :, 0]
    hl = label[:, :, 1]
    idx = np.argmax(np.reshape(pre, (shape[0], shape[1], shape[2]*shape[3])),
                    axis=2)
    hp = idx / shape[3]
    wp = idx % shape[3]
    eye = np.sqrt((hl[:, 36] - hl[:, 45])*(hl[:, 36] - hl[:, 45])
                  + (wl[:, 36] - wl[:, 45])*(wl[:, 36] - wl[:, 45]))
    err = np.sqrt((hp - hl) * (hp - hl) + (wp - wl) * (wp - wl)) \
        / np.reshape(eye, (shape[0], 1))
    return np.sum(err) / shape[0] / shape[1]


def errrate_v3(pre, label):
    shape = pre.shape
    (hp, wp) = get_index(pre)
    wl = label[:, :, 0]
    hl = label[:, :, 1]
    eye = np.sqrt(np.square(hl[:, 36:42].mean(axis=1) -
                            hl[:, 42:48].mean(axis=1)) +
                  np.square(wl[:, 36:42].mean(axis=1) -
                            wl[:, 42:48].mean(axis=1)))
    err = np.sqrt((hp - hl) * (hp - hl) + (wp - wl) * (wp - wl)) \
        / np.reshape(eye, (shape[0], 1))
    return np.sum(err) / shape[0] / shape[1]


def errrate_v4(pre, label):
    shape = pre.shape
    wp = pre[:, :, 0]
    hp = pre[:, :, 1]
    wl = label[:, :, 0]
    hl = label[:, :, 1]
    eye = np.sqrt(np.square(hl[:, 36:42].mean(axis=1) -
                            hl[:, 42:48].mean(axis=1)) +
                  np.square(wl[:, 36:42].mean(axis=1) -
                            wl[:, 42:48].mean(axis=1)))
    err = np.sqrt((hp - hl) * (hp - hl) + (wp - wl) * (wp - wl)) \
        / np.reshape(eye, (shape[0], 1))
    return np.sum(err) / shape[0] / shape[1]


def get_index(pre):
    shape = pre.shape
    idx = np.argmax(np.reshape(pre, (shape[0], shape[1], shape[2]*shape[3])),
                    axis=2)
    out = np.ndarray((shape[0], shape[1], 2))
    out[:, :, 0] = idx % shape[3]
    out[:, :, 1] = idx / shape[3]
    return out


def get_index_mean(pre):
    shape = pre.shape
    h = np.reshape(np.array(range(shape[2])), (1, 1, shape[2], 1))
    w = np.reshape(np.array(range(shape[3])), (1, 1, 1, shape[3]))
    shape = np.ndarray((shape[0], shape[1], 2))
    shape[:, :, 0] = np.sum(np.sum(pre * w, axis=3), axis=2)
    shape[:, :, 1] = np.sum(np.sum(pre * h, axis=3), axis=2)
    return shape


def softmax(feat):
    '''
    :param feat: a M*N matrix,
    :return: the softmax of the second axis.
    '''
    feat_max = feat.max(1)
    feat = feat - feat_max.reshape((feat_max.shape[0], 1))
    feat_exp = np.exp(feat)
    feat_sum = feat_exp.sum(1)
    prob = np.divide(feat_exp, feat_sum.reshape((feat_sum.shape[0], 1)))
    return prob


def normalize(feat):
    '''
    :param feat: a N*N matrix
    :return: normalize of the second axis.
    '''
    feat_sum = feat.sum(1)
    feat = feat / feat_sum.reshape((feat_sum.shape[0], 1))
    return feat


def show_predict(img, shape, label=None):
    '''
    :param img: H*W*3 array
    :param label: 68*2 point labels
    :return: None
    '''
    if img is not None:
        plt.imshow(img)
        plt.axis('image')
    plt.hold(True)
    plt.plot(shape[::2], shape[1::2], 'r.', markersize=12)
    if label is not None:
        plt.plot(label[::2], label[1::2], 'g.', markersize=12)
    plt.axis('off')
    plt.show()


def vectorize(array):
    return array.reshape((array.size,))

if __name__ == '__main__':
    #maps = np.load('general_map.npy')
    #show(maps[6], True)
    #show(maps[35], True)

    train_label = 'dataset/300W_train_label.txt'
    #train_label = 'dataset/temp.txt'
    f = open(train_label)
    in_size = 224
    scale = 4
    std = 10
    maps = np.zeros([68, in_size / scale, in_size / scale], dtype=float)
    for l in f.readlines():
        label = l.split()[1:]
        label = [float(i) for i in label]
        maps += gauss_map(label, in_size, std, scale)
    f.close()
