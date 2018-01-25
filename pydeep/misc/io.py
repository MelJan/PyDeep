""" This class contains methods to read and write data.

    :Implemented:
        - Save/Load arbitrary objects.
        - Save/Load images.
        - Load MNIST.
        - Load CIFAR.
        - Load Caltech.

    :Version:
        1.1.0

    :Date:
        19.03.2017

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2017 Jan Melchior

        This file is part of the Python library PyDeep.

        PyDeep is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import cPickle
import os
import gzip
import numpy as numx
import scipy.io
import scipy.misc
import requests
import pydeep.misc.measuring as mea


def save_object(obj, path, info=True, compressed=True):
    """ Saves an object to file.

    :param obj: object to be saved.
    :type obj: object

    :param path: Path and name of the file
    :type path: string

    :param info: Prints statements if True
    :type info: bool

    :param compressed: Object will be compressed before storage.
    :type compressed: bool

    :return:
    :rtype:
    """
    if info is True:
        print('-> Saving File  ... ')
    try:
        if compressed:
            fp = gzip.open(path, 'wb')
            cPickle.dump(obj, fp)
            fp.close()
        else:
            file_path = open(path, 'w')
            cPickle.dump(obj, file_path)
        if info is True:
            print('-> done!')
    except:
        raise Exception('-> File writing Error: ')


def save_image(array, path, ext='bmp'):
    """ Saves a numpy array to an image file.

    :param array: Data to save
    :type array: numpy array [width, height]

    :param path: Path and name of the directory to save the image at.
    :type path: string

    :param ext: Extension for the image.
    :type ext: string
    """
    scipy.misc.imsave(path + '.' + ext, array)


def load_object(path, info=True, compressed=True):
    """ Loads an object from file.

    :param path: Path and name of the file
    :type path: string

    :param info: If True, prints status information.
    :type info: bool

    :param compressed:
    :type compressed: bool

    :return: Loaded object
    :rtype: object
    """
    if not os.path.isfile(path):
        if info is True:
            print('-> File not existing: ' + path)
        return None
    else:
        if info is True:
            print('-> Loading File  ... ')
        try:
            if compressed is True:
                fp = gzip.open(path, 'rb')
                obj = cPickle.load(fp)
                fp.close()
                if info is True:
                    print('-> done!')
                return obj
            else:
                file_path = open(path, 'r')
                obj = cPickle.load(file_path)
                if info is True:
                    print('-> done!')
                return obj
        except:
            raise Exception('-> File reading Error: ')


def load_image(path, grayscale=False):
    """ Loads an image to numpy array.

    :param path: Path and name of the directory to save the image at.
    :type path: string

    :param grayscale: If true image is converted to gray scale.
    :type grayscale: bool

    :return: Loaded image.
    :rtype: numpy array [width, height]
    """
    return scipy.misc.imread(path, flatten=grayscale)


def download_file(url, path, buffer_size=1024 ** 2):
    """ Downloads an saves a dataset from a given url.

    :param url: URL including filename (e.g. www.testpage.com/file1.zip)
    :type url: string

    :param path: Path the dataset should be stored including filename (e.g. /home/file1.zip).
    :type path: string, None

    :param buffer_size: Size of the streaming buffer in bytes.
    :type buffer_size: int
    """
    print('-> Downloading ' + url + ' to ' + path)
    with open(path, 'wb') as handle:
        url_stream = requests.get(url, stream=True)
        file_size = numx.float64(url_stream.headers.get('content-length'))
        num_steps = numx.int32(file_size / buffer_size)
        if not url_stream.ok:
            raise Exception("-> Connection lost")
        i = 0
        for block in url_stream.iter_content(buffer_size):
            handle.write(block)
            mea.print_progress(i, num_steps, True)
            i += 1


def load_mnist(path, binary=False):
    """ Loads the MNIST digit data in binary [0,1] or real values [0,1].

    :param path: Path and name of the file to load.
    :type path: string

    :param binary: If True returns binary images, real valued between [0,1] if False.
    :type binary: bool

    :return: MNIST dataset [train_set, train_lab, valid_set, valid_lab, test_set, test_lab]
    :rtype: list of numpy arrays
    """
    if not os.path.isfile(path):
        print('-> File not existing: ' + path)
        try:
            download_file('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz', path)
        except:
            raise Exception('-> Download failed, make sure you have internet connection!')
    print('-> loading data ... ')
    try:
        f = gzip.open(path, 'rb')
        print('-> done!')
    except:
        raise Exception('-> File reading Error: ')
    print('-> uncompress data ... ')
    try:
        train_set, valid_set, test_set = cPickle.load(f)
        train_lab = train_set[1]
        valid_lab = valid_set[1]
        test_lab = test_set[1]
        f.close()
        print('-> done!')
    except:
        raise Exception('-> File reading Error: ')
    if binary:
        train_set = numx.where(train_set[0] < 0.5, 0, 1)
        valid_set = numx.where(valid_set[0] < 0.5, 0, 1)
        test_set = numx.where(test_set[0] < 0.5, 0, 1)
        train_set = numx.array(train_set, dtype=numx.int)
        valid_set = numx.array(valid_set, dtype=numx.int)
        test_set = numx.array(test_set, dtype=numx.int)
    else:
        train_set = numx.array(train_set[0], dtype=numx.double)
        valid_set = numx.array(valid_set[0], dtype=numx.double)
        test_set = numx.array(test_set[0], dtype=numx.double)
    train_lab = numx.array(train_lab, dtype=numx.int)
    valid_lab = numx.array(valid_lab, dtype=numx.int)
    test_lab = numx.array(test_lab, dtype=numx.int)
    return train_set, train_lab, valid_set, valid_lab, test_set, test_lab


def load_caltech(path):
    """ Loads the Caltech dataset.

    :param path: Path and name of the file to load.
    :type path: string

    :return: CAltech dataset [train_set, train_lab, valid_set, valid_lab, test_set, test_lab]
    :rtype: list of numpy arrays
    """
    if not os.path.isfile(path):
        print('-> File not existing: ' + path)
        try:
            download_file('http://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat', path,
                          buffer_size=1024 * 128)
        except:
            raise Exception('-> Download failed, make sure you have internet connection!')
    print('-> loading data ... ')
    try:
        train_set = scipy.io.loadmat(path)['train_data']
        test_set = scipy.io.loadmat(path)['test_data']
        valid_set = scipy.io.loadmat(path)['val_data']

        train_lab = scipy.io.loadmat(path)['train_labels']
        test_lab = scipy.io.loadmat(path)['test_labels']
        valid_lab = scipy.io.loadmat(path)['val_labels']
        print('-> done!')
    except:
        raise Exception('-> File reading Error: ')
    train_set = numx.array(train_set, dtype=numx.int)
    valid_set = numx.array(valid_set, dtype=numx.int)
    test_set = numx.array(test_set, dtype=numx.int)
    train_lab = numx.array(train_lab, dtype=numx.int)
    valid_lab = numx.array(valid_lab, dtype=numx.int)
    test_lab = numx.array(test_lab, dtype=numx.int)
    return train_set, train_lab, valid_set, valid_lab, test_set, test_lab


def load_cifar(path, grayscale=True):
    """ Loads the CIFAR dataset in real values [0,1]

    :param path: Path and name of the file to load.
    :type path: string

    :param grayscale: If true converts the data to grayscale.
    :type grayscale: bool

    :return:  CIFAR data and labels.
    :rtype: list of numpy arrays ([# samples, 1024],[# samples])
    """
    import tarfile
    import cPickle

    import os
    if not os.path.isfile(path):
        print('-> File not existing: ' + path)
        try:
            download_file('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', path,
                          buffer_size=10 * 1024 ** 2)
        except:
            raise Exception('Download failed, make sure you have internet connection!')
    print('-> Extracting ...')
    try:
        tar = tarfile.open(path, 'r:gz')
        batch_test = cPickle.load(tar.extractfile(tar.getmembers()[3]))  # test
        print('-> test data extracted')
        batch_valid = cPickle.load(tar.extractfile(tar.getmembers()[7]))  # 5
        print('-> validation data extracted')
        batch_1 = cPickle.load(tar.extractfile(tar.getmembers()[8]))  # 1
        batch_2 = cPickle.load(tar.extractfile(tar.getmembers()[6]))  # 2
        batch_3 = cPickle.load(tar.extractfile(tar.getmembers()[4]))  # 3
        batch_4 = cPickle.load(tar.extractfile(tar.getmembers()[1]))  # 4
        print('-> training data extracted')

        train_set = numx.vstack((batch_1['data'], batch_2['data'], batch_3['data'], batch_4['data']))
        train_lab = numx.hstack((batch_1['labels'], batch_2['labels'], batch_3['labels'], batch_4['labels']))
        valid_set = batch_valid['data']
        valid_lab = batch_valid['labels']
        test_set = batch_test['data']
        test_lab = batch_test['labels']
    except:
        raise Exception('-> File reading Error, failed to uncompress data. ')
    if grayscale:
        train_set = (0.3 * train_set[:, 0:1024] + 0.59 * train_set[:, 1024:2048] + 0.11 * train_set[:, 2048:3072])
        valid_set = (0.3 * valid_set[:, 0:1024] + 0.59 * valid_set[:, 1024:2048] + 0.11 * valid_set[:, 2048:3072])
        test_set = (0.3 * test_set[:, 0:1024] + 0.59 * test_set[:, 1024:2048] + 0.11 * test_set[:, 2048:3072])
    train_set = numx.array(train_set, dtype=numx.double)
    valid_set = numx.array(valid_set, dtype=numx.double)
    test_set = numx.array(test_set, dtype=numx.double)
    train_lab = numx.array(train_lab, dtype=numx.int)
    valid_lab = numx.array(valid_lab, dtype=numx.int)
    test_lab = numx.array(test_lab, dtype=numx.int)
    return train_set, train_lab, valid_set, valid_lab, test_set, test_lab


def load_natural_image_patches(path):
    """ Loads the natural image patches used in the publication 'Gaussian-binary restricted Boltzmann machines for \
        modeling natural image statistics'.
         .. seealso:: http://journals.plos.org/plosone/article/authors?id=10.1371/journal.pone.0171015

    :param path: Path and name of the file to load.
    :type path: string

    :return: Natural image dataset
    :rtype: numpy array
    """
    if not os.path.isfile(path):
        print('-> File not existing: ' + path)
        try:
            download_file('https://zenodo.org/record/167823/files/NaturalImage.mat', path, buffer_size=10 * 1024 ** 2)
        except:
            raise Exception('Download failed, make sure you have internet connection!')
    print('-> loading data ... ')
    try:
        # https://zenodo.org/record/167823/files/NaturalImage.mat
        data = scipy.io.loadmat(path)['rawImages'].T
        print('-> done!')
    except:
        raise Exception('-> File reading Error: ')
    data = numx.array(data, dtype=numx.double)
    return data

def load_olivetti_faces(path, correct_orientation=True):
    """ Loads the Olivetti face dataset 400 images, size 64x64
    :param path: Path and name of the file to load.
    :type path: string
    :param correct_orientation: Corrects the orientation of the images.
    :type correct_orientation: bool
    :return: Olivetti face dataset
    :rtype: numpy array
    """
    if not os.path.isfile(path):
        print('-> File not existing: ' + path)
        try:
            download_file('http://www.cs.nyu.edu/~roweis/data/olivettifaces.mat', path, buffer_size=1 * 1024 ** 2)
        except:
            try:
                download_file('https://github.com/probml/pmtk3/tree/master/bigData/facesOlivetti/facesOlivetti.mat',
                              path, buffer_size=1 * 1024 ** 2)
            except:
                raise Exception('Download failed, make sure you have internet connection!')
    print('-> loading data ... ')
    try:
        data = scipy.io.loadmat(path)['faces'].T
        if correct_orientation:
            import pydeep.base.numpyextension as npext
            for i in range(data.shape[0]):
                data[i] = npext.rotate(data[i].reshape(64,64),270).reshape(64*64)
            print('-> orientation corrected!')
        print('-> done!')
    except:
        raise Exception('-> File reading Error: ')
    data = numx.array(data, dtype=numx.double)
    return data