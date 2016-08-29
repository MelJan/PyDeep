''' This class contains methods to read and generate problem data.

    :Implemented:
        - Save/Load arbitrary objects.
        - Save/Load images.
        - Load MNIST.
        - Load CIFAR.
        - Load CALTECH.

    :Version:
        1.0

    :Date:
        29.08.2016

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2016 Jan Melchior

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
import cPickle
import os
import gzip
import numpy as numx
import scipy.io
import scipy.misc


def save_object(obj, path, info=True, compressed=True):
    ''' Saves an object to file.

    :Parameters:
        obj: object to be saved.
            -type: object

        path: Path and name of the file
             -type: string

        info: Prints statements if TRUE
             -type: bool

    '''
    if info == True:
        print '-> Saving File  ... ',
    try:
        if compressed:
            fp = gzip.open(path, 'wb')
            cPickle.dump(obj, fp)
            fp.close()
        else:
            file_path = open(path, 'w')
            cPickle.dump(obj, file_path)
        if info == True:
            print 'done!'
    except:
        print "-> File writing Error: "
        return None


def save_image(array, path, ext='bmp'):
    ''' Saves a matrix to a image file.

    :Parameters:
        matrix: Data to save
               -type: numpy array [width, height]

        path: Path and name of the directory to save the image at.
             -type: string

        ext: Extension for the image.
            -type: string

    '''
    scipy.misc.imsave(path + '.' + ext, array)


def load_object(path, info=True, compressed=True):
    ''' Loads an object from file.

    :Parameters:
      path: Path and name of the file
           -type: string

      info: Prints statements if TRUE
           -type: bool

    :Returns:
        Loaded object
       -type: object

    '''
    if not os.path.isfile(path):
        if info is True:
            print "-> File not existing: " + path
        return None
    else:
        if info is True:
            print '-> Loading File  ... ',
        try:
            if compressed is True:
                fp = gzip.open(path, 'rb')
                obj = cPickle.load(fp)
                fp.close()
                if info is True:
                    print 'done!'
                return obj
            else:
                file_path = open(path, 'r')
                obj = cPickle.load(file_path)
                if info is True:
                    print 'done!'
                return obj
        except:
            if info is True:
                print "-> File reading Error: "
            return None


def load_image(path):
    ''' Loads an image to a numpy array.

    :Parameters:
        path: Path and name of the directory to save the image at.
             -type: string

    :Returns:
        Image to load.
       -type: numpy array [width, height]

    '''
    return scipy.misc.imread(path)

def load_MNIST(path, binary=False):
    ''' Loads the MNIST digit data in binary [0,1] or real values [0,1]

    :Parameters:
        path: Path and name of the file to load.
             -type: string

        binary: If True returns binary images, real valued between [0,1]
                if False.
               -type: bool

    :Returns:
        MNIST dataset
       -type: list of numpy arrays

    '''
    if not os.path.isfile(path):
        print "-> File not existing: " + path
        return None
    else:
        print '-> loading data ... ',
        try:
            f = gzip.open(path, 'rb')
            print 'done!'
        except:
            print "-> File reading Error: "
            return None
        print '-> uncompress data ... ',
        try:
            train_set, valid_set, test_set = cPickle.load(f)
            train_lab = train_set[1]
            valid_lab = valid_set[1]
            test_lab = test_set[1]
            f.close()
            print 'done!'
        except:
            print "-> File reading Error: "
            return None
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
        return train_set, train_lab, valid_set, valid_lab, test_set, test_lab


def load_CALTECH(path):
    ''' Loads the Caltech dataset.

    :Parameters:
        path: Path and name of the file to load.
             -type: string

    :Returns:
        CalTech dataset
       -type: list of numpy arrays

    '''
    if not os.path.isfile(path):
        print "-> File not existing: " + path
        return None
    else:
        print '-> loading data ... ',
        try:
            train_set = scipy.io.loadmat(path)["train_data"]
            test_set = scipy.io.loadmat(path)["test_data"]
            valid_set = scipy.io.loadmat(path)["val_data"]

            train_lab = scipy.io.loadmat(path)["train_labels"]
            test_lab = scipy.io.loadmat(path)["test_labels"]
            valid_lab = scipy.io.loadmat(path)["val_labels"]
            print 'done!'
        except:
            print "-> File reading Error: "
            return None
        return train_set, train_lab, valid_set, valid_lab, test_set, test_lab


def load_CIFAR(path, grayscale=True):
    ''' Loads the MNIST digit data in binary [0,1] or real values [0,1]

    :Parameters:
        path: Path and name of the file to load.
             -type: string

        grayscale: If true converts the data to grayscale.
                  -type: bool

    :Returns:
        CIFAR data and labels.
       -type: list of numpy arrays ([# samples, 1024],[# samples])

    '''
    try:
        fo = open(path, 'rb')
        dictionary = cPickle.load(fo)
        fo.close()
    except:
        print 'file loading error!'
    label = dictionary['labels']
    data = dictionary['data']
    if (grayscale):
        data = (0.3 * data[:, 0:1024] + 0.59 * data[:, 1024:2048]
                + 0.11 * data[:, 2048:3072])
    return numx.array(data), numx.array(label)

