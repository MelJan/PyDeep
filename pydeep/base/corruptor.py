""" This module provides implementations for corrupting the training data. 

    :Implemented:
        - Identity
        - Sampling Binary
        - BinaryNoise
        - Additive Gauss Noise
        - Multiplicative Gauss Noise
        - Dropout
        - Random Permutation
        - KeepKWinner
        - KWinnerTakesAll

    :Info: 
        http://ufldl.stanford.edu/wiki/index.php/Sparse_Coding:_Autoencoder_Interpretation
   
    :Version:
        1.1.0

    :Date:
        13.03.2017

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
import numpy as numx


class Identity(object):
    """ Dummy corruptor object.
    """

    @classmethod
    def corrupt(cls, data):
        """ The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        return data


class AdditiveGaussNoise(object):
    """ An object that corrupts data by adding Gauss noise.
    """

    def __init__(self, mean, std):
        """ The function corrupts the data.

        :param mean: Constant the data is shifted
        :type mean: float

        :param std: Standard deviation Added to the data.
        :type std: float
        """
        self.mean = mean
        self.std = std

    def corrupt(self, data):
        """ The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        return data + self.mean + numx.random.standard_normal(data.shape) * self.std


class MultiGaussNoise(object):
    """ An object that corrupts data by multiplying Gauss noise.
    """

    def __init__(self, mean, std):
        """ Corruptor contructor.

        :param mean: Constant the data is shifted
        :type mean: float

        :param std: Standard deviation Added to the data.
        :type std: float
        """
        self.mean = mean
        self.std = std

    def corrupt(self, data):
        """ The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        return data * (self.mean + numx.random.standard_normal(data.shape) * self.std)


class SamplingBinary(object):
    """ Sample binary states (zero out) corruption.
    """

    @classmethod
    def corrupt(cls, data):
        """ The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        return data > numx.random.random(data.shape)

    
class BinaryNoise(object):
    """ Binary Noise.
    """

    def __init__(self, percentage):
        """ Corruptor contructor.

        :param percentage: Percent of random chosen pixel/states.
        :type percentage: float [0,1]

        :param std: Standard deviation Added to the data.
        """
        self.percentage = percentage

    def corrupt(self, data):
        """ The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        return numx.abs(data - numx.random.binomial(1, self.percentage, data.shape))

    
class Dropout(object):
    """ Dropout (zero out) corruption.
    """

    def __init__(self, dropout_percentage=0.2):
        """ Corruptor contructor.

        :param dropout_percentage: Dropout percentage
        :type dropout_percentage: float
        """
        self.dropout_percentage = dropout_percentage

    def corrupt(self, data):
        """ The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        return data * numx.random.binomial(1, 1.0 - self.dropout_percentage, data.shape) / (
            1.0 - self.dropout_percentage)


class RandomPermutation(object):
    """ RandomPermutation corruption, a fix number of units change their activation values.
    """

    def __init__(self, permutation_percentage=0.2):
        """ Corruptor contructor.

        :param permutation_percentage: permutation_percentage: Percentage of states to permute
        :type permutation_percentage: float
        """
        self.permutation_percentage = permutation_percentage

    def corrupt(self, data):
        """ The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        result = numx.copy(data)
        num_switches = numx.int32(data.shape[1] * self.permutation_percentage * 0.5)
        for d in range(data.shape[0]):
            # Proof of concept
            # setA = numx.random.randint(0, pattern.shape[1], num_states_to_change / 2)
            # setB = numx.random.randint(0, pattern.shape[1], num_states_to_change / 2)
            # result[d][setA] = pattern[d][setB]
            # result[d][setB] = pattern[d][setA]
            tempset = numx.random.permutation(numx.arange(data.shape[1]))
            result[d][tempset[0:num_switches]] = data[d][tempset[num_switches:2 * num_switches]]
            result[d][tempset[num_switches:2 * num_switches]] = data[d][tempset[0:num_switches]]
        return result


class KeepKWinner(object):
    """ Implements K Winner stay. Keep the k max values and set the rest to 0.
    """

    def __init__(self, k=10, axis=0):
        """ Corruptor contructor.

        :param k: Keep the k max values and set the rest to 0.
        :type k: int

        :param axis: Axis =0 across min batch, axis = 1 across hidden units
        :type axis: int
        """
        self.k = k
        self.axis = axis

    def corrupt(self, data):
        """ The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        data = data
        if self.axis == 0:
            return data * (data >= numx.atleast_2d(numx.sort(data, axis=self.axis)[-self.k, :]))
        else:
            return data * (data >= numx.atleast_2d(numx.sort(data, axis=self.axis)[:, -self.k]).T)


class KWinnerTakesAll(object):
    """ Implements K Winner takes all. Keep the k max values and set the rest to 0.
    """

    def __init__(self, k=10, axis=0):
        """ Corruptor constructor.

        :param k: Keep the k max values and set the rest to 0.
        :type k: int

        :param axis: Axis =0 across min batch, axis = 1 across hidden units
        :type axis: int
        """
        self.k = k
        self.axis = axis

    def corrupt(self, data):
        """ The function corrupts the data.

        :param data: Input of the layer.
        :type data: numpy array [num samples, layer dim]

        :return: Corrupted data.
        :rtype: numpy array [num samples, layer dim]
        """
        data = data
        if self.axis == 0:
            return 1.0 * (data >= numx.atleast_2d(numx.sort(data, axis=self.axis)[-self.k, :]))
        else:
            return 1.0 * (data >= numx.atleast_2d(numx.sort(data, axis=self.axis)[:, -self.k]).T)
