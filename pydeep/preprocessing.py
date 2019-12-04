""" This module contains several classes for data preprocessing.

    :Implemented:
        - Standarizer
        - Principal Component Analysis (PCA)
        - Zero Phase Component Analysis (ZCA)
        - Independent Component Analysis (ICA)
        - Binarize data
        - Rescale data
        - Remove row means
        - Remove column means

    :Version:
        1.1.0

    :Date:
        04.04.2017

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
import pydeep.base.numpyextension as npext


def binarize_data(data):
    """ Converts data to binary values. \
        For data out of [a,b] a data point p will become zero if p < 0.5*(b-a) one otherwise.

    :param data: Data to be binarized.
    :type data: numpy array [num data point, data dimension]

    :return: Binarized data.
    :rtype: numpy array [num data point, data dimension]
    """
    return numx.array(numx.where(data < 0.5, 0, 1))


def rescale_data(data, new_min=0.0, new_max=1.0):
    """ Normalize the values of a matrix. e.g. [min,max] ->  [new_min,new_max]

    :param data: Data to be normalized.
    :type data: numpy array [num data point, data dimension]

    :param new_min: New min value.
    :type new_min: float

    :param new_max: Rescaled data
    :type new_max: float

    :return:
    :rtype: numpy array [num data point, data dimension]
    """
    datac = numx.array(data, numx.float64)
    minimum = numx.min(numx.min(datac, axis=1), axis=0)
    datac -= minimum
    maximum = numx.max(numx.max(datac, axis=1), axis=0)
    datac *= (new_max - new_min) / maximum
    datac += new_min
    return datac


def remove_rows_means(data, return_means=False):
    """ Remove the individual mean of each row.

    :param data: Data to be normalized
    :type data: numpy array [num data point, data dimension]

    :param return_means: If True returns also the means
    :type return_means: bool

    :return: Data without row means, row means (optional).
    :rtype: numpy array [num data point, data dimension], Means of the data (optional).
    """
    means = numx.mean(data, axis=1).reshape(data.shape[0], 1)
    output = data - means
    if return_means is True:
        return output, means
    else:
        return output


def remove_cols_means(data, return_means=False):
    """ Remove the individual mean of each column.

    :param data: Data to be normalized
    :type data: numpy array [num data point, data dimension]

    :param return_means: If True returns also the means
    :type return_means: bool

    :return: Data without column means, column means (optional).
    :rtype: numpy array [num data point, data dimension], Means of the data (optional).
    """
    means = numx.mean(data, axis=0).reshape(1, data.shape[1])
    output = data - means
    if return_means is True:
        return output, means
    else:
        return output


class STANDARIZER(object):
    """ Shifts the data having zero mean and scales it having unit variances along the axis.

    """

    def __init__(self, input_dim):
        """ Constructor.

        :param input_dim: Data dimensionality.
        :type input_dim: int
        """
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.mean = None
        self.covariance_matrix = None
        self.standard_deviation = None
        self.trained = False

    def train(self, data):
        """ Training the model (full batch).

        :param data: Data for training.
        :type data: numpy array [num data point, data dimension]
        """
        if self.input_dim != data.shape[1]:
            raise ValueError("Wrong data dimensionality.")
        # Center data and compute covariance matrix
        self.mean = numx.mean(data, axis=0).reshape(1, data.shape[1])
        self.covariance_matrix = numx.cov(data - self.mean, rowvar=0)
        self.standard_deviation = numx.std(data, axis=0).reshape(1, data.shape[1])
        self.trained = True

    def project(self, data):
        """ Projects the data to normalized space.

        :param data: Data to project.
        :type data: numpy array [num data point, data dimension]

        :return: Projected data.
        :rtype: numpy array [num data point, data dimension]
        """
        if self.input_dim != data.shape[1]:
            raise ValueError("Wrong data dimensionality.")
        if not self.trained:
            raise ValueError("Train model first!")
        return (data - self.mean) / self.standard_deviation

    def unproject(self, data):
        """ Projects the data back to the input space.

        :param data: Data to unproject.
        :type data: numpy array [num data point, data dimension]

        :return: Projected data.
        :rtype: numpy array [num data point, data dimension]
        """
        if self.input_dim != data.shape[1]:
            raise ValueError("Wrong data dimensionality.")
        if not self.trained:
            raise ValueError("Train model first!")
        return data * self.standard_deviation + self.mean


class PCA(STANDARIZER):
    """ Principle component analysis (PCA) using Singular Value Decomposition (SVD)
    """

    def __init__(self, input_dim, whiten=False):
        """ Constructor.

        :param input_dim: Data dimensionality.
        :type input_dim: int

        :param whiten: If true the projected data will be de-correlated in all directions.
        :type whiten: bool
        """
        super(PCA, self).__init__(input_dim)
        self.whiten = whiten
        self.eigen_values = None
        self.projection_matrix = None
        self.unprojection_matrix = None

    def train(self, data):
        """ Training the model (full batch).

        :param data: data for training.
        :type data: numpy array [num data point, data dimension]
        """
        super(PCA, self).train(data)
        # Compute Eigenvalue and Eigenvectors of Covariance matrix
        self.projection_matrix, self.eigen_values, _ = numx.linalg.svd(
            self.covariance_matrix)

        # Sort Eigenvalues and Eigenvectors by Eigenvalues in decreasing order
        index = numx.argsort(self.eigen_values)[::-1]
        self.eigen_values = self.eigen_values[index].reshape(1, index.shape[0])
        self.projection_matrix = self.projection_matrix[:, index]
        self.unprojection_matrix = self.projection_matrix.T

        # If true the projected data will be decorrelated in all directions
        if self.whiten is True:
            self.unprojection_matrix = (self.projection_matrix * numx.sqrt(self.eigen_values)).T
            self.projection_matrix = self.projection_matrix / numx.sqrt(self.eigen_values)

        self.trained = True

    def project(self, data, num_components=None):
        """ Projects the data to Eigenspace.

        :Info:
            projection_matrix has its projected vectors as its columns. i.e. if we project x by W into y where W is \
            the projection_matrix, then y = W.T * x

        :param data: Data to project.
        :type data: numpy array [num data point, data dimension]

        :param num_components:
        :type num_components: int or None

        :return: Projected data.
        :rtype: numpy array [num data point, data dimension]
        """
        if not self.trained:
            raise ValueError("Train model first!")
        n = self.output_dim
        if num_components is not None:
            n = num_components
        return numx.dot(data - self.mean, self.projection_matrix[:, 0:n])

    def unproject(self, data, num_components=None):
        """ Projects the data from Eigenspace to normal space.

        :param data: Data to be unprojected.
        :type data: numpy array [num data point, data dimension]

        :param num_components: Number of components to project.
        :type num_components: int

        :return: Unprojected data.
        :rtype: numpy array [num data point, num_components]
        """
        if not self.trained:
            raise ValueError("Train model first!")
        n = self.input_dim
        if num_components is not None:
            n = num_components
        return numx.dot(data, self.unprojection_matrix[0:data.shape[1], 0:n]) + self.mean[:, 0:n]


class ZCA(PCA):
    """ Principle component analysis (PCA) using Singular Value Decomposition (SVD).
    """

    def __init__(self, input_dim):
        """ Constructor.

        :param input_dim: Data dimensionality.
        :type input_dim: int
        """
        super(ZCA, self).__init__(input_dim, False)

    def train(self, data):
        """ Training the model (full batch).

        :param data: data for training.
        :type data: numpy array [num data point, data dimension]
        """
        super(ZCA, self).train(data)
        self.projection_matrix = numx.dot(self.projection_matrix / numx.sqrt(self.eigen_values),
                                          self.projection_matrix.T)
        self.unprojection_matrix = numx.dot(self.unprojection_matrix.T * numx.sqrt(self.eigen_values),
                                            self.unprojection_matrix)


class ICA(PCA):
    """ Independent Component Analysis using FastICA.
    """

    def __init__(self, input_dim):
        """ Constructor.

        :param input_dim: Data dimensionality.
        :type input_dim: int
        """
        super(ICA, self).__init__(input_dim, False)
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.trained = False

    def train(self,
              data,
              iterations=1000,
              convergence=0.0,
              status=False):
        """ Training the model (full batch).

        :param data: data for training.
        :type data: numpy array [num data point, data dimension]

        :param iterations: Number of iterations
        :type iterations: int

        :param convergence: If the angle (in degrees) between filters of two updates is less than the given value, \
                            training is terminated.
        :type convergence: double

        :param status: If true the progress is printed to the console.
        :type status: bool
        """
        if self.input_dim != data.shape[1]:
            raise ValueError("Wrong data dimensionality.")
        # Random init
        self.projection_matrix = numx.random.randn(data.shape[1],
                                                   data.shape[1])
        projection_matrix_old = numx.copy(self.projection_matrix)
        for epoch in range(0, iterations):
            # One iteration.
            # TODO: PendingDeprecationWarning: the matrix subclass is not the recommended
            # way to represent matrices or deal with linear algebra (see
            # https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). Please
            # adjust your code to use regular ndarray.
            hyptan = 1.0 - 2.0 / (numx.exp(2.0 * numx.dot(data, self.projection_matrix)) + 1.0)
            self.projection_matrix = (numx.dot(data.T, hyptan) / data.shape[0] - numx.array(numx.dot(numx.ones(
                (data.shape[1], 1)), numx.matrix(numx.mean(1.0 - hyptan ** 2.0, axis=0)))) * self.projection_matrix)
            tmp = numx.linalg.inv(numx.dot(self.projection_matrix.T, self.projection_matrix))

            ew, ev = numx.linalg.eig(tmp)
            self.projection_matrix = numx.dot(self.projection_matrix, numx.real(numx.dot(numx.dot(ev,
                                                                                                  numx.diag(ew) ** 0.5),
                                                                                         ev.T)))

            angle = numx.mean(
                numx.diagonal(npext.angle_between_vectors(projection_matrix_old.T, self.projection_matrix.T, True)))
            if angle < convergence or 180.0 - angle < convergence:
                break
            projection_matrix_old = numx.copy(self.projection_matrix)

            if status is True:
                import pydeep.misc.measuring as mea
                mea.print_progress(epoch, iterations, True)

        # Set results
        self.mean = numx.zeros((1, data.shape[1]))
        self.unprojection_matrix = self.projection_matrix.T

        self.trained = True

    def log_likelihood(self, data):
        """ Calculates the Log-Likelihood (LL) for the given data.

        :param data: data to calculate the Log-Likelihood for.
        :type data: numpy array [num data point, data dimension]

        :return: log-likelihood.
        :rtype: numpy array [num data point]
        """
        if not self.trained:
            raise ValueError("Train model first!")
        return numx.sum(numx.log(0.5 / (numx.cosh(numx.dot(self.unprojection_matrix, data.T)) ** 2.0)),
                        axis=0) + numx.log(numx.abs(numx.linalg.det(self.projection_matrix)))
