""" This module contains the weight layer for DBMs.

    :Implemented:
        - DBM model

    :Version:
        1.0.0

    :Date:
        26.05.2019

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2019 Jan Melchior

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
import pydeep.base.numpyextension as numxExt
from scipy.signal import convolve2d


class Weight_layer(object):
    ''' This class implements a weight layer that connects one unit
        layer to another.

    '''

    @classmethod
    def generate_2D_connection_matrix(cls,
                                      input_x_dim,
                                      input_y_dim,
                                      field_x_dim,
                                      field_y_dim,
                                      overlap_x_dim,
                                      overlap_y_dim,
                                      wrap_around=True):
        ''' This function constructs a connection matrix, which can be
            used to force the weights to have local receptive fields.

        :Parameters:
            input_x_dim:    Input dimension.
                           -type: int

            input_y_dim     Output dimension.
                           -type: int

            field_x_dim:    Size of the receptive field in dimension x.
                           -type: int

            field_y_dim:    Size of the receptive field in dimension y.
                           -type: int

            overlap_x_dim:  Overlap of the receptive fields in dimension x.
                           -type: int

            overlap_y_dim:  Overlap of the receptive fields in dimension y.
                           -type: int

            wrap_around:    If true teh overlap has warp around in both dimensions.
                           -type: bool

        :Returns:
            Connection matrix.
           -type: numpy arrays [input dim, output dim]

        '''
        if field_x_dim > input_x_dim:
            raise NotImplementedError("field_x_dim > input_x_dim is invalid!")
        if field_y_dim > input_y_dim:
            raise NotImplementedError("field_y_dim > input_y_dim is invalid!")
        if overlap_x_dim >= field_x_dim:
            raise NotImplementedError("overlap_x_dim >= field_x_dim is invalid!")
        if overlap_y_dim >= field_y_dim:
            raise NotImplementedError("overlap_y_dim >= field_y_dim is invalid!")

        matrix = None
        start_x = 0
        start_y = 0
        end_x = input_x_dim
        end_y = input_y_dim
        if wrap_around == False:
            end_x -= field_x_dim - 1
            end_y -= field_y_dim - 1
        step_x = field_x_dim - overlap_x_dim
        step_y = field_y_dim - overlap_y_dim

        for x in range(start_x, end_x, step_x):
            for y in range(start_y, end_y, step_y):
                column = numx.zeros((input_x_dim, input_y_dim))
                for i in range(x, x + field_x_dim, 1):
                    for j in range(y, y + field_y_dim, 1):
                        column[i % input_x_dim, j % input_y_dim] = 1.0
                column = column.reshape((input_x_dim * input_y_dim))
                if matrix == None:
                    matrix = column
                else:
                    matrix = numx.vstack((matrix, column))
        return matrix.T

    def __init__(self,
                 input_dim,
                 output_dim,
                 initial_weights='AUTO',
                 connections=None,
                 dtype=numx.float64):
        ''' This function initializes the weight layer.

        :Parameters:
            input_dim:          Input dimension.
                               -type: int

            output_dim          Output dimension.
                               -type: int

            initial_weights:    Initial weights.
                                'AUTO' and a scalar are random init.
                               -type: 'AUTO', scalar or
                                      numpy array [input dim, output_dim]

            connections:        Connection matrix containing 0 and 1 entries, where
                                0 connections disable the corresponding weight.
                                generate_2D_connection_matrix() can be used to contruct a matrix.
                               -type: numpy array [input dim, output_dim] or None

            dtype:               Used data type i.e. numpy.float64
                                -type: numpy.float32 or numpy.float64 or
                                       numpy.longdouble

        '''
        # Set internal datatype
        self.dtype = dtype

        # Set input and output dimension
        self.input_dim = input_dim
        self.output_dim = output_dim

        # AUTO   -> Small random values out of
        #           +-4*numx.sqrt(6/(self.input_dim+self.output_dim)
        # Scalar -> Small Gaussian distributed random values with std_dev
        #           initial_weights
        # Array  -> The corresponding values are used
        if initial_weights is 'AUTO':
            self.weights = numx.array((2.0 * numx.random.rand(self.input_dim,
                                                              self.output_dim) - 1.0)
                                      * (4.0 * numx.sqrt(6.0 / (self.input_dim
                                                                + self.output_dim)))
                                      , dtype=dtype)
        else:
            if (numx.isscalar(initial_weights)):
                self.weights = numx.array(numx.random.randn(self.input_dim,
                                                            self.output_dim)
                                          * initial_weights, dtype=dtype)
            else:
                if initial_weights.shape[0] != self.input_dim or initial_weights.shape[1] != self.output_dim or len(
                        initial_weights.shape) != 2:
                    raise NotImplementedError(
                        "Initial weight matrix must habe the shape defined by input_dim and output_dim!")
                else:
                    self.weights = numx.array(initial_weights, dtype=dtype)

        # Set connection matrix
        self.connections = None
        if connections != None:
            if connections.shape[0] != self.input_dim or connections.shape[1] != self.output_dim or len(
                    connections.shape) != 2:
                raise NotImplementedError("Connections matrix must have the shape defined by input_dim and output_dim!")
            else:
                self.connections = numx.array(connections, dtype=dtype)
                self.weights *= self.connections

    def propagate_up(self, bottom_up_states):
        ''' This function propagates the input to the next layer.

        :Parameters:
            bottom_up_states: States of the unit layer below.
                             -type: numpy array [batch_size, input dim]

        :Returns:
            Top down states.
           -type: numpy arrays [batch_size, output dim]

        '''
        return numx.dot(bottom_up_states, self.weights)

    def propagate_down(self, top_down_states):
        ''' This function propagates the output to the previous layer.

        :Parameters:
            top_down_states: States of the unit layer above.
                            -type: numpy array [batch_size, output dim]

        :Returns:
            Bottom up states.
           -type: numpy arrays [batch_size, input dim]

        '''
        return numx.dot(top_down_states, self.weights.T)

    def calculate_weight_gradients(self,
                                   bottom_up_pos,
                                   top_down_pos,
                                   bottom_up_neg,
                                   top_down_neg,
                                   offset_bottom_up,
                                   offset_top_down):
        ''' This function calculates the average gradient from input
            and output samples of positive and negative sampling phase.

        :Parameters:
            bottom_up_pos:    Input samples from the positive phase.
                             -type: numpy array [batch_size, input dim]

            top_down_pos:     Output samples from the positive phase.
                             -type: numpy array [batch_size, output dim]

            bottom_up_neg:    Input samples from the negative phase.
                             -type: numpy array [batch_size, input dim]

            top_down_neg:     Output samples from the negative phase.
                             -type: numpy array [batch_size, output dim]

            offset_bottom_up: Offset for the input data.
                             -type: numpy array [1, input dim]

            offset_top_down:  Offset for the output data.
                             -type: numpy array [1, output dim]

        :Returns:
            Weight gradient.
           -type: numpy arrays [input dim, output dim]

        '''
        pos = numx.dot((bottom_up_pos - offset_bottom_up).T, (top_down_pos - offset_top_down))
        neg = numx.dot((bottom_up_neg - offset_bottom_up).T, (top_down_neg - offset_top_down))
        grad = (pos - neg) / bottom_up_pos.shape[0]
        if self.connections != None:
            grad *= self.connections
        return grad

    def update_weights(self, weight_updates, restriction, restriction_typ):
        ''' This function updates the weight parameters.

        :Parameters:
            weight_updates:     Update for the weight parameter.
                               -type: numpy array [input dim, output dim]

            restriction:        If a scalar is given the weights will be
                                forced after an update not to exceed this value.
                                restriction_typ controls how the values are
                                restricted.
                               -type: scalar or None

            restriction_typ:    If a value for the restriction is given, this parameter
                                determines the restriction typ. 'Cols', 'Rows', 'Mat'
                                or 'Abs' to restricted the colums, rows or matrix norm
                                or the matrix absolut values.
                               -type: string

        '''

        # Restricts the gradient
        if numx.isscalar(restriction):
            if restriction > 0:
                if restriction_typ is 'Cols':
                    weight_updates = numxExt.restrict_norms(
                        weight_updates,
                        restriction, 0)
                if restriction_typ is 'Rows':
                    weight_updates = numxExt.restrict_norms(
                        weight_updates,
                        restriction, 1)
                if restriction_typ is 'Mat':
                    weight_updates = numxExt.restrict_norms(
                        weight_updates,
                        restriction, None)

                if restriction_typ is 'Val':
                    numx.clip(weight_updates, -restriction, restriction, weight_updates)

        # Update weights
        self.weights += weight_updates


class Convolving_weight_layer(Weight_layer):
    ''' This class implements a weight layer that connects one unit
        layer to another with convolutional weights.

    '''

    @classmethod
    def construct_gauss_filter(cls, width, height, variance=1.0):
        ''' This function constructs a 2D-Gauss filter.

        :Parameters:
            width:     Filter width.
                      -type: int

            height     Filter Height.
                      -type: int

            variance   Variance of the Gaussian
                      -type: scalar


        :Returns:
            Convolved matrix with the same shape as matrix.
           -type: 2D numpy arrays

        '''
        if width % 2 == 0:
            print("Width needs to be odd!")
            pass
        if height % 2 == 0:
            print("Height needs to be odd!")
            pass
        lowerW = (width - 1) / 2
        lowerH = (height - 1) / 2
        mat = numx.zeros((width, height))
        for x in range(0, width):
            for y in range(0, height):
                mat[x, y] = (numx.exp(-0.5 * ((x - lowerW) ** 2 + (y - lowerH) ** 2) / variance)) / (
                        2 * numx.pi * variance)
        return mat / numx.sum(mat)

    def _convolve(self, matrix, mask):
        ''' This function performs a 2D convolution on every column of a given matrix.

        :Parameters:
            matrix: 2D-input matrix.
                   -type: small 2D numpy array

            mask    2D-mask matrix.
                   -type: small 2D numpy array


        :Returns:
            Convolved matrix with the same shape as matrix.
           -type: 2D numpy arrays

        '''
        s = numx.int32(numx.sqrt(matrix.shape[1]))
        result = numx.empty(matrix.shape)
        for i in range(matrix.shape[0]):
            temp = matrix[i].reshape(s, s)
            result[i] = convolve2d(temp, mask, mode='same', boundary='wrap').reshape(matrix.shape[1])
        return result

    def __init__(self,
                 input_dim,
                 output_dim,
                 mask,
                 initial_weights='AUTO',
                 connections=None,
                 dtype=numx.float64):
        ''' This function initializes the convolutional weight layer.

        :Parameters:
            input_dim:          Input dimension.
                               -type: int

            output_dim          Output dimension.
                               -type: int

            mask                Convolution mask.
                                construct_gauss_filter can be used for example.
                               -type: small numpy array

            initial_weights:    Initial weights.
                                'AUTO' and a scalar are random init.
                               -type: 'AUTO', scalar or
                                      numpy array [input dim, output_dim]

            connections:        Connection matrix containing 0 and 1 entries, where
                                0 connections disable the corresponding weight.
                                generate_2D_connection_matrix() can be used to contruct a matrix.
                               -type: numpy array [input dim, output_dim] or None

            dtype:               Used data type i.e. numpy.float64
                                -type: numpy.float32 or numpy.float64 or
                                       numpy.longdouble

        '''
        super(Convolving_weight_layer,
              self).__init__(input_dim=input_dim,
                             output_dim=output_dim,
                             initial_weights=initial_weights,
                             connections=connections,
                             dtype=dtype)
        self.mask = mask
        self.orginalW = numx.copy(self.weights)
        self.weights = self._convolve(self.orginalW, self.mask)

    def calculate_weight_gradients(self,
                                   bottom_up_pos,
                                   top_down_pos,
                                   bottom_up_neg,
                                   top_down_neg,
                                   offset_bottom_up,
                                   offset_top_down):
        ''' This function calculates the average gradient from input
            and output samples of positive and negative sampling phase.

        :Parameters:
            bottom_up_pos:    Input samples from the positive phase.
                             -type: numpy array [batch_size, input dim]

            top_down_pos:     Output samples from the positive phase.
                             -type: numpy array [batch_size, output dim]

            bottom_up_neg:    Input samples from the negative phase.
                             -type: numpy array [batch_size, input dim]

            top_down_neg:     Output samples from the negative phase.
                             -type: numpy array [batch_size, output dim]

            offset_bottom_up: Offset for the input data.
                             -type: numpy array [1, input dim]

            offset_top_down:  Offset for the output data.
                             -type: numpy array [1, output dim]

        :Returns:
            Weight gradient.
           -type: numpy arrays [input dim, output dim]

        '''
        pos = numx.dot((bottom_up_pos - offset_bottom_up).T,
                       (self._convolve(top_down_pos - offset_top_down, self.mask)))
        neg = numx.dot((bottom_up_neg - offset_bottom_up).T,
                       (self._convolve(top_down_neg - offset_top_down, self.mask)))
        grad = (pos - neg) / bottom_up_pos.shape[0]
        if self.connections != None:
            grad *= self.connections
        return grad

    def update_weights(self, weight_updates, restriction, restriction_typ):
        ''' This function updates the weight parameters.

        :Parameters:
            weight_updates:     Update for the weight parameter.
                               -type: numpy array [input dim, output dim]

            restriction:        If a scalar is given the weights will be
                                forced after an update not to exceed this value.
                                restriction_typ controls how the values are
                                restricted.
                               -type: scalar or None

            restriction_typ:    If a value for the restriction is given, this parameter
                                determines the restriction typ. 'Cols', 'Rows', 'Mat'
                                or 'Abs' to restricted the colums, rows or matrix norm
                                or the matrix absolut values.
                               -type: string

        '''
        # Update weights
        self.orginalW += weight_updates

        # Restricts the gradient
        if numx.isscalar(restriction):
            if restriction > 0:
                if restriction_typ is 'Cols':
                    self.orginalW = numxExt.restrict_norms(
                        self.orginalW,
                        restriction, 0)
                if restriction_typ is 'Rows':
                    self.orginalW = numxExt.restrict_norms(
                        self.orginalW,
                        restriction, 1)
                if restriction_typ is 'Mat':
                    self.orginalW = numxExt.restrict_norms(
                        self.orginalW,
                        restriction, None)

                if restriction_typ is 'Val':
                    numx.clip(self.orginalW, -restriction, restriction, self.orginalW)

        self.weights = self._convolve(self.orginalW, self.mask)
