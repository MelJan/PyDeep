''' This module provides different math functions that extend the numpy library.
              
    :Implemented:
        - log_sum_exp
        - log_diff_exp
        - get_norms
        - restrict_norms
        - resize_norms
        - angle_between_vectors
        - generate_binary_code
        
    :Version:
        1.0

    :Date:
        10.08.2016

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
import numpy as numx
from scipy.ndimage.interpolation import rotate

def log_sum_exp(x, axis=0):
    """ Calculates the logarithm of the sum of e to the power of input 'x'. The method tries to avoid
        overflows by using the relationship: log(sum(exp(x))) = alpha + log(sum(exp(x-alpha))).
        
    :Parameter:
        x:    data.
             -type: float or numpy array 
          
        axis: Sums along the given axis.
             -type: int
        
    :Return:
        Logarithm of the sum of exp of x. 
       -type: float or numpy array.
        
    """
    alpha = x.max(axis) - numx.log(numx.finfo(numx.float64).max)/2.0
    if axis == 1:
        return numx.squeeze(alpha + numx.log(
                                             numx.sum(
                                                      numx.exp(x.T - alpha)
                                                      , axis=0)))
    else:
        return numx.squeeze(alpha + numx.log(
                                             numx.sum(
                                                      numx.exp(x - alpha)
                                                      , axis=0)))

def log_diff_exp(x, axis=0):
    """ Calculates the logarithm of the diffs of e to the power of input 'x'. The method tries to avoid
        overflows by using the relationship: log(diff(exp(x))) = alpha + log(diff(exp(x-alpha))).
        
    :Parameter:
        x:    data.
             -type: float or numpy array 
          
        axis: Sums along the given axis.
             -type: int
        
    :Return:
        Logarithm of the sum of exp of x. 
       -type: float or numpy array.
        
    """
    alpha = x.max(axis) - numx.log(numx.finfo(numx.float64).max)/2.0
    if axis == 1:
        return numx.squeeze(alpha + numx.log(
                                             numx.diff(
                                                       numx.exp(x.T - alpha)
                                                       , n=1, axis=0)))
    else:
        return numx.squeeze(alpha + numx.log(
                                             numx.diff(
                                                       numx.exp(x - alpha)
                                                       , n=1, axis=0)))

def get_norms(matrix, axis=0):
    """ Computes the norms of the matrix along a given axis.
    
    :Parameters:
        matrix:   Matrix to get the norm of.
                 -type: numpy array [num rows, num columns]

        axis:     Axis along the norm should be calculated.
                  0 = rows, 1 = cols, None = Matrix norm
                 -type: int, None
        
    :Return:
        Norms along the given axis. 
       -type: numpy array or float
             
    """
    return numx.sqrt(numx.sum(matrix*matrix, axis=axis))

def restrict_norms(matrix, max_norm, axis=0):
    """ This function restricts a matrix, its columns or rows to a given norm.

    :Parameters:
        matrix:   Matrix that should be restricted.
                 -type: numpy array [num rows, num columns]
                          
        max_norm: The maximal data norm.
                 -type: double 

        axis:     Restriction of the matrix along the given axis.
                  or the full matrix.
                 -type: int, None

    :Return:
        Restricted matrix
       -type: numpy array [num rows, num columns]
                      
    """
    res = numx.double(matrix)
    if axis is None:
        norm = numx.sqrt(numx.sum(res*res))
        if norm > max_norm:
            res *= max_norm / norm
    else:

        # If no value is bigger than max_norm/SQRT(N) then the norm is smaller
        # as the threshold!
        if numx.max(res) > max_norm/numx.sqrt(res.shape[numx.abs(1-axis)]):
            # Calculate norms
            norms = get_norms(res, axis=axis)
            # Restrict the vectors
            for r in range(norms.shape[0]):
                if norms[r] > max_norm:
                    if axis == 0:
                        res[:, r] *= max_norm / norms[r]
                    else:
                        res[r, :] *= max_norm / norms[r]
    return res

def resize_norms(matrix, norm, axis=0):
    """ This function resizes a matrix, its columns or rows to a given norm.

    :Parameters:
        matrix:   Matrix that should be resized.
                 -type: numpy array [num rows, num columns]
                          
        norm:     The norm to restrict the matrix to.
                 -type: double 

        axis:     Resize of the matrix along the given axis.
                 -type: int, None

    :Return:
        Resized matrix, however it is inplace
       -type: numpy array [num rows, num columns]
                      
    """
    res = numx.double(matrix)
    if axis is None:
        norm_temp = numx.sqrt(numx.sum(res*res))
        res *= norm / norm_temp
    else:

        # Calculate norms
        norms = get_norms(res, axis=axis)
        # Restrict the vectors
        for r in range(norms.shape[0]):
            if axis == 0:
                res[:, r] *= norm / norms[r]
            else:
                res[r, :] *= norm / norms[r]
    return res

def angle_between_vectors(v1, v2, degree = True):
    ''' Computes the angle between two vectors.

    :Parameters:
        vector1: Vector 1.
                -type: numpy array

        vector2: Vector 2.
                -type: numpy array

        degree:  If true degrees is return, rad otherwise
                -type: numpy array

    :Returns:
        angle
        -type: scalar

    '''
    v1 = numx.atleast_2d(v1)
    v2 = numx.atleast_2d(v2)
    c = numx.dot(v1,v2.T)/(get_norms(v1,axis = 1)*get_norms(v2,axis = 1))
    c = numx.arccos(numx.clip(c, -1, 1))
    if degree:
        c = numx.degrees(c)
    return c

def generate_binary_code(bit_length, batch_size_exp=None, batch_number=0):
    """ This function can be used to generate all possible binary vectors of length 'bit_length'. It is possible to
        generate only a particular batch of the data, where 'batch_size_exp' controls the size of the batch
        (batch_size = 2**batch_size_exp) and 'batch_number' is the index of the batch that should be generated.
         

        :Example: bit_length = 2, batchSize = 2
                  -> All combination = 2^bit_length = 2^2 = 4 
                  -> All_combinations / batchSize = 4 / 2 = 2 batches
                  -> _generate_bit_array(2, 2, 0) = [0,0],[0,1]
                  -> _generate_bit_array(2, 2, 1) = [1,0],[1,1]
                   
        :Parameter:
            bit_length:     Length of the bit vectors.
                           -type: int
                          
            batch_size_exp: Size of the batch of data.
                            :INFO: batch_size = 2**batch_size_exp
                           -type: int
                          
            batch_number:   Index of the batch.
                           -type: int
          
        :Return:
            Bit array containing the states  .
           -type: numpy array [num samples, bit_length]
           
    """
    # No batch size is given, all data is returned
    if batch_size_exp is None:
        batch_size_exp = bit_length
    batch_size = 2**batch_size_exp
    # Generate batch
    bit_combinations = numx.zeros((batch_size, bit_length))
    for number in range(batch_size):
        dividend = number + batch_number * batch_size
        bit_index = 0
        while dividend != 0:
            bit_combinations[number, bit_index] = numx.remainder(dividend, 2)
            dividend = numx.floor_divide(dividend, 2)
            bit_index += 1
    return bit_combinations