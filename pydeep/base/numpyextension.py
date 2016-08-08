''' This module provides different math functions that extend the numpy library.
              
    :Implemented:
        - log_sum_exp
        - log_diff_exp
        - get_norms
        - restrict_norms
        - restrict_matrix_norm
        - resize_norms
        - angle_between_vectors
        - get_2D_gauss_kernel
        - generate_binary_code
        - get_binary_label
        - compare_index_of_max
        - rotationSequence
        - generate_2D_connection_matrix
        
    :Version:
        1.0

    :Date:
        08.08.2016

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

def multinominal_batch_sampling(probabilties, isnormalized = True):
    """ Sample states where only one entry is one and the rest is 
        zero according to the given probablities.
    
    :Parameters:
        probabilties: Matrix containing probabilities the rows have to sum to one, 
                      otherwise chosen normalized = False.
                     -type: numpy array [batchsize, number of states]

        isnormalized: If True the probabilities are assumed to be normalized. 
                      If False the probabilities are normalized.
                     -type: bool
        
    :Return:
        Sampled multinominal states.
       -type: numpy array [batchsize, number of states]
             
    """
    probs = numx.float64(probabilties)
    if not isnormalized:
        probs = probs/numx.sum(probs,axis = 1).reshape(probs.shape[0],1)
    mini = probs.cumsum(axis = 1)
    maxi = mini-probs
    sample = numx.random.random((probs.shape[0],1))
    return (mini > sample)*(sample >= maxi)


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

def get_2D_gauss_kernel(N, M, shift = 0.0, var = 1.0):
    """ Creates a 2D Gauss kernel of size NxM with variance 1
                   
        :Parameter:
            N:     Number of pixels first dimension.
                  -type: int

            M:     Number of pixels second dimension.
                  -type: int
                          
            shift: The Gaussian is shifted by this amout fromthe center of the image.
                   Passing a scalar -> x,y shifted by the same value
                   Passing a vector -> x,y shifted accordingly
                  -type: int, 1D numpy array
                          
            var:   Variances or Covariance matrix.
                   Passing a scalar -> Isotropic Gaussian
                   Passing a vector -> Spherical covariance with vector values on the diagonals. 
                   Passing a matrix -> Full Gaussian
                  -type: int, 1D numpy array or 2D numpy array
          
        :Return:
            Bit array containing the states  .
           -type: numpy array [num samples, bit_length]
           
    """  
    def gauss(x, mean, cov):
        return 1.0/(2.0*numx.pi*numx.sqrt(numx.linalg.det(cov)))*numx.exp(-0.5*numx.dot(numx.dot((x-mean).T,numx.linalg.inv(cov)),x-mean))
    
    if numx.isscalar(shift):
        m = numx.array([shift,shift])
    else:
        m = shift
        
    if numx.isscalar(var):
        cov = numx.array([[var,0],[0,var]])
    else:
        if len(var.shape) == 1:
            cov = numx.array([[var[0],0],[0,var[1]]])
        else:
            cov = var
    
    if N % 2 == 0:
        print "N needs to be odd!"
        pass
    if M % 2 == 0:
        print "M needs to be odd!"
        pass
    lowerN = (N-1)/2
    lowerM = (M-1)/2
    mat = numx.zeros((N,M))
    for x in range(0,N):
        for y in range(0,M):
            mat[x,y] = gauss(numx.array([x-lowerN,y-lowerM]), mean = m, cov = cov)
    return mat

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


def get_binary_label(int_array):
    """ This function converts a 1D-array with integers labels into a 2D-array containing binary labels.

        :Example: -> [3,1,0]
                  -> [[1,0,0,0],[0,0,1,0],[0,0,0,1]]

        :Parameter:
            int_array:  1D array containing integers
                      - type: int

        :Return:
            2D array with binary labels.
           -type: numpy array [num samples, num labels]

    """
    max_label = numx.max(int_array)+1
    result = numx.zeros((int_array.shape[0], max_label))
    for i in range(int_array.shape[0]):
        result[i, int_array[i]] = 1
    return result

def compare_index_of_max(output, target):
    """ Compares data rows by comparing the index of the maximal value.
        e.g. Classifier output and true labels.
        
        :Example: [0.3,0.5,0.2],[0.2,0.6,0.2] -> 0
                  [0.3,0.5,0.2],[0.6,0.2,0.2] -> 1

        :Parameter:
            output:  vectors usually containing label probabilties,
                    - type: numpy array [batchsize, output_dim]

            target:  vectors usually containing true labels.
                    - type: numpy array [batchsize, output_dim]

        :Return:
            Int array containging 0 is the two rows hat the maximum at the same index, 1 otherwise.
           -type: numpy array [num samples, num labels]

    """
    return numx.int32(numx.argmax(output,axis=1) != numx.argmax(target,axis=1))

def rotationSequence(image, width, height, steps):
    """ Rotates a 2D image given as a 1D vector with shape[width*height] in 'steps' number of steps. 
        
        :Parameter:
            image:  Image as 1D vector.
                   -type: numpy array [1, width*height]

            width:  Width of the image such that image.shape[0] = width*height.
                   -type: int
                  
            height: Height of the image such that image.shape[0] = width*height.
                   -type: int

            steps:  Number of rotation steps e.g. 360 each steps is 1 degree.
                   -type: int

        :Return:
            Bool array containging True is the two rows hat the maximum at the same index, False otherwise.
           -type: numpy array [num samples, num labels]

    """
    results = numx.zeros((steps,image.shape[0]))
    results[0] = image
    for i in range(1,steps):
        angle = i*360.0/steps
        sample = rotate(image.reshape(width,height),angle)
        sample = sample[(sample.shape[0]-width)/2:(sample.shape[0]+width)/2,(sample.shape[0]-height)/2:(sample.shape[0]+height)/2]
        results[i] = sample.reshape(1,image.shape[0])
    return results

def generate_2D_connection_matrix(input_x_dim,
                                  input_y_dim,
                                  field_x_dim,
                                  field_y_dim,
                                  overlap_x_dim,
                                  overlap_y_dim,
                                  wrap_around=True):
    """ This function constructs a connection matrix, which can be
        used to force the weights to have local receptive fields.
        Example: input_x_dim = 3,
                 input_y_dim = 3,
                 field_x_dim = 2,
                 field_y_dim = 2,
                 overlap_x_dim = 1,
                 overlap_y_dim = 1,
                 wrap_around=False)
        leads to numx.array([[1,1,0,
                              1,1,0,
                              0,0,0],
                             [0,1,1,
                              0,1,1,
                              0,0,0],
                             [0,0,0,
                              1,1,0,
                              1,1,0],
                             [0,0,0,
                              0,1,1,
                              0,1,1]]).T

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

    """
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
    if wrap_around is False:
        end_x -= field_x_dim - 1
        end_y -= field_y_dim - 1
    step_x = field_x_dim - overlap_x_dim
    step_y = field_y_dim - overlap_y_dim

    for x in range(start_x,end_x,step_x):
        for y in range(start_y,end_y,step_y):
            column = numx.zeros((input_x_dim,input_y_dim))
            for i in range(x,x+field_x_dim,1):
                for j in range(y,y+field_y_dim,1):
                    column[i%input_x_dim,j%input_y_dim] = 1.0
            column = column.reshape((input_x_dim*input_y_dim))
            if matrix is None:
                matrix = column
            else:
                matrix = numx.vstack((matrix,column))
    return matrix.T