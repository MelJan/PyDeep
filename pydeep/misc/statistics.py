''' This class contains methods to calculate some statistics for a given RBM 
    and data.

    :Implemented:
        - hidden_activation
        - reorder_filter_by_hidden_activation
        - generate_samples
        - gaussian_anchor_component_scaling
        - gaussian_first_order_component_scaling
        - filter_frequency_and_angle
        - filter_angle_response
        - calculate_amari_distance

    :Version:
        1.0

    :Date:
        06.06.2016

    :Author:
        Jan Melchior, Nan Wang

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
import copy

def hidden_activation(rbm, 
                      data, 
                      states = False):
    ''' Calculates the hidden activation.
    
    :Parameters:
        rbm:    RBM model object.
               -type: RBM model object
        
        data:   Data for the activation calculation
               -type: numpy array [num samples, dimensions]
        
        states: If True uses states rather then probabilities by rounding to
                0 or 1.
                type:bool
    
    :Return:
        hidden activation and the mean and standard deviation over the data.
       -type: numpy array, float, float
        
    '''
    activation = rbm.probability_h_given_v(data)
    if states:
        activation = numx.round(activation)
    return activation,numx.mean(activation,
                                axis = 0),numx.std(activation,axis = 0)

def reorder_filter_by_hidden_activation(rbm, 
                                        data):
    ''' Reorders the weights by its activation over the data set in 
        decreasing order. 
    
    :Parameters:
        rbm:    RBM model object.
               -type: RBM model object
        
        data:   Data for the activation calculation.
               -type: numpy array [num samples, dimensions]
    
    '''
    probs = numx.sum(rbm.probability_h_given_v(data),axis=0)
    index = numx.argsort(probs, axis = 0)
    rbm_ordered = copy.deepcopy(rbm)
    for i in range(probs.shape[0]):
        u = probs.shape[0] - i - 1
        rbm_ordered.w[:,u] = rbm.w[:,index[i]]
        rbm_ordered.bh[0,u] = rbm.bh[0,index[i]]
    return rbm_ordered 

def generate_samples(rbm,
                     data,
                     iterations,
                     stepsize,
                     v1,
                     v2, 
                     sample_states = False, 
                     whitening = None):
    ''' Generates samples from the given RBM model.     
    
    :Parameters:
        rbm:           RBM model.
                      -type: RBM model object.
        
        data:          Data to sample from.
                      -type: numpy array [num samples, dimensions]
        
        iterations:    Number of Gibbs sampling steps.
                      -type: int
        
        stepsize:      After how many steps a sample should be plotted.
                      -type: int
        
        v1:            X-Axis of the reorder image patch.
                      -type: int
        
        v2:            Y-Axis of the reorder image patch.
                      -type: int
        
        sample_states: If true returns the sates , probabilities otherwise.
                      -type: bool
        
        whitening:     If the data has been preprocessed it needs to be 
                       undone.
                      -type: preprocessing object or None
        
    :Returns:
        Matrix with image patches order along X-Axis and it's evolution in 
        Y-Axis.
       -type: numpy array
    
    '''
    result = data
    if whitening != None:
        result = whitening.unproject(data)
    vis_states = data
    for i in xrange(1,iterations+1):    
        hid_probs = rbm.probability_h_given_v(vis_states)
        hid_states = rbm.sample_h(hid_probs)
        vis_probs = rbm.probability_v_given_h(hid_states)   
        vis_states = rbm.sample_v(vis_probs)
        if i % stepsize == 0: 
            if whitening != None:
                if sample_states:
                    result = numx.vstack((result,
                                        whitening.unproject(vis_states)))
                else:
                    result = numx.vstack((result,
                                        whitening.unproject(vis_probs)))
            else:
                if sample_states:
                    result = numx.vstack((result,vis_states))
                else:
                    result = numx.vstack((result,vis_probs))
                
    import pydeep.misc.visualization as Vis
    return Vis.tile_matrix_rows(result.T, v1,v2,iterations/stepsize+1
                               ,data.shape[0], border_size = 1,
                               normalized = False)

''' Up from here only for Gaussian-Binary RBMs '''

def gaussian_anchor_component_scaling(rbm, 
                                      logZ):
    ''' Computes the anchor scaling factor for Gaussian RBM.
    
    :Parameters:
        rbm:  RBM.
             -type: RBM model object.
                    
        logZ: Logarithm of the partition function.
             -type: float
              
    :Return:
        scaling factors for the anchor components.
       -type: float
              
    '''
    detSigma = 1.0
    for i in range(rbm.variance.shape[1]):
        detSigma *= rbm.variance[0,i]
    return numx.exp(rbm.input_dim/2.0*numx.log(2.0*numx.pi)
                  +0.5*numx.log(detSigma)-logZ)

def gaussian_first_order_component_scaling(rbm, logZ):
    ''' Computes the first order scaling factors.
    
    :Parameters:
        rbm:        RBM.
                   -type: RBM model object.
                    
        logZ: Logarithm of the partition function.
             -type: float
              
    :Return:
        scaling factors for the first order components as well as its mean and
        standard deviation.
       -type: numpy array
        
    '''
    normB= (rbm.bv*rbm.bv)/(2.0*rbm.variance)
    WB = rbm.w+rbm.bv.T
    normWB = (WB*WB)/(2.0*rbm.variance.T)
    scalings = numx.exp(numx.sum(normWB,axis = 0) 
            - numx.sum(numx.sum(normB, axis = 1),axis =0) 
            + rbm.bh.reshape(rbm.output_dim))
    return scalings,numx.mean(scalings,axis = 0),numx.std(scalings,axis = 0)

def filter_frequency_and_angle(filters, 
                               num_of_angles = 40):
    ''' Analyze the filters by calculating the responses when gratings, 
        i.e. sinusoidal functions, are input to them.
    
    :Info:
        Hyv/"arinen, A. et al. (2009) Natural image statistics, Page 144-146
    
    :Parameters:
        filters:    the filter matrix with the size of input_dim x output_dim 
                   -type: numpy array
                    
        num_of_ang: the number of angles for generating gratings
                   -type: numpy array
                        
    :Return:
        the optimal frequency (pixels/cycle) of the filters,
        the optimal orientation angle (rad) of the filters
       -type: numpy array, numpy array
        
    '''
    Rsp_max_ang, Rsp_max_ang_idx = filter_frequency_response(filters, 
                                                             num_of_angles)
    opt_frq = Rsp_max_ang.argmax(0)+2
    opt_ang = numx.diag(Rsp_max_ang_idx[opt_frq-2][:]) * numx.pi/num_of_angles
    return opt_frq, opt_ang

def filter_frequency_response(filters, 
                              num_of_angles = 40):
    '''Compute the response of filters w.r.t. different frequency

    :Parameters:
        filters: filters to analyze 
                -type: numpy array
              
        num_of_angles: number of angles steps to check
                      -type: int
    
    :Returns:
        frequency response as output_dim x max_wavelength-1 index of the 
        angles
       -type: numpy array, numpy array
        
    '''
    input_dim = filters.shape[0] # input dimensionality, 196
    max_wavelength = int(numx.sqrt(filters.shape[0]))
    output_dim = filters.shape[1]
    frq_rsp = numx.zeros([max_wavelength-1, output_dim])
    frq_rsp_ang_idx = numx.zeros([max_wavelength-1, output_dim])
    
    vec_theta = numx.array(range(0, num_of_angles))
    vec_theta = vec_theta * numx.pi / num_of_angles
    sinMatrix = numx.tile(numx.sin(vec_theta), (input_dim,1))
    cosMatrix = numx.tile(numx.cos(vec_theta), (input_dim,1))
    
    vec_xy = numx.array(range(0, input_dim))
    vec_x = numx.floor_divide(vec_xy, max_wavelength)
    vec_y = vec_xy + 1 - vec_x * max_wavelength
    xMatrix = numx.tile(vec_x.transpose(), (num_of_angles, 1))
    yMatrix = numx.tile(vec_y.transpose(), (num_of_angles, 1))  
    uMatrix = sinMatrix.transpose() * xMatrix + cosMatrix.transpose()*yMatrix
       
    for frq_idx in range(2, max_wavelength+1):
        alpha = float(1)/float(frq_idx)
        # sine gratings of all angles under a specific freq.
        gratingMatrix_sin = numx.sin( 2 * numx.pi * alpha * uMatrix) 
        # cosine gratings of all angles under a specific freq.
        gratingMatrix_cos = numx.cos( 2 * numx.pi * alpha * uMatrix) 
        Rsp_fix_frq = (numx.dot(gratingMatrix_sin, filters)**2 
                    + numx.dot(gratingMatrix_cos, filters)**2)
        frq_rsp[frq_idx-2] = Rsp_fix_frq.max(0)
        frq_rsp_ang_idx[frq_idx-2] = Rsp_fix_frq.argmax(0)
       
    return frq_rsp, frq_rsp_ang_idx
    
def filter_angle_response(filters, 
                          num_of_angles = 40):
    '''Compute the angle response of the given filter.
    
    :Parameters:
        filters:       filters to analyze 
                      -type: numpy array
                    
        num_of_angles: number of angles steps to check
                      -type: int
    
    :Returns:
        angle response as output_dim x num_of_ang, index of angles
       -type: numpy array, numpy array
        
    '''
    input_dim = filters.shape[0]
    max_wavelength = int(numx.sqrt(filters.shape[0]))
    output_dim = filters.shape[1]
    ang_rsp = numx.zeros([num_of_angles, output_dim])
    ang_rsp_frq_idx = numx.zeros([num_of_angles, output_dim])
    
    vec_frq = numx.array(range(2, max_wavelength+1))
    vec_frq = float(1) / vec_frq
    frqMatrix = numx.tile( vec_frq * numx.pi * 2, (input_dim,1))
    
    vec_xy = numx.array(range(0, input_dim))
    vec_x = numx.floor_divide(vec_xy, max_wavelength)
    vec_y = vec_xy + 1 - vec_x * max_wavelength
    xMatrix = numx.tile(vec_x.transpose(), (max_wavelength-1, 1))
    yMatrix = numx.tile(vec_y.transpose(), (max_wavelength-1, 1))   
    
    for ang_idx in range(0, num_of_angles):
        theta = ang_idx * numx.pi / num_of_angles
        uMatrix = numx.sin(theta) * xMatrix + numx.cos(theta) * yMatrix
        gratingMatrix_sin = numx.sin(frqMatrix.transpose() * uMatrix)
        gratingMatrix_cos = numx.cos(frqMatrix.transpose() * uMatrix)
        Rsp_fix_ang = (numx.dot(gratingMatrix_sin, filters)**2 
                    + numx.dot(gratingMatrix_cos, filters)**2)
        ang_rsp[ang_idx] = Rsp_fix_ang.max(0)
        ang_rsp_frq_idx[ang_idx] = Rsp_fix_ang.argmax(0)
        
    return ang_rsp, ang_rsp_frq_idx 

def calculate_amari_distance(matrix_one, 
                             matrix_two, 
                             version = 1):
    '''Calculate the Amari distance between two input matrices.

    :Parameters:
        matrix_one: the first matrix
                   -type: numpy array
                    
        matrix_two: the second matrix
                   -type: numpy array
                    
    :Returns:
        amari_distance: the amari distance between two input matrices.
                       -type: float

    '''
    if matrix_one.shape!=matrix_two.shape:
        return "Two matrices must have the same shape."
    product_matrix = numx.abs(numx.dot(matrix_one, 
                                       numx.linalg.inv(matrix_two)))

    product_matrix_max_col = numx.array(product_matrix.max(0))
    product_matrix_max_row = numx.array(product_matrix.max(1))

    N = product_matrix.shape[0]

    if version != 1:
        ''' Formula from Teh
        Here they refered to as "amari distance"
        The value is in [2*N-2N^2, 0]. 
        reference:
            Teh, Y. W.; Welling, M.; Osindero, S. & Hinton, G. E. Energy-based
            models for sparse overcomplete representations J MACH LEARN RES,
            2003, 4, 1235--1260
        '''
        amari_distance = product_matrix / numx.tile(product_matrix_max_col
                                                  , (N, 1))
        amari_distance += product_matrix / (numx.tile(product_matrix_max_row
                                                    , (N, 1)).T)
        amari_distance = amari_distance.sum() - 2 * N * N
    else:
        ''' Formula from ESLII
        Here they refered to as "amari error"
        The value is in [0, N-1].
        reference:
            Bach, F. R.; Jordan, M. I. Kernel Independent Component
            Analysis, J MACH LEARN RES, 2002, 3, 1--48
        '''
        amari_distance = product_matrix / numx.tile(product_matrix_max_col
                                                  , (N, 1))
        amari_distance += product_matrix / (numx.tile(product_matrix_max_row
                                                    , (N, 1)).T)
        amari_distance = amari_distance.sum() / (2 * N) - 1

    return amari_distance
