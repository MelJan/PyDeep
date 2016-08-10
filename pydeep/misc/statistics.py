''' This class contains methods to calculate some statistics for a given RBM 
    and data.

    :Implemented:
        - hidden_activation
        - reorder_filter_by_hidden_activation
        - generate_samples

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
