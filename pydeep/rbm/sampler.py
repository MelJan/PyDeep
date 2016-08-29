''' This module provides different sampling algorithms for RBMs running on CPU.
    The structure is kept modular to simplify the understanding of the code and
    the mathematics. In addition the modularity helps to create other kind of
    sampling algorithms by inheritance.

    :Implemented:
        - Gibbs Sampling   
        - Persistent Gibbs Sampling

    :Info:
        For the derivations see:
        http://www.ini.rub.de/data/documents/tns/masterthesis_janmelchior.pdf

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
import numpy as numx
import exceptions as ex

class Gibbs_sampler(object):
    ''' Implementation of k-step Gibbs-sampling for bipartite graphs.
          
    '''
          
    def __init__(self, model):
        ''' Initializes the sampler with the model.             
            
        :Parameters:
            model: The model to sample from. 
                  -type: Valid model class like BinaryBinary-RBM.
            
        ''' 
        # Set the model                
        if not hasattr(model,'probability_h_given_v'):
            raise ex.ValueError("The model needs to implement the function "
                                "probability_h_given_v!") 
        if not hasattr(model,'probability_v_given_h'):
            raise ex.ValueError("The model needs to implement the function "
                                "probability_v_given_h!") 
        if not hasattr(model,'sample_h'):
            raise ex.ValueError("The model needs to implement the function "
                                "sample_h!")  
        if not hasattr(model,'sample_v'):
            raise ex.ValueError("The model needs to implement the function "
                                "sample_v!")  
        self.model = model
     
    def sample(self, 
               vis_states, 
               k = 1, 
               betas = None, 
               ret_states = True):
        ''' Performs k steps Gibbs-sampling starting from given visible data.

        :Parameters:
            vis_states:   The initial visible states to sample from.
                         -type: numpy array [num samples, input dimension]
                         
            k:            The number of Gibbs sampling steps. 
                         -type: int

            betas:        Inverse temperature to sample from.
                          (energy based models) 
                         -type: None, float, numpy array [num_betas,1]
                          
            ret_states:   If False returns the visible probabilities instead of
                          the states.
                         -type: bool
        
        :Returns:
            The visible samples of the Markov chains.
           -type: numpy array [num samples, input dimension]    
            
        '''
        # Sample hidden states
        hid = self.model.probability_h_given_v(vis_states, betas)  
        hid = self.model.sample_h(hid, betas)
        
        # sample further
        return self.sample_from_h(hid, k, betas, ret_states)
      
    def sample_from_h(self, 
                      hid_states, 
                      k = 1, 
                      betas = None, 
                      ret_states = True):
        ''' Performs k steps Gibbs-sampling starting from given hidden states.

        :Parameters:
            hid_states: The initial hidden states to sample from.
                       -type: numpy array [num samples, output dimension]
                          
            k:          The number of Gibbs sampling steps. 
                       -type: int
                          
            betas:      Inverse temperature to sample from.
                        (energy based models) 
                       -type: None, float, numpy array [num_betas,1]
                          
            ret_states: If False returns the visible probabilities instead of
                        the states.
                       -type: bool
        
        :Returns:
            The visible samples of the Markov chains.
           -type: numpy array [num samples, input dimension]    
            
        '''
        # Sample k times
        vis = self.model.probability_v_given_h(hid_states, betas)  
        for _ in xrange(k-1):
            vis = self.model.sample_v(vis, betas) 
            hid = self.model.probability_h_given_v(vis, betas)  
            hid = self.model.sample_h(hid, betas)
            vis = self.model.probability_v_given_h(hid, betas) 
        
        # Return states or probs
        if ret_states:
            return self.model.sample_v(vis, betas)
        else:
            return vis                      

class Persistent_Gibbs_sampler(object):
    ''' Implementation of k-step persistent Gibbs sampling.
          
    '''
    
    def __init__(self, model, num_chains):
        ''' Initializes the sampler with the model.
        
        :Parameters:
            model:      The model to sample from. 
                       -type: Valid model class.
                   
            num_chains: The number of Markov chains.  
                        NOTE: Optimal performance is achieved if the number of
                        samples and the number of chains equal the batch_size.
                       -type: int
            
        '''    
        
        # Check and set the model                
        if not hasattr(model,'probability_h_given_v'):
            raise ex.ValueError("The model needs to implement the function "
                                "probability_h_given_v!") 
        if not hasattr(model,'probability_v_given_h'):
            raise ex.ValueError("The model needs to implement the function "
                                "probability_v_given_h!") 
        if not hasattr(model,'sample_h'):
            raise ex.ValueError("The model needs to implement the function "
                                "sample_h!")  
        if not hasattr(model,'sample_v'):
            raise ex.ValueError("The model needs to implement the function "
                                "sample_v!") 
        if not hasattr(model,'input_dim'):
            raise ex.ValueError("The model needs to implement the parameter "
                                "input_dim!")   
        self.model = model
        
        # Initialize persistent Markov chains to Gaussian random samples.
        if numx.isscalar(num_chains):
            self.chains = model.sample_v(numx.random.randn(num_chains, 
                                                           model.input_dim)*0.01) 
        else:
            raise ex.ValueError("Number of chains needs to be an integer "
                                "or None.")  

    def sample(self, 
               num_samples, 
               k = 1, 
               betas = None, 
               ret_states = True):
        ''' Performs k steps persistent Gibbs-sampling.
        
        :Parameters:            
            k:           The number of Gibbs sampling steps. 
                        -type: int

            num_samples: The number of samples to generate.
                         NOTE: Optimal performance is achieved if the number of
                         samples and the number of chains equal the batch_size.
                        -type: int, numpy array

            betas:       Inverse temperature to sample from.
                         (energy based models) 
                        -type: None, float, numpy array [num_betas,1]
                          
            ret_states:  If False returns the visible probabilities instead of
                         the states.
                        -type: bool
                            
        :Returns:
            The visible samples of the Markov chains.
           -type: numpy array [num samples, input dimension]     
            
        '''  
        # Sample k times
        for _ in xrange(k):
            hid = self.model.probability_h_given_v(self.chains, betas) 
            hid = self.model.sample_h(hid, betas) 
            vis = self.model.probability_v_given_h(hid, betas)  
            self.chains = self.model.sample_v(vis, betas)
        if ret_states:
            samples = self.chains
        else:
            samples = vis

        if num_samples == self.chains.shape[0]:
            return samples
        else:
            # If more samples than chains,
            repeats = numx.int32(num_samples / self.chains.shape[0])

            for _ in xrange(repeats):

                # Sample k times
                for _ in xrange(k):
                    hid = self.model.probability_h_given_v(self.chains, betas)
                    hid = self.model.sample_h(hid, betas)
                    vis = self.model.probability_v_given_h(hid, betas)
                    self.chains = self.model.sample_v(vis, betas)
                if ret_states:
                    samples = numx.vstack([samples,self.chains])
                else:
                    samples = numx.vstack([samples,vis])
            return samples[0:num_samples,:]