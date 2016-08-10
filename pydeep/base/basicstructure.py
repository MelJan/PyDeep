''' This module provides basic structural elements, which different models have in common.

    :Implemented:
        - BipartiteGraph
   
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
from pydeep.base.activationfunction import Sigmoid
import exceptions as ex

class BipartiteGraph(object):
    ''' Implementation of a bipartite graph structure.     

    '''
  
    def __init__(self, 
                 number_visibles, 
                 number_hiddens, 
                 data = None, 
                 visible_activation_function = Sigmoid, 
                 hidden_activation_function = Sigmoid, 
                 initial_weights = 'AUTO', 
                 initial_visible_bias = 'AUTO', 
                 initial_hidden_bias = 'AUTO', 
                 initial_visible_offsets = 'AUTO', 
                 initial_hidden_offsets = 'AUTO', 
                 dtype = numx.float64):
        ''' This function initializes all necessary parameters and data 
            structures. It is recommended to pass the training data to 
            initialize the network automatically.
            
        :Parameters:
            number_visibles:             Number of the visible variables.
                                        -type: int
                                  
            number_hiddens               Number of hidden variables.
                                        -type: int
                                  
            data:                        The training data for parameter 
                                         initialization if 'AUTO' is chosen 
                                         for the corresponding parameter.
                                        -type: None or 
                                         numpy array [num samples, input dim]

            visible_activation_function  Activation function for the visible units.
                                        -type: ActivationFunction

            hidden_activation_function   Activation function for the hidden units.
                                        -type: ActivationFunction

            initial_weights:             Initial weights.
                                         'AUTO' and a scalar are random init.
                                        -type: 'AUTO', scalar or 
                                         numpy array [input dim, output_dim]
                                  
            initial_visible_bias:        Initial visible bias.
                                         'AUTO' is random, 'INVERSE_SIGMOID' is the
                                         inverse Sigmoid of the visilbe mean.
                                         If a scalar is passed all values are 
                                         initialized with it. 
                                        -type: 'AUTO','INVERSE_SIGMOID', scalar or 
                                         numpy array [1, input dim]
                                  
            initial_hidden_bias:         Initial hidden bias.
                                         'AUTO' is random, 'INVERSE_SIGMOID' is the
                                         inverse Sigmoid of the hidden mean.
                                         If a scalar is passed all values are 
                                         initialized with it. 
                                        -type: 'AUTO','INVERSE_SIGMOID', scalar or 
                                         numpy array [1, output_dim]
                                  
            initial_visible_offsets:     Initial visible offset values.
                                         AUTO=data mean or 0.5 if no data is given.
                                         If a scalar is passed all values are 
                                         initialized with it. 
                                        -type: 'AUTO', scalar or 
                                         numpy array [1, input dim]
                                  
            initial_hidden_offsets:      Initial hidden offset values.
                                         AUTO = 0.5
                                         If a scalar is passed all values are 
                                         initialized with it. 
                                        -type: 'AUTO', scalar or 
                                         numpy array [1, output_dim]
                        
            dtype:                       Used data type i.e. numpy.float64
                                        -type: numpy.float32 or numpy.float64 or 
                                         numpy.float128  
            
        '''
        
        # Set internal datatype
        self.dtype = dtype

        # Set input and output dimension
        self.input_dim = number_visibles
        self.output_dim = number_hiddens
        
        self.visible_activation_function = visible_activation_function 
        self.hidden_activation_function = hidden_activation_function 
        
        self._data_mean = 0.5*numx.ones((1, self.input_dim),self.dtype)
        self._data_std = numx.ones((1, self.input_dim),self.dtype)
        if data is not None:
            if isinstance(data,list):
                data = numx.concatenate(data)
            if self.input_dim != data.shape[1]:
                raise ex.ValueError("Data dimension and model input \
                                    dimension have to be equal!")  
            self._data_mean = data.mean(axis=0).reshape(1,data.shape[1])
            self._data_std = data.std(axis=0).reshape(1,data.shape[1])

        # AUTO   -> Small random values out of 
        #           +-4*numx.sqrt(6/(self.input_dim+self.output_dim)  
        # Scalar -> Small Gaussian distributed random values with std_dev 
        #           initial_weights
        # Array  -> The corresponding values are used   
        if initial_weights is 'AUTO':
            self.w = numx.array((2.0 * numx.random.rand(self.input_dim,
                                                self.output_dim) - 1.0)
                              * (4.0 * numx.sqrt(6.0 / (self.input_dim 
                                                    + self.output_dim)))
                              ,dtype=dtype)
        else:
            if(numx.isscalar(initial_weights)):
                self.w = numx.array(numx.random.randn(self.input_dim, 
                                                      self.output_dim) 
                                  * initial_weights, dtype=dtype)
            else:
                self.w = numx.array(initial_weights, dtype=dtype)

        # AUTO   -> data != None -> Initialized to the data mean
        #           data == None -> Initialized to Visible range mean
        # Scalar -> Initialized to given value
        # Array  -> The corresponding values are used  
        self.ov = numx.zeros((1,self.input_dim))
        if initial_visible_offsets is 'AUTO':
            if data is not None:
                self.ov += self._data_mean
            else:
                self.ov += 0.5
        else:
            if(numx.isscalar(initial_visible_offsets)):
                self.ov += initial_visible_offsets
            else:
                self.ov += initial_visible_offsets.reshape(1,self.input_dim)
        self.ov = numx.array(self.ov, dtype=dtype)
        
        # AUTO   -> data != None -> Initialized to the inverse sigmoid of 
        #           data mean
        #           data == Initialized to randn()*0.01
        # Scalar -> Initialized to given value + randn()*0.01
        # Array  -> The corresponding values are used 
        if initial_visible_bias is 'AUTO':
            if data is None:
                self.bv = numx.zeros((1, self.input_dim))
            else:
                self.bv = numx.array(Sigmoid.g(numx.clip(self._data_mean,0.001,
                          0.999)), dtype=dtype).reshape(self.ov.shape)
        else:
            if initial_visible_bias is 'INVERSE_SIGMOID':
                self.bv = numx.array(Sigmoid.g(numx.clip(self.ov,0.001,
                          0.999)), dtype=dtype).reshape(1,self.input_dim)
            else:
                if(numx.isscalar(initial_visible_bias)):
                    self.bv = numx.array(initial_visible_bias 
                                     + numx.zeros((1, self.input_dim))
                                     , dtype=dtype) 
                else:
                    self.bv = numx.array(initial_visible_bias, 
                                         dtype=dtype)
        
        # AUTO   -> Initialized to Hidden range mean
        # Scalar -> Initialized to given value
        # Array  -> The corresponding values are used 
        self.oh = numx.zeros((1,self.output_dim))
        if initial_hidden_offsets is 'AUTO':
            self.oh += 0.5
        else:
            if(numx.isscalar(initial_hidden_offsets)):
                self.oh += initial_hidden_offsets
            else:
                self.oh += initial_hidden_offsets.reshape(1,self.output_dim)
        self.oh = numx.array(self.oh, dtype=dtype)
        
        # AUTO   -> Initialized to randn()*0.01
        # Scalar -> Initialized to given value + randn()*0.01
        # Array  -> The corresponding values are used 
        if initial_hidden_bias is 'AUTO':
            self.bh = numx.zeros((1, self.output_dim))
        else:
            if initial_hidden_bias is 'INVERSE_SIGMOID':
                    self.bh = numx.array(
                        Sigmoid.g(numx.clip(self.oh,0.001,0.999))
                                  , dtype=dtype).reshape(self.oh.shape)
            else:
                if(numx.isscalar(initial_hidden_bias)):
                    self.bh = numx.array(initial_hidden_bias 
                                     + numx.zeros((1, self.output_dim))
                                     , dtype=dtype) 
                else:
                    self.bh = numx.array(initial_hidden_bias, dtype=dtype)

    def _visible_pre_activation(self, h):
        ''' Computes the visible pre-activations from hidden activations.
        
        :Parameters:
            h: Hidden activations.
              -type: numpy array [num data points, output_dim] 
                                   
        :Returns: 
            Visible pre-synaptic activations.
           -type: numpy array [num data points, input_dim] 
           
        ''' 
        return numx.dot(h-self.oh, self.w.T) + self.bv
        
    def _visible_post_activation(self, pre_act_v):
        ''' Computes the visible (post) activations from 
            visible pre-activations.
        
        :Parameters:
            pre_act_v: Visible pre-activations.
                      -type: numpy array [num data points, input_dim] 
                                   
        :Returns: 
            Visible activations.
           -type: numpy array [num data points, input_dim] 
           
        '''  
        return self.visible_activation_function.f(pre_act_v)
    
    def visible_activation(self, h):
        ''' Computes the visible (post) activations from 
            hidden activations.
        
        :Parameters:
            h: Hidden activations.
              -type: numpy array [num data points, output_dim] 
                                   
        :Returns: 
            Visible activations.
           -type: numpy array [num data points, input_dim] 
           
        '''  
        return self._visible_post_activation(self._visible_pre_activation(h))
        
    def _hidden_pre_activation(self, v):
        ''' Computes the Hidden pre-activations from visible activations.
        
        :Parameters:
            v: Visible activations.
              -type: numpy array [num data points, input_dim] 
                                   
        :Returns: 
            Hidden pre-synaptic activations.
           -type: numpy array [num data points, output_dim] 
           
        ''' 
        return numx.dot(v-self.ov, self.w) + self.bh
        
    def _hidden_post_activation(self, pre_act_h):
        ''' Computes the Hidden (post) activations from 
            hidden pre-activations.
        
        :Parameters:
            pre_act_h: Hidden pre-activations.
                      -type: numpy array [num data points, output_dim] 
                                   
        :Returns: 
            Hidden activations.
           -type: numpy array [num data points, output_dim] 
           
        '''  
        return self.hidden_activation_function.f(pre_act_h)
    
    def hidden_activation(self, v):
        ''' Computes the Hidden (post) activations from 
            visible activations.
        
        :Parameters:
            v: Visible activations.
              -type: numpy array [num data points, input_dim] 
                                   
        :Returns: 
            Hidden activations.
           -type: numpy array [num data points, output_dim] 
           
        '''  
        return self._hidden_post_activation(self._hidden_pre_activation(v))
        
    def _add_hidden_units(self, 
                          num_new_hiddens, 
                          position = 0, 
                          initial_weights='AUTO', 
                          initial_bias='AUTO', 
                          initial_offsets = 'AUTO'):
        ''' This function adds new hidden units at the given position to the
            model.
            Warning: If the parameters are changed. the trainer needs to be 
                     reinitialized.
        
        :Parameters:
            num_new_hiddens: The number of new hidden units to add.
                            -type: int
                            
            position:        Position where the units should be added.
                            -type: int

            initial_weights: The initial weight values for the hidden units. 
                            -type: 'AUTO' or scalar or
                                   numpy array [input_dim, num_new_hiddens]

            initial_bias:    The initial hidden bias values. 
                            -type: 'AUTO' or scalar or 
                                   numpy array [1, num_new_hiddens]
                            
            initial_offsets: The initial hidden mean values. 
                            -type: 'AUTO' or scalar or 
                                   numpy array [1, num_new_hiddens]
                             
        ''' 
        # AUTO   -> Small random values out of 
        #           +-4*numx.sqrt(6/(self.input_dim+self.output_dim)  
        # Scalar -> Small Gaussian distributed random values with std_dev 
        #           initial_weights
        # Array  -> The corresponding values are used   
        new_weights = None
        if initial_weights == 'AUTO':
            new_weights = ((2.0 * numx.random.rand(self.input_dim, 
                            num_new_hiddens) - 1.0) * (4.0 * numx.sqrt(6.0 
                            / (self.input_dim + self.output_dim 
                               + num_new_hiddens))))
        else:
            if(numx.isscalar(initial_weights)):
                new_weights = numx.random.randn(self.input_dim, 
                              num_new_hiddens) * initial_weights
            else:
                new_weights = initial_weights
        self.w = numx.array(numx.insert(self.w, numx.ones((num_new_hiddens)
                           ) * position, new_weights, axis=1),self.dtype)
    
        # AUTO   -> Initialized to Hidden range mean
        # Scalar -> Initialized to given value
        # Array  -> The corresponding values are used 
        new_oh = None
        if initial_offsets == 'AUTO':
            new_oh = numx.zeros((1, num_new_hiddens)) + 0.5
        else:
            if(numx.isscalar(initial_offsets)):
                new_oh = numx.zeros((1, num_new_hiddens)) + initial_offsets
            else:
                new_oh = initial_offsets   
        self.oh = numx.array(numx.insert(self.oh, numx.ones((num_new_hiddens)
                            ) * position, new_oh, axis=1),self.dtype)
        
        # AUTO   -> Initialized to randn()*0.01
        # Scalar -> Initialized to given value + randn()*0.01
        # Array  -> The corresponding values are used 
        if initial_bias == 'AUTO':
            new_bias = numx.zeros((1, num_new_hiddens))
        else:
            if initial_bias == 'INVERSE_SIGMOID':
                new_bias = Sigmoid.g(numx.clip(new_oh,0.01,
                           0.99)).reshape(new_oh.shape)
            else:
                if(numx.isscalar(initial_bias)):
                    new_bias = initial_bias + numx.zeros((1, num_new_hiddens))
                else:
                    new_bias = numx.array(initial_bias, dtype=self.dtype)
        self.bh = numx.array(numx.insert(self.bh, numx.ones((num_new_hiddens)
                             ) * position, new_bias, axis=1),self.dtype)
        
        self.output_dim = self.w.shape[1] 
        
    def _remove_hidden_units(self, indices):
        ''' This function removes the hidden units whose indices are given.
            Warning: If the parameters are changed. the trainer needs to be 
                     reinitialized.
                     
        :Parameters:
            indices: Indices to remove.
                    -type: int or list of int or numpy array of int
            
        ''' 
        self.w = numx.delete(self.w, numx.array(indices), axis=1)
        self.bh = numx.delete(self.bh, numx.array(indices), axis=1)
        self.oh = numx.delete(self.oh, numx.array(indices), axis=1)
        self.output_dim = self.w.shape[1] 

    def _add_visible_units(self, 
                           num_new_visibles, 
                           position=0, 
                           initial_weights='AUTO', 
                           initial_bias='AUTO', 
                           initial_offsets = 'AUTO',
                           data = None):
        ''' This function adds new visible units at the given position to 
            the model.
            Warning: If the parameters are changed. the trainer needs to be 
                     reinitialized.
        
        :Parameters:
            num_new_visibles: The number of new hidden units to add
                             -type: int

            position:         Position where the units should be added.
                             -type: int

            initial_weights:  The initial weight values for the hidden units. 
                             -type: 'AUTO' or scalar or 
                                    numpy array [num_new_visibles, output_dim]

            initial_bias:     The initial hidden bias values. 
                             -type: 'AUTO' or scalar or
                                     numpy array [1, num_new_visibles]
                            
            initial_offsets:  The initial visible offset values. 
                             -type: 'AUTO' or scalar or
                                     numpy array [1, num_new_visibles]

            data:             Data for AUTO initilization.
                             -type: numpy array [num datapoints, num_new_visibles]
                                        
        '''          
        new_data_mean = 0.5*numx.ones((1, num_new_visibles),self.dtype)
        new_data_std = numx.ones((1, num_new_visibles),self.dtype)/12.0
        if data is not None:
            if isinstance(data,list):
                data = numx.concatenate(data)
            new_data_mean = data.mean(axis=0).reshape(1,num_new_visibles)
            new_data_std = data.std(axis=0).reshape(1,num_new_visibles)
        self._data_mean = numx.array(numx.insert(self._data_mean, numx.ones((
                          num_new_visibles))*position, new_data_mean, axis=1)
                          ,self.dtype)
        self._data_std  = numx.array(numx.insert(self._data_std, numx.ones((
                          num_new_visibles))*position, new_data_std, axis=1)
                          ,self.dtype)
        
        # AUTO   -> Small random values out of 
        #           +-4*numx.sqrt(6/(self.input_dim+self.output_dim)  
        # Scalar -> Small Gaussian distributed random values with std_dev 
        #           initial_weights
        # Array  -> The corresponding values are used  
        new_weights = None
        if initial_weights == 'AUTO':
            new_weights = numx.array((2.0 * numx.random.rand(num_new_visibles,
                          self.output_dim) - 1.0)* (4.0 * numx.sqrt(6.0 / (
                          self.input_dim + self.output_dim 
                          + num_new_visibles))), dtype=self.dtype)
        else:
            if(numx.isscalar(initial_weights)):
                new_weights = numx.random.randn(num_new_visibles, 
                                        self.output_dim) * initial_weights
            else:
                new_weights = initial_weights
        self.w = numx.array(numx.insert(self.w, numx.ones((num_new_visibles)
                                )*position, new_weights, axis=0),self.dtype)

        if initial_offsets == 'AUTO':
            if data is not None:
                new_ov = new_data_mean
            else:
                new_ov = numx.zeros((1, num_new_visibles)) + 0.5
        else:
            if(numx.isscalar(initial_offsets)):
                new_ov = numx.zeros((1, num_new_visibles)) + initial_offsets
            else:
                new_ov = initial_offsets
        self.ov = numx.array(numx.insert(self.ov, numx.ones((num_new_visibles)
                            )*position, new_ov, axis=1),self.dtype)
        
        # AUTO   -> data != None -> Initialized to the inverse sigmoid of 
        #           data mean
        #           data == Initialized to randn()*0.01
        # Scalar -> Initialized to given value + randn()*0.01
        # Array  -> The corresponding values are used 
        new_bias = None
        if initial_bias == 'AUTO':
            if data is not None:
                new_bias = numx.zeros((1, num_new_visibles))
            else:
                new_bias = new_data_mean
        else:
            if(numx.isscalar(initial_bias)):
                new_bias = numx.zeros((1, num_new_visibles)) + initial_bias
            else:
                new_bias = initial_bias
        self.bv = numx.array(numx.insert(self.bv,
                                         numx.ones((num_new_visibles))*position,
                                         new_bias, axis=1),self.dtype)   
        self.input_dim = self.w.shape[0]      

    def _remove_visible_units(self, indices):
        ''' This function removes the visible units whose indices are given.
            Warning: If the parameters are changed. the trainer needs to be 
                     reinitialized.
                     
        :Parameters:
            indices: Indices of units to be remove.
                    -type: int or list of int or numpy array of int
            
        ''' 
        self.w = numx.delete(self.w, numx.array(indices), axis=0)
        self.bv = numx.delete(self.bv, numx.array(indices), axis=1)
        self.ov = numx.delete(self.ov, numx.array(indices), axis=1)
        self._data_mean = numx.delete(self._data_mean, 
                                      numx.array(indices), axis=1)
        self._data_std = numx.delete(self._data_std, 
                                      numx.array(indices), axis=1)
        self.input_dim = self.w.shape[0]  

    def get_parameters(self):
        ''' This function returns all model parameters in a list.
        
        :Returns: 
            The parameter references in a list.
           -type: list 

        ''' 
        return [self.w, self.bv, self.bh]

    def update_parameters(self, updates):
        ''' This function updates all parameters given the updates derived by 
            the training methods.
        
        :Parameters:
            updates: Parameter gradients.
                    -type: list of numpy arrays (num para. x [para.shape])
                       
        ''' 
        i = 0
        for p in self.get_parameters():
            p += updates[i]
            i += 1

    def update_offsets(self, 
                       new_visible_offsets = 0.0, 
                       new_hidden_offsets = 0.0, 
                       update_visible_offsets = 1.0,
                       update_hidden_offsets = 1.0):
        ''' This function updates the visible and hidden offsets. 
            update_offsets(0,0,1,1) reparameterizes to the normal binary RBM.
        
        :Parameters:
            new_visible_offsets:    New visible means.
                                   -type: numpy arrays [1, input dim]
                              
            new_hidden_offsets:     New hidden means.
                                   -type: numpy arrays [1, output dim]

            update_visible_offsets: Update/Shifting factor for the visible 
                                    means.
                                   -type: float
                              
            update_hidden_offsets:  Update/Shifting factor for the hidden means.
                                   -type: float
                                   
        ''' 
        # update the centers
        if update_hidden_offsets != 0.0:
            self.bv += (update_hidden_offsets*numx.dot(new_hidden_offsets
                                                       -self.oh,self.w.T))
            self.oh = ((1.0-update_hidden_offsets)*self.oh 
                       + update_hidden_offsets*new_hidden_offsets)
        # update the centers
        if update_visible_offsets != 0.0:
            self.bh += (update_visible_offsets* numx.dot(new_visible_offsets
                                                         -self.ov, self.w))
            self.ov = ((1.0-update_visible_offsets)* self.ov 
                       + update_visible_offsets* new_visible_offsets)
