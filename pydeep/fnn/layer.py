'''  Feed Forward Neural Network Layers.

    .. Note::
    
        Due to computational benefits the common notation for the delta terms is 
        split in a delta term for the common layer and the error signal passed 
        to the layer below. See the following Latex code for details. This allows
        to store all layer depending results in the corresponding layer and avoid 
        useless computations without messing up the code.
        
        \begin{eqnarray}
            \delta^{(n)} &=& Cost'(a^{(n)} ,label) \bullet \sigma'(z^{(n)}) \\
            error^{(i)} &=& (W^{(i)})^T \delta^{(i)} \\
            \delta^{(i)} &=&  error^{(i+1)} \bullet \sigma'(z^{(i)})
        \end{eqnarray}

    :Version:
        1.0

    :Date:
        08.02.2016

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
import pydeep.base.activationfunction as AFct
import pydeep.base.costfunction as CFct

class FullConnLayer(object):
    ''' Represents a simple 1D Hidden-Layer.
    
    ''' 
    

    def __init__(self, 
                 input_dim, 
                 output_dim,  
                 activation_function=AFct.SoftSign,
                 initial_weights='AUTO',
                 initial_bias=0.0,
                 initial_offset=0.5,
                 connections=None,
                 dtype=numx.float64):
        ''' This function initializes all necessary parameters and data structures.
            
        :Parameters:
            input_dim:            Number of input dimensions.
                                 -type: int
                                  
            output_dim            Number of output dimensions.
                                 -type: int
                                  
            activation_function:  Activation function.
                                 -type: pydeep.base.activationfunction

            initial_weights:      Initial weights.
                                  'AUTO' = .. seealso:: "Understanding the difficulty of training deep feedforward neural
                                           networks - X Glo, Y Bengio - 2015"
                                  scalar = sampling values from a zero mean Gaussian with std='scalar',
                                  numpy array  = pass an array, for example to implement tied weights.
                                 -type: 'AUTO', scalar or numpy array [input dim, output_dim]
                                  
            initial_bias:         Initial bias.
                                  scalar = all values will be set to 'initial_bias',
                                  numpy array  = pass an array
                                 -type: 'AUTO', scalar or numpy array [1, output dim]
                                  
            initial_offset:       Initial offset values.
                                  scalar = all values will be set to 'initial_offset',
                                  numpy array  = pass an array
                                 -type: 'AUTO', scalar or numpy array [1, input dim]

            connections:          Connection matrix containing 0 and 1 entries, where 0 connections disable the
                                  corresponding weight.
                                  Example: pydeep.base.numpyextension.generate_2D_connection_matrix() can be used to
                                  construct such a matrix.
                                 -type: numpy array [input dim, output_dim] or None
                        
            dtype:                Used data type i.e. numpy.float64
                                 -type: numpy.float32 or numpy.float64 or numpy.longdouble
            
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_function = activation_function
        self.connections = connections
        self.dtype = dtype 

        # Temp pre-synaptic output
        self.temp_z = None
        # Temp post-synaptic output
        self.temp_a = None
        # Temp input
        self.temp_x = None
        # Temp error/delta values
        self.temp_deltas = None

        # Initialize the weights
        if initial_weights is 'AUTO':
            # See" Understanding the difficulty of training deep feedforward neural networks - X Glo, Y Bengio - 2015"
            sig_factor = 1.0
            if activation_function == AFct.Sigmoid:
                sig_factor = 4.0
            self.weights = numx.array((2.0 * numx.random.rand(self.input_dim,self.output_dim) - 1.0)
                           * (sig_factor * numx.sqrt(6.0 / (self.input_dim + self.output_dim))), dtype=self.dtype)
        else:
            if numx.isscalar(initial_weights):
                self.weights = numx.array(numx.random.randn(self.input_dim, self.output_dim) * initial_weights,
                                          dtype=self.dtype)
            else:
                self.weights = initial_weights
                if self.weights.shape != (self.input_dim,self.output_dim):
                    raise Exception("Weight matrix dim. and input dim and output dim. have to match!")

        # Drop connections is connection matrix is given
        if connections is not None:
            self.weights *= self.connections

        # Initialize the bias
        if initial_bias is 'AUTO':
            self.bias = numx.array(numx.zeros((1,self.output_dim)), dtype=self.dtype)
        elif numx.isscalar(initial_bias):
            self.bias = numx.array(numx.zeros((1,self.output_dim)) + initial_bias, dtype=self.dtype)
        else:
            self.bias = numx.array(initial_bias, dtype=self.dtype)
            if initial_bias.shape != (1,self.output_dim):
                raise Exception("Bias dim. and output dim. have to match!")

        # Initialize the offset
        if initial_offset is 'AUTO':
            self.offset = numx.array(numx.zeros((1,self.input_dim)) + 0.5, dtype=self.dtype)
        elif numx.isscalar(initial_offset):
            self.offset = numx.array(numx.zeros((1,self.input_dim)) + initial_offset, dtype=self.dtype)
        else:
            self.offset = numx.array(initial_offset, dtype=self.dtype)
            if self.offset.shape != (1,self.input_dim):
                raise Exception("Offset dim. and input dim. have to match!")

    def clear_temp_data(self):
        ''' Sets all temp variables to None.

        '''
        # Temp pre-synaptic output
        self.temp_z = None
        # Temp post-synaptic output
        self.temp_a = None
        # Temp input
        self.temp_x = None
        # Temp error/delta values
        self.temp_deltas = None

    def get_parameters(self):
        ''' This function returns all model parameters in a list.

        :Returns:
            The parameter references in a list.
           -type: list

        '''
        return [self.weights, self.bias]

    def update_parameters(self, parameter_updates):
        ''' This function updates all parameters given the updates derived by the training methods.

        :Parameters:
            parameter_updates:  Parameter gradients.
                               -type: list of numpy arrays (num para. x [para.shape])

        '''
        for p, u in zip(self.get_parameters(), parameter_updates):
            p -= u

    def update_offsets(self, shift=1.0, new_mean = None):
        ''' This function updates the offsets.
            Example: update_offsets(1,0) reparameterizes to an uncentered model.

        :Parameters:

            shift:    Shifting factor for the offset shift.
                     -type: float

            new_mean: New mean value if None the activation from the last forward propagation ist used to calculate the
                      mean.
                     -type: float, numpy array or None

        '''
        if shift > 0.0:
            # Calculate data mean
            if new_mean is None:
                new_mean = numx.mean(self.temp_x, axis =0).reshape(1,self.input_dim)
            # Reparameterize the network to the new mean
            self.bias += shift * numx.dot(new_mean-self.offset, self.weights)
            # Exp. mov. avg. update
            self.offset *= (1.0-shift)
            self.offset += shift * new_mean

    def forward_propagate(self, x):
        ''' Forward-propagates the data through the network and stores pre-syn activation, post-syn activation and input
            mean internally.
        
        :Parameters:
            x:      Data
                   -type: numpy arrays [batchsize, input dim]

        :Returns: 
            Post activation
           -type: numpy arrays [batchsize, output dim] 
                               
        '''
        # Store data
        self.temp_x = x
        # Calculate pre-synaptic output
        self.temp_z = numx.dot(self.temp_x - self.offset,self.weights) + self.bias
        # Calculate post-synaptic output
        self.temp_a = self.activation_function.f(self.temp_z)
        return self.temp_a

    def _backward_propagate(self):
        ''' Back-propagates the error signal.

        :Returns:
            Backprop Signal, delta value for the layer below.
           -type: numpy arrays [batchsize, input dim]

        '''
        return numx.dot(self.temp_deltas, self.weights.T)

    def _calculate_gradient(self):
        ''' Calculates the gradient for the parameters,

        :Returns:
            The parameters gradient in a list.
           -type: list

        '''
        # Weight gradient
        gradW = numx.dot((self.temp_x-self.offset).T, self.temp_deltas)/self.temp_x.shape[0]
        # If connection matrix is given drop corresponding connections
        if self.connections is not None:
            gradW *= self.connections
        # Return weight gradient and bias gradient
        return [gradW, numx.mean(self.temp_deltas, axis=0).reshape(1,self.temp_deltas.shape[1])]

    def _get_deltas(self, deltas,
                          labels,
                          cost,
                          reg_cost,
                          desired_sparseness,
                          cost_sparseness,
                          reg_sparseness,
                          check_gradient=False):
        ''' Computes the delta value/ error terms for the layer.
        
        :Parameters:
            deltas:             Delta values from the layer above or None if top-layer.
                               -type: None or numpy arrays [batchsize, output dim]

            labels:             numpy array or None if the current layer has no cost.
                               -type: None or numpy arrays [batchsize, output dim]

            cost:               Cost function for the layer.
                               -type: pydeep.base.costfunction

            reg_cost:           Strength of the cost function.
                               -type: scalar

            desired_sparseness: Desired sparseness value/average hidden activity.
                               -type: scalar

            cost_sparseness:    Cost function for the sparseness regularization.
                               -type: pydeep.base.costfunction

            reg_sparseness:     Strength of the sparseness term.
                               -type: scalar

            check_gradient:     Flase for gradient checking mode.
                               -type: bool

        :Returns: 
            Delta values for the current layer.
           -type: numpy arrays [batchsize, output dim]
                               
        '''
        """
        # Optimization possible if all cost functions are NegLogLikelihood and the activation function is Softmax.
        optimized = True
        if deltas is not None:
            # If we have delta value we have to compute the Jacobian anyway so no optimization
            optimized = False
        else:
            deltas = 0.0

        # If sparseness regularization is enabled ...
        if reg_sparseness != 0.0:
            # ... compute the cost derivative
            deltas_sparse = cost_sparseness.df(numx.atleast_2d(numx.mean(self.temp_a,axis = 0)),desired_sparseness)
            deltas += reg_sparseness*deltas_sparse
            # Optimization only possible if NegLogLikelihood + Softmax
            optimized = False

        # If labels are provided ...
        if reg_cost != 0.0:
            # ... compute the cost derivative
            deltas_targets = cost.df(self.temp_a, labels)
            deltas += reg_cost*deltas_targets
            # Optimization only possible if NegLogLikelihood + Softmax
            if cost != CFct.NegLogLikelihood and cost != CFct.CrossEntropyError and not isinstance(cost, CFct.NegLogLikelihood) \
                    and not isinstance(cost, CFct.CrossEntropyError):
                optimized = False

        self.temp_deltas = None
        # The SoftMax function is a special case since the derivative of a variable depends on all others and thus the
        # derivative is a Jacobian. The delta values can thefore not be given by a elementwise product of top-deltas
        # and activation function derivative, instead an additional sum is involved.
        if ((self.activation_function == AFct.SoftMax or isinstance(self.activation_function,AFct.SoftMax)) and
            (cost == CFct.NegLogLikelihood or isinstance(cost, CFct.NegLogLikelihood)) or
            (self.activation_function == AFct.Sigmoid or isinstance(self.activation_function,AFct.Sigmoid)) and
            (cost == CFct.CrossEntropyError or isinstance(cost, CFct.CrossEntropyError))):
            # All costs are NegLogLikelihood and the activation function is Softmax use the optimized way
            if optimized:
                # Labels given ...
                if reg_cost != 0.0:
                    self.temp_deltas = reg_cost * (self.temp_a - labels)
                # Sparseness given
                if reg_sparseness != 0.0:
                    self.temp_deltas += reg_sparseness * (self.temp_a - desired_sparseness)
            else:
                # Calcuclate Jacobian
                J = AFct.SoftMax.df(self.temp_a)
                # For each partial derivative ...
                for i in range(self.temp_a.shape[0]):
                    # Compute the delta value
                    if deltas is not None:
                        temp = numx.dot(deltas[i].reshape(1,deltas.shape[1]),J[i])
                    else:
                        raise Exception("No cost was specified in top layer!")
                    # Stack results in matrix
                    if self.temp_deltas is None:
                        self.temp_deltas = temp
                    else:
                        self.temp_deltas = numx.vstack((self.temp_deltas,temp))
        else:
            # Independent function, simpler calculation
            self.temp_deltas = self.activation_function.df(self.temp_z)*deltas
        return self.temp_deltas
        """

        # Optimization possible if all cost functions are CrossEntropyError and the activation function is Softmax.
        optimized = False
        if deltas is None:
            deltas = 0.0
            if ((cost == CFct.CrossEntropyError or isinstance(cost, CFct.CrossEntropyError)) and
                (self.activation_function == AFct.SoftMax or isinstance(self.activation_function, AFct.SoftMax) or
                 self.activation_function == AFct.Sigmoid or isinstance(self.activation_function, AFct.Sigmoid))):
                optimized = True
                if check_gradient:
                    optimized = False

        # If sparseness regularization is enabled -> no optimization...
        if reg_sparseness != 0.0:
            # ... compute the cost derivative
            deltas_sparse = cost_sparseness.df(numx.atleast_2d(numx.mean(self.temp_a, axis=0)), desired_sparseness)
            deltas += reg_sparseness * deltas_sparse
            optimized = False

        self.temp_deltas = None
        # The SoftMax function is a special case since the derivative of a variable depends on all others and thus the
        # derivative is a Jacobian. The delta values can thefore not be given by a elementwise product of top-deltas
        # and activation function derivative, instead an additional sum is involved.
        if not optimized:

            # If labels are provided ...
            if reg_cost != 0.0:
                # ... compute the cost derivative
                deltas_targets = cost.df(self.temp_a, labels)
                deltas += reg_cost * deltas_targets

            if self.activation_function == AFct.SoftMax or isinstance(self.activation_function,AFct.SoftMax):
                # Calcuclate Jacobian
                J = AFct.SoftMax.df(self.temp_a)
                # For each partial derivative ...
                for i in range(self.temp_a.shape[0]):
                    # Compute the delta value
                    if deltas is not None:
                        temp = numx.dot(deltas[i].reshape(1, deltas.shape[1]), J[i])
                    else:
                        raise Exception("No cost was specified in top layer!")
                    # Stack results in matrix
                    if self.temp_deltas is None:
                        self.temp_deltas = temp
                    else:
                        self.temp_deltas = numx.vstack((self.temp_deltas, temp))
            else:
                # Independent function, simpler calculation
                self.temp_deltas = self.activation_function.df(self.temp_z) * deltas
        else:
            # Labels given ...
            if reg_cost != 0.0:
                self.temp_deltas = reg_cost * (self.temp_a - labels)

        return self.temp_deltas