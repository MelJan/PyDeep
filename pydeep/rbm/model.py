''' This module provides restricted Boltzmann machines (RBMs) with different
    types of units. The structure is very close to the mathematical derivations
    to simplify the understanding. In addition, the modularity helps to create
    other kind of RBMs without adapting the training algorithms.

    :Implemented:
        - centered BinaryBinary RBM (BB-RBM)

    :Info:
        For the derivations see:
        http://www.ini.rub.de/data/documents/tns/masterthesis_janmelchior.pdf

        A usual way to create a new unit is to inherit from a given RBM class
        and override the functions that changed, e.g. Gaussian-Binary RBM
        inherited from the Binary-Binary RBM.

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
import pydeep.base.activationfunction as Act
from pydeep.base.basicstructure import BipartiteGraph

class BinaryBinaryRBM(BipartiteGraph):
    ''' Implementation of a centered restricted Boltzmann machine with binary
        visible and binary hidden units.

    '''

    def __init__(self,
                 number_visibles,
                 number_hiddens,
                 data = None,
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
            number_visibles:         Number of the visible variables.
                                    -type: int

            number_hiddens           Number of hidden variables.
                                    -type: int

            data:                    The training data for parameter
                                     initialization if 'AUTO' is chosen
                                     for the corresponding parameter.
                                    -type: None or
                                           numpy array [num samples, input dim]

            initial_weights:         Initial weights.
                                     'AUTO' and a scalar are random init.
                                    -type: 'AUTO', scalar or
                                          numpy array [input dim, output_dim]

            initial_visible_bias:    Initial visible bias.
                                     'AUTO' is random, 'INVERSE_SIGMOID' is the
                                     inverse Sigmoid of the visilbe mean.
                                     If a scalar is passed all values are
                                     initialized with it.
                                    -type: 'AUTO','INVERSE_SIGMOID', scalar or
                                          numpy array [1, input dim]

            initial_hidden_bias:     Initial hidden bias.
                                     'AUTO' is random, 'INVERSE_SIGMOID' is the
                                     inverse Sigmoid of the hidden mean.
                                     If a scalar is passed all values are
                                     initialized with it.
                                    -type: 'AUTO','INVERSE_SIGMOID', scalar or
                                          numpy array [1, output_dim]

            initial_visible_offsets: Initial visible offset values.
                                     AUTO=data mean or 0.5 if no data is given.
                                     If a scalar is passed all values are
                                     initialized with it.
                                    -type: 'AUTO', scalar or
                                           numpy array [1, input dim]

            initial_hidden_offsets:  Initial hidden offset values.
                                     AUTO = 0.5
                                     If a scalar is passed all values are
                                     initialized with it.
                                    -type: 'AUTO', scalar or
                                           numpy array [1, output_dim]

            dtype:                   Used data type i.e. numpy.float64
                                    -type: numpy.float32 or numpy.float64 or
                                           numpy.float128

        '''
        # Call constructor of superclass
        super(BinaryBinaryRBM,
              self).__init__(number_visibles = number_visibles,
                             number_hiddens = number_hiddens,
                             data = data,
                             visible_activation_function = Act.Sigmoid,
                             hidden_activation_function = Act.Sigmoid,
                             initial_weights = initial_weights,
                             initial_visible_bias = initial_visible_bias,
                             initial_hidden_bias = initial_hidden_bias,
                             initial_visible_offsets = initial_visible_offsets,
                             initial_hidden_offsets = initial_hidden_offsets,
                             dtype = dtype)

        self.bv_base = self._getbasebias()
        self._fast_PT = True

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

            initial_offsets:     The initial visible offset values.
                             -type: 'AUTO' or scalar or
                                     numpy array [1, num_new_visibles]

        '''
        super(BinaryBinaryRBM, self)._add_visible_units(num_new_visibles,
                                                        position,
                                                        initial_weights,
                                                        initial_bias,
                                                        initial_offsets,
                                                        data)
        self.bv_base = self._getbasebias()

    def _remove_visible_units(self, indices):
        ''' This function removes the visible units whose indices are given.
            Warning: If the parameters are changed. the trainer needs to be
                     reinitialized.

        :Parameters:
            indices: Indices of units to be remove.
                    -type: int or list of int or numpy array of int

        '''
        super(BinaryBinaryRBM, self)._remove_visible_units(indices)
        self.bv_base = numx.delete(self.bv_base, numx.array(indices), axis=1)

    def _calculate_weight_gradient(self, v, h):
        ''' This function calculates the gradient for the weights from the
            visible and hidden activations.

        :Parameters:
            v: Visible activations.
              -type: numpy arrays [batchsize, input dim]

            h: Hidden activations.
              -type: numpy arrays [batchsize, output dim]

        :Returns:
            Weight gradient.
           -type: numpy arrays [input dim, output dim]

        '''
        return numx.dot((v - self.ov).T , h - self.oh)

    def _calculate_visible_bias_gradient(self, v):
        ''' This function calculates the gradient for the visible biases.

        :Parameters:
            v: Visible activations.
              -type: numpy arrays [batch_size, input dim]

        :Returns:
            Visible bias gradient.
           -type: numpy arrays [1, input dim]

        '''
        return numx.sum(v - self.ov, axis=0).reshape(1,v.shape[1])

    def _calculate_hidden_bias_gradient(self, h):
        ''' This function calculates the gradient for the hidden biases.

        :Parameters:
            h:  Hidden activations.
               -type: numpy arrays [batch size, output dim]

        :Returns:
            Hidden bias gradient.
           -type: numpy arrays [1, output dim]

        '''
        return numx.sum(h - self.oh, axis=0).reshape(1, h.shape[1])

    def calculate_gradients(self, v, h):
        ''' This function calculates all gradients of this RBM and returns
            them as a list of arrays. This keeps the flexibility of adding
            parameters which will be updated by the training algorithms.

        :Parameters:
            v: Visible activations.
              -type: numpy arrays [batch size, output dim]

            h: Hidden activations.
              -type: numpy arrays [batch size, output dim]

        :Returns:
            Gradients for all parameters.
           -type: list of numpy arrays (num parameters x [parameter.shape])

        '''
        return [self._calculate_weight_gradient(v, h)
                ,self._calculate_visible_bias_gradient(v)
                ,self._calculate_hidden_bias_gradient(h)]

    def sample_v(self, v, beta = None, use_base_model = False):
        ''' Samples the visible variables from the
            conditional probabilities v given h.

        :Parameters:
            v:    Conditional probabilities of v given h.
                 -type: numpy array [batch size, input dim]

            beta: DUMMY Variable
                  The sampling in other types of units like Gaussian-Binary
                  RBMs will be affected by beta.
                 -type: None


            use_base_model: If true uses the base model, i.e. the MLE of
                            the bias values. (DUMMY in this case)
                           -type: bool

        :Returns:
            States for v.
           -type: numpy array [batch size, input dim]

        '''
        return self.dtype(v > numx.random.random(v.shape))

    def sample_h(self, h, beta = None, use_base_model = False):
        ''' Samples the hidden variables from the
            conditional probabilities h given v.

        :Parameters:
            h:    Conditional probabilities of h given v.
                 -type: numpy array [batch size, output dim]

            beta: DUMMY Variable
                  The sampling in other types of units like Gaussian-Binary
                  RBMs will be affected by beta. (DUMMY in this case)
                 -type: None

            use_base_model: If true uses the base model, i.e. the MLE of
                            the bias values.
                           -type: bool

        :Returns:
            States for h.
           -type: numpy array [batch size, output dim]

        '''
        return self.dtype(h> numx.random.random(h.shape))

    def probability_v_given_h(self, h, beta = None, use_base_model = False):
        ''' Calculates the conditional probabilities of v given h.

        :Parameters:
            h:              Hidden states.
                           -type: numpy array [batch size, output dim]

            beta:           Allows to sample from a given inverse temperature
                            beta, or if a vector is given to sample from
                            different betas simultaneously.
                            None is equivalent to pass the value 1.0
                           -type: None, float or numpy array [batch size, 1]

            use_base_model: If true uses the base model, i.e. the MLE of
                            the bias values.
                           -type: bool

        :Returns:
            Conditional probabilities v given h.
           -type: numpy array [batch size, input dim]

        '''
        activation = self._visible_pre_activation(h)
        if beta is not None:
            activation *= beta
            if use_base_model is True:
                activation += (1.0-beta)*self.bv_base
        return self._visible_post_activation(activation)

    def probability_h_given_v(self, v, beta = None, use_base_model = False):
        ''' Calculates the conditional probabilities of h given v.

        :Parameters:
            v:              Visible states.
                           -type: numpy array [batch size, input dim]

            beta:           Allows to sample from a given inverse temperature
                            beta, or if a vector is given to sample from
                            different betas simultaneously.
                            None is equivalent to pass the value 1.0
                           -type: None, float or numpy array [batch size, 1]

            use_base_model: DUMMY variable since we do not use a base hidden
                            bias.

        :Returns:
            Conditional probabilities h given v.
           -type: numpy array [batch size, output dim]

        '''
        activation = self._hidden_pre_activation(v)
        if beta is not None:
            activation *= beta
        return self._hidden_post_activation(activation)

    def energy(self, v, h, beta = None, use_base_model = False):
        ''' Compute the energy of the RBM given observed variable states v
            and hidden variables state h.

        :Parameters:
            v:              Visible states.
                           -type: numpy array [batch size, input dim]

            h:              Hidden states.
                           -type: numpy array [batch size, output dim]

            beta:           Allows to sample from a given inverse temperature
                            beta, or if a vector is given to sample from
                            different betas simultaneously.
                            None is equivalent to pass the value 1.0
                           -type: None, float or numpy array [batch size, 1]

            use_base_model: If true uses the base model, i.e. the MLE of
                            the bias values.
                           -type: bool

        :Returns:
            Energy of v and h.
           -type: numpy array [batch size,1]

        '''
        temp_v = v-self.ov
        temp_h = h-self.oh
        energy = (- numx.dot(temp_v, self.bv.T)
                  - numx.dot(temp_h, self.bh.T)
                  - numx.sum(numx.dot(temp_v, self.w)
                  * temp_h,axis=1).reshape(v.shape[0], 1))
        if beta is not None:
            energy *= beta
            if use_base_model is True:
                energy -= (1.0-beta)*numx.dot(temp_v, self.bv_base.T)
        return energy

    def unnormalized_log_probability_v(self,
                                       v,
                                       beta = None,
                                       use_base_model = False):
        ''' Computes the unnormalized log probabilities of v.

        :Parameters:
            v:              Visible states.
                           -type: numpy array [batch size, input dim]

            beta:           Allows to sample from a given inverse temperature
                            beta, or if a vector is given to sample from
                            different betas simultaneously.
                            None is equivalent to pass the value 1.0
                           -type: None, float or numpy array [batch size, 1]

            use_base_model: If true uses the base model, i.e. the MLE of
                            the bias values.
                           -type: bool

        :Returns:
            Unnormalized log probability of v.
           -type: numpy array [batch size, 1]

        '''
        temp_v = v - self.ov
        activation = numx.dot(temp_v, self.w) + self.bh
        bias = numx.dot(temp_v, self.bv.T).reshape(temp_v.shape[0], 1)
        if beta is not None:
            activation *= beta
            bias *= beta
            if use_base_model is True:
                bias += (1.0-beta)*numx.dot(temp_v, self.bv_base.T
                                            ).reshape(temp_v.shape[0], 1)
        return bias + numx.sum(
                               numx.log(
                                        numx.exp(activation*(1.0 - self.oh))
                                      + numx.exp(-activation*self.oh)
                                        )
                               , axis=1).reshape(v.shape[0], 1)

    def unnormalized_log_probability_h(self,
                                       h,
                                       beta=None,
                                       use_base_model = False):
        ''' Computes the unnormalized log probabilities of h.

        :Parameters:
            h:              Hidden states.
                           -type: numpy array [batch size, input dim]

            beta:           Allows to sample from a given inverse temperature
                            beta, or if a vector is given to sample from
                            different betas simultaneously.
                            None is equivalent to pass the value 1.0
                           -type: None, float or numpy array [batch size, 1]

            use_base_model: If true uses the base model, i.e. the MLE of
                            the bias values.
                           -type: bool

        :Returns:
            Unnormalized log probability of h.
           -type: numpy array [batch size, 1]

        '''
        temp_h = h - self.oh
        activation = numx.dot(temp_h, self.w.T) + self.bv
        bias = numx.dot(temp_h, (self.bh).T).reshape(h.shape[0], 1)
        if beta is not None:
            activation *= beta
            bias *= beta
            if use_base_model is True:
                activation += (1.0-beta)*self.bv_base
        return bias + numx.sum(
                               numx.log(
                                        numx.exp(activation*(1.0 - self.ov))
                                      + numx.exp(-activation*self.ov))
                               , axis=1).reshape(h.shape[0], 1)

    def log_probability_v(self, logZ, v, beta = None, use_base_model = False):
        ''' Computes the log-probability / LogLikelihood(LL) for the given
            visible units for this model.
            To estimate the LL we need to know the logarithm of the partition
            function Z. For small models it is possible to calculate Z,
            however since this involves calculating all possible hidden
            states, it is intractable for bigger models. As an estimation
            method annealed importance sampling (AIS) can be used instead.

        :Parameters:
            logZ:           The logarithm of the partition function.
                           -type: float

            v:              Visible states.
                           -type: numpy array [batch size, input dim]

            beta:           Allows to sample from a given inverse temperature
                            beta, or if a vector is given to sample from
                            different betas simultaneously.
                            None is equivalent to pass the value 1.0
                           -type: None, float or numpy array [batch size, 1]

            use_base_model: If true uses the base model, i.e. the MLE of
                            the bias values.
                           -type: bool

        :Returns:
            Log probability for visible_states.
           -type: numpy array [batch size, 1]

        '''
        return self.unnormalized_log_probability_v(v, beta,
                                                   use_base_model) - logZ

    def log_probability_h(self, logZ,  h, beta = None, use_base_model = False):
        ''' Computes the log-probability / LogLikelihood(LL) for the given
            hidden units for this model.
            To estimate the LL we need to know the logarithm of the partition
            function Z. For small models it is possible to calculate Z,
            however since this involves calculating all possible hidden
            states, it is intractable for bigger models. As an estimation
            method annealed importance sampling (AIS) can be used instead.

        :Parameters:
            logZ:           The logarithm of the partition function.
                           -type: float

            h:              Hidden states.
                           -type: numpy array [batch size, output dim]

            beta:           Allows to sample from a given inverse temperature
                            beta, or if a vector is given to sample from
                            different betas simultaneously.
                            None is equivalent to pass the value 1.0
                           -type: None, float or numpy array [batch size, 1]

            use_base_model: If true uses the base model, i.e. the MLE of
                            the bias values.
                           -type: bool

        :Returns:
            Log probability for hidden_states.
           -type: numpy array [batch size, 1]

        '''
        return self.unnormalized_log_probability_h(h, beta,
                                                   use_base_model) - logZ

    def log_probability_v_h(self,
                            logZ,
                            v,
                            h,
                            beta = None,
                            use_base_model = False):
        ''' Computes the joint log-probability / LogLikelihood(LL) for the
            given visible and hidden units for this model.
            To estimate the LL we need to know the logarithm of the partition
            function Z. For small models it is possible to calculate Z,
            however since this involves calculating all possible hidden
            states, it is intractable for bigger models. As an estimation
            method annealed importance sampling (AIS) can be used instead.

        :Parameters:
            logZ:           The logarithm of the partition function.
                           -type: float

            v:              Visible states.
                           -type: numpy array [batch size, input dim]

            h:              Hidden states.
                           -type: numpy array [batch size, output dim]

            beta:           Allows to sample from a given inverse temperature
                            beta, or if a vector is given to sample from
                            different betas simultaneously.
                            None is equivalent to pass the value 1.0
                           -type: None, float or numpy array [batch size, 1]

            use_base_model: If true uses the base model, i.e. the MLE of
                            the bias values.
                           -type: bool
        :Returns:
            Joint log probability for v and h.
           -type: numpy array [batch size, 1]

        '''
        return -self.energy(v, h, beta, use_base_model) - logZ

    def _base_log_partition(self, use_base_model = False):
        ''' Returns the base partition function for a given visible bias.
            Note that for AIS we need to be able to calculate the partition
            function of the base distribution exactly. Furthermore it is
            beneficial if the base distribution is a good approximation of
            the target distribution. A good choice is therefore the maximum
            likelihood estimate of the visible bias, given the data.

        :Parameters:
            use_base_model: If true uses the base model, i.e. the MLE of
                            the bias values.
                           -type: bool

        :Returns:
            Partition function for zero parameters.
           -type: float

        '''
        if use_base_model is True:
            return (numx.sum(numx.log(numx.exp(-self.ov*self.bv_base)
                                      + numx.exp((1.0-self.ov)*self.bv_base)))
                            +(self.output_dim)*numx.log(2.0))
        else:
            return ((self.input_dim)*numx.log(2.0)
                    +(self.output_dim)*numx.log(2.0))

    def _getbasebias(self):
        ''' Returns the maximum likelihood estimate of the visible bias,
            given the data. If no data is given the RBMs bias value is return,
            but is highly rcommended to pass the data.

        :Returns:
            Base bias
           -type: numpy array [1,  input dim]

        '''
        save_mean = numx.clip(self.dtype(self._data_mean)
                        , 0.00001,0.99999
                        ).reshape(1,self._data_mean.shape[1])
        return numx.log( save_mean) - numx.log(1.0-save_mean)