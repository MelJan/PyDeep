""" This module provides restricted Boltzmann machines (RBMs) with different types of units. The structure is very
    close to the mathematical derivations to simplify the understanding. In addition, the modularity helps to create
    other kind of RBMs without adapting the training algorithms.

    :Implemented:
        - centered BinaryBinary RBM (BB-RBM)
        - centered GaussianBinary RBM (GB-RBM) with fixed variance
        - centered GaussianBinaryVariance RBM (GB-RBM) with trainable variance

        # Models without implementation of p(v),p(h),p(v,h) -> AIS, PT, true gradient, ... cannot be used!
        - centered BinaryBinaryLabel RBM (BBL-RBM)
        - centered GaussianBinaryLabel RBM (GBL-RBM)

        # Models with intractable p(v),p(h),p(v,h) -> AIS, PT, true gradient, ... cannot be used!
        - centered BinaryRect RBM (BR-RBM)
        - centered RectBinary RBM (RB-RBM)
        - centered RectRect RBM (RR-RBM)
        - centered GaussianRect RBM (GR-RBM)
        - centered GaussianRectVariance RBM (GRV-RBM)

    :Info:
        For the derivations .. seealso::
        https://www.ini.rub.de/PEOPLE/wiskott/Reprints/Melchior-2012-MasterThesis-RBMs.pdf

        A usual way to create a new unit is to inherit from a given RBM class
        and override the functions that changed, e.g. Gaussian-Binary RBM
        inherited from the Binary-Binary RBM.

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
from pydeep.base.activationfunction import Sigmoid, SoftMax, SoftPlus
from pydeep.base.basicstructure import BipartiteGraph
from pydeep.base.numpyextension import multinominal_batch_sampling


class BinaryBinaryRBM(BipartiteGraph):
    """ Implementation of a centered restricted Boltzmann machine with binary visible and binary hidden units.

    """

    def __init__(self,
                 number_visibles,
                 number_hiddens,
                 data=None,
                 initial_weights='AUTO',
                 initial_visible_bias='AUTO',
                 initial_hidden_bias='AUTO',
                 initial_visible_offsets='AUTO',
                 initial_hidden_offsets='AUTO',
                 dtype=numx.float64):
        """ This function initializes all necessary parameters and data  structures. It is recommended to pass the \
         training data to initialize the network automatically.

        :param number_visibles: Number of the visible variables.
        :type number_visibles: int

        :param number_hiddens: Number of hidden variables.
        :type number_hiddens: int

        :param data: The training data for parameter initialization if 'AUTO' is chosen for the corresponding parameter.
        :type data: None or numpy array [num samples, input dim]

        :param initial_weights: Initial weights. 'AUTO' and a scalar are random init.
        :type initial_weights: 'AUTO', scalar or numpy array [input dim, output_dim]

        :param initial_visible_bias: Initial visible bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid \
                                     of the visilbe mean. If a scalar is passed all values are initialized with it.
        :type initial_visible_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, input dim]

        :param initial_hidden_bias: Initial hidden bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid of \
                                    the hidden mean. If a scalar is passed all values are initialized with it.
        :type initial_hidden_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, output_dim]

        :param initial_visible_offsets: Initial visible offset values. AUTO=data mean or 0.5 if no data is given. If a \
                                        scalar is passed all values are initialized with it.
        :type initial_visible_offsets: 'AUTO', scalar or numpy array [1, input dim]

        :param initial_hidden_offsets: Initial hidden offset values. AUTO = 0.5 If a scalar is passed all values are \
                                       initialized with it.
        :type initial_hidden_offsets: 'AUTO', scalar or numpy array [1, output_dim]

        :param dtype: Used data type i.e. numpy.float64
        :type dtype: numpy.float32 or numpy.float64 or numpy.longdouble
        """
        # Call constructor of superclass
        super(BinaryBinaryRBM,
              self).__init__(number_visibles=number_visibles,
                             number_hiddens=number_hiddens,
                             data=data,
                             visible_activation_function=Sigmoid,
                             hidden_activation_function=Sigmoid,
                             initial_weights=initial_weights,
                             initial_visible_bias=initial_visible_bias,
                             initial_hidden_bias=initial_hidden_bias,
                             initial_visible_offsets=initial_visible_offsets,
                             initial_hidden_offsets=initial_hidden_offsets,
                             dtype=dtype)

        self.bv_base = self._getbasebias()
        self._fast_PT = True

    def _add_visible_units(self,
                           num_new_visibles,
                           position=0,
                           initial_weights='AUTO',
                           initial_bias='AUTO',
                           initial_offsets='AUTO',
                           data=None):
        """ This function adds new visible units at the given position to the model. \
            .. Warning:: If the parameters are changed. the trainer needs to be
                     reinitialized.

        :param num_new_visibles: The number of new hidden units to add
        :type num_new_visibles: int

        :param position: Position where the units should be added.
        :type position: int

        :param initial_weights: Initial weights. 'AUTO' and a scalar are random init.
        :type initial_weights: 'AUTO', scalar or numpy array [input num_new_visibles, output_dim]

        :param initial_bias: Initial visible bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid of \
                                     the visilbe mean. If a scalar is passed all values are initialized with it.
        :type initial_bias: 'AUTO' or scalar or numpy array [1, num_new_visibles]

        :param initial_offsets: The initial visible offset values.
        :type initial_offsets: 'AUTO' or scalar or numpy array [1, num_new_visibles]

        :param data: If data is given and the offset and bias is initzialized accordingly, if 'AUTO' is chosen.
        :type data: numpy array [num datapoints, num_new_visibles]
        """
        super(BinaryBinaryRBM, self)._add_visible_units(num_new_visibles,
                                                        position,
                                                        initial_weights,
                                                        initial_bias,
                                                        initial_offsets,
                                                        data)
        self.bv_base = self._getbasebias()

    def _remove_visible_units(self, indices):
        """ This function removes the visible units whose indices are given.
            .. Warning:: If the parameters are changed. the trainer needs to be
                     reinitialized.

        :param indices: Indices of units to be remove.
        :type indices: int or list of int or numpy array of int
        """
        super(BinaryBinaryRBM, self)._remove_visible_units(indices)
        self.bv_base = numx.delete(self.bv_base, numx.array(indices), axis=1)

    def _calculate_weight_gradient(self, v, h):
        """ This function calculates the gradient for the weights from the visible and hidden activations.

        :param v: Visible activations.
        :type v: numpy arrays [batchsize, input dim]

        :param h: Hidden activations.
        :type h: numpy arrays [batchsize, output dim]

        :return: Weight gradient.
        :rtype: numpy arrays [input dim, output dim]
        """
        return numx.dot((v - self.ov).T, h - self.oh)

    def _calculate_visible_bias_gradient(self, v):
        """ This function calculates the gradient for the visible biases.

        :param v: Visible activations.
        :type v: numpy arrays [batch_size, input dim]

        :return: Visible bias gradient.
        :rtype: numpy arrays [1, input dim]
        """
        return numx.sum(v - self.ov, axis=0).reshape(1, v.shape[1])

    def _calculate_hidden_bias_gradient(self, h):
        """ This function calculates the gradient for the hidden biases.

        :param h: Hidden activations.
        :type h: numpy arrays [batch size, output dim]

        :return: Hidden bias gradient.
        :rtype: numpy arrays [1, output dim]
        """
        return numx.sum(h - self.oh, axis=0).reshape(1, h.shape[1])

    def calculate_gradients(self, v, h):
        """ This function calculates all gradients of this RBM and returns them as a list of arrays. This keeps the \
            flexibility of adding parameters which will be updated by the training algorithms.

        :param v: Visible activations.
        :type v: numpy arrays [batch size, output dim]

        :param h: Hidden activations.
        :type h: numpy arrays [batch size, output dim]

        :return: Gradients for all parameters.
        :rtype: list of numpy arrays (num parameters x [parameter.shape])
        """
        return [self._calculate_weight_gradient(v, h), self._calculate_visible_bias_gradient(v),
                self._calculate_hidden_bias_gradient(h)]

    def sample_v(self, v, beta=None, use_base_model=False):
        """ Samples the visible variables from the conditional probabilities v given h.

        :param v: Conditional probabilities of v given h.
        :type v: numpy array [batch size, input dim]

        :param beta: DUMMY Variable. \
                     The sampling in other types of units like Gaussian-Binary RBMs will be affected by beta.
        :type beta: None

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values. (DUMMY in this case)
        :type use_base_model: bool

        :return: States for v.
        :rtype: numpy array [batch size, input dim]
        """
        return self.dtype(v > numx.random.random(v.shape))

    def sample_h(self, h, beta=None, use_base_model=False):
        """ Samples the hidden variables from the conditional probabilities h given v.

        :param h: Conditional probabilities of h given v.
        :type h: numpy array [batch size, output dim]

        :param beta: DUMMY Variable. \
                     The sampling in other types of units like Gaussian-Binary RBMs will be affected by beta.
        :type beta: None

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values. (DUMMY in this case)
        :type use_base_model: bool

        :return: States for h.
        :rtype: numpy array [batch size, output dim]
        """
        return self.dtype(h > numx.random.random(h.shape))

    def probability_v_given_h(self, h, beta=None, use_base_model=False):
        """ Calculates the conditional probabilities of v given h.

        :param h: Hidden states.
        :type h: numpy array [batch size, output dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from  \
                     different betas simultaneously. None is equivalent to pass the value 1.0
        :type beta: None, float or numpy array [batch size, 1]

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
        :type use_base_model: bool

        :return: Conditional probabilities v given h.
        :rtype: numpy array [batch size, input d
        """
        activation = self._visible_pre_activation(h)
        if beta is not None:
            activation *= beta
            if use_base_model is True:
                activation += (1.0 - beta) * self.bv_base
        return self._visible_post_activation(activation)

    def probability_h_given_v(self, v, beta=None, use_base_model=False):
        """ Calculates the conditional probabilities of h given v.

        :param v: Visible states.
        :type v: numpy array [batch size, input dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously. None is equivalent to pass the value 1.0
        :type beta: None, float or numpy array [batch size, 1]

        :param use_base_model: DUMMY variable, since we do not use a base hidden bias.
        :type use_base_model: bool

        :return: Conditional probabilities h given v.
        :rtype: numpy array [batch size, output dim]
        """
        activation = self._hidden_pre_activation(v)
        if beta is not None:
            activation *= beta
        return self._hidden_post_activation(activation)

    def energy(self, v, h, beta=None, use_base_model=False):
        """ Compute the energy of the RBM given observed variable states v and hidden variables state h.

        :param v: Visible states.
        :type v: numpy array [batch size, input dim]

        :param h: Hidden states.
        :type h: numpy array [batch size, output dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously. None is equivalent to pass the value 1.0
        :type beta: None, float or numpy array [batch size, 1]

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
        :type use_base_model: bool

        :return: Energy of v and h.
        :rtype: numpy array [batch size,1]
        """
        temp_v = v - self.ov
        temp_h = h - self.oh
        energy = - numx.dot(temp_v, self.bv.T) - numx.dot(temp_h, self.bh.T) - numx.sum(
            numx.dot(temp_v, self.w) * temp_h, axis=1).reshape(v.shape[0], 1)
        if beta is not None:
            energy *= beta
            if use_base_model is True:
                energy -= (1.0 - beta) * numx.dot(temp_v, self.bv_base.T)
        return energy

    def unnormalized_log_probability_v(self,
                                       v,
                                       beta=None,
                                       use_base_model=False):
        """ Computes the unnormalized log probabilities of v.

        :param v: Visible states.
        :type v: numpy array [batch size, input dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously.None is equivalent to pass the value 1.0.

        :type beta: None, float or numpy array [batch size, 1]

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
        :type use_base_model: bool

        :return: Unnormalized log probability of v.
        :rtype: numpy array [batch size, 1]
        """
        temp_v = v - self.ov
        activation = numx.dot(temp_v, self.w) + self.bh
        bias = numx.dot(temp_v, self.bv.T).reshape(temp_v.shape[0], 1)
        if beta is not None:
            activation *= beta
            bias *= beta
            if use_base_model is True:
                bias += (1.0 - beta) * numx.dot(temp_v, self.bv_base.T).reshape(temp_v.shape[0], 1)
        return bias + numx.sum(numx.log(numx.exp(activation * (1.0 - self.oh)) + numx.exp(
            -activation * self.oh)), axis=1).reshape(v.shape[0], 1)

    def unnormalized_log_probability_h(self,
                                       h,
                                       beta=None,
                                       use_base_model=False):
        """ Computes the unnormalized log probabilities of h.

        :param h: Hidden states.
        :type h: numpy array [batch size, output dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously.None is equivalent to pass the value 1.0.
        :type beta: None, float or numpy array [batch size, 1]

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
        :type use_base_model: bool

        :return: Unnormalized log probability of h.
        :rtype: numpy array [batch size, 1]
        """
        temp_h = h - self.oh
        activation = numx.dot(temp_h, self.w.T) + self.bv
        bias = numx.dot(temp_h, self.bh.T).reshape(h.shape[0], 1)
        if beta is not None:
            activation *= beta
            bias *= beta
            if use_base_model is True:
                activation += (1.0 - beta) * self.bv_base
        return bias + numx.sum(numx.log(numx.exp(activation * (1.0 - self.ov)) +
                                        numx.exp(-activation * self.ov)), axis=1).reshape(h.shape[0], 1)

    def log_probability_v(self, logz, v, beta=None, use_base_model=False):
        """ Computes the log-probability / LogLikelihood(LL) for the given visible units for this model. To estimate \
            the LL we need to know the logarithm of the partition function Z. For small models it is possible to \
            calculate Z, however since this involves calculating all possible hidden states, it is intractable for \
            bigger models. As an estimation method annealed importance sampling (AIS) can be used instead.

        :param logz: The logarithm of the partition function.
        :type logz: float

        :param v: Visible states.
        :type v: numpy array [batch size, input dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously.None is equivalent to pass the value 1.0.
        :type beta: None, float or numpy array [batch size, 1]

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
        :type use_base_model: bool

        :return: Log probability for visible_states.
        :rtype: numpy array [batch size, 1]
        """
        return self.unnormalized_log_probability_v(v, beta, use_base_model) - logz

    def log_probability_h(self, logz, h, beta=None, use_base_model=False):
        """ Computes the log-probability / LogLikelihood(LL) for the given hidden units for this model. To estimate \
            the LL we need to know the logarithm of the partition function Z. For small models it is possible to \
            calculate Z, however since this involves calculating all possible hidden states, it is intractable for \
            bigger models. As an estimation method annealed importance sampling (AIS) can be used instead.

        :param logz: The logarithm of the partition function.
        :type logz: float

        :param h: Hidden states.
        :type h: numpy array [batch size, output dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously. None is equivalent to pass the value 1.0
        :type beta: None, float or numpy array [batch size, 1]

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
        :type use_base_model: bool

        :return: Log probability for hidden_states.
        :rtype: numpy array [batch size, 1]
        """
        return self.unnormalized_log_probability_h(h, beta, use_base_model) - logz

    def log_probability_v_h(self,
                            logz,
                            v,
                            h,
                            beta=None,
                            use_base_model=False):
        """ Computes the joint log-probability / LogLikelihood(LL) for the given visible and hidden units for this \
            model. To estimate the LL we need to know the logarithm of the partition function Z. For small models it \
            is possible to calculate Z, however since this involves calculating all possible hidden states, it is \
            intractable for bigger models. As an estimation method annealed importance sampling (AIS) can be used \
            instead.

        :param logz: The logarithm of the partition function.
        :type logz: float

        :param v: Visible states.
        :type v: numpy array [batch size, input dim]

        :param h: Hidden states.
        :type h: numpy array [batch size, output dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously. None is equivalent to pass the value 1.0
        :type beta: None, float or numpy array [batch size, 1]

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
        :type use_base_model: bool

        :return: Joint log probability for v and h.
        :rtype: numpy array [batch size, 1]
        """
        return -self.energy(v, h, beta, use_base_model) - logz

    def _base_log_partition(self, use_base_model=False):
        """ Returns the base partition function for a given visible bias. .. Note:: that for AIS we need to be able to \
            calculate the partition function of the base distribution exactly. Furthermore it is beneficial if the \
            base distribution is a good approximation of the target distribution. A good choice is therefore the \
            maximum likelihood estimate of the visible bias, given the data.

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
        :type use_base_model: bool

        :return: Partition function for zero parameters.
        :rtype: float
        """
        if use_base_model is True:
            return numx.sum(numx.log(numx.exp(-self.ov * self.bv_base) + numx.exp((1.0 - self.ov) * self.bv_base))
                            ) + self.output_dim * numx.log(2.0)
        else:
            return self.input_dim * numx.log(2.0) + self.output_dim * numx.log(2.0)

    def _getbasebias(self):
        """ Returns the maximum likelihood estimate of the visible bias, given the data. If no data is given the RBMs \
            bias value is return, but is highly recommended to pass the data.

        :return: Base bias.
        :rtype: numpy array [1,  input dim]
        """
        save_mean = numx.clip(self.dtype(self._data_mean), 0.00001, 0.99999).reshape(1, self._data_mean.shape[1])
        return numx.log(save_mean) - numx.log(1.0 - save_mean)


class GaussianBinaryRBM(BinaryBinaryRBM):
    """ Implementation of a centered Restricted Boltzmann machine with Gaussian visible and binary hidden units.

    """

    def __init__(self,
                 number_visibles,
                 number_hiddens,
                 data=None,
                 initial_weights='AUTO',
                 initial_visible_bias='AUTO',
                 initial_hidden_bias='AUTO',
                 initial_sigma='AUTO',
                 initial_visible_offsets='AUTO',
                 initial_hidden_offsets='AUTO',
                 dtype=numx.float64):
        """ This function initializes all necessary parameters and data structures. It is recommended to pass the \
            training data to initialize the network automatically.

        :param number_visibles: Number of the visible variables.
        :type number_visibles: int

        :param number_hiddens: Number of hidden variables.
        :type number_hiddens: int

        :param data: The training data for parameter initialization if 'AUTO' is chosen for the corresponding parameter.
        :type data: None or numpy array [num samples, input dim]

        :param initial_weights: Initial weights. 'AUTO' and a scalar are random init.
        :type initial_weights: 'AUTO', scalar or numpy array [input dim, output_dim]

        :param initial_visible_bias: Initial visible bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid \
                                     of the visilbe mean. If a scalar is passed all values are initialized with it.
        :type initial_visible_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, input dim]

        :param initial_hidden_bias: Initial hidden bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid of \
                                    the hidden mean. If a scalar is passed all values are initialized with it.
        :type initial_hidden_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, output_dim]

        :param initial_sigma: Initial standard deviation for the model.
        :type initial_sigma: 'AUTO', scalar or numpy array [1, input_dim]

        :param initial_visible_offsets: Initial visible offset values. AUTO=data mean or 0.5 if no data is given. If a \
                                        scalar is passed all values are initialized with it.
        :type initial_visible_offsets: 'AUTO', scalar or numpy array [1, input dim]

        :param initial_hidden_offsets: Initial hidden offset values. AUTO = 0.5 If a scalar is passed all values are \
                                       initialized with it.
        :type initial_hidden_offsets: 'AUTO', scalar or numpy array [1, output_dim]

        :param dtype: Used data type i.e. numpy.float64
        :type dtype: numpy.float32 or numpy.float64 or numpy.longdouble
        """
        if initial_visible_bias is 'AUTO' or initial_visible_bias is 'INVERSE_SIGMOID':
            if data is not None:
                initial_visible_bias = numx.mean(data, axis=0).reshape(1, data.shape[1])
            else:
                initial_visible_bias = 0.0

        if initial_visible_offsets is 'AUTO':
            initial_visible_offsets = 0.0

        if initial_hidden_offsets is 'AUTO':
            initial_hidden_offsets = 0.0

        # Call constructor of superclass
        super(GaussianBinaryRBM,
              self).__init__(number_visibles=number_visibles,
                             number_hiddens=number_hiddens,
                             data=data,
                             initial_weights=initial_weights,
                             initial_visible_bias=initial_visible_bias,
                             initial_hidden_bias=initial_hidden_bias,
                             initial_visible_offsets=initial_visible_offsets,
                             initial_hidden_offsets=initial_hidden_offsets,
                             dtype=dtype)

        if data is None:
            self._data_std = numx.ones((1, self.input_dim), dtype=self.dtype)
            self._data_mean = numx.zeros((1, self.input_dim), dtype=self.dtype)
        else:
            self._data_std = numx.clip(self._data_std, 0.001, numx.finfo(self.dtype).max)

        # No Simoid units lead to 4 times smaller initial values
        if initial_weights is 'AUTO':
            self.w /= 4.0

        self.sigma = numx.ones((1, self.input_dim), dtype=self.dtype)
        if initial_sigma is 'AUTO':
            self.sigma *= self._data_std
        else:
            if numx.isscalar(initial_sigma):
                self.sigma *= initial_sigma
            else:
                self.sigma = numx.array(initial_sigma, dtype=dtype)
        self.sigma_base = numx.copy(self._data_std)
        self.bv_base = numx.copy(self._data_mean)
        self._fast_PT = False

    def _add_visible_units(self,
                           num_new_visibles,
                           position=0,
                           initial_weights='AUTO',
                           initial_bias='AUTO',
                           initial_sigmas=1.0,
                           initial_offsets='AUTO',
                           data=None):
        """ This function adds new visible units at the given position to the model.
            .. Warning:: If the parameters are changed. the trainer needs to be
                     reinitialized.

        :param num_new_visibles: The number of new hidden units to add
        :type num_new_visibles: int

        :param position: Position where the units should be added.
        :type position: int

        :param initial_weights: Initial weights. 'AUTO' and a scalar are random init.
        :type initial_weights: 'AUTO', scalar or numpy array [input num_new_visibles, output_dim]

        :param initial_bias: Initial visible bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid of \
                                     the visilbe mean. If a scalar is passed all values are initialized with it.
        :type initial_bias: 'AUTO' or scalar or numpy array [1, num_new_visibles]

        :param initial_sigmas: The initial standard deviation for the model.
        :type initial_sigmas: 'AUTO' or scalar or numpy array [1, num_new_visibles]

        :param initial_offsets: The initial visible offset values.
        :type initial_offsets: 'AUTO' or scalar or numpy array [1, num_new_visibles]

        :param data: If data is given and the offset and bias is initzialized accordingly, if 'AUTO' is chosen.
        :type data: numpy array [num datapoints, num_new_visibles]

        """
        if initial_weights is 'AUTO':
            initial_weights = numx.array((2.0 * numx.random.rand(
                num_new_visibles, self.output_dim) - 1.0) * (numx.sqrt(6.0 / (self.input_dim + self.output_dim +
                                                                              num_new_visibles))), dtype=self.dtype)

        if initial_bias is 'AUTO' or initial_bias is 'INVERSE_SIGMOID':
            if data is not None:
                initial_bias = data.mean(axis=0).reshape(1, self.input_dim)
            else:
                initial_bias = 0.0

        if initial_offsets is 'AUTO':
            initial_offsets = 0.0
        bv_base_old = self.bv_base
        super(GaussianBinaryRBM, self)._add_visible_units(num_new_visibles,
                                                          position,
                                                          initial_weights,
                                                          initial_bias,
                                                          initial_offsets,
                                                          data)

        self._data_std = numx.clip(self._data_std, 0.001, numx.finfo(self.dtype).max)

        if initial_sigmas is 'AUTO':
            if data is None:
                new_sigma = numx.ones((1, num_new_visibles), dtype=self.dtype)
            else:
                new_sigma = numx.clip(data.std(axis=0), 0.001, numx.finfo(self.dtype).max).reshape(1, num_new_visibles)
        else:
            if numx.isscalar(initial_sigmas):
                new_sigma = numx.ones((1, num_new_visibles)) * initial_sigmas
            else:
                new_sigma = numx.array(initial_sigmas)
        self.sigma = numx.insert(self.sigma, numx.array(numx.ones(num_new_visibles) * position, dtype=int), new_sigma,
                                 axis=1)

        new_bv_base = numx.zeros((1, num_new_visibles))
        if data is not None:
            new_bv_base = data.mean(axis=0).reshape(1, data.shape[1])
        self.bv_base = numx.insert(bv_base_old, numx.array(numx.ones(num_new_visibles) * position, dtype=int),
                                   new_bv_base, axis=1)

    def _add_hidden_units(self,
                          num_new_hiddens,
                          position=0,
                          initial_weights='AUTO',
                          initial_bias='AUTO',
                          initial_offsets='AUTO'):
        """ This function adds new hidden units at the given position to the model.
            .. Warning:: If the parameters are changed. the trainer needs to be
                     reinitialized.

        :param num_new_hiddens: The number of new hidden units to add.
        :type num_new_hiddens: int

        :param position: Position where the units should be added.
        :type position: int

        :param initial_weights: The initial weight values for the hidden units.
        :type initial_weights: 'AUTO' or scalar or numpy array [input_dim, num_new_hiddens]

        :param initial_bias: The initial hidden bias values.
        :type initial_bias: 'AUTO' or scalar or numpy array [1, num_new_hiddens]

        :param initial_offsets: he initial hidden mean values.
        :type initial_offsets: 'AUTO' or scalar or numpy array [1, num_new_hidden
        """
        # No Simoid units lead to 4 times smaller initial values
        if initial_weights is 'AUTO':
            initial_weights = numx.array((2.0 * numx.random.rand(
                self.input_dim, num_new_hiddens) - 1.0) * (numx.sqrt(6.0 / (self.input_dim + self.output_dim +
                                                                            num_new_hiddens))), dtype=self.dtype)
        if initial_bias is 'AUTO' or initial_bias is 'INVERSE_SIGMOID':
            initial_bias = 0.0

        if initial_offsets is 'AUTO':
            initial_offsets = 0.0

        super(GaussianBinaryRBM, self)._add_hidden_units(num_new_hiddens,
                                                         position,
                                                         initial_weights,
                                                         initial_bias,
                                                         initial_offsets)

    def _remove_visible_units(self, indices):
        """ This function removes the visible units whose indices are given.
            .. Warning:: If the parameters are changed. the trainer needs to be
                     reinitialized.

        :param indices: Indices of units to be remove.
        :type indices: int or list of int or numpy array of int
        """
        super(GaussianBinaryRBM, self)._remove_visible_units(indices)
        self.bv_base = numx.delete(self.bv_base, numx.array(indices), axis=1)
        self.sigma = numx.delete(self.sigma, numx.array(indices), axis=1)
        self.sigma_base = numx.delete(self.sigma_base, numx.array(indices), axis=1)

    def _calculate_weight_gradient(self, v, h):
        """ This function calculates the gradient for the weights from the visible and hidden activations.

        :param v: Visible activations.
        :type v: numpy arrays [batchsize, input dim]

        :param h: Hidden activations.
        :type h: numpy arrays [batchsize, output dim]

        :return: Weight gradient.
        :rtype: numpy arrays [input dim, output dim]
        """
        return numx.dot(((v - self.ov) / (self.sigma ** 2)).T, h - self.oh)

    def _calculate_visible_bias_gradient(self, v):
        """ This function calculates the gradient for the visible biases.

                :param v: Visible activations.
                :type v: numpy arrays [batch_size, input dim]

                :return: Visible bias gradient.
                :rtype: numpy arrays [1, input dim]
        """
        return (numx.sum(v - self.ov - self.bv, axis=0).reshape(1, v.shape[1])) / (self.sigma ** 2)

    def sample_v(self, v, beta=None, use_base_model=False):
        """ Samples the visible variables from the conditional probabilities v given h.

        :param v: Conditional probabilities of v given h.
        :type v: numpy array [batch size, input dim]

        :param beta: DUMMY Variable
                     The sampling in other types of units like Gaussian-Binary RBMs will be affected by beta.
        :type beta: None

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values. (DUMMY in this case)
        :type use_base_model: bool

        :return: States for v.
        :rtype: numpy array [batch size, input dim]
        """
        temp_sigma = self.sigma
        if beta is not None:
            temp_sigma = beta * self.sigma + (1.0 - beta) * self.sigma_base
        return v + numx.random.randn(v.shape[0], v.shape[1]) * temp_sigma

    def probability_v_given_h(self, h, beta=None, use_base_model=False):
        """ Calculates the conditional probabilities of v given h.

        :param h: Hidden states.
        :type h: numpy array [batch size, output dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously. None is equivalent to pass the value 1.0
        :type beta: None, float or numpy array [batch size, 1]

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
        :type use_base_model: bool

        :return: Conditional probabilities v given h.
        :rtype: numpy array [batch size, input dim]
        """
        if beta is not None:
            temp_bv = (1.0 - beta) * self.bv_base + self.bv * beta + self.ov
            activation = beta * numx.dot(h - self.oh, self.w.T) + temp_bv
        else:
            activation = numx.dot(h - self.oh, self.w.T) + self.bv + self.ov

        return activation

    def probability_h_given_v(self, v, beta=None, use_base_model=False):
        """ Calculates the conditional probabilities h given v.

        :param v: Visible states / data.
        :type v: numpy array [batch size, input dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously. None is equivalent to pass the value 1.0
        :type beta: None, float or numpy array [batch size, 1]

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
        :type use_base_model: bool

        :return: Conditional probabilities h given v.
        :rtype: numpy array [batch size, output dim]
        """
        if beta is not None:
            temp_sigma = beta * self.sigma + (1.0 - beta) * self.sigma_base
            activation = self.bh + beta * numx.dot((v - self.ov) / (temp_sigma ** 2), self.w)
        else:
            temp_sigma = self.sigma
            activation = self.bh + numx.dot((v - self.ov) / (temp_sigma ** 2), self.w)
        return self._hidden_post_activation(activation)

    def energy(self, v, h, beta=None, use_base_model=False):
        """ Compute the energy of the RBM given observed variable states v and hidden variables state h.

        :param v: Visible states.
        :type v: numpy array [batch size, input dim]

        :param h: Hidden states.
        :type h: numpy array [batch size, output dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously. None is equivalent to pass the value 1.0
        :type beta: None, float or numpy array [batch size, 1]

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
        :type use_base_model: bool

        :return: Energy of v and h.
        :rtype: numpy array [batch size,1]
        """
        temp_v = v - self.ov
        temp_h = h - self.oh
        if beta is not None:
            temp_bv = self.bv * beta + self.bv_base * (1.0 - beta)
            temp_bh = beta * self.bh
            temp_sigma = (self.sigma * beta + self._data_std * (1.0 - beta))
            temp_vw = beta * numx.dot(temp_v / (temp_sigma ** 2), self.w)
        else:
            temp_bv = self.bv
            temp_bh = self.bh
            temp_sigma = self.sigma
            temp_vw = numx.dot(temp_v / (temp_sigma ** 2), self.w)
        return (0.5 * numx.sum(((temp_v - temp_bv) / temp_sigma) ** 2, axis=1).reshape(h.shape[0], 1)
                - numx.dot(temp_h, temp_bh.T) - numx.sum(temp_vw * temp_h, axis=1).reshape(h.shape[0], 1))

    def unnormalized_log_probability_v(self,
                                       v,
                                       beta=None,
                                       use_base_model=False):
        """ Computes the unnormalized log probabilities of v.
            ln(z*p(v)) = ln(p(v))-ln(z)+ln(z) = ln(p(v)).

        :param v: Visible states.
        :type v: numpy array [batch size, input dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously.None is equivalent to pass the value 1.0.

        :type beta: None, float or numpy array [batch size, 1]

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
        :type use_base_model: bool

        :return: Unnormalized log probability of v.
        :rtype: numpy array [batch size, 1]
        """
        temp_v = v - self.ov
        temp_bh = self.bh
        if beta is not None:
            temp_sigma = beta * self.sigma + (1.0 - beta) * self.sigma_base
            temp_w = self.w * beta
            temp_bv = self.bv * beta + self.bv_base * (1.0 - beta)
        else:
            temp_sigma = self.sigma
            temp_bv = self.bv
            temp_w = self.w
        activation = numx.dot(temp_v / (temp_sigma ** 2), temp_w) + temp_bh
        bias = ((temp_v - temp_bv) / temp_sigma) ** 2
        activation = (-0.5 * numx.sum(bias, axis=1).reshape(v.shape[0], 1) + numx.sum(numx.log(numx.exp(
            activation * (1 - self.oh)) + numx.exp(-activation * self.oh)), axis=1).reshape(v.shape[0], 1))
        return activation

    def unnormalized_log_probability_h(self,
                                       h,
                                       beta=None,
                                       use_base_model=False):
        """ Computes the unnormalized log probabilities of h.

        :param h: Hidden states.
        :type h: numpy array [batch size, output dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously.None is equivalent to pass the value 1.0.
        :type beta: None, float or numpy array [batch size, 1]

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
        :type use_base_model: bool

        :return: Unnormalized log probability of h.
        :rtype: numpy array [batch size, 1]
        """
        temp_h = h - self.oh
        temp_bh = self.bh
        if beta is not None:
            temp_sigma = beta * self.sigma + (1.0 - beta) * self.sigma_base
            temp_w = self.w * beta
            temp_bv = self.bv * beta + self.bv_base * (1.0 - beta) + self.ov
        else:
            temp_sigma = self.sigma
            temp_w = self.w
            temp_bv = self.bv + self.ov
        temp_wh = numx.dot(temp_h, temp_w.T)
        return (self.input_dim * 0.5 * numx.log(2.0 * numx.pi)
                + numx.sum(numx.log(temp_sigma))
                + numx.dot(temp_h, temp_bh.T).reshape(h.shape[0], 1)
                + numx.sum(((temp_bv + temp_wh) / (numx.sqrt(2) * temp_sigma)) ** 2, axis=1).reshape(h.shape[0], 1)
                - numx.sum((temp_bv / (numx.sqrt(2) * temp_sigma)) ** 2))

    def _base_log_partition(self, use_base_model=False):
        """ Returns the base partition function which needs to be calculateable.

        :param use_base_model: DUMMY sicne the integral does not change if the mean is shifted.
        :type use_base_model: bool

        :return: Partition function for zero parameters.
        :rtype: float
        """
        return (self.input_dim * 0.5 * numx.log(2.0 * numx.pi) + numx.sum(numx.log(self.sigma_base))
                + numx.sum(numx.log(numx.exp(-self.oh * self.bh) + numx.exp((1.0 - self.oh) * self.bh))))


class GaussianBinaryVarianceRBM(GaussianBinaryRBM):
    """ Implementation of a Restricted Boltzmann machine with Gaussian visible having trainable variances and binary \
        hidden units.

    """

    def __init__(self,
                 number_visibles,
                 number_hiddens,
                 data=None,
                 initial_weights='AUTO',
                 initial_visible_bias='AUTO',
                 initial_hidden_bias='AUTO',
                 initial_sigma='AUTO',
                 initial_visible_offsets=0.0,
                 initial_hidden_offsets=0.0,
                 dtype=numx.float64):
        """ This function initializes all necessary parameters and data structures. It is recommended to pass the \
            training data to initialize the network automatically.

        :param number_visibles: Number of the visible variables.
        :type number_visibles: int

        :param number_hiddens: Number of hidden variables.
        :type number_hiddens: int

        :param data: The training data for parameter initialization if 'AUTO' is chosen for the corresponding parameter.
        :type data: None or numpy array [num samples, input dim]

        :param initial_weights: Initial weights. 'AUTO' and a scalar are random init.
        :type initial_weights: 'AUTO', scalar or numpy array [input dim, output_dim]

        :param initial_visible_bias: Initial visible bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid \
                                     of the visilbe mean. If a scalar is passed all values are initialized with it.
        :type initial_visible_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, input dim]

        :param initial_hidden_bias: Initial hidden bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid of \
                                    the hidden mean. If a scalar is passed all values are initialized with it.
        :type initial_hidden_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, output_dim]

        :param initial_sigma: Initial standard deviation for the model.
        :type initial_sigma: 'AUTO', scalar or numpy array [1, input_dim]

        :param initial_visible_offsets: Initial visible offset values. AUTO=data mean or 0.5 if no data is given. If a \
                                        scalar is passed all values are initialized with it.
        :type initial_visible_offsets: 'AUTO', scalar or numpy array [1, input dim]

        :param initial_hidden_offsets: Initial hidden offset values. AUTO = 0.5 If a scalar is passed all values are \
                                       initialized with it.
        :type initial_hidden_offsets: 'AUTO', scalar or numpy array [1, output_dim]

        :param dtype: Used data type i.e. numpy.float64
        :type dtype: numpy.float32 or numpy.float64 or numpy.longdouble
        """
        # Call constructor of superclass
        super(GaussianBinaryVarianceRBM,
              self).__init__(number_visibles=number_visibles,
                             number_hiddens=number_hiddens,
                             data=data,
                             initial_weights=initial_weights,
                             initial_visible_bias=initial_visible_bias,
                             initial_hidden_bias=initial_hidden_bias,
                             initial_sigma=initial_sigma,
                             initial_visible_offsets=initial_visible_offsets,
                             initial_hidden_offsets=initial_hidden_offsets,
                             dtype=dtype)

    def _calculate_sigma_gradient(self, v, h):
        """ This function calculates the gradient for the variance of the RBM.

        :param v: States of the visible variables.
        :type v: numpy arrays [batchsize, input dim]

        :param h: Probs/States of the hidden variables.
        :type h: numpy arrays [batchsize, output dim]

        :return: Sigma gradient.
        :rtype: list of numpy arrays [input dim,1]
        """
        var_diff = (v - self.bv - self.ov) ** 2
        return ((var_diff - 2.0 * (v - self.ov) * numx.dot(h, self.w.T)).sum(axis=0) / (self.sigma *
                                                                                        self.sigma * self.sigma))

    def get_parameters(self):
        """ This function returns all mordel parameters in a list.

        :return: The parameter references in a list.
        :rtype: list
        """
        return [self.w, self.bv, self.bh, self.sigma]

    def calculate_gradients(self, v, h):
        """ his function calculates all gradients of this RBM and returns them as an ordered array. This keeps the \
            flexibility of adding parameters which will be updated by the training algorithms.

        :param v: States of the visible variables.
        :type v: numpy arrays [batchsize, input dim]

        :param h: Probabilities of the hidden variables.
        :type h: numpy arrays [batchsize, output dim]

        :return: Gradients for all parameters.
        :rtype: numpy arrays (num parameters x [parameter.shape])
        """
        return [self._calculate_weight_gradient(v, h), self._calculate_visible_bias_gradient(v),
                self._calculate_hidden_bias_gradient(h), self._calculate_sigma_gradient(v, h)]


class BinaryBinaryLabelRBM(BinaryBinaryRBM):
    """ Implementation of a centered Restricted Boltzmann machine with Binary visible plus Softmax label units and \
        binary hidden units.

    """

    def __init__(self,
                 number_visibles,
                 number_labels,
                 number_hiddens,
                 data=None,
                 initial_weights='AUTO',
                 initial_visible_bias='AUTO',
                 initial_hidden_bias='AUTO',
                 initial_visible_offsets='AUTO',
                 initial_hidden_offsets='AUTO',
                 dtype=numx.float64):
        """ This function initializes all necessary parameters and data  structures. It is recommended to pass the \
            training data to initialize the network automatically.

        :param number_visibles: Number of the visible variables.
        :type number_visibles: int

        :param number_labels: Number of the label variables.
        :type number_labels: int

        :param number_hiddens: Number of hidden variables.
        :type number_hiddens: int

        :param data: The training data for parameter initialization if 'AUTO' is chosen for the corresponding parameter.
        :type data: None or numpy array [num samples, input dim]

        :param initial_weights: Initial weights. 'AUTO' and a scalar are random init.
        :type initial_weights: 'AUTO', scalar or numpy array [input dim, output_dim]

        :param initial_visible_bias: Initial visible bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid \
                                     of the visilbe mean. If a scalar is passed all values are initialized with it.
        :type initial_visible_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, input dim]

        :param initial_hidden_bias: Initial hidden bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid of \
                                    the hidden mean. If a scalar is passed all values are initialized with it.
        :type initial_hidden_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, output_dim]

        :param initial_visible_offsets: Initial visible offset values. AUTO=data mean or 0.5 if no data is given. If a \
                                        scalar is passed all values are initialized with it.
        :type initial_visible_offsets: 'AUTO', scalar or numpy array [1, input dim]

        :param initial_hidden_offsets: Initial hidden offset values. AUTO = 0.5 If a scalar is passed all values are \
                                       initialized with it.
        :type initial_hidden_offsets: 'AUTO', scalar or numpy array [1, output_dim]

        :param dtype: Used data type i.e. numpy.float64
        :type dtype: numpy.float32 or numpy.float64 or numpy.longdouble
        """
        # Call constructor of superclass
        super(BinaryBinaryLabelRBM,
              self).__init__(number_visibles=number_visibles + number_labels,
                             number_hiddens=number_hiddens,
                             data=data,
                             initial_weights=initial_weights,
                             initial_visible_bias=initial_visible_bias,
                             initial_hidden_bias=initial_hidden_bias,
                             initial_visible_offsets=initial_visible_offsets,
                             initial_hidden_offsets=initial_hidden_offsets,
                             dtype=dtype)

        self.data_dim = number_visibles
        self.label_dim = number_labels

        class SoftMaxSigmoid(object):
            """ SoftMax + Sigmoid conbination.

            """

            @classmethod
            def f(cls, x):
                """ Calculates the SoftPlus function value for a given input x.

                :param x: Input data.
                :type x: scalar or numpy array.

                :return: Value of the SoftPlus function for x.
                :rtype: scalar or numpy array with the same shape as x.
                """
                return numx.hstack((Sigmoid.f(x[:, 0:self.data_dim]),
                                    SoftMax.f(x[:, self.data_dim:])))

        self.visible_activation_function = SoftMaxSigmoid

    def sample_v(self, v, beta=None, use_base_model=False):
        """ Samples the visible variables from the conditional probabilities v given h.

        :param v: Conditional probabilities of v given h.
        :type v: numpy array [batch size, input dim]

        :param beta: DUMMY Variable. The sampling in other types of units like Gaussian-Binary RBMs will be affected \
                     by beta.
        :type beta: None

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values. (DUMMY in this case)
        :type use_base_model: bool

        :return: States for v.
        :rtype: numpy array [batch size, input dim]
        """
        ''' # Proof of concept
        result = numx.random.multinomial(1,v[0,self.visible_dim:self.input_dim])
        for i in range(1,v.shape[0]):
            result = numx.vstack((result,numx.random.multinomial(1,v[i,self.visible_dim:self.input_dim])))
        result = numx.hstack((v[:,0:self.visible_dim] > numx.random.random((v.shape[0],self.visible_dim)),result))
        return self.dtype(result)
        '''
        return numx.hstack((v[:, 0:self.data_dim] > numx.random.random((v.shape[0], self.data_dim)),
                            self.dtype(multinominal_batch_sampling(v[:, self.data_dim:], False))))

    def _add_visible_units(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def _remove_visible_units(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def energy(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def unnormalized_log_probability_v(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def unnormalized_log_probability_h(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def log_probability_v(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def log_probability_h(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def log_probability_v_h(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def _base_log_partition(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")


class GaussianBinaryLabelRBM(GaussianBinaryRBM):
    """ Implementation of a centered Restricted Boltzmann machine with Gaussian visible plus Softmax label units and \
        binary hidden units.

    """

    def __init__(self,
                 number_visibles,
                 number_labels,
                 number_hiddens,
                 data=None,
                 initial_weights='AUTO',
                 initial_visible_bias='AUTO',
                 initial_hidden_bias='AUTO',
                 initial_sigma='AUTO',
                 initial_visible_offsets='AUTO',
                 initial_hidden_offsets='AUTO',
                 dtype=numx.float64):
        """ This function initializes all necessary parameters and data  structures. It is recommended to pass the \
             training data to initialize the network automatically.

         :param number_visibles: Number of the visible variables.
         :type number_visibles: int

         :param number_labels: Number of the label variables.
         :type number_labels: int

         :param number_hiddens: Number of hidden variables.
         :type number_hiddens: int

         :param data: The training data for parameter initialization if 'AUTO' is chosen for the corresponding
                      parameter.
         :type data: None or numpy array [num samples, input dim]

         :param initial_weights: Initial weights. 'AUTO' and a scalar are random init.
         :type initial_weights: 'AUTO', scalar or numpy array [input dim, output_dim]

         :param initial_visible_bias: Initial visible bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid \
                                      of the visilbe mean. If a scalar is passed all values are initialized with it.
         :type initial_visible_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, input dim]

         :param initial_hidden_bias: Initial hidden bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid \
                                     of the hidden mean. If a scalar is passed all values are initialized with it.
         :type initial_hidden_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, output_dim]

         :param initial_sigma: Initial standard deviation for the model.
         :type initial_sigma: 'AUTO', scalar or numpy array [1, input_dim]

         :param initial_visible_offsets: Initial visible offset values. AUTO=data mean or 0.5 if no data is given. If \
                                         a scalar is passed all values are initialized with it.
         :type initial_visible_offsets: 'AUTO', scalar or numpy array [1, input dim]

         :param initial_hidden_offsets: Initial hidden offset values. AUTO = 0.5 If a scalar is passed all values are \
                                        initialized with it.
         :type initial_hidden_offsets: 'AUTO', scalar or numpy array [1, output_dim]

         :param dtype: Used data type i.e. numpy.float64
         :type dtype: numpy.float32 or numpy.float64 or numpy.longdouble
         """
        # Call constructor of superclass
        super(GaussianBinaryLabelRBM,
              self).__init__(number_visibles=number_visibles + number_labels,
                             number_hiddens=number_hiddens,
                             data=data,
                             initial_weights=initial_weights,
                             initial_visible_bias=initial_visible_bias,
                             initial_hidden_bias=initial_hidden_bias,
                             initial_sigma=initial_sigma,
                             initial_visible_offsets=initial_visible_offsets,
                             initial_hidden_offsets=initial_hidden_offsets,
                             dtype=dtype)
        self.data_dim = number_visibles
        self.label_dim = number_labels

        self.sigma[:, self.data_dim:] = numx.ones((1, self.label_dim), dtype=self.dtype)

        class SoftMaxLinear(object):
            """ SoftMax + Sigmoid conbination.

            """

            @classmethod
            def f(cls, x):
                """ Calculates the SoftPlus function value for a given input x.

                :param x: Input data.
                :type x: scalar or numpy array.

                :return: Value of the SoftPlus function for x.
                :rtype: scalar or numpy array with the same shape as x.
                """
                return numx.hstack((x[:, 0:self.visible_dim],
                                    SoftMax.f(x[:, self.visible_dim:self.input_dim])))

        self.visible_activation_function = SoftMaxLinear

    def sample_v(self, v, beta=None, use_base_model=False):
        """ Samples the visible variables from the conditional probabilities v given h.

        :param v: Conditional probabilities of v given h.
        :type v: numpy array [batch size, input dim]

        :param beta: DUMMY Variable. The sampling in other types of units like Gaussian-Binary RBMs will be affected \
                     by beta.
        :type beta: None

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values. (DUMMY in this case)
        :type use_base_model: bool

        :return: States for v.
        :rtype: numpy array [batch size, input dim]
        """
        if beta is None:
            res = v[:, 0:self.data_dim] + numx.random.randn(v.shape[0], self.data_dim) * self.sigma[:, 0:self.data_dim]
        else:
            res = (v[:, 0:self.data_dim] + numx.random.randn(v.shape[0], self.data_dim)
                   * (beta * (self.sigma - self._data_std) + self._data_std)[:, 0:self.data_dim])

        return numx.hstack((res, self.dtype(multinominal_batch_sampling(v[:, self.data_dim:], False))))

    def _add_visible_units(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def _remove_visible_units(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def energy(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def unnormalized_log_probability_v(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def unnormalized_log_probability_h(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def log_probability_v(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def log_probability_h(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def log_probability_v_h(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")

    def _base_log_partition(self):
        """ Not available!
        """
        raise Exception("Not yet implemented!")


class BinaryRectRBM(BinaryBinaryRBM):
    """ Implementation of a centered Restricted Boltzmann machine with Binary visible and Noisy linear rectified
        hidden units.

    """

    def __init__(self,
                 number_visibles,
                 number_hiddens,
                 data=None,
                 initial_weights='AUTO',
                 initial_visible_bias='AUTO',
                 initial_hidden_bias='AUTO',
                 initial_visible_offsets='AUTO',
                 initial_hidden_offsets='AUTO',
                 dtype=numx.float64):
        """ This function initializes all necessary parameters and data  structures. It is recommended to pass the \
            training data to initialize the network automatically.

         :param number_visibles: Number of the visible variables.
         :type number_visibles: int

         :param number_hiddens: Number of hidden variables.
         :type number_hiddens: int

         :param data: The training data for parameter initialization if 'AUTO' is chosen for the corresponding \
                      parameter.
         :type data: None or numpy array [num samples, input dim]

         :param initial_weights: Initial weights. 'AUTO' and a scalar are random init.
         :type initial_weights: 'AUTO', scalar or numpy array [input dim, output_dim]

         :param initial_visible_bias: Initial visible bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid \
                                      of the visilbe mean. If a scalar is passed all values are initialized with it.
         :type initial_visible_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, input dim]

         :param initial_hidden_bias: Initial hidden bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid \
                                     of the hidden mean. If a scalar is passed all values are initialized with it.
         :type initial_hidden_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, output_dim]

         :param initial_visible_offsets: Initial visible offset values. AUTO=data mean or 0.5 if no data is given. If \
                                         a scalar is passed all values are initialized with it.
         :type initial_visible_offsets: 'AUTO', scalar or numpy array [1, input dim]

         :param initial_hidden_offsets: Initial hidden offset values. AUTO = 0.5 If a scalar is passed all values are \
                                        initialized with it.
         :type initial_hidden_offsets: 'AUTO', scalar or numpy array [1, output_dim]

         :param dtype: Used data type i.e. numpy.float64
         :type dtype: numpy.float32 or numpy.float64 or numpy.longdouble
         """
        # Call constructor of superclass
        super(BinaryBinaryRBM,
              self).__init__(number_visibles=number_visibles,
                             number_hiddens=number_hiddens,
                             data=data,
                             initial_weights=initial_weights,
                             initial_visible_bias=initial_visible_bias,
                             initial_hidden_bias=initial_hidden_bias,
                             initial_visible_offsets=initial_visible_offsets,
                             initial_hidden_offsets=initial_hidden_offsets,
                             dtype=dtype)
        self.temp = 0
        self.max_act = 1000.0
        self.hidden_activation_function = SoftPlus

    def probability_h_given_v(self, v, beta=None):
        """ Calculates the conditional probabilities h given v.

        :param v: Visible states / data.
        :type v: numpy array [batch size, input dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously.
        :type beta: float or numpy array [batch size, 1]

        :return: Conditional probabilities h given v.
        :rtype: numpy array [batch size, output dim]
        """
        activation = numx.dot(v - self.ov, self.w) + self.bh
        if beta is not None:
            activation *= beta
        self.temp = activation
        activation = self._hidden_post_activation(activation)
        return activation

    def sample_h(self, h, beta=None, use_base_model=False):
        """ Samples the hidden variables from the conditional probabilities h given v.

        :param h: Conditional probabilities of h given v.
        :type h: numpy array [batch size, output dim]

        :param beta: DUMMY Variable. The sampling in other types of units like Gaussian-Binary RBMs will be affected \
                     by beta.
        :type beta: None

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values. (DUMMY in this case)
        :type use_base_model: bool

        :return: States for h.
        :rtype: numpy array [batch size, output dim]
        """
        x = self.temp  # numx.log(numx.exp(h)-1.0)
        activation = x + numx.random.randn(x.shape[0], x.shape[1]) * Sigmoid.f(x)
        return numx.clip(activation, 0.0, self.max_act)

    def _add_visible_units(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def _remove_visible_units(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def energy(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def unnormalized_log_probability_v(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def unnormalized_log_probability_h(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def log_probability_v(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def log_probability_h(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def log_probability_v_h(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def _base_log_partition(self):
        """ Not available!
        """
        raise Exception("Not implemented!")


class RectBinaryRBM(BinaryBinaryRBM):
    """ Implementation of a centered Restricted Boltzmann machine with Noisy linear rectified visible units and binary \
        hidden units.

    """

    def __init__(self,
                 number_visibles,
                 number_hiddens,
                 data=None,
                 initial_weights='AUTO',
                 initial_visible_bias='AUTO',
                 initial_hidden_bias='AUTO',
                 initial_visible_offsets='AUTO',
                 initial_hidden_offsets='AUTO',
                 dtype=numx.float64):
        """ This function initializes all necessary parameters and data  structures. It is recommended to pass the \
             training data to initialize the network automatically.

         :param number_visibles: Number of the visible variables.
         :type number_visibles: int

         :param number_hiddens: Number of hidden variables.
         :type number_hiddens: int

         :param data: The training data for parameter initialization if 'AUTO' is chosen for the corresponding \
                      parameter.
         :type data: None or numpy array [num samples, input dim]

         :param initial_weights: Initial weights. 'AUTO' and a scalar are random init.
         :type initial_weights: 'AUTO', scalar or numpy array [input dim, output_dim]

         :param initial_visible_bias: Initial visible bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid \
                                      of the visilbe mean. If a scalar is passed all values are initialized with it.
         :type initial_visible_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, input dim]

         :param initial_hidden_bias: Initial hidden bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid \
                                     of the hidden mean. If a scalar is passed all values are initialized with it.
         :type initial_hidden_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, output_dim]

         :param initial_visible_offsets: Initial visible offset values. AUTO=data mean or 0.5 if no data is given. If \
                                         a scalar is passed all values are initialized with it.
         :type initial_visible_offsets: 'AUTO', scalar or numpy array [1, input dim]

         :param initial_hidden_offsets: Initial hidden offset values. AUTO = 0.5 If a scalar is passed all values are \
                                        initialized with it.
         :type initial_hidden_offsets: 'AUTO', scalar or numpy array [1, output_dim]

         :param dtype: Used data type i.e. numpy.float64
         :type dtype: numpy.float32 or numpy.float64 or numpy.longdouble
         """
        # Call constructor of superclass
        super(BinaryBinaryRBM,
              self).__init__(number_visibles=number_visibles,
                             number_hiddens=number_hiddens,
                             data=data,
                             initial_weights=initial_weights,
                             initial_visible_bias=initial_visible_bias,
                             initial_hidden_bias=initial_hidden_bias,
                             initial_visible_offsets=initial_visible_offsets,
                             initial_hidden_offsets=initial_hidden_offsets,
                             dtype=dtype)
        self.temp = 0
        self.max_act = 1000.0
        self.visible_activation_function = SoftPlus

    def probability_v_given_h(self, h, beta=None, use_base_model=False):
        """ Calculates the conditional probabilities of v given h.

        :param h: Hidden states.
        :type h: numpy array [batch size, output dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously. None is equivalent to pass the value 1.0
        :type beta: None, float or numpy array [batch size, 1]

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
        :type use_base_model: bool

        :return: Conditional probabilities v given h.
        :rtype: numpy array [batch size, input d
        """
        activation = numx.dot(h - self.oh, self.w.T) + self.bv
        if beta is not None:
            activation *= beta
        self.temp = activation
        activation = self._visible_post_activation(activation)
        return activation

    def sample_v(self, v, beta=None, use_base_model=False):
        """ Samples the visible variables from the conditional probabilities v given h.

        :param v: Conditional probabilities of v given h.
        :type v: numpy array [batch size, input dim]

        :param beta: DUMMY Variable. The sampling in other types of units like Gaussian-Binary \
                     RBMs will be affected by beta.
        :type beta: None

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values. (DUMMY in this case)
        :type use_base_model: bool

        :return: States for v.
        :rtype: numpy array [batch size, input dim]
        """
        x = self.temp  # numx.log(numx.exp(h)-1.0)
        activation = v + numx.random.randn(x.shape[0], x.shape[1]) * Sigmoid.f(x)
        return numx.clip(activation, 0.0, self.max_act)

    def _add_visible_units(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def _remove_visible_units(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def energy(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def unnormalized_log_probability_v(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def unnormalized_log_probability_h(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def log_probability_v(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def log_probability_h(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def log_probability_v_h(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def _base_log_partition(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def _getbasebias(self):
        """ Not available!
        """
        raise Exception("Not implemented!")


class RectRectRBM(BinaryRectRBM):
    """ Implementation of a centered Restricted Boltzmann machine with Noisy linear rectified visible and hidden units.

    """

    def __init__(self,
                 number_visibles,
                 number_hiddens,
                 data=None,
                 initial_weights='AUTO',
                 initial_visible_bias='AUTO',
                 initial_hidden_bias='AUTO',
                 initial_visible_offsets='AUTO',
                 initial_hidden_offsets='AUTO',
                 dtype=numx.float64):
        """ This function initializes all necessary parameters and data  structures. It is recommended to pass the \
             training data to initialize the network automatically.

         :param number_visibles: Number of the visible variables.
         :type number_visibles: int

         :param number_hiddens: Number of hidden variables.
         :type number_hiddens: int

         :param data: The training data for parameter initialization if 'AUTO' is chosen for the corresponding \
                      parameter.
         :type data: None or numpy array [num samples, input dim]

         :param initial_weights: Initial weights. 'AUTO' and a scalar are random init.
         :type initial_weights: 'AUTO', scalar or numpy array [input dim, output_dim]

         :param initial_visible_bias: Initial visible bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid \
                                      of the visilbe mean. If a scalar is passed all values are initialized with it.
         :type initial_visible_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, input dim]

         :param initial_hidden_bias: Initial hidden bias. 'AUTO' is random, 'INVERSE_SIGMOID' is the inverse Sigmoid \
                                     of the hidden mean. If a scalar is passed all values are initialized with it.
         :type initial_hidden_bias: 'AUTO','INVERSE_SIGMOID', scalar or numpy array [1, output_dim]

         :param initial_visible_offsets: Initial visible offset values. AUTO=data mean or 0.5 if no data is given. If \
                                         a scalar is passed all values are initialized with it.
         :type initial_visible_offsets: 'AUTO', scalar or numpy array [1, input dim]

         :param initial_hidden_offsets: Initial hidden offset values. AUTO = 0.5 If a scalar is passed all values are \
                                        initialized with it.
         :type initial_hidden_offsets: 'AUTO', scalar or numpy array [1, output_dim]

         :param dtype: Used data type i.e. numpy.float64
         :type dtype: numpy.float32 or numpy.float64 or numpy.longdouble
         """
        # Call constructor of superclass
        super(BinaryRectRBM,
              self).__init__(number_visibles=number_visibles,
                             number_hiddens=number_hiddens,
                             data=data,
                             initial_weights=initial_weights,
                             initial_visible_bias=initial_visible_bias,
                             initial_hidden_bias=initial_hidden_bias,
                             initial_visible_offsets=initial_visible_offsets,
                             initial_hidden_offsets=initial_hidden_offsets,
                             dtype=dtype)
        self.temp = 0
        self.max_act = 1000.0
        self.visible_activation_function = SoftPlus

    def probability_v_given_h(self, h, beta=None, use_base_model=False):
        """ Calculates the conditional probabilities of v given h.

        :param h: Hidden states.
        :type h: numpy array [batch size, output dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously. None is equivalent to pass the value 1.0
        :type beta: None, float or numpy array [batch size, 1]

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values.
        :type use_base_model: bool

        :return: Conditional probabilities v given h.
        :rtype: numpy array [batch size, input d
        """
        activation = numx.dot(h - self.oh, self.w.T) + self.bv
        if beta is not None:
            activation *= beta
        self.temp = activation
        activation = self._visible_post_activation(activation)
        return activation

    def sample_v(self, v, beta=None, use_base_model=False):
        """ Samples the visible variables from the conditional probabilities v given h.

        :param v: Conditional probabilities of v given h.
        :type v: numpy array [batch size, input dim]

        :param beta: DUMMY Variable
                     The sampling in other types of units like Gaussian-Binary
                     RBMs will be affected by beta.
        :type beta: None

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values. (DUMMY in this case)
        :type use_base_model: bool

        :return: States for v.
        :rtype: numpy array [batch size, input dim]
        """
        x = self.temp  # numx.log(numx.exp(h)-1.0)
        activation = v + numx.random.randn(x.shape[0], x.shape[1]) * Sigmoid.f(x)
        return numx.clip(activation, 0.0, self.max_act)


class GaussianRectRBM(GaussianBinaryRBM):
    """ Implementation of a centered Restricted Boltzmann machine with Gaussian visible and Noisy linear rectified
        hidden units.

    """

    def __init__(self,
                 number_visibles,
                 number_hiddens,
                 data=None,
                 initial_weights='AUTO',
                 initial_visible_bias='AUTO',
                 initial_hidden_bias='AUTO',
                 initial_sigma='AUTO',
                 initial_visible_offsets='AUTO',
                 initial_hidden_offsets='AUTO',
                 dtype=numx.float64):
        """ This function initializes all necessary parameters and data structures. See comments for automatically \
            chosen values.

        :param number_visibles: Number of the visible variables.
        :type number_visibles: int

        :param number_hiddens: Number of the hidden variables.
        :type number_hiddens: int

        :param data: The training data for initializing the visible bias.
        :type data: None or numpy array [num samples, input dim] or List of numpy arrays [num samples, input dim]

        :param initial_weights: Initial weights.
        :type initial_weights: 'AUTO', scalar or numpy array [input dim, output_dim]

        :param initial_visible_bias: Initial visible bias.
        :type initial_visible_bias: 'AUTO', scalar or numpy array [1,input dim]

        :param initial_hidden_bias: Initial hidden bias.
        :type initial_hidden_bias: 'AUTO', scalar or numpy array [1, output_dim]

        :param initial_sigma: Initial standard deviation for the model.
        :type initial_sigma: 'AUTO', scalar or numpy array [1, input_dim]

        :param initial_visible_offsets: Initial visible mean values.
        :type initial_visible_offsets: 'AUTO', scalar or numpy array [1, input dim]

        :param initial_hidden_offsets: Initial hidden mean values.
        :type initial_hidden_offsets: 'AUTO', scalar or numpy array [1, output_dim]

        :param dtype: Used data type.
        :type dtype: numpy.float32, numpy.float64 and, numpy.longdouble
        """

        # Call constructor of superclass
        super(GaussianRectRBM,
              self).__init__(number_visibles=number_visibles,
                             number_hiddens=number_hiddens,
                             data=data,
                             initial_weights=initial_weights,
                             initial_visible_bias=initial_visible_bias,
                             initial_hidden_bias=initial_hidden_bias,
                             initial_sigma=initial_sigma,
                             initial_visible_offsets=initial_visible_offsets,
                             initial_hidden_offsets=initial_hidden_offsets,
                             dtype=dtype)
        self.max_act = 10.0
        self.temp = 0
        self.hidden_activation_function = SoftPlus

    def probability_h_given_v(self, v, beta=None):
        """ Calculates the conditional probabilities h given v.

        :param v: Visible states / data.
        :type v: numpy array [batch size, input dim]

        :param beta: Allows to sample from a given inverse temperature beta, or if a vector is given to sample from \
                     different betas simultaneously.
        :type beta: float or numpy array [batch size, 1]

        :return: Conditional probabilities h given v.
        :rtype: numpy array [batch size, output dim]
        """
        temp_sigma = self.sigma
        if beta is not None:
            temp_sigma = (self.sigma * beta + self._data_std * (1.0 - beta))
        activation = self.bh + numx.dot((v - self.ov) / (temp_sigma ** 2), self.w)
        self.temp = activation
        activation = self._hidden_post_activation(activation)
        return activation

    def sample_h(self, h, beta=None, use_base_model=False):
        """ Samples the hidden variables from the conditional probabilities h given v.

        :param h: Conditional probabilities of h given v.
        :type h: numpy array [batch size, output dim]

        :param beta: DUMMY Variable
                     The sampling in other types of units like Gaussian-Binary RBMs will be affected by beta.
        :type beta: None

        :param use_base_model: If true uses the base model, i.e. the MLE of the bias values. (DUMMY in this case)
        :type use_base_model: bool

        :return: States for h.
        :rtype: numpy array [batch size, output dim]
        """
        x = self.temp  # numx.log(numx.exp(h)-1.0)
        activation = x + numx.random.randn(x.shape[0], x.shape[1]) * Sigmoid.f(x)
        return numx.clip(activation, 0.0, self.max_act)

    def _add_visible_units(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def _remove_visible_units(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def energy(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def unnormalized_log_probability_v(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def unnormalized_log_probability_h(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def log_probability_v(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def log_probability_h(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def log_probability_v_h(self):
        """ Not available!
        """
        raise Exception("Not implemented!")

    def _base_log_partition(self):
        """ Not available!
        """
        raise Exception("Not implemented!")


class GaussianRectVarianceRBM(GaussianRectRBM):
    """ Implementation of a Restricted Boltzmann machine with Gaussian visible having trainable variances and noisy \
        rectified hidden units.

    """

    def __init__(self,
                 number_visibles,
                 number_hiddens,
                 data=None,
                 initial_weights='AUTO',
                 initial_visible_bias='AUTO',
                 initial_hidden_bias='AUTO',
                 initial_sigma='AUTO',
                 initial_visible_offsets=0.0,
                 initial_hidden_offsets=0.0,
                 dtype=numx.float64):
        """ This function initializes all necessary parameters and data structures. See comments for automatically \
            chosen values.

        :param number_visibles: Number of the visible variables.
        :type number_visibles: int

        :param number_hiddens: Number of the hidden variables.
        :type number_hiddens: int

        :param data: The training data for initializing the visible bias.
        :type data: None or numpy array [num samples, input dim] or List of numpy arrays [num samples, input dim]

        :param initial_weights: Initial weights.
        :type initial_weights: 'AUTO', scalar or numpy array [input dim, output_dim]

        :param initial_visible_bias: Initial visible bias.
        :type initial_visible_bias: 'AUTO', scalar or numpy array [1,input dim]

        :param initial_hidden_bias: Initial hidden bias.
        :type initial_hidden_bias: 'AUTO', scalar or numpy array [1, output_dim]

        :param initial_sigma: Initial standard deviation for the model.
        :type initial_sigma: 'AUTO', scalar or numpy array [1, input_dim]

        :param initial_visible_offsets: Initial visible mean values.
        :type initial_visible_offsets: 'AUTO', scalar or numpy array [1, input dim]

        :param initial_hidden_offsets: Initial hidden mean values.
        :type initial_hidden_offsets: 'AUTO', scalar or numpy array [1, output_dim]

        :param dtype: Used data type.
        :type dtype: numpy.float32, numpy.float64 and, numpy.longdouble
        """
        # Call constructor of superclass
        super(GaussianRectVarianceRBM,
              self).__init__(number_visibles=number_visibles,
                             number_hiddens=number_hiddens,
                             data=data,
                             initial_weights=initial_weights,
                             initial_visible_bias=initial_visible_bias,
                             initial_hidden_bias=initial_hidden_bias,
                             initial_sigma=initial_sigma,
                             initial_visible_offsets=initial_visible_offsets,
                             initial_hidden_offsets=initial_hidden_offsets,
                             dtype=dtype)

    def _calculate_sigma_gradient(self, v, h):
        """ This function calculates the gradient for the variance of the RBM.

        :param v: States of the visible variables.
        :type v: numpy arrays [batchsize, input dim]

        :param h: Probabilities of the hidden variables.
        :type h: numpy arrays [batchsize, output dim]

        :return: Sigma gradient.
        :rtype: list of numpy arrays [input dim,1]
        """
        var_diff = (v - self.bv - self.ov) ** 2
        return (var_diff - 2.0 * (v - self.ov) * numx.dot(h, self.w.T)).sum(axis=0) / (self.sigma ** 3)

    def get_parameters(self):
        """ This function returns all model parameters in a list.

        :return: The parameter references in a list.
        :rtype: list
        """
        return [self.w, self.bv, self.bh, self.sigma]

    def calculate_gradients(self, v, h):
        """ This function calculates all gradients of this RBM and returns them as an ordered array. This keeps the \
            flexibility of adding parameters which will be updated by the training algorithms.

        :param v: States of the visible variables.
        :type v: numpy arrays [batchsize, input dim]

        :param h: Probabilities of the hidden variables.
        :type h: numpy arrays [batchsize, output dim]

        :return: Gradients for all parameters.
        :rtype: numpy arrays (num parameters x [parameter.shape])
        """
        return [self._calculate_weight_gradient(v, h), self._calculate_visible_bias_gradient(v),
                self._calculate_hidden_bias_gradient(h), self._calculate_sigma_gradient(v, h)]
