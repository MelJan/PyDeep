''' This module provides a general implementation of a 3 layer tied weights Auto-encoder (x-h-y).
    The code is focused on readability and clearness, while keeping the efficiency and flexibility high.
    Several activation functions are available for visible and hidden units which can be mixed arbitrarily.
    The code can easily be adapted to AEs without tied weights. For deep AEs the FFN code can be adapted.

    :Implemented:
        -  AE  - Auto-encoder (centered)
        - DAE  - Denoising Auto-encoder (centered)
        - SAE  - Sparse Auto-encoder (centered)
        - CAE  - Contractive Auto-encoder (centered)
        - SLAE - Slow Auto-encoder (centered)

    :Info:
        http://ufldl.stanford.edu/wiki/index.php/Sparse_Coding:_Autoencoder_Interpretation

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

from pydeep.base.activationfunction import Sigmoid, SoftMax
from pydeep.base.basicstructure import BipartiteGraph
from pydeep.base.costfunction import CrossEntropyError


class AutoEncoder(BipartiteGraph):
    ''' Class for a 3 Layer Auto-encoder (x-h-y) with tied weights.
    '''

    def __init__(self,
                 number_visibles,
                 number_hiddens,
                 data=None,
                 visible_activation_function=Sigmoid,
                 hidden_activation_function=Sigmoid,
                 cost_function=CrossEntropyError,
                 initial_weights='AUTO',
                 initial_visible_bias='AUTO',
                 initial_hidden_bias='AUTO',
                 initial_visible_offsets='AUTO',
                 initial_hidden_offsets='AUTO',
                 dtype=numx.float64):

        '''
        This function initializes all necessary parameters and data
        structures. It is recommended to pass the training data to
        initialize the network automatically.

        :Parameters:
            number_visibles:              Number of the visible variables.
                                         -type: int

            number_hiddens                Number of hidden variables.
                                         -type: int

            data:                         The training data for parameter
                                          initialization if 'AUTO' is chosen.
                                         -type: None or
                                                numpy array [num samples, input dim]
                                                or List of numpy arrays
                                                [num samples, input dim]

            visible_activation_function:  A non linear transformation function
                                          for the visible units (default: Sigmoid)
                                         -type: Subclass of ActivationFunction()

            hidden_activation_function:   A non linear transformation function
                                          for the hidden units (default: Sigmoid)
                                         -type: Subclass of ActivationFunction

            cost_function                 A cost function (default: CrossEntropyError())
                                         -type: subclass of FNNCostFunction()

            initial_weights:              Initial weights.'AUTO' is random
                                         -type: 'AUTO', scalar or
                                                numpy array [input dim, output_dim]

            initial_visible_bias:         Initial visible bias.
                                          'AUTO' is random
                                          'INVERSE_SIGMOID' is the inverse Sigmoid of
                                           the visilbe mean
                                         -type:  'AUTO','INVERSE_SIGMOID', scalar or
                                                 numpy array [1, input dim]

            initial_hidden_bias:          Initial hidden bias.
                                          'AUTO' is random
                                          'INVERSE_SIGMOID' is the inverse Sigmoid of
                                          the hidden mean
                                         -type:  'AUTO','INVERSE_SIGMOID', scalar or
                                                 numpy array [1, output_dim]

            initial_visible_offsets:      Initial visible mean values.
                                          AUTO=data mean or 0.5 if not data is given.
                                         -type:  'AUTO', scalar or
                                                 numpy array [1, input dim]

            initial_hidden_offsets:       Initial hidden mean values.
                                          AUTO = 0.5
                                         -type: 'AUTO', scalar or
                                                 numpy array [1, output_dim]

            dtype:                        Used data type i.e. numpy.float64
                                         -type: numpy.float32 or numpy.float64 or
                                                numpy.longdouble

        '''

        if (cost_function == CrossEntropyError) and not (visible_activation_function == Sigmoid):
            raise Exception("The Cross entropy cost should only be used with Sigmoid units or units of "
                            "interval (0,1)", UserWarning)

        # Set the initial_visible_bias to zero if the activation function is not a SIGMOID
        if ((initial_visible_bias is 'AUTO' or initial_visible_bias is 'INVERSE_SIGMOID')
            and not (visible_activation_function == Sigmoid)):
            initial_visible_bias = 0.0

        # Set the AUTO initial_hidden_bias to zero if the activation function is not a SIGMOID
        if ((initial_hidden_bias is 'AUTO' or initial_hidden_bias is 'INVERSE_SIGMOID')
            and not (hidden_activation_function == Sigmoid)):
            initial_hidden_bias = 0.0

        if visible_activation_function == SoftMax or hidden_activation_function == SoftMax:
            raise Exception("Softmax not supported but you can use FNN instead!")

        # Call constructor of superclass
        super(AutoEncoder,
              self).__init__(number_visibles=number_visibles,
                             number_hiddens=number_hiddens,
                             data=data,
                             visible_activation_function=visible_activation_function,
                             hidden_activation_function=hidden_activation_function,
                             initial_weights=initial_weights,
                             initial_visible_bias=initial_visible_bias,
                             initial_hidden_bias=initial_hidden_bias,
                             initial_visible_offsets=initial_visible_offsets,
                             initial_hidden_offsets=initial_hidden_offsets,
                             dtype=dtype)

        # Store the cost function
        self.cost_function = cost_function

    def _get_contractive_penalty(self, a_h, factor):
        ''' Calculates contractive penalty cost for a data point x.

        :Parameters:

            a_h:     Pre-synaptic activation of h: a_h = (Wx+c).
                    -type: numpy array [num samples, hidden dim]

            factor:  Influence factor (lambda) for the penalty.
                    -type: float

        :Returns:
            Contractive penalty costs for x.
           -type: numpy array [num samples]
        '''
        w2_sum = numx.sum(self.w ** 2.0, axis=0).reshape(1, self.output_dim)
        df2 = self.hidden_activation_function.df(a_h) ** 2.0
        return factor * numx.sum(df2 * w2_sum, axis=1)

    def _get_sparse_penalty(self, h, factor, desired_sparseness):
        ''' Calculates sparseness penalty cost for a data point x.
            .. Warning:: Different penalties are used depending on the
                     hidden activation function.

        :Parameters:

            h:                   hidden activation.
                                -type: numpy array [num samples, hidden dim]

            factor:              Influence factor (beta) for the penalty.
                                -type: float

            desired_sparseness:  Desired average hidden activation.
                                -type: float

        :Returns:
            Sparseness penalty costs for x.
           -type: numpy array [num samples]
        '''
        mean_h = numx.atleast_2d(numx.mean(h, axis=0))
        if self.hidden_activation_function == Sigmoid:
            min_value = 1e-10
            max_value = 1.0 - min_value
            mean_h = numx.atleast_2d(numx.clip(mean_h, min_value, max_value))
            sparseness = desired_sparseness * numx.log(desired_sparseness / mean_h) + (1.0
                                                                                       - desired_sparseness) * numx.log(
                (1.0 - desired_sparseness) / (1.0 - mean_h))
        else:
            sparseness = (desired_sparseness - mean_h) ** 2.0
        return factor * numx.sum(sparseness, axis=1)

    def _get_slowness_penalty(self, h, h_next, factor):
        ''' Calculates slowness penalty cost for a data point x.
            .. Warning:: Different penalties are used depending on the
                     hidden activation function.

        :Parameters:

            h:                   hidden activation.
                                -type: numpy array [num samples, hidden dim]

            h_next:              hidden activation of the next data point in a sequence.
                                -type: numpy array [num samples, hidden dim]

            factor:              Influence factor (beta) for the penalty.
                                -type: float

        :Returns:
            Sparseness penalty costs for x.
           -type: numpy array [num samples]
        '''
        return factor * numx.sum((h - h_next) ** 2.0, axis=1)

    def energy(self,
               x,
               contractive_penalty=0.0,
               sparse_penalty=0.0,
               desired_sparseness=0.01,
               x_next=None,
               slowness_penalty=0.0):
        ''' Calculates the energy/cost for a data point x.

        :Parameters:

            x:                   Data points.
                                -type: numpy array [num samples, input dim]

            contractive_penalty: If a value > 0.0 is given the cost is also
                                 calculated on the contractive penalty.
                                -type: float

            sparse_penalty:      If a value > 0.0 is given the cost is also
                                 calculated on the sparseness penalty.
                                -type: float

            desired_sparseness:  Desired average hidden activation.
                                -type: float

            x_next:              Next data points.
                                -type: None or numpy array [num samples, input dim]

            slowness_penalty:    If a value > 0.0 is given the cost is also
                                 calculated on the slowness penalty.
                                -type: float

        :Returns:
            Costs for x.
           -type: numpy array [num samples]
        '''
        a_h, h = self._encode(x)
        _, y = self._decode(h)
        cost = self.cost_function.f(y, x)
        if contractive_penalty > 0.0:
            cost += self._get_contractive_penalty(a_h, contractive_penalty)
        if sparse_penalty > 0.0:
            cost += self._get_sparse_penalty(h, sparse_penalty, desired_sparseness)
        if slowness_penalty > 0.0 and x_next is not None:
            h_next = self.encode(x_next)
            cost += self._get_slowness_penalty(h, h_next, slowness_penalty)
        return cost

    def _encode(self, x):
        ''' The function propagates the activation of the input
            layer through the network to the hidden/output layer.

        :Parameters:

            x:    Input of the network.
                 -type: numpy array [num samples, input dim]

        :Returns:
            Pre and Post synaptic output.
           -type: List of arrays [num samples, hidden dim]
        '''
        # Compute pre-synaptic activation
        pre_act_h = self._hidden_pre_activation(x)
        # Compute post-synaptic activation
        h = self._hidden_post_activation(pre_act_h)
        return pre_act_h, h

    def encode(self, x):
        ''' The function propagates the activation of the input
            layer through the network to the hidden/output layer.

        :Parameters:

            x:    Input of the network.
                 -type: numpy array [num samples, input dim]

        :Returns:
            Output of the network.
           -type: array [num samples, hidden dim]
        '''
        return self._encode(numx.atleast_2d(x))[1]

    def _decode(self, h):
        ''' The function propagates the activation of the hidden
            layer reverse through the network to the input layer.

        :Parameters:

            h:    Output of the network
                 -type: numpy array [num samples, hidden dim]

        :Returns:
            Input of the network.
           -type: array [num samples, input dim]
        '''
        # Compute pre-synaptic activation
        pre_act_y = self._visible_pre_activation(h)
        # Compute post-synaptic activation
        y = self._visible_post_activation(pre_act_y)
        return pre_act_y, y

    def decode(self, h):
        ''' The function propagates the activation of the hidden
            layer reverse through the network to the input layer.

        :Parameters:

            h:    Output of the network
                 -type: numpy array [num samples, hidden dim]

        :Returns:
            Pre and Post synaptic input.
           -type: List of arrays [num samples, input dim]
        '''
        return self._decode(numx.atleast_2d(h))[1]

    def reconstruction_error(self, x, absolut=False):
        ''' Calculates the reconstruction error for given training data.

        :Parameters:

            x:       Datapoints
                    -type: numpy array [num samples, input dim]

            absolut: If true the absolute error is caluclated.
                    -type: bool

        :Returns:
            Reconstruction error.
           -type: List of arrays [num samples, 1]
        '''
        diff = x - self.decode(self.encode(x))
        if absolut is True:
            diff = numx.abs(diff)
        else:
            diff = diff ** 2
        return numx.sum(diff, axis=1)

    def _get_gradients(self,
                       x,
                       a_h,
                       h,
                       a_y,
                       y,
                       reg_contractive,
                       reg_sparseness,
                       desired_sparseness,
                       reg_slowness,
                       x_next,
                       a_h_next,
                       h_next):

        '''
        Computes the gradients of weights, visible and the hidden bias.
        Depending on whether contractive penalty and or sparse penalty
        is used the gradient changes.

        :Parameters:

            x:                    Training data.
                                 -type: numpy array [num samples, input dim]

            a_h:                  Pre-synaptic activation of h: a_h = (Wx+c).
                                 -type: numpy array [num samples, output dim]

            h                     Post-synaptic activation of h: h = f(a_h).
                                 -type: numpy array [num samples, output dim]

            a_y:                  Pre-synaptic activation of y: a_y = (Wh+b).
                                 -type: numpy array [num samples, input dim]

            y                     Post-synaptic activation of y: y = f(a_y).
                                 -type: numpy array [num samples, input dim]

            reg_contractive:      Contractive influence factor (lambda).
                                 -type: float

            reg_sparseness:       Sparseness influence factor (lambda).
                                 -type: float

            desired_sparseness:   Desired average hidden activation.
                                 -type: float

            reg_slowness:         Slowness influence factor.
                                 -type: float

            x_next:               Next Training data in Sequence.
                                 -type: numpy array [num samples, input dim]

            a_h_next:             Next pre-synaptic activation of h: a_h = (Wx+c).
                                 -type: numpy array [num samples, output dim]

            h_next                Next post-synaptic activation of h: h = f(a_h).
                                 -type: numpy array [num samples, input dim]

        '''
        # Calculate derivatives for the activation functions
        df_a_y = self.visible_activation_function.df(a_y)
        df_a_h = self.hidden_activation_function.df(a_h)

        # Calculate the visible bias gradient
        grad_b = self.cost_function.df(y, x) * df_a_y

        # Calculate the hidden bias gradient
        grad_c = numx.dot(grad_b, self.w)

        # Add the sparse penalty gradient part
        if reg_sparseness > 0.0:
            grad_c += reg_sparseness * self.__get_sparse_penalty_gradient_part(h, desired_sparseness)
        grad_c *= df_a_h

        # Calculate weight gradient
        grad_W = numx.dot((x - self.ov).T, grad_c) + numx.dot(grad_b.T, (h - self.oh))

        # Average / Normalize over batches
        grad_b = numx.mean(grad_b, axis=0).reshape(1, self.input_dim)
        grad_c = numx.mean(grad_c, axis=0).reshape(1, self.output_dim)
        grad_W /= x.shape[0]

        # Add contractive penalty gradient
        if reg_contractive > 0.0:
            pW, pc = self._get_contractive_penalty_gradient(x, a_h, df_a_h)
            grad_c += reg_contractive * pc
            grad_W += reg_contractive * pW

        if reg_slowness > 0.0 and x_next is not None:
            df_a_h_next = self.hidden_activation_function.df(a_h_next)
            pW, pc = self._get_slowness_penalty_gradient(x, x_next, h, h_next, df_a_h, df_a_h_next)
            grad_c += reg_slowness * pc
            grad_W += reg_slowness * pW
        return [grad_W, grad_b, grad_c]

    def __get_sparse_penalty_gradient_part(self, h, desired_sparseness):

        '''
        This function computes the desired part of the gradient
        for the sparse penalty term. Only used for efficiency.

        :Parameters:

            h:                        hidden activations
                                     -type: numpy array [num samples, input dim]

            desired_sparseness:       Desired average hidden activation.
                                     -type: float

        :Returs:
            The computed gradient part is returned
           -type: numpy array [1, hidden dim]
        '''
        mean_h = numx.atleast_2d(numx.mean(h, axis=0))
        if self.hidden_activation_function == Sigmoid or isinstance(self.hidden_activation_function, Sigmoid):
            min_value = 1e-10
            max_value = 1.0 - min_value
            mean_h = numx.clip(mean_h, min_value, max_value)
            grad = -desired_sparseness / mean_h + (1.0 - desired_sparseness) / (1.0 - mean_h)
        else:
            grad = -2.0 * (desired_sparseness - mean_h)
        return grad

    def _get_sparse_penalty_gradient(self, h, df_a_h, desired_sparseness):

        '''
        This function computes the gradient for the sparse penalty term.

        :Parameters:

            h:                  hidden activations
                               -type: numpy array [num samples, input dim]

            df_a_h:             Derivative of untransformed hidden activations
                               -type: numpy array [num samples, input dim]

            desired_sparseness: Desired average hidden activation.
                               -type: float

        :Returs:
            The computed gradient part is returned
           -type: numpy array [1, hidden dim]
        '''
        return self.__get_sparse_penalty_gradient_part(h, desired_sparseness) * df_a_h

    def _get_contractive_penalty_gradient(self, x, a_h, df_a_h):
        '''
        This function computes the gradient for the contractive penalty term.

        :Parameters:

            x:      Training data.
                   -type: numpy array [num samples, input dim]

            a_h:    Untransformed hidden activations
                   -type: numpy array [num samples, input dim]

            df_a_h: Derivative of untransformed hidden activations
                   -type: numpy array [num samples, input dim]

        :Returs:
            The computed gradient is returned
           -type: numpy array [input dim, hidden dim]
        '''
        ddf_a_h = self.hidden_activation_function.ddf(a_h)
        w2_sum = numx.sum(self.w ** 2, axis=0).reshape(1, self.output_dim)
        grad_c = 2.0 * df_a_h * ddf_a_h * w2_sum
        grad_w = numx.dot((x - self.ov).T, 2.0 * df_a_h * ddf_a_h) * w2_sum / x.shape[0] + 2.0 * self.w * (
        numx.mean(df_a_h ** 2.0, axis=0))
        grad_c = numx.mean(grad_c, axis=0).reshape(1, self.output_dim)
        return [grad_w, grad_c]

    def _get_slowness_penalty_gradient(self, x, x_next, h, h_next, df_a_h, df_a_h_next):
        ''' This function computes the gradient for the slowness penalty term.

        :Parameters:

            x:           Training data.
                        -type: numpy array [num samples, input dim]

            x_next:      Next training data points in Sequence.
                        -type: numpy array [num samples, input dim]

            h:           Corresponding hidden activations.
                        -type: numpy array [num samples, output dim]

            h_next:      Corresponding next hidden activations.
                        -type: numpy array [num samples, output dim]

            df_a_h:      Derivative of untransformed hidden activations.
                        -type: numpy array [num samples, input dim]

            df_a_h_next: Derivative of untransformed next hidden activations.
                        -type: numpy array [num samples, input dim]

        :Returs:
            The computed gradient is returned
           -type: numpy array [input dim, hidden dim]
        '''

        diff = 2.0 * (h - h_next)
        grad_w = (numx.dot((x - self.ov).T, diff * df_a_h) - numx.dot((x_next - self.ov).T, diff * df_a_h_next)) / \
                 x.shape[0]
        grad_c = numx.mean(diff * (df_a_h - df_a_h_next), axis=0)
        return [grad_w, grad_c]

    def finit_differences(self, data, delta, reg_sparseness, desired_sparseness, reg_contractive, reg_slowness,
                          data_next):
        ''' Finite differences test for AEs.
            The finite differences test involves all functions of the model except init and reconstruction_error

            data:                    The training data
                                    -type: numpy array [num samples, input dim]

            delta:                   The learning rate.
                                    -type: numpy array[num parameters]


            reg_sparseness:          The parameter (epsilon) for the sparseness regularization.
                                    -type: float

            desired_sparseness:      Desired average hidden activation.
                                    -type: float

            reg_contractive:         The parameter (epsilon) for the contractive regularization.
                                    -type: float

            reg_slowness:            The parameter (epsilon) for the slowness regularization.
                                    -type: float

            data_next:               The next training data in the sequence.
                                    -type: numpy array [num samples, input dim]



        '''
        data = numx.atleast_2d(data)

        diff_w = numx.zeros((data.shape[0], self.w.shape[0], self.w.shape[1]))
        diff_b = numx.zeros((data.shape[0], self.bv.shape[0], self.bv.shape[1]))
        diff_c = numx.zeros((data.shape[0], self.bh.shape[0], self.bh.shape[1]))

        for d in range(data.shape[0]):
            batch = data[d].reshape(1, data.shape[1])
            a_h, h = self._encode(batch)
            a_y, y = self._decode(h)
            if data_next is not None:
                batch_next = data_next[d].reshape(1, data.shape[1])
                a_h_next, h_next = self._encode(batch_next)
            else:
                batch_next = None
                a_h_next, h_next = None, None

            for i in range(self.input_dim):
                for j in range(self.output_dim):
                    grad_w_ij = self._get_gradients(batch, a_h, h, a_y, y, reg_contractive, reg_sparseness,
                                                    desired_sparseness,
                                                    reg_slowness, batch_next, a_h_next, h_next)[0][i, j]
                    self.w[i, j] += delta
                    E_pos = self.energy(batch, reg_contractive, reg_sparseness, desired_sparseness, batch_next,
                                        reg_slowness)
                    self.w[i, j] -= 2 * delta
                    E_neg = self.energy(batch, reg_contractive, reg_sparseness, desired_sparseness, batch_next,
                                        reg_slowness)
                    self.w[i, j] += delta
                    diff_w[d, i, j] = grad_w_ij - (E_pos - E_neg) / (2.0 * delta)

            for i in range(self.input_dim):
                grad_b_i = \
                self._get_gradients(batch, a_h, h, a_y, y, reg_contractive, reg_sparseness, desired_sparseness,
                                    reg_slowness, batch_next, a_h_next, h_next)[1][0, i]
                self.bv[0, i] += delta
                E_pos = self.energy(batch, reg_contractive, reg_sparseness, desired_sparseness, batch_next,
                                    reg_slowness)
                self.bv[0, i] -= 2 * delta
                E_neg = self.energy(batch, reg_contractive, reg_sparseness, desired_sparseness, batch_next,
                                    reg_slowness)
                self.bv[0, i] += delta
                diff_b[d, 0, i] = grad_b_i - (E_pos - E_neg) / (2.0 * delta)

                for j in range(self.output_dim):
                    grad_c_j = \
                    self._get_gradients(batch, a_h, h, a_y, y, reg_contractive, reg_sparseness, desired_sparseness,
                                        reg_slowness, batch_next, a_h_next, h_next)[2][0, j]
                    self.bh[0, j] += delta
                    E_pos = self.energy(batch, reg_contractive, reg_sparseness, desired_sparseness, batch_next,
                                        reg_slowness)
                    self.bh[0, j] -= 2 * delta
                    E_neg = self.energy(batch, reg_contractive, reg_sparseness, desired_sparseness, batch_next,
                                        reg_slowness)
                    self.bh[0, j] += delta
                    diff_c[d, 0, j] = grad_c_j - (E_pos - E_neg) / (2.0 * delta)

        return [diff_w, diff_b, diff_c]