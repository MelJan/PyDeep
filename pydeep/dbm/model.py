""" This module contains the model class for DBMs.

    :Implemented:
        - DBM model

    :Version:
        1.0.0

    :Date:
        26.05.2019

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2019 Jan Melchior

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


class DBM_model(object):

    def __init__(self, layers):
        ''' This function initializes the weight layer.

        :Parameters:
            model:      Model, basically a list of layers
                       -type: Weight_layer or None

            dtype:      Used data type i.e. numpy.float64
                       -type: numpy.float32 or numpy.float64 or
                                           numpy.longdouble

        '''
        self.layers = layers
        self.num_layers = len(layers)
        if self.num_layers % 2 == 0:
            self.even_num_layers = True
        else:
            self.even_num_layers = False

    def meanfield(self, chains, k, fixed_layer=None, inplace=True):
        ''' Samples all states k times.

        :Parameters:
            chains:     Markov chains.
                       -type: list of numpy arrays

            k:          Number of Gibbs sampling steps.
                       -type: int

            target:     Resulting variable, if None inplace calculation is performed,
                       -type: int

        '''
        if isinstance(fixed_layer, list):
            if len(fixed_layer) != len(chains):
                raise Exception("fixed_layer has to have the same length as chains!")
        else:
            fixed_layer = []
            for i in range(0, self.num_layers):
                fixed_layer.append(False)

        # Create empty list with pre caluclated values
        pre_top = []
        pre_bottom = []
        for i in range(self.num_layers):
            pre_top.append(None)
            pre_bottom.append(None)

        # Get all pre calculatable values
        for i in range(self.num_layers):
            if fixed_layer[i] is True:
                if i > 0:
                    pre_top[i - 1] = self.layers[i].input_weight_layer.propagate_down(chains[i])
                if i < self.num_layers - 1:
                    pre_bottom[i + 1] = self.layers[i].output_weight_layer.propagate_up(chains[i])

        if inplace is True:
            target = chains
        else:
            target = []
            for i in range(0, self.num_layers):
                target.append(numx.copy(chains[i]))

        for _ in range(k):
            for i in range(1, self.num_layers - 1, 2):
                if fixed_layer[i] is False:
                    target[i] = self.layers[i].activation(target[i - 1], target[i + 1], pre_bottom[i], pre_top[i])[0]
            # Top layer if even
            if self.even_num_layers is True:
                if fixed_layer[self.num_layers - 1] is False:
                    target[self.num_layers - 1] = \
                    self.layers[self.num_layers - 1].activation(target[self.num_layers - 2], None,
                                                                pre_bottom[self.num_layers - 1], None)[0]
            # First Layer
            if fixed_layer[0] is False:
                target[0] = self.layers[0].activation(None, target[1], None, pre_top[0])[0]
            for i in range(2, self.num_layers - 1, 2):
                if fixed_layer[i] is False:
                    target[i] = self.layers[i].activation(target[i - 1], target[i + 1], pre_bottom[i], pre_top[i])[0]
            # Top layer if odd
            if self.even_num_layers is False:
                if fixed_layer[self.num_layers - 1] is False:
                    target[self.num_layers - 1] = \
                    self.layers[self.num_layers - 1].activation(target[self.num_layers - 2], None,
                                                                pre_bottom[self.num_layers - 1], None)[0]
        return target

    def sample(self, chains, k, fixed_layer=None, inplace=True):
        ''' Samples all states k times.

        :Parameters:
            chains:     Markov chains.
                       -type: list of numpy arrays

            k:          Number of Gibbs sampling steps.
                       -type: int

            target:     Resulting variable, if None inplace calculation is performed,
                       -type: int

        '''
        if isinstance(fixed_layer, list):
            if len(fixed_layer) != len(chains):
                raise Exception("fixed_layer has to have the same length as chains!")
        else:
            fixed_layer = []
            for i in range(0, self.num_layers):
                fixed_layer.append(False)
        # print("FIX LAYERS:",fixed_layer)
        # Create empty list with pre caluclated values
        pre_top = []
        pre_bottom = []
        for i in range(self.num_layers):
            pre_top.append(None)
            pre_bottom.append(None)

        # print("PRE TOP:",pre_top)
        # print("PRE BOTTOM:",pre_bottom)

        # Get all pre calculatable values
        for i in range(self.num_layers):
            if fixed_layer[i] is True:
                # print("PRE-CALC LAYER "+str(i))
                if i > 0:
                    # print("->DOWN ACTIVITY")
                    pre_top[i - 1] = self.layers[i].input_weight_layer.propagate_down(chains[i])
                if i < self.num_layers - 1:
                    # print("->UP ACTIVITY")
                    pre_bottom[i + 1] = self.layers[i].output_weight_layer.propagate_up(chains[i])

        # print("PRE TOP:",pre_top)
        # print("PRE BOTTOM:",pre_bottom)

        if inplace is True:
            target = chains
        else:
            target = []
            for i in range(0, self.num_layers):
                target.append(numx.copy(chains[i]))

        # print("TARGET:",target)

        for _ in range(k):
            # ODD LAYERS
            # print("UPDATE ODD LAYERS")
            for i in range(1, self.num_layers - 1, 2):
                if fixed_layer[i] is False:
                    # print("->Layer "+str(i)+" updated")
                    target[i] = self.layers[i].sample(
                        self.layers[i].activation(target[i - 1], target[i + 1], pre_bottom[i], pre_top[i]))
                # else:
                # print("->Layer "+str(i)+" is fixed")
            # Top layer if even
            if self.even_num_layers is True:
                if fixed_layer[self.num_layers - 1] is False:
                    # print("->Layer "+str(self.num_layers-1)+"(TOP EVEN) updated")
                    target[self.num_layers - 1] = self.layers[self.num_layers - 1].sample(
                        self.layers[self.num_layers - 1].activation(target[self.num_layers - 2], None,
                                                                    pre_bottom[self.num_layers - 1], None))
                # else:
                # print("->Layer " + str(self.num_layers-1) + "(TOP EVEN) is fixed")
            # print("UPDATE EVEN LAYERS")
            # First Layer
            if fixed_layer[0] is False:
                # print("->Layer " + str(0) + " updated")
                target[0] = self.layers[0].sample(self.layers[0].activation(None, target[1], None, pre_top[0]))
            # else:
            # print("->Layer " + str(0) + " is fixed")
            # EVEN LAYERS
            for i in range(2, self.num_layers - 1, 2):
                if fixed_layer[i] is False:
                    # print("->Layer "+str(i)+" updated")
                    target[i] = self.layers[i].sample(
                        self.layers[i].activation(target[i - 1], target[i + 1], pre_bottom[i], pre_top[i]))
                # else:
                # print("->Layer "+str(i)+" is fixed")
            # Top layer if odd
            if self.even_num_layers is False:
                if fixed_layer[self.num_layers - 1] is False:
                    # print("->Layer "+str(self.num_layers-1)+"(TOP ODD) updated")
                    target[self.num_layers - 1] = self.layers[self.num_layers - 1].sample(
                        self.layers[self.num_layers - 1].activation(target[self.num_layers - 2], None,
                                                                    pre_bottom[self.num_layers - 1], None))
                # else:
                # print("->Layer " + str(self.num_layers - 1) + "(TOP ODD) is fixed")
        return target

    def update(self, chains_d, chains_m, lr_W, lr_b, lr_o, restriction=None, restrict_typ=None):
        if numx.isscalar(lr_W):
            lr_W = numx.ones((self.num_layers - 1)) * lr_W
        if numx.isscalar(lr_b):
            lr_b = numx.ones((self.num_layers)) * lr_b
        if numx.isscalar(lr_o):
            lr_o = numx.ones((self.num_layers)) * lr_o

        i = 0
        for l, x_d in zip(self.layers, chains_d):
            l.update_offsets(numx.mean(x_d, axis=0).reshape(1, x_d.shape[1]), lr_o[i])
            i += 1

        grad_w = []
        for l in range(1, self.num_layers):
            grad = self.layers[l].input_weight_layer.calculate_weight_gradients(chains_d[l - 1], chains_d[l],
                                                                                chains_m[l - 1], chains_m[l],
                                                                                self.layers[l - 1].offset,
                                                                                self.layers[l].offset)
            # grad -= 0.0002*self.layers[l].input_weight_layer.weights
            self.layers[l].input_weight_layer.update_weights(lr_W[l - 1] * grad, restriction, restrict_typ)
            grad_w.append(grad)

        for l in range(self.num_layers):
            if l == 0:
                grad_b = self.layers[0].calculate_gradient_b(chains_d[0], chains_m[0], None, self.layers[1].offset,
                                                             None, grad_w[0])
            elif l == self.num_layers - 1:
                grad_b = self.layers[l].calculate_gradient_b(chains_d[l], chains_m[l], self.layers[l - 1].offset, None,
                                                             grad_w[l - 1], None)
            else:
                grad_b = self.layers[l].calculate_gradient_b(chains_d[l], chains_m[l], self.layers[l - 1].offset,
                                                             self.layers[l + 1].offset, grad_w[l - 1], grad_w[l])
            self.layers[l].update_biases(lr_b[l] * grad_b, restriction, restrict_typ)
