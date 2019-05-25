'''  Feed Forward Neural Network Model.

    .. Note::
    
        Due to computational benefits the common notation for the delta terms is 
        split in a delta term for the common layer and the error signal passed 
        to the layer below. See the following Latex code for details. This allows
        to store all layer depending results in the corresponding layer and avoid
        useless computations without messing up the code.
        .. math::
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

class Model(object):
    ''' Model which stores the layers.

    '''

    def __init__(self,
                 layers = []):
        ''' Constructor takes a list of layers or an empty list.

        :Parameters:
            layers:  List of layers or empty list.
                    -type: list of layers.

        '''
        # Set variables
        self.layers = layers   
        self.num_layers = len(self.layers)
        if self.num_layers > 0:
            self.input_dim = self.layers[0].input_dim
            self.output_dim = self.layers[self.num_layers-1].output_dim
            self.consistency_check()
        else:
            self.input_dim = 0
            self.output_dim = 0

    def clear_temp_data(self):
        ''' Sets all temp variables to None.

        '''
        for l in self.layers:
            l.clear_temp_data()

    def calculate_cost(self,
                       labels,
                       costs,
                       reg_costs,
                       desired_sparseness,
                       costs_sparseness,
                       reg_sparseness,
                       ):
        ''' Calculates the cost for given labels.
            You need to call model.forward_propagate before!

        :Parameters:

            labels:             list of numpy arrays, entries can be None but the last layer needs labels!
                               -type: list of None and/or numpy arrays

            costs:              Cost functions for the layers, entries can be None but the last layer needs a cost
                                function!
                               -type: list of None and/or pydeep.base.costfunction

            reg_costs:          list of scalars controlling the strength of the cost functions.
                               -type: list of scalars

            desired_sparseness: List of desired sparseness values/average hidden activities.
                               -type: list of scalars

            costs_sparseness:   Cost functions for the sparseness, entries can be None.
                               -type: list of None and/or pydeep.base.costfunction

            reg_sparseness:     Strength of the sparseness term.
                               -type: list of scalars

        :return:
            Cost values for the datapoints.
           -type: numpy array [batchsize, 1]

        '''
        cost = 0.0
        # Add all intermediate costs
        for l in range(self.num_layers):
            if reg_sparseness[l] != 0.0:
                mean_h = numx.atleast_2d(numx.mean(self.layers[l].temp_a, axis=0))
                cost += reg_sparseness[l]*costs_sparseness[l].f(mean_h, desired_sparseness[l])
            if reg_costs[l] != 0.0:
                cost += reg_costs[l]*costs[l].f(self.layers[l].temp_a, labels[l])
        return cost

    def consistency_check(self):
        ''' Raises exceptions if the network structure is incorrect, e.g. output dim layer 0 != input dim layer 1

        '''
        num_layers = len(self.layers)
        for i in range(num_layers-1):
            if not self.layers[i].output_dim == self.layers[i+1].input_dim:
                raise Exception("Output dimensions mismatch layer "+str(i)+" and "+str(i+1)+" !", UserWarning) 

    def append_layer(self, layer):
        ''' Appends a layer to the network.

        :Parameters:
            layer:  Neural network layer.
                   -type: Neural network layer.

        '''
        if self.num_layers > 0:
            # Check consistency
            if not self.layers[len(self.layers)-1].output_dim == layer.input_dim:
                raise Exception("Output dimensions mismatch last layer and new layer !", UserWarning)
        else:
            # First layer set input dim
            self.input_dim = layer.input_dim
        # Everything is okay, append layer
        self.output_dim = layer.output_dim
        self.layers.append(layer)
        self.num_layers +=1

    def pop_layer(self):
        ''' Pops the last layer in the network.

        '''
        if self.num_layers > 0:
            self.layers.pop(self.num_layers-1)
            self.num_layers -= 1
            if self.num_layers > 0:
                self.output_dim = self.layers[self.num_layers-1].output_dim
            else:
                self.input_dim = 0
                self.output_dim = 0

    def forward_propagate(self, data, corruptor= None):
        ''' Propagates the inout data through the network.

        :Parameters:
            data:      Input data to propagate.
                      -type: numpy array [batchsize, self.input_dim]

            corruptor: None or list of corruptors, one for the input followed by one for every hidden layers output.
                      -type: Noen or list of corruptors

        :Returns:
            Output of the network, every unit state is also stored in the corresponding layer.
           -type: numpy arrays [batchsize, self.output dim]

        '''
        # No corruptor just progate data as is
        if corruptor is None:
            output = data
            for l in range(len(self.layers)):
                output = self.layers[l].forward_propagate(output)
        else:
            # copy data
            output = numx.copy(data)
            for l in range(self.num_layers):
                # If corruptor is given , distored activations
                if corruptor[l] is not None:
                    output = corruptor[l].corrupt(output)
                output = self.layers[l].forward_propagate(output)
            if corruptor[self.num_layers] is not None:
                output = corruptor[self.num_layers].corrupt(output)
            '''
            output = numx.copy(data)
            if corruptor[0] is not None:
                output = corruptor[0].corrupt(output)
            for l in range(self.num_layers):
                # If corruptor is given , distored activations
                if corruptor[l+1] is not None:
                    output = corruptor[l+1].corrupt(output)
                output = self.layers[l].forward_propagate(output)
            '''
        return output

    def _get_gradients(self, labels, costs, reg_costs, desired_sparseness, costs_sparseness, reg_sparseness, check_gradient=False):
        ''' Calculates the gradient for the network (Used in finit_differences()).
            You need to call model.forward_propagate before!

        :Parameters:

            labels:             list of numpy arrays, entries can be None but the last layer needs labels!
                               -type: list of None and/or numpy arrays

            costs:              Cost functions for the layers, entries can be None but the last layer needs a cost
                                function!
                               -type: list of None and/or pydeep.base.costfunction

            reg_costs:          list of scalars controlling the strength of the cost functions.
                               -type: list of scalars

            desired_sparseness: List of desired sparseness values/average hidden activities.
                               -type: list of scalars

            costs_sparseness:   Cost functions for the sparseness, entries can be None.
                               -type: list of None and/or pydeep.base.costfunction

            reg_sparseness:     Strength of the sparseness term.
                               -type: list of scalars

            check_gradient:     Flase for gradient checking mode.
                               -type: bool

        :return:
            gradient for the network.
           -type: list of list of numpy arrays

        '''
        grad = []
        deltas = None
        # Go from top layer to last layer
        for l in range(self.num_layers-1, -1, -1):
            # Caluclate the delta values
            deltas = self.layers[l]._get_deltas(deltas = deltas,
                                                      labels = labels[l],
                                                      cost = costs[l],
                                                      reg_cost = reg_costs[l],
                                                      desired_sparseness = desired_sparseness[l],
                                                      cost_sparseness = costs_sparseness[l],
                                                      reg_sparseness = reg_sparseness[l],
                                                      check_gradient=check_gradient)
            # backprop the error if it is not first/bottom most layer.
            if l > 0:
                deltas = self.layers[l]._backward_propagate()

            # Now we are ready to calculate the gradient
            grad.insert(0,self.layers[l]._calculate_gradient())
        return grad

    def finit_differences(self, delta, data, labels, costs, reg_costs, desired_sparseness, costs_sparseness, reg_sparseness):
        ''' Calculates the finite differences for the networks gradient.

        :Parameters:

            delta:              Small delta value added to the parameters.
                               -type: float

            data:               Input data of the network.
                               -type: numpy arrays

            labels:             list of numpy arrays, entries can be None but the last layer needs labels!
                               -type: list of None and/or numpy arrays

            costs:              Cost functions for the layers, entries can be None but the last layer needs a cost
                                function!
                               -type: list of None and/or pydeep.base.costfunction

            reg_costs:          list of scalars controlling the strength of the cost functions.
                               -type: list of scalars

            desired_sparseness: List of desired sparseness values/average hidden activities.
                               -type: list of scalars

            costs_sparseness:   Cost functions for the sparseness, entries can be None.
                               -type: list of None and/or pydeep.base.costfunction

            reg_sparseness:     Strength of the sparseness term.
                               -type: list of scalars

        :return:
            Finite differences W,b, max W, max b
           -type: list of list of numpy arrays

        '''
        data = numx.atleast_2d(data)
        # Lists of the difference
        diffs_w = []
        diffs_b = []
        # Vars for tracking the maximal value
        max_diffb = -99999
        max_diffw = -99999
        # Loop through all layers
        for l in range(len(self.layers)):
            # Select curretn layer and initialize temp variable for the differences
            layer = self.layers[l]
            diff_w = numx.zeros(layer.weights.shape)
            diff_b = numx.zeros(layer.bias.shape)
            # Loop over each weight
            for i in range(layer.input_dim):
                for j in range(layer.output_dim):
                    # Only caluclate the dinite difference if the will exist in the model, that is the connection matrix
                    if layer.connections is None or (layer.connections is not None and layer.connections[i][j] > 0.0):
                        # Propagate through and calculate gradient
                        self.forward_propagate(data = data)
                        grad_w_ij = self._get_gradients(labels = labels,
                                                        costs = costs,
                                                        reg_costs = reg_costs,
                                                        desired_sparseness = desired_sparseness,
                                                        costs_sparseness = costs_sparseness,
                                                        reg_sparseness = reg_sparseness,
                                                        check_gradient=True)[l][0][i][j]
                        # Add delta to the current weight, propagate through and calculate cost
                        layer.weights[i,j] += delta
                        self.forward_propagate(data)
                        E_pos = self.calculate_cost(labels = labels,
                                                    costs = costs,
                                                    reg_costs = reg_costs,
                                                    desired_sparseness = desired_sparseness,
                                                    costs_sparseness = costs_sparseness,
                                                    reg_sparseness = reg_sparseness)
                        # Subtract 2 times delta from the current weight (subtract 1 times the original), propagate through and calculate cost
                        layer.weights[i,j] -= 2*delta
                        self.forward_propagate(data)
                        E_neg = self.calculate_cost(labels = labels,
                                                    costs = costs,
                                                    reg_costs = reg_costs,
                                                    desired_sparseness = desired_sparseness,
                                                    costs_sparseness = costs_sparseness,
                                                    reg_sparseness = reg_sparseness)
                        # Restore the original weight
                        layer.weights[i,j] += delta
                        # Calculate the difference
                        approx = (E_pos-E_neg)/(2.0*delta)
                        # Calculate the absolute difference
                        diff_w[i,j] = numx.abs(grad_w_ij - approx)
                        # Update the maximal value
                        if numx.max(numx.abs(diff_w[i,j])) > max_diffw:
                            max_diffw = numx.abs(diff_w[i,j])
            # Loop over each bias
            for j in range(layer.output_dim):
                # Propagate through and calculate gradient
                self.forward_propagate(data)
                grad_b_j = self._get_gradients(labels = labels,
                                               costs = costs,
                                               reg_costs = reg_costs,
                                               desired_sparseness = desired_sparseness,
                                               costs_sparseness = costs_sparseness,
                                               reg_sparseness = reg_sparseness,
                                               check_gradient=True)[l][1][0][j]
                # Add delta to the current bias, propagate through and calculate cost
                layer.bias[0,j] += delta
                self.forward_propagate(data)
                E_pos = self.calculate_cost(labels = labels,
                                            costs = costs,
                                            reg_costs = reg_costs,
                                            desired_sparseness = desired_sparseness,
                                            costs_sparseness = costs_sparseness,
                                            reg_sparseness = reg_sparseness)
                # Subtract 2 times delta from the current bias (subtract 1 times the original), propagate through and calculate cost
                layer.bias[0,j] -= 2*delta
                self.forward_propagate(data)
                E_neg = self.calculate_cost(labels = labels,
                                            costs = costs,
                                            reg_costs = reg_costs,
                                            desired_sparseness = desired_sparseness,
                                            costs_sparseness = costs_sparseness,
                                            reg_sparseness = reg_sparseness)
                # Restore the original weight
                layer.bias[0,j] += delta
                approx = (E_pos-E_neg)/(2.0*delta)
                # Calculate the absolute difference
                diff_b[0,j] = numx.abs(grad_b_j - approx)
                # Update the maximal value
                if numx.max(numx.abs(diff_b[0,j])) > max_diffb:
                    max_diffb = numx.abs(diff_b[0,j])
            # Append the difference matrices
            diffs_w.append(diff_w)
            diffs_b.append(diff_b)
        # Return all diffs as well as the maximum weight and bias differences
        return diffs_w, diffs_b, max_diffw, max_diffb