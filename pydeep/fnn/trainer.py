'''  Feed Forward Neural Network Trainer.

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
import pydeep.base.numpyextension as numxExt
import pydeep.base.activationfunction as AFct
import pydeep.base.costfunction as CFct
import pydeep.base.corruptor as Corr

class GDTrainer(object):
    ''' Gradient decent feed forward neural network trainer.

    '''

    def __init__(self, model):
        ''' Constructor takes a model.

        :Parameters:
            model: FNN model to train.
                  -type: FNN model.
        '''
        self.model = model
        # Storage variable for the old gradient
        self._old_grad = []
        for l in self.model.layers:
            self._old_grad.append([numx.zeros((l.input_dim,l.output_dim)),numx.zeros((1,l.output_dim))])

    def calculate_errors(self, output_label):
        ''' Calculates the errors for the output of the model and given output_labels.
            You need to call model.forward_propagate before!

        :Parameters:

            output_label: numpy array containing the labels for the network output.
                         -type: list of None and/or numpy arrays

        :return:
            Bool array 0=True is prediction was corect 1=False otherwise.
           -type: numpy array [batchsize, 1]

        '''
        # Get the index of the maximum value along axis 1 from label and output and compare it
        return numxExt.compare_index_of_max(self.model.layers[self.model.num_layers-1].temp_a, output_label)

    def check_setup(self,
                    data,
                    labels,
                    costs,
                    reg_costs,
                    epsilon,
                    momentum,
                    update_offsets,
                    corruptor,
                    reg_L1Norm,
                    reg_L2Norm,
                    reg_sparseness,
                    desired_sparseness,
                    costs_sparseness,
                    restrict_gradient,
                    restriction_norm):
        ''' The function checks for valid training and network configuration.
            Warning are printed if a valid is wrong or suspicious.

        :Parameters:
            data:               Training data as numpy array.
                               -type: numpy arrays [batchsize, inpput dim]

            labels:             List of numpy arrays or None if a layer has no cost, the last layer has to have a cost
                                and thus the last item in labels has to be an array.
                               -type: List of numpy arrays and/or Nones

            costs:              List of Cost functions. The last layer has to have a cost.
                               -type: pydeep.base.costfunction

            reg_costs:          List of scalars controlling the strength of the cost functions. Last entry i.e. 1.
                               -type: scalar

            epsilon:            List of Learning rates.
                               -type: list of scalars

            momentum:           List of Momentum terms.
                               -type: list of scalars

            update_offsets:     List of Shifting factors for centering.
                               -type: list of scalars

            corruptor:          List of Corruptor objects e.g. Dropout.
                               -type: list of pydeep.base.corruptors

            reg_L1Norm:          List of L1 Norm Regularization terms.
                               -type: list of scalars

            reg_L2Norm:         List of L2 Norm Regularization terms.
                               -type: list of scalars

            reg_sparseness:     List of scalars controlling the strength of the sparseness regularization.
                               -type: list of scalars

            desired_sparseness: List of scalars / target sparseness.
                               -type: list of scalars

            costs_sparseness:   List of sparseness cost and/or None values
                               -type: list of pydeep.base.costfunction and/or None

            restrict_gradient:  Maximal norm for the gradient or None
                               -type: list of scalars

            restriction_norm:   Defines how the weights will be restricted 'Cols', 'Rows' or 'Mat'.
                               -type: Strings 'Cols', 'Rows' or 'Mat'

        :return:
            True is the test was sucessfull
           -type: bool

        '''
        failed = False
        if data.shape[1] != self.model.input_dim:
            print(Warning("Data dimension does not match the models output dimension"))
            failed = True
        if labels[len(labels)-1].shape[1] != self.model.output_dim:
            print(Warning("Labels["+str(len(labels)-1)+"] dimension does not match the models output dimension"))
            failed = True

        # Check main cost exists
        if not numx.isscalar(reg_costs[len(reg_costs)-1]) or reg_costs[len(reg_costs)-1] != 1:
            print(Warning("reg_costs["+str(len(reg_costs)-1)+"], which is the main cost should be 1.0"))
            failed = True
            if reg_costs[len(reg_costs)-1] > 0.0:
                if labels[len(reg_costs)-1] is None:
                    print(Warning("reg_costs["+str(len(reg_costs)-1)+"] > 0 then labels["+str(len(reg_costs)-1)+
                                         "] has to be an array!"))
                    failed = True
        if labels[len(labels)-1] is None:
            print(Warning("labels["+str(len(labels)-1)+"] has to contain values."))
            failed = True

        # Check norm restrictions
        if restriction_norm != 'Cols' and restriction_norm != 'Rows' and restriction_norm != 'Mat':
            print(Warning("restriction_norm has to be Cols, Rows or Mat"))
            failed = True

        if not isinstance(restrict_gradient,list):
            print(Warning("restrict_gradient has to be a list of length "+str(self.model.num_layers)))
            failed = True
        if not isinstance(epsilon,list):
            print(Warning("epsilon has to be a list of length "+str(self.model.num_layers)))
            failed = True
        if not isinstance(momentum,list):
            print(Warning("momentum has to be a list of length "+str(self.model.num_layers)))
            failed = True
        if not isinstance(update_offsets,list):
            print(Warning("update_offsets has to be a list of length "+str(self.model.num_layers)))
            failed = True
        if not isinstance(reg_L1Norm,list):
            print(Warning("reg_L1Norm has to be a list of length "+str(self.model.num_layers)))
            failed = True
        if not isinstance(reg_L2Norm,list):
            print(Warning("reg_L2Norm has to be a list of length "+str(self.model.num_layers)))
            failed = True
        if not isinstance(corruptor,list) and corruptor is not None:
            print(Warning("corruptor has to be None or a list of length "+str(self.model.num_layers)))
            failed = True
        if not isinstance(reg_sparseness,list):
            print(Warning("reg_sparseness has to be a list of length "+str(self.model.num_layers)))
            failed = True
        if not isinstance(desired_sparseness,list):
            print(Warning("desired_sparseness has to be a list of length "+str(self.model.num_layers)))
            failed = True
        if not isinstance(costs_sparseness,list):
            print(Warning("costs_sparseness has to be a list of length "+str(self.model.num_layers)))
            failed = True
        if not isinstance(reg_costs,list):
            print(Warning("reg_costs has to be a list of length "+str(self.model.num_layers)))
            failed = True
        if not isinstance(costs,list):
            print(Warning("costs has to be a list of length "+str(self.model.num_layers)))
            failed = True
        if not isinstance(labels,list):
            print(Warning("labels has to be a list of length "+str(self.model.num_layers)))
            failed = True

        if len(epsilon) != self.model.num_layers:
            print(Warning("len(epsilon) has to be equal to num _layers"))
            failed = True
        if len(momentum) != self.model.num_layers:
            print(Warning("len(momentum) has to be equal to num _layers"))
            failed = True
        if len(update_offsets) != self.model.num_layers:
            print(Warning("len(update_offsets) has to be equal to num _layers"))
            failed = True
        if len(reg_L1Norm) != self.model.num_layers:
            print(Warning("len(reg_L1Norm) has to be equal to num _layers"))
            failed = True
        if len(reg_L2Norm) != self.model.num_layers:
            print(Warning("len(reg_L2Norm) has to be equal to num _layers"))
            failed = True
        if corruptor is not None:
            if len(corruptor) != self.model.num_layers+1:
                print(Warning("len(corruptor) has to be equal to num _layers+1"))
                failed = True
        if len(reg_sparseness) != self.model.num_layers:
            print(Warning("len(reg_sparseness) has to be equal to num _layers"))
            failed = True
        if len(desired_sparseness) != self.model.num_layers:
            print(Warning("len(desired_sparseness) has to be equal to num _layers"))
            failed = True
        if len(costs_sparseness) != self.model.num_layers:
            print(Warning("len(costs_sparseness) has to be equal to num _layers"))
            failed = True
        if len(reg_costs) != self.model.num_layers:
            print(Warning("len(reg_costs) has to be equal to num _layers"))
            failed = True

        if len(costs) != self.model.num_layers:
            print(Warning("len(costs) has to be equal to num _layers"))
            failed = True
        if len(labels) != self.model.num_layers:
            print(Warning("len(labels) has to be equal to num _layers"))
            failed = True

        if corruptor is not None:
            if not isinstance(corruptor[0], Corr.Identity) and not corruptor[0] is None:
                print(Warning("corruptor["+str(0)+"] has to be None or CFct.CostFunction"))
                failed = True
        # For each layer
        for l in range(self.model.num_layers):
            # Check simple hyperparameter
            if epsilon[l] < 0.0 or epsilon[l] > 1.0:
                print(Warning("epsilon["+str(l)+"] should to be a positive scalar in range [0,1]"))
                failed = True
            if momentum[l] < 0.0 or momentum[l] > 1.0:
                print(Warning("momentum["+str(l)+"] should to be a positive scalar in range [0,1]"))
                failed = True
            if update_offsets[l] < 0.0 or update_offsets[l] > 1:
                print(Warning("reg_L2Norm["+str(l)+"] has to be a positive scalar in range [0,1]"))
                failed = True
            if reg_L1Norm[l] < 0.0 or reg_L1Norm[l] > 0.001:
                print(Warning("reg_L1Norm["+str(l)+"] should to be a positive scalar in range [0,0.001]"))
                failed = True
            if reg_L2Norm[l] < 0.0 or reg_L2Norm[l] > 0.001 :
                print(Warning("reg_L2Norm["+str(l)+"] should to be a positive scalar in range [0,0.001]"))
                failed = True
            if corruptor is not None:
                if corruptor[l+1] is not None and not isinstance(corruptor[l+1],Corr.Identity):
                    print(Warning("corruptor["+str(l+1)+"] has to be None or CFct.CostFunction"))
                    failed = True

            # Check sparseness
            if not numx.isscalar(reg_sparseness[l]) or reg_sparseness[l] < 0.0:
                print(Warning("reg_sparseness["+str(l)+"] has to be a positive scalar"))
                failed = True
            if reg_sparseness[l] > 0.0:
                if reg_sparseness[l] > 1.0:
                    print(Warning("reg_sparseness["+str(l)+"] should not be greater than 1"))
                    failed = True
                if not numx.isscalar(desired_sparseness[l]) or not desired_sparseness[l] > 0.0:
                    print(Warning("reg_sparseness["+str(l)+"] > 0 then desired_sparseness["+str(l)+
                                         "] has to be a positive scalar!"))
                    failed = True
                if not costs_sparseness[l] is not None:
                    print(Warning("costs_sparseness["+str(l)+"] should not be None"))
                    failed = True


            # Check cost
            if not numx.isscalar(reg_costs[l]) or reg_costs[l] < 0.0:
                print(Warning("reg_costs["+str(l)+"] has to be a positive scalar"))
                failed = True
            if reg_costs[l] > 0.0:
                if reg_costs[l] > 1.0:
                    print(Warning("reg_costs["+str(l)+"] should not be greater than 1"))
                    failed = True
                if labels[l] is None:
                    print(Warning("reg_costs["+str(l)+"] > 0 then labels["+str(l)+
                                         "] has to be an array!"))
                    failed = True
                else:
                    if labels[l].shape[1] != self.model.layers[l].output_dim:
                        print(Warning("Label["+str(l)+"] dim. does not match layer["+str(l)+"] output dim"))
                        failed = True
                if costs[l] is not None:
                    if ((costs[l] == CFct.CrossEntropyError or costs[l] == CFct.NegLogLikelihood) and not
                        (self.model.layers[l].activation_function == AFct.SoftMax or
                        self.model.layers[l].activation_function == AFct.Sigmoid)):
                        print(Warning("Layer "+str(l)+": Activation function "+str(self.model.layers[l].activation_function)
                                        +" and cost "+str(costs[l])+" incompatible"))
                        failed = True
                else:
                    print(Warning("costs["+str(l)+"] should not be None"))
                    failed = True
        return not failed


    def train(self,
              data,
              labels,
              costs,
              reg_costs,
              epsilon,
              momentum,
              update_offsets,
              corruptor,
              reg_L1Norm,
              reg_L2Norm,
              reg_sparseness,
              desired_sparseness,
              costs_sparseness,
              restrict_gradient, 
              restriction_norm):
        ''' Train function which performes one step of gradient descent.
            Use check_setup() to check whether your training setup is valid.

        :Parameters:
            data:               Training data as numpy array.
                               -type: numpy arrays [batchsize, inpput dim]

            labels:             List of numpy arrays or None if a layer has no cost, the last layer has to have a cost
                                and thus the last item in labels has to be an array.
                               -type: List of numpy arrays and/or Nones

            costs:              List of Cost functions. The last layer has to have a cost.
                               -type: pydeep.base.costfunction

            reg_costs:          List of scalars controlling the strength of the cost functions. Last entry i.e. 1.
                               -type: scalar

            epsilon:            List of Learning rates.
                               -type: list of scalars

            momentum:           List of Momentum terms.
                               -type: list of scalars

            update_offsets:     List of Shifting factors for centering.
                               -type: list of scalars

            corruptor:          List of Corruptor objects e.g. Dropout.
                               -type: list of pydeep.base.corruptors

            reg_L1Norm:          List of L1 Norm Regularization terms.
                               -type: list of scalars

            reg_L2Norm:         List of L2 Norm Regularization terms.
                               -type: list of scalars

            reg_sparseness:     List of scalars controlling the strength of the sparseness regularization.
                               -type: list of scalars

            desired_sparseness: List of scalars / target sparseness.
                               -type: list of scalars

            costs_sparseness:   List of sparseness cost and/or None values
                               -type: list of pydeep.base.costfunction and/or None

            restrict_gradient:  Maximal norm for the gradient or None
                               -type: list of scalars

            restriction_norm:   Defines how the weights will be restricted 'Cols', 'Rows' or 'Mat'.
                               -type: Strings 'Cols', 'Rows' or 'Mat'

        '''
        # Forward propagate through the entire network, possibly use corrupter states
        output = self.model.forward_propagate(data = data, corruptor = corruptor)

        # Reparameterize the network to the new mean - Update all offests and biases
        for l in range(len(self.model.layers)):
            self.model.layers[l].update_offsets(shift = update_offsets[l], new_mean = None)

        deltas = None
        # Go from top layer to last layer
        for l in range(self.model.num_layers-1, -1, -1):
            # caluclate the delta values
            deltas = self.model.layers[l]._get_deltas(deltas = deltas,
                                                      labels = labels[l],
                                                      cost = costs[l],
                                                      reg_cost = reg_costs[l],
                                                      desired_sparseness = desired_sparseness[l],
                                                      cost_sparseness = costs_sparseness[l],
                                                      reg_sparseness = reg_sparseness[l])

            # backprop the error if it is not first/bottom most layer.
            if l > 0:
                deltas = self.model.layers[l]._backward_propagate()

            # Now we are ready to calculate the gradient
            grad = self.model.layers[l]._calculate_gradient()
            # Possibly add weight decay terms
            if reg_L1Norm[l] > 0.0:
                grad[0] += (reg_L1Norm[l] * numx.sign(self.model.layers[l].weights))
            if reg_L2Norm[l] > 0.0:
                grad[0] += (reg_L2Norm[l] * self.model.layers[l].weights)

            # Apply learning rate
            grad[0] *= epsilon[l]
            grad[1] *= epsilon[l]

            # Restricts the gradient is desired
            if numx.isscalar(restrict_gradient[l]):
                if restrict_gradient[l] > 0:
                    if restriction_norm is 'Cols':
                        grad[0] = numxExt.restrict_norms(grad[0], restrict_gradient[l], 0)
                    if restriction_norm is 'Rows':
                        grad[0] = numxExt.restrict_norms(grad[0], restrict_gradient[l], 1)
                    if restriction_norm is 'Mat':
                        grad[0] = numxExt.restrict_norms(grad[0], restrict_gradient[l], None)

            # Use a momentum
            if momentum[l] > 0.0:
                grad[0] += momentum[l]*self._old_grad[l][0]
                grad[1] += momentum[l]*self._old_grad[l][1]

            # Update the model parameters
            self.model.layers[l].update_parameters([grad[0],grad[1]])
            self._old_grad[l][0] = grad[0]
            self._old_grad[l][1] = grad[1]

class ADAGDTrainer(GDTrainer):
    ''' ADA-Gradient decent feed forward neural network trainer.

    '''

    def __init__(self, model, numerical_stabilty = 1e-6 ):
        ''' Constructor takes a model.

        :Parameters:
            model:              FNN model to train.
                               -type: FNN model.

            master_epsilon:     Master/Default learning rate.
                               -type: float.

            numerical_stabilty: Value added to avoid numerical instabilties by devicion by zero.
                               -type: float.

        '''
        self._numerical_stabilty = numerical_stabilty
        # Call constructor of superclass
        super(ADAGDTrainer, self).__init__(model = model)


    def train(self,
              data,
              labels,
              costs,
              reg_costs,
              epsilon,
              update_offsets,
              corruptor,
              reg_L1Norm,
              reg_L2Norm,
              reg_sparseness,
              desired_sparseness,
              costs_sparseness,
              restrict_gradient,
              restriction_norm):
        ''' Train function which performes one step of gradient descent.
            Use check_setup() to check whether your training setup is valid.

        :Parameters:
            data:               Training data as numpy array.
                               -type: numpy arrays [batchsize, inpput dim]

            labels:             List of numpy arrays or None if a layer has no cost, the last layer has to have a cost
                                and thus the last item in labels has to be an array.
                               -type: List of numpy arrays and/or Nones

            costs:              List of Cost functions. The last layer has to have a cost.
                               -type: pydeep.base.costfunction

            reg_costs:          List of scalars controlling the strength of the cost functions. Last entry i.e. 1.
                               -type: scalar

            epsilon:            List of Learning rates.
                               -type: list of scalars

            update_offsets:     List of Shifting factors for centering.
                               -type: list of scalars

            corruptor:          List of Corruptor objects e.g. Dropout.
                               -type: list of pydeep.base.corruptors

            reg_L1Norm:          List of L1 Norm Regularization terms.
                               -type: list of scalars

            reg_L2Norm:         List of L2 Norm Regularization terms.
                               -type: list of scalars

            reg_sparseness:     List of scalars controlling the strength of the sparseness regularization.
                               -type: list of scalars

            desired_sparseness: List of scalars / target sparseness.
                               -type: list of scalars

            costs_sparseness:   List of sparseness cost and/or None values
                               -type: list of pydeep.base.costfunction and/or None

            restrict_gradient:  Maximal norm for the gradient or None
                               -type: list of scalars

            restriction_norm:   Defines how the weights will be restricted 'Cols', 'Rows' or 'Mat'.
                               -type: Strings 'Cols', 'Rows' or 'Mat'

        '''
        # Forward propagate through the entire network, possibly use corrupter states
        output = self.model.forward_propagate(data = data, corruptor = corruptor)

        # Reparameterize the network to the new mean - Update all offests and biases
        for l in range(len(self.model.layers)):
            self.model.layers[l].update_offsets(shift = update_offsets[l], new_mean = None)

        deltas = None
        # Go from top layer to last layer
        for l in range(self.model.num_layers-1, -1, -1):
            # caluclate the delta values
            deltas = self.model.layers[l]._get_deltas(deltas = deltas,
                                                      labels = labels[l],
                                                      cost = costs[l],
                                                      reg_cost = reg_costs[l],
                                                      desired_sparseness = desired_sparseness[l],
                                                      cost_sparseness = costs_sparseness[l],
                                                      reg_sparseness = reg_sparseness[l])

            # backprop the error if it is not first/bottom most layer.
            if l > 0:
                deltas = self.model.layers[l]._backward_propagate()

            # Now we are ready to calculate the gradient
            grad = self.model.layers[l]._calculate_gradient()
            # Possibly add weight decay terms
            if reg_L1Norm[l] > 0.0:
                grad[0] += (reg_L1Norm[l] * numx.sign(self.model.layers[l].weights))
            if reg_L2Norm[l] > 0.0:
                grad[0] += (reg_L2Norm[l] * self.model.layers[l].weights)

            # Apply learning rate ny ADA rule
            self._old_grad[l][0] += grad[0]**2
            self._old_grad[l][1] += grad[1]**2
            grad[0] /= (self._numerical_stabilty + numx.sqrt(self._old_grad[l][0]))
            grad[1] /= (self._numerical_stabilty + numx.sqrt(self._old_grad[l][1]))
            grad[0] *= epsilon[l]
            grad[1] *= epsilon[l]

            # Restricts the gradient is desired
            if numx.isscalar(restrict_gradient):
                if restrict_gradient > 0:
                    if restriction_norm is 'Cols':
                        grad[0] = numxExt.restrict_norms(grad[0], restrict_gradient, 0)
                    if restriction_norm is 'Rows':
                        grad[0] = numxExt.restrict_norms(grad[0], restrict_gradient, 1)
                    if restriction_norm is 'Mat':
                        grad[0] = numxExt.restrict_norms(grad[0], restrict_gradient, None)
            # Update the model parameters
            self.model.layers[l].update_parameters([grad[0],grad[1]])