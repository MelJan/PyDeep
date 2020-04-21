''' This module provides implementations for training different variants of Auto-encoders,
    modifications on standard gradient decent are provided (centering, denoising, dropout,
    sparseness, contractiveness, slowness L1-decay, L2-decay, momentum, gradient restriction)

    :Implemented:
        - GDTrainer

    :Info:
        http://ufldl.stanford.edu/wiki/index.php/Sparse_Coding:_Autoencoder_Interpretation

    :Version:
        1.0

    :Date:
        21.01.2018

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2018 Jan Melchior

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
import pydeep.base.numpyextension as npExt
import pydeep.ae.model as MODEL

class GDTrainer(object):
    ''' Auto encoder trainer using gradient descent.

    '''

    def __init__(self, model):

        '''
        The constructor takes the model as input

        :Parameters:
            model:    An auto-encoder object which should be trained.
                     -type: AutoEncoder

        '''
        # Store passed model
        if isinstance(model, MODEL.AutoEncoder):
            self.model = model
        else:
            raise Exception("Model has to be an Auto-encoder object!")

        # Count the number of parameters
        parameters = self.model.get_parameters()
        self.num_parameters = len(parameters)

        # Storage variables for the gradients
        self.parameter_updates = []
        for i in range(self.num_parameters):
            self.parameter_updates.append(numx.zeros((
                                          parameters[i].shape[0],
                                          parameters[i].shape[1]),
                                          dtype=model.dtype))

    def _train(self,
               data,
               epsilon,
               momentum,
               update_visible_offsets,
               update_hidden_offsets,
               corruptor,
               reg_L1Norm,
               reg_L2Norm,
               reg_sparseness,
               desired_sparseness,
               reg_contractive,
               reg_slowness,
               data_next,
               restrict_gradient,
               restriction_norm):

        ''' The training for one batch is performed using gradient descent.

        :Parameters:
            data:                     The training data
                                     -type: numpy array [num samples, input dim]

            epsilon:                  The learning rate.
                                     -type: numpy array[num parameters]

            momentum:                 The momentum term.
                                     -type: numpy array[num parameters]


            update_visible_offsets:  The update step size for the models
                                     visible offsets.
                                     Good value if functionality is used: 0.001
                                    -type: float

            update_hidden_offsets:   The update step size for the models hidden
                                     offsets.
                                     Good value if functionality is used: 0.001
                                    -type: float

            corruptor:                Defines if and how the data gets corrupted.
                                      (e.g. Gauss noise, dropout, Max out)
                                     -type: corruptor

            reg_L1Norm:                The parameter for the L1 regularization
                                     -type: float

            reg_L2Norm:                The parameter for the L2 regularization,
                                      also know as weight decay.
                                     -type: float

            reg_sparseness:           The parameter (epsilon) for the sparseness regularization.
                                    -type: float

            desired_sparseness:      Desired average hidden activation.
                                    -type: float

            reg_contractive:          The parameter (epsilon) for the contractive regularization.
                                    -type: float

            reg_slowness:             The parameter (epsilon) for the slowness regularization.
                                    -type: float

            data_next:               The next training data in the sequence.
                                    -type: numpy array [num samples, input dim]

            restrict_gradient:       If a scalar is given the norm of the
                                     weight gradient is restricted to stay
                                     below this value.
                                    -type: None, float

            restriction_norm:        restricts the column norm, row norm or
                                     Matrix norm.
                                    -type: string: 'Cols','Rows', 'Mat'

        '''
        x_next = None
        h_next = None
        a_h_next = None
        #orginal_h = None
        # Forward propagation, if corruptor is given the data is corrupted
        if corruptor == None:
            x = data
            x_next = data_next
            a_h,h = self.model._encode(x)
            #orginal_h = h
            a_y,y = self.model._decode(h)
            if reg_slowness > 0.0 and data_next is not None:
                a_h_next,h_next = self.model._encode(x_next)
        else:
            #_,orginal_h = self.model._encode(data)
            if isinstance(corruptor, list):
                x = corruptor[0].corrupt(data)
                a_h,h = self.model._encode(x)
                h = corruptor[1].corrupt(h)
                a_y,y = self.model._decode(h)
                y = corruptor[2].corrupt(y)
                if reg_slowness > 0.0 and data_next != None:
                    x_next = corruptor[0].corrupt(data_next)
                    a_h_next,h_next = self.model._encode(x_next)
            else:
                x = corruptor.corrupt(data)
                a_h,h = self.model._encode(x)
                h = corruptor.corrupt(h)
                a_y,y = self.model._decode(h)
                y = corruptor.corrupt(y)
                if reg_slowness > 0.0 and data_next != None:
                    x_next = corruptor.corrupt(data_next)
                    a_h_next,h_next = self.model._encode(x_next)

        # Update offsets
        mean_h = 0.0
        mean_x = 0.0
        if update_visible_offsets > 0.0:
            mean_x = numx.mean(x,axis=0).reshape(1,self.model.input_dim)
        if update_hidden_offsets > 0.0:
            mean_h = numx.mean(h,axis=0).reshape(1,self.model.output_dim)

        self.model.update_offsets(mean_x,
                                  mean_h,
                                  update_visible_offsets,
                                  update_hidden_offsets)

        # Get the gradients for the model
        gradients = self.model._get_gradients(data, a_h, h, a_y, y, reg_contractive, reg_sparseness, desired_sparseness,
                                              reg_slowness, x_next, a_h_next, h_next)

        # adapt parameters
        for i in range(self.num_parameters):
            self.parameter_updates[i] *= momentum[i]
            self.parameter_updates[i] -= epsilon[i] * gradients[i]

        # add weight decay L1 norm
        if reg_L1Norm != 0:
            self.parameter_updates[0] -= (epsilon[0] * reg_L1Norm
                                          * numx.sign(self.model.w))
        # add weight decay L2 norm
        if reg_L2Norm != 0:
            self.parameter_updates[0] -= (epsilon[0] * reg_L2Norm
                                          * self.model.w)

        # Restricts the gradient
        if numx.isscalar(restrict_gradient):
            if restrict_gradient > 0:
                if restriction_norm is 'Cols':
                    typ = 0
                if restriction_norm is 'Rows':
                    typ = 1
                if restriction_norm is 'Mat':
                    typ = None
                self.parameter_updates[0] = npExt.restrict_norms(self.parameter_updates[0], restrict_gradient, typ )

        # update the parameters with the calculated gradient
        self.model.update_parameters(self.parameter_updates)

    def train(self,
              data,
              num_epochs = 1,
              epsilon =  0.1,
              momentum = 0.0,
              update_visible_offsets = 0.0,
              update_hidden_offsets = 0.0,
              corruptor = None,
              reg_L1Norm = 0.0,
              reg_L2Norm = 0.0,
              reg_sparseness = 0.0,
              desired_sparseness = 0.01,
              reg_contractive = 0.0,
              reg_slowness = 0.0,
              data_next = None,
              restrict_gradient = False,
              restriction_norm = 'Mat'):

        ''' The training for one batch is performed using gradient descent.

        :Parameters:
            data:                    The data used for training.
                                    -type: list of numpy arrays
                                           [num samples input dimension]

            num_epochs:              Number of epochs to train.
                                    -type: int

            epsilon:                 The learning rate.
                                     -type: numpy array[num parameters]

            momentum:                The momentum term.
                                     -type: numpy array[num parameters]

            update_visible_offsets:  The update step size for the models
                                     visible offsets.
                                     Good value if functionality is used: 0.001
                                    -type: float

            update_hidden_offsets:   The update step size for the models hidden
                                     offsets.
                                     Good value if functionality is used: 0.001
                                    -type: float

            corruptor:               Defines if and how the data gets corrupted.
                                     -type: corruptor

            reg_L1Norm:              The parameter for the L1 regularization
                                     -type: float

            reg_L2Norm:              The parameter for the L2 regularization,
                                     also know as weight decay.
                                     -type: float


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

            restrict_gradient:       If a scalar is given the norm of the
                                     weight gradient is restricted to stay
                                     below this value.
                                    -type: None, float

            restriction_norm:        restricts the column norm, row norm or
                                     Matrix norm.
                                    -type: string: 'Cols','Rows', 'Mat'

        '''
        # Set learning rates
        if(numx.isscalar(epsilon)):
            epsilon = numx.zeros(self.num_parameters) + epsilon

        # Set momenti
        if(numx.isscalar(momentum)):
            momentum = numx.zeros(self.num_parameters) + momentum

        if isinstance(data,list):
            for _ in range(num_epochs):
                # gradient update for all batches
                for batch in data:
                    self._train(data = batch,
                                epsilon = epsilon,
                                momentum = momentum,
                                update_visible_offsets = update_visible_offsets,
                                update_hidden_offsets = update_hidden_offsets,
                                corruptor = corruptor,
                                reg_L1Norm = reg_L1Norm,
                                reg_L2Norm = reg_L2Norm,
                                reg_sparseness = reg_sparseness,
                                desired_sparseness = desired_sparseness,
                                reg_contractive = reg_contractive,
                                reg_slowness = reg_slowness,
                                data_next = data_next,
                                restrict_gradient = restrict_gradient,
                                restriction_norm = restriction_norm)

        else:
            for _ in range(num_epochs):
                self._train(data = data,
                            epsilon = epsilon,
                            momentum = momentum,
                            update_visible_offsets = update_visible_offsets,
                            update_hidden_offsets = update_hidden_offsets,
                            corruptor = corruptor,
                            reg_L1Norm = reg_L1Norm,
                            reg_L2Norm = reg_L2Norm,
                            reg_sparseness = reg_sparseness,
                            desired_sparseness = desired_sparseness,
                            reg_contractive = reg_contractive,
                            reg_slowness = reg_slowness,
                            data_next = data_next,
                            restrict_gradient = restrict_gradient,
                            restriction_norm = restriction_norm)
