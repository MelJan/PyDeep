''' This module provides different types of training algorithms for RBMs
    running on CPU. The structure is kept modular to simplify the
    understanding of the code and the mathematics. In addition the modularity
    helps to create other kind of training algorithms by inheritance.

    :Implemented:
        - CD   (Contrastive Divergence)
        - PCD  (Persistent Contrastive Divergence)
        - PT   (Parallel Tempering)
        - IPT  (Independent Parallel Tempering)
        - GD   (Exact Gradient descent (only for small binary models))

    :Info:
        For the derivations see:
        http://www.ini.rub.de/data/documents/tns/masterthesis_janmelchior.pdf

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
import exceptions as ex
import pydeep.rbm.sampler as SAMPLER
import pydeep.rbm.estimator as ESTIMATOR
import pydeep.rbm.model as MODEL
import pydeep.base.numpyextension as npExt
import numpy as numx

class CD(object):
    ''' Implementation of the training algorithm Contrastive Divergence (CD).

        :INFO:
            A fast learning algorithm for deep belief nets, Geoffrey E. Hinton
            and Simon Osindero Yee-Whye Teh Department of Computer Science
            University of Toronto Yee-Whye Teh 10 Kings College Road National
            University of Singapore.

    '''

    @classmethod
    def _calculate_centered_gradient(cls,
                                         gradients,
                                         visible_offsets,
                                         hidden_offsets):
        ''' Calculates the centered gradient from the normal CD gradient for
            the parameters W, bv, bh and the corresponding offset values.

        :Parameters:
            gradients:       Original gradients.
                            -type: List of 2D numpy arrays

            visible_offsets: Visible offsets to be used.
                            -type: numpy array[1,input dim]

            hidden_offsets:  Hidden offsets to be used.
                            -type: numpy array[1,output dim]

        :Returns:
            Enhanced gradients for all parameters.
           -type: numpy arrays (num parameters x [parameter.shape])

        '''
        gradients[0] -= numx.dot(gradients[1].T,hidden_offsets) + \
                        numx.dot(visible_offsets.T,gradients[2])
        gradients[1] -= numx.dot(hidden_offsets, gradients[0].T)
        gradients[2] -= numx.dot(visible_offsets, gradients[0])

        return gradients

    def __init__(self, model,data = None):
        ''' The constructor initializes the CD trainer with a given model and
            data.

        :Parameters:
            model: The model to sample from.
                  -type: Valid model class.

            data:  Data for initialization, only has effect if the centered
                   gradient is used.
                  -type: numpy array [num. samples x input dim]
        '''
        # Store model variable
        self.model = model

        # Create the Gibbs-sampler
        self.sampler = SAMPLER.Gibbs_sampler(model)

        # Count the number of parameters
        parameters = self.model.get_parameters()
        self.num_parameters = len(parameters)

        self.hidden_offsets = 0.5*numx.ones((1,model.output_dim))
        if data is not None:
            if self.model.input_dim != data.shape[1]:
                raise ex.ValueError("Data dimension and model input dimension \
                                     have to be equal!")
            self.visible_offsets = data.mean(axis=0).reshape(1,data.shape[1])
        else:
            self.visible_offsets = 0.5*numx.ones((1,model.input_dim))
        # Storage variables for the gradients
        self.parameter_updates = []
        for i in range(self.num_parameters):
            self.parameter_updates.append(numx.zeros((
                                          parameters[i].shape[0],
                                          parameters[i].shape[1]),
                                          dtype=model.dtype))

    def _adapt_gradient(self,
                        pos_gradients,
                        neg_gradients,
                        batch_size,
                        epsilon,
                        momentum,
                        regL1Norm,
                        regL2Norm,
                        regSparseness,
                        desired_sparseness,
                        mean_hidden_activity,
                        visible_offsets,
                        hidden_offsets,
                        use_centered_gradient,
                        restrict_gradient,
                        restriction_norm):
        ''' This function updates the parameter gradients.

        :Parameters:

            pos_gradients:            Positive Gradients.
                                     -type: numpy array[parameter index,
                                                        parameter shape]

            neg_gradients:            Negative Gradients.
                                     -type: numpy array[parameter index,
                                                        parameter shape]

            batch_size:               The batch_size of the data.
                                     -type: float

            epsilon:                  The learning rate.
                                     -type: numpy array[num parameters]

            momentum:                 The momentum term.
                                     -type: numpy array[num parameters]

            regL1Norm:                The parameter for the L1 regularization
                                     -type: float

            regL2Norm:                The parameter for the L2 regularization,
                                      also know as weight decay.
                                     -type: float

            regSparseness:            The parameter for the desired_sparseness.
                                      regularization.
                                     -type: None or float

            desired_sparseness:       Desired average hidden activation or
                                      None for no regularization.
                                     -type: None or float

            mean_hidden_activity:     Average hidden activation
                                      <P(h_i=1|x)>_h_i
                                     -type: numpy array [num samples]

            visible_offsets:          If not zero the gradient is centered
                                      around this value.
                                     -type: float

            hidden_offsets:           If not zero the gradient is centered
                                      around this value.
                                     -type: float

            use_centered_gradient:    Uses the centered gradient instead of
                                      centering.
                                     -type: bool

            restrict_gradient:        If a scalar is given the norm of the
                                      weight gradient (along the input dim) is
                                      restricted to stay below this value.
                                     -type: None, float

            restriction_norm:         restricts the column norm, row norm or
                                      Matrix norm.
                                     -type: string: 'Cols','Rows', 'Mat'



        '''
        # calculate normal gradient
        gradients = []
        for i in range(self.num_parameters):
            gradients.append((pos_gradients[i] - neg_gradients[i])/batch_size)

        # adapt to centered gradient
        if use_centered_gradient:
            gradients = self._calculate_centered_gradient(gradients,
                                                          visible_offsets,
                                                          hidden_offsets)

        # adapt parameters
        for i in range(self.num_parameters):
            self.parameter_updates[i] *= momentum[i]
            self.parameter_updates[i] += epsilon[i] * gradients[i]

        # Add sparse penalty
        if regSparseness != 0:
            if desired_sparseness is not None:
                self.parameter_updates[2] += (epsilon[2] * regSparseness * (
                                              desired_sparseness
                                              - mean_hidden_activity))
                #st = numx.clip(mean_hidden_activity,0.001,0.999)
                #st = -desired_sparseness/st+(1.0-desired_sparseness)/(1.0-st)
                #self.parameter_updates[2] -= epsilon[2] * regSparseness * st

        # add weight decay
        if regL1Norm != 0:
            self.parameter_updates[0] -= (epsilon[0] * regL1Norm
                                          * numx.sign(self.model.w))

        if regL2Norm != 0:
            self.parameter_updates[0] -= (epsilon[0] * regL2Norm
                                          * self.model.w)

        # Restricts the gradient
        if numx.isscalar(restrict_gradient):
            if restrict_gradient > 0:
                if restriction_norm is 'Cols':
                    self.parameter_updates[0] = npExt.restrict_norms(
                                                    self.parameter_updates[0],
                                                    restrict_gradient, 0 )
                if restriction_norm is 'Rows':
                    self.parameter_updates[0] = npExt.restrict_norms(
                                                    self.parameter_updates[0],
                                                    restrict_gradient, 1 )
                if restriction_norm is 'Mat':
                    self.parameter_updates[0] = npExt.restrict_norms(
                                                    self.parameter_updates[0],
                                                    restrict_gradient, None)

    def _train(self,
               data,
               epsilon,
               k,
               momentum,
               regL1Norm,
               regL2Norm,
               regSparseness,
               desired_sparseness,
               update_visible_offsets,
               update_hidden_offsets,
               offset_typ,
               use_centered_gradient,
               restrict_gradient,
               restriction_norm,
               use_hidden_states):
        ''' The training for one batch is performed using Contrastive
            Divergence (CD) for k Gibbs-sampling steps.

        :Parameter:
            data:                    The data used for training.
                                    -type: numpy array [batch_size,
                                                        input dimension]

            epsilon:                 The learning rates.
                                    -type: scalar or
                                           numpy array[num parameters] or
                                           numpy array[num parameters,
                                                       parameter shape]

            k:                       The number of Gibbs sampling steps.
                                    -type: int

            momentum:                The momentum term.
                                    -type: scalar or
                                           numpy array[num parameters] or
                                           numpy array[num parameters,
                                                       parameter shape]

            regL1Norm:               The parameter for the L1 regularization
                                    -type: float

            regL2Norm:               The parameter for the L2 regularization,
                                     also know as weight decay.
                                    -type: float

            regSparseness:           The parameter for the desired_sparseness.
                                     regularization.
                                    -type: None or float

            desired_sparseness:      Desired average hidden activation or
                                     None for no regularization.
                                    -type: None or float

            update_visible_offsets:  The update step size for the models
                                     visible offsets.
                                    -type: float

            update_hidden_offsets:   The update step size for the models
                                     hidden offsets.
                                     -type: float

            offset_typ:              Different offsets can be used to center
                                     the gradient.
                                     Example: 'DM' uses the positive phase
                                              visible mean and the negative
                                              phase hidden mean. 'A0' uses the
                                              average of positive and negative
                                              phase mean for visible, zero for
                                              the hiddens.
                                              Possible values are out of
                                              {A,D,M,0}x{A,D,M,0}
                                    -type: string

            use_centered_gradient:   Uses the centered gradient instead of
                                     centering.
                                    -type: bool

            restrict_gradient:       If a scalar is given the norm of the
                                     weight gradient is restricted to stay
                                     below this value.
                                    -type: None, float

            restriction_norm:        restricts the column norm, row norm or
                                     Matrix norm.
                                    -type: string: 'Cols','Rows', 'Mat'

            use_hidden_states:       If True, the hidden states are used for
                                     the gradient calculations, the hiddens
                                     probabilities otherwise.
                                    -type: bool

        '''

        # Sample the first time
        hid_probs_pos = self.model.probability_h_given_v(data)
        hid_states_pos = self.model.sample_h(hid_probs_pos)

        if update_visible_offsets != 0.0:
            xmean_pos = numx.mean(data,axis=0).reshape(1,self.model.input_dim)
        hmean_pos = 0.0
        if update_hidden_offsets != 0.0 or regSparseness != 0.0:
            if use_hidden_states:
                hmean_pos = numx.mean(hid_states_pos,axis=0
                                      ).reshape(1,self.model.output_dim)
            else:
                hmean_pos = numx.mean(hid_probs_pos,axis=0
                                      ).reshape(1,self.model.output_dim)

        # Perform k steps of Gibbs sampling
        if isinstance(self.sampler, SAMPLER.Gibbs_sampler):
            vis_states_neg = self.sampler.sample_from_h(hid_states_pos,k = k)
        else:
            vis_states_neg = self.sampler.sample(data.shape[0],k = k)
        hid_probs_neg = self.model.probability_h_given_v(vis_states_neg)

        if use_hidden_states:
            hid_states_neg = self.model.sample_h(hid_probs_neg)

        if update_visible_offsets!= 0.0:
            xmean_neg = numx.mean(vis_states_neg,axis=0
                                  ).reshape(1,self.model.input_dim)
        hmean_neg = 0.0
        if update_hidden_offsets != 0.0:
            if use_hidden_states:
                hmean_neg = numx.mean(hid_states_neg,axis=0
                                      ).reshape(1,self.model.output_dim)
            else:
                hmean_neg = numx.mean(hid_probs_neg,axis=0
                                      ).reshape(1,self.model.output_dim)
        new_visible_offsets = 0.0
        if update_visible_offsets != 0.0:
            if offset_typ[0] is 'A':
                new_visible_offsets = (xmean_pos+xmean_neg)*0.5
            if offset_typ[0] is 'D':
                new_visible_offsets = xmean_pos
            if offset_typ[0] is 'M':
                new_visible_offsets = xmean_neg
            if offset_typ[0] is '0':
                new_visible_offsets = 0.0 * xmean_pos
        new_hidden_offsets = 0.0
        if update_hidden_offsets != 0.0:
            if offset_typ[1] is 'A':
                new_hidden_offsets = (hmean_pos+hmean_neg)*0.5
            if offset_typ[1] is 'D':
                new_hidden_offsets = hmean_pos
            if offset_typ[1] is 'M':
                new_hidden_offsets = hmean_neg
            if offset_typ[1] is '0':
                new_hidden_offsets = 0.0 * hmean_pos

        if use_centered_gradient == False:
            # update the centers
            self.model.update_offsets(new_visible_offsets,
                                      new_hidden_offsets,
                                      update_visible_offsets,
                                      update_hidden_offsets)
            self.visible_offsets = 0.0
            self.hidden_offsets = 0.0
        else:
            self.hidden_offsets = ((1.0-update_hidden_offsets)
                                      *self.hidden_offsets
                                      + update_hidden_offsets
                                      *new_hidden_offsets)
            self.visible_offsets = ((1.0-update_visible_offsets)
                                       * self.visible_offsets
                                       + update_visible_offsets
                                       * new_visible_offsets)

        # Calculate positive phase gradient using states or probabilities
        if use_hidden_states:
            pos_gradients = self.model.calculate_gradients(data,
                                                           hid_states_pos)
            neg_gradients = self.model.calculate_gradients(vis_states_neg,
                                                           hid_states_neg)
        else:
            pos_gradients = self.model.calculate_gradients(data,
                                                           hid_probs_pos)
            neg_gradients = self.model.calculate_gradients(vis_states_neg,
                                                           hid_probs_neg)


        # Adapt the gradients by weight decay momentum and learning rate
        self._adapt_gradient(pos_gradients = pos_gradients,
                             neg_gradients = neg_gradients,
                             batch_size = data.shape[0],
                             epsilon = epsilon,
                             momentum = momentum,
                             regL1Norm = regL1Norm,
                             regL2Norm = regL2Norm,
                             regSparseness = regSparseness,
                             desired_sparseness = desired_sparseness,
                             mean_hidden_activity = hmean_pos,
                             visible_offsets = self.visible_offsets,
                             hidden_offsets = self.hidden_offsets,
                             use_centered_gradient = use_centered_gradient,
                             restrict_gradient = restrict_gradient,
                             restriction_norm = restriction_norm)

        # update the parameters with the calculated gradient
        self.model.update_parameters(self.parameter_updates)

    def train(self,
              data,
              num_epochs=1,
              epsilon=0.01,
              k=1,
              momentum = 0.0,
              regL1Norm = 0.0,
              regL2Norm = 0.0,
              regSparseness = 0.0,
              desired_sparseness = None,
              update_visible_offsets = 0.01,
              update_hidden_offsets = 0.01,
              offset_typ = 'DD',
              use_centered_gradient = False,
              restrict_gradient = False,
              restriction_norm = 'Mat',
              use_hidden_states = False):
        ''' Train the models with all batches.

        :Parameters:
            data:                    The data used for training.
                                    -type: list of numpy arrays
                                           [num samples input dimension]

            num_epochs:              Number of epochs to train.
                                    -type: int

            epsilon:                 The learning rate. A scalar sets the
                                     learning for all parameters to the same
                                     value. Good value is often 0.01
                                    -type: float, numpy array [num parameters]

            k:                       The number of Gibbs sampling steps.
                                     Good value if functionality is used:
                                     The bigger the better but also
                                     computationally more expensive.
                                    -type: int

            momentum:                The momentum term. A scalar sets the
                                     momentum for all parameters to the same
                                     value.
                                     Good value if functionality is used: 0.9.
                                    -type: float , numpy array [num parameters]

            regL1Norm:               The parameter for the L1 regularization
                                    -type: float

            regL2Norm:               The parameter for the L2 regularization,
                                     also know as weight decay.
                                    -type: float

            regSparseness:           The parameter for the desired_sparseness.
                                     regularization.
                                    -type: None or float

            desired_sparseness:      Desired average hidden activation or None
                                     for no regularization. Good value if
                                     functionality is used: 0.05.
                                    -type: float or None

            update_visible_offsets:  The update step size for the models
                                     visible offsets.
                                     Good value if functionality is used: 0.001
                                    -type: float

            update_hidden_offsets:   The update step size for the models hidden
                                     offsets.
                                     Good value if functionality is used: 0.001
                                    -type: float

            offset_typ:              Different offsets can be used to center
                                     the gradient.
                                     Example: 'DM' uses the positive phase
                                               visible mean and the negative
                                               phase hidden mean. 'A0' uses the
                                               average of positive and negative
                                               phase mean for visible, zero for
                                               the hiddens.
                                               Possible values are out of
                                               {A,D,M,0}x{A,D,M,0}
                                    -type: string

            use_centered_gradient:   Uses the centered gradient instead of
                                     centering.
                                    -type: bool

            restrict_gradient:       If a scalar is given the norm of the
                                     weight gradient is restricted to stay
                                     below this value.
                                    -type: None, float

            restriction_norm:        restricts the column norm, row norm or
                                     Matrix norm.
                                    -type: string: 'Cols','Rows', 'Mat'

            use_hidden_states:       If True, the hidden states are used for
                                     the gradient calculations, the hiddens
                                     probabilities otherwise. Adds noise on the
                                     gradient, helps for training GB-RBMs.
                                    -type: bool

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
                                k = k,
                                momentum = momentum,
                                regL1Norm = regL1Norm,
                                regL2Norm = regL2Norm,
                                regSparseness = regSparseness,
                                desired_sparseness = desired_sparseness,
                                update_visible_offsets = update_visible_offsets,
                                update_hidden_offsets = update_hidden_offsets,
                                offset_typ = offset_typ,
                                use_centered_gradient = use_centered_gradient,
                                restrict_gradient = restrict_gradient,
                                restriction_norm = restriction_norm,
                                use_hidden_states = use_hidden_states)
        else:
            for _ in range(num_epochs):
                self._train(data = data,
                            epsilon = epsilon,
                            k = k,
                            momentum = momentum,
                            regL1Norm = regL1Norm,
                            regL2Norm = regL2Norm,
                            regSparseness = regSparseness,
                            desired_sparseness = desired_sparseness,
                            update_visible_offsets = update_visible_offsets,
                            update_hidden_offsets = update_hidden_offsets,
                            offset_typ = offset_typ,
                            use_centered_gradient = use_centered_gradient,
                            restrict_gradient = restrict_gradient,
                            restriction_norm = restriction_norm,
                            use_hidden_states = use_hidden_states)

class PCD(CD):
    ''' Implementation of the training algorithm Persistent Contrastive
        Divergence (PCD).

        :Reference:
            Training Restricted Boltzmann Machines using Approximations to the
            Likelihood Gradient, Tijmen Tieleman, Department of Computer
            Science, University of Toronto, Toronto, Ontario M5S 3G4, Canada

    '''

    def __init__(self,
                 model,
                 num_chains,
                 data = None):
        ''' The constructor initializes the PCD trainer with a given model and
            data.

        :Parameters:
            model:      The model to sample from.
                       -type: Valid model class.

            num_chains: The number of chains that should be used.
                        Note: You should use the data's batch size!
                       -type: int

            data:       Data for initialization, only has effect if the
                        centered
                        gradient is used.
                       -type: numpy array [num. samples x input dim]
        '''
        # Call super constructor of CD
        super(PCD, self).__init__(model, data)
        self.sampler = SAMPLER.Persistent_Gibbs_sampler(model, num_chains)

class PT(CD):
    ''' Implementation of the training algorithm Parallel Tempering Contrastive
        Divergence (PT).

        :Reference:
            Parallel Tempering for Training of Restricted Boltzmann Machines,
            Guillaume Desjardins, Aaron Courville, Yoshua Bengio, Pascal
            Vincent, Olivier Delalleau, Dept. IRO, Universite de Montreal P.O.
            Box 6128, Succ. Centre-Ville, Montreal, H3C 3J7, Qc, Canada.
    '''

    def __init__(self,
                  model,
                  betas=3,
                  data = None):
        ''' The constructor initializes the IPT trainer with a given model and
            data.

        :Parameters:
            model: The model to sample from.
                  -type: Valid model class.

            betas: List of inverse temperatures to sample from. If a scalar is
                   given, the temperatures will be set linearly from 0.0 to
                   1.0 in 'betas' steps.
                  -type: int, numpy array [num betas]

        '''
        # Call super constructor of CD
        super(PT, self).__init__(model, data)
        if numx.isscalar(betas):
            self.sampler = SAMPLER.Parallel_Tempering_sampler(model,betas,None)
        else:
            self.sampler = SAMPLER.Parallel_Tempering_sampler(model,
                                                              betas.shape[0],
                                                              betas)

class IPT(CD):
    ''' Implementation of the training algorithm Independent Parallel Tempering
        Contrastive Divergence (IPT). As normal PT but the chain's switches are
        done only from one batch to the next instead of from one sample to the
        next.

        :Reference:
            Parallel Tempering for Training of Restricted Boltzmann Machines,
            Guillaume Desjardins, Aaron Courville, Yoshua Bengio, Pascal
            Vincent, Olivier Delalleau, Dept. IRO, Universite de Montreal P.O.
            Box 6128, Succ. Centre-Ville, Montreal, H3C 3J7, Qc, Canada.

    '''

    def __init__(self,
                 model,
                 num_samples,
                 betas=3,
                 data = None):
        ''' The constructor initializes the IPT trainer with a given model and
            data.

        :Parameters:
            model:       The model to sample from.
                        -type: Valid model class.

            num_samples: The number of Samples to produce.
                         Note you should use the batchsize.
                        -type: int

            betas:       List of inverse temperatures to sample from. If a
                         scalar is given, the temperatures will be set linearly
                         from 0.0 to  1.0 in 'betas' steps.
                        -type: int, numpy array [num betas]

            data:        Data for initialization, only has effect if the
                         centered gradient is used.
                        -type: numpy array [num. samples x input dim]

        '''
        # Call super constructor of CD
        super(IPT, self).__init__(model,data)

        if numx.isscalar(betas):
            self.sampler = SAMPLER.Independent_Parallel_Tempering_sampler(
                                                                model,
                                                                num_samples,
                                                                betas,
                                                                None)
        else:
            self.sampler = SAMPLER.Independent_Parallel_Tempering_sampler(
                                                        model,
                                                        num_samples,
                                                        betas.shape[0],
                                                        betas)

class GD(CD):
    ''' Implementation of the training algorithm Gradient descent. Since it
        involves the calculation of the partition function for each update,
        it is only possible for small BBRBMs.

    '''

    def __init__(self,
                 model,
                 data = None):
        ''' The constructor initializes the Gradient trainer with a given
            model.

        :Parameters:
            model: The model to sample from.
                  -type: Valid model class.


            data:  Data for initialization, only has effect if the centered
                   gradient is used.
                  -type: numpy array [num. samples x input dim]

        '''
        if not isinstance(model,MODEL.BinaryBinaryRBM):
            raise ex.ValueError("True gradient only possible for "
                                +"Binary Binary RBMs!")

        # Call super constructor of CD
        super(GD, self).__init__(model, data)

    def _train(self,
               data,
               epsilon,
               k,
               momentum,
               regL1Norm,
               regL2Norm,
               regSparseness,
               desired_sparseness,
               update_visible_offsets,
               update_hidden_offsets,
               offset_typ,
               use_centered_gradient,
               restrict_gradient,
               restriction_norm,
               use_hidden_states):
        ''' The training for one batch is performed using True
            Gradient (GD) for k Gibbs-sampling steps.

        :Parameter:
            data:                    The data used for training.
                                    -type: numpy array [batch_size,
                                                        input dimension]

            epsilon:                 The learning rates.
                                    -type: scalar or
                                           numpy array[num parameters] or
                                           numpy array[num parameters,
                                                       parameter shape]

            k:                       The number of Gibbs sampling steps.
                                    -type: int

            momentum:                The momentum term.
                                    -type: scalar or
                                           numpy array[num parameters] or
                                           numpy array[num parameters,
                                                       parameter shape]

            regL1Norm:               The parameter for the L1 regularization
                                    -type: float

            regL2Norm:               The parameter for the L2 regularization,
                                     also know as weight decay.
                                    -type: float

            regSparseness:           The parameter for the desired_sparseness.
                                     regularization.
                                    -type: None or float

            desired_sparseness:      Desired average hidden activation or
                                     None for no regularization.
                                    -type: None or float

            update_visible_offsets:  The update step size for the models
                                     visible offsets.
                                    -type: float

            update_hidden_offsets:   The update step size for the models
                                     hidden offsets.
                                     -type: float

            offset_typ:              Different offsets can be used to center
                                     the gradient.
                                     Example: 'DM' uses the positive phase
                                              visible mean and the negative
                                              phase hidden mean. 'A0' uses the
                                              average of positive and negative
                                              phase mean for visible, zero for
                                              the hiddens.
                                              Possible values are out of
                                              {A,D,M,0}x{A,D,M,0}
                                    -type: string

            use_centered_gradient:   Uses the centered gradient instead of
                                     centering.
                                    -type: bool

            restrict_gradient:       If a scalar is given the norm of the
                                     weight gradient is restricted to stay
                                     below this value.
                                    -type: None, float

            restriction_norm:        restricts the column norm, row norm or
                                     Matrix norm.
                                    -type: string: 'Cols','Rows', 'Mat'

            use_hidden_states:       If True, the hidden states are used for
                                     the gradient calculations, the hiddens
                                     probabilities otherwise.
                                    -type: bool

        '''
        # Sample the first time
        hid_probs_pos = self.model.probability_h_given_v(data)

        if update_visible_offsets != 0.0:
            xmean_pos = numx.mean(data,axis=0).reshape(1,self.model.input_dim)
        hmean_pos = 0.0
        if update_hidden_offsets != 0.0 or regSparseness != 0.0:
            if use_hidden_states:
                hid_states_pos = self.model.sample_h(hid_probs_pos)
                hmean_pos = numx.mean(hid_states_pos,axis=0
                                      ).reshape(1,self.model.output_dim)
            else:
                hmean_pos = numx.mean(hid_probs_pos,axis=0
                                      ).reshape(1,self.model.output_dim)

        # Calculate the partition function
        ln_Z = 0.0
        if self.model.input_dim < self.model.output_dim:
            batch_size = numx.min([self.model.input_dim,12])
            ln_Z = ESTIMATOR.partition_function_factorize_v(
                                                 self.model, beta=1.0,
                                                 batchsize_exponent=batch_size,
                                                 status=False)
        else:
            batch_size = numx.min([self.model.output_dim,12])
            ln_Z = ESTIMATOR.partition_function_factorize_h(
                                                 self.model,
                                                 beta=1.0,
                                                 batchsize_exponent=batch_size,
                                                 status=False)

        # empty negative phase parts
        neg_gradients = [numx.zeros((self.model.w.shape)),
                         numx.zeros((self.model.bv.shape)),
                         numx.zeros((self.model.bh.shape))]

        # Calculate gradient stepwise in batches
        bit_length = self.model.input_dim

        batchSize = numx.power(2, batch_size)
        num_combinations = numx.power(2, bit_length)
        num_batches = num_combinations / batchSize
        bitCombinations = numx.zeros((batchSize, self.model.input_dim))

        for batch in range(0, num_batches ):
            # Generate current batch
            bitCombinations = npExt.generate_binary_code(bit_length,
                                                         batch_size, batch)
            # P(x)
            prob_x = numx.exp(
                     self.model.log_probability_v(ln_Z,bitCombinations))
            # P(h|x)
            prob_h_x = self.model.probability_h_given_v(bitCombinations)
            # Calculate gradient
            neg_gradients[1] += numx.sum(numx.tile(prob_x,
                                                   (1, self.model.input_dim))
                                         *(bitCombinations -
                                           self.model.ov)
                                         , axis = 0)
            prob_x = (numx.tile(prob_x,
                               (1,self.model.output_dim))
                      *(prob_h_x-self.model.oh))
            neg_gradients[0] += numx.dot((bitCombinations-
                                          self.model.ov).T,prob_x)
            neg_gradients[2] += numx.sum(prob_x, axis = 0)

        if update_visible_offsets != 0.0 and (offset_typ[0] is 'A' or offset_typ[0] is 'M'):
            bitCombinations = npExt.generate_binary_code(self.model.input_dim, None, 0)
            prob_x = numx.exp(self.model.log_probability_v(ln_Z,bitCombinations))
            xmean_neg = numx.sum(prob_x*bitCombinations,axis=0).reshape(1,self.model.input_dim)

        if update_hidden_offsets != 0.0 and (offset_typ[1] is 'A' or offset_typ[1] is 'M'):
            bitCombinations = npExt.generate_binary_code(self.model.output_dim, None, 0)
            prob_h = numx.exp(self.model.log_probability_h(ln_Z,bitCombinations))
            hmean_neg = numx.sum(prob_h*bitCombinations,axis=0).reshape(1,self.model.output_dim)

        new_visible_offsets = 0.0
        if update_visible_offsets != 0.0:
            if offset_typ[0] is 'A':
                new_visible_offsets = (xmean_pos+xmean_neg)*0.5
            if offset_typ[0] is 'D':
                new_visible_offsets = xmean_pos
            if offset_typ[0] is 'M':
                new_visible_offsets = xmean_neg
            if offset_typ[0] is '0':
                new_visible_offsets = 0.0 * xmean_pos
        new_hidden_offsets = 0.0
        if update_hidden_offsets != 0.0:
            if offset_typ[1] is 'A':
                new_hidden_offsets = (hmean_pos+hmean_neg)*0.5
            if offset_typ[1] is 'D':
                new_hidden_offsets = hmean_pos
            if offset_typ[1] is 'M':
                new_hidden_offsets = hmean_neg
            if offset_typ[1] is '0':
                new_hidden_offsets = 0.0 * hmean_pos

        if use_centered_gradient == False:
            # update the centers
            self.model.update_offsets(new_visible_offsets,
                                      new_hidden_offsets,
                                      update_visible_offsets,
                                      update_hidden_offsets)
            self.visible_offsets = 0.0
            self.hidden_offsets = 0.0
        else:
            self.hidden_offsets = ((1.0-update_hidden_offsets)
                                      *self.hidden_offsets
                                      + update_hidden_offsets
                                      *new_hidden_offsets)
            self.visible_offsets = ((1.0-update_visible_offsets)
                                       * self.visible_offsets
                                       + update_visible_offsets
                                       * new_visible_offsets)

        # Calculate positive phase gradient using states or probabilities
        if use_hidden_states:
            pos_gradients = self.model.calculate_gradients(data,
                                                           hid_states_pos)
        else:
            pos_gradients = self.model.calculate_gradients(data,
                                                           hid_probs_pos)

        # Times batch size since adpat gradient devides by batchsize
        neg_gradients[0] *= data.shape[0]
        neg_gradients[1] *= data.shape[0]
        neg_gradients[2] *= data.shape[0]

        # Adapt the gradients by weight decay momentum and learning rate
        self._adapt_gradient(pos_gradients = pos_gradients,
                             neg_gradients = neg_gradients,
                             batch_size = data.shape[0],
                             epsilon = epsilon,
                             momentum = momentum,
                             regL1Norm = regL1Norm,
                             regL2Norm = regL2Norm,
                             regSparseness = regSparseness,
                             desired_sparseness = desired_sparseness,
                             mean_hidden_activity = hmean_pos,
                             visible_offsets = self.visible_offsets,
                             hidden_offsets = self.hidden_offsets,
                             use_centered_gradient = use_centered_gradient,
                             restrict_gradient = restrict_gradient,
                             restriction_norm = restriction_norm)

        # update the parameters with the calculated gradient
        self.model.update_parameters(self.parameter_updates)

