''' Test module for AE trainer.

    :Version:
        1.0

    :Date:
        08.02.2016

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2016

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
import unittest
import sys

from pydeep.misc.toyproblems import generate_bars_and_stripes_complete
import pydeep.ae.model as MODEL
import pydeep.ae.trainer as TRAINER
import pydeep.base.activationfunction as AFct
import pydeep.base.costfunction as CFct

print("\n... pydeep.ae.trainer.py")

class Test_AE_Trainer(unittest.TestCase):

    def perform_training(self,
                     ae,
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
                     restriction_norm,
                     num_epochs = 1000):
        numx.random.seed(42)
        tr = TRAINER.GDTrainer(ae)
        tr.train(data = data,
                 num_epochs= num_epochs,
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
        rec1 = numx.mean(ae.energy(x = data,
                         contractive_penalty = reg_contractive,
                         sparse_penalty = reg_sparseness,
                         desired_sparseness=desired_sparseness,
                         x_next=data_next,
                         slowness_penalty = reg_slowness))
        tr.train(data = data,
                 num_epochs= 10,
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
        rec2 = numx.mean(ae.energy(x = data,
                         contractive_penalty = reg_contractive,
                         sparse_penalty = reg_sparseness,
                         desired_sparseness=desired_sparseness,
                         x_next=data_next,
                         slowness_penalty = reg_slowness))
        assert numx.all(rec1 - rec2 >= 0.0)


    def test_trainer(self):
        ''' Checks if Auto encoder converges in terms of rec error.

        '''
        sys.stdout.write('Auto encoder -> Performing trainer convergences check ...')
        sys.stdout.flush()
        data = generate_bars_and_stripes_complete(4)
        data_next = numx.random.permutation(generate_bars_and_stripes_complete(4))

        for act_out in [AFct.Identity,AFct.SoftSign,AFct.Rectifier,AFct.SoftPlus,AFct.Sigmoid,AFct.HyperbolicTangent]:
            for act_in in [AFct.Sigmoid]:
                ae = MODEL.AutoEncoder(number_visibles = 16,
                                               number_hiddens = 20,
                                               data=data,
                                               visible_activation_function=act_in,
                                               hidden_activation_function=act_out,
                                               cost_function=CFct.CrossEntropyError,
                                               initial_weights='AUTO',
                                               initial_visible_bias='AUTO',
                                               initial_hidden_bias='AUTO',
                                               initial_visible_offsets='AUTO',
                                               initial_hidden_offsets='AUTO')
                self.perform_training(ae = ae,
                                              data = data,
                                              epsilon = 0.01,
                                              momentum = 0.0,
                                              update_visible_offsets = 0.0,
                                              update_hidden_offsets = 0.0,
                                              corruptor = None,
                                              reg_L1Norm = 0.0,
                                              reg_L2Norm = 0.0,
                                              reg_sparseness = 0.0,
                                              desired_sparseness = 0.0,
                                              reg_contractive = 0.0,
                                              reg_slowness = 0.0,
                                              data_next = None,
                                              restrict_gradient = 0.0,
                                              restriction_norm = 'Cols')

        for act_out in [AFct.Identity,AFct.SoftSign,AFct.Rectifier,AFct.SoftPlus,AFct.Sigmoid,AFct.HyperbolicTangent]:
            for act_in in [AFct.Identity,AFct.SoftSign,AFct.Rectifier,AFct.SoftPlus,AFct.Sigmoid,AFct.HyperbolicTangent]:
                ae = MODEL.AutoEncoder(number_visibles = 16,
                                               number_hiddens = 20,
                                               data=data,
                                               visible_activation_function=act_in,
                                               hidden_activation_function=act_out,
                                               cost_function=CFct.SquaredError,
                                               initial_weights='AUTO',
                                               initial_visible_bias='AUTO',
                                               initial_hidden_bias='AUTO',
                                               initial_visible_offsets='AUTO',
                                               initial_hidden_offsets='AUTO')
                self.perform_training(ae = ae,
                                              data = data,
                                              epsilon = 0.01,
                                              momentum = 0.0,
                                              update_visible_offsets = 0.0,
                                              update_hidden_offsets = 0.0,
                                              corruptor = None,
                                              reg_L1Norm = 0.0,
                                              reg_L2Norm = 0.0,
                                              reg_sparseness = 0.0,
                                              desired_sparseness = 0.0,
                                              reg_contractive = 0.0,
                                              reg_slowness = 0.0,
                                              data_next = None,
                                              restrict_gradient = 0.0,
                                              restriction_norm = 'Cols')

        for act_out in [AFct.Identity,AFct.SoftSign,AFct.Rectifier,AFct.SoftPlus,AFct.Sigmoid,AFct.HyperbolicTangent]:
            for act_in in [AFct.Identity,AFct.SoftSign,AFct.Rectifier,AFct.SoftPlus,AFct.Sigmoid,AFct.HyperbolicTangent]:
                ae = MODEL.AutoEncoder(number_visibles = 16,
                                               number_hiddens = 20,
                                               data=data,
                                               visible_activation_function=act_in,
                                               hidden_activation_function=act_out,
                                               cost_function=CFct.AbsoluteError,
                                               initial_weights='AUTO',
                                               initial_visible_bias='AUTO',
                                               initial_hidden_bias='AUTO',
                                               initial_visible_offsets='AUTO',
                                               initial_hidden_offsets='AUTO')
                self.perform_training(ae = ae,
                                              data = data,
                                              epsilon = 0.005,
                                              momentum = 0.0,
                                              update_visible_offsets = 0.0,
                                              update_hidden_offsets = 0.0,
                                              corruptor = None,
                                              reg_L1Norm = 0.0,
                                              reg_L2Norm = 0.0,
                                              reg_sparseness = 0.0,
                                              desired_sparseness = 0.0,
                                              reg_contractive = 0.0,
                                              reg_slowness = 0.0,
                                              data_next = None,
                                              restrict_gradient = 0.0,
                                              restriction_norm = 'Cols')

        # Normal
        ae = MODEL.AutoEncoder(number_visibles = 16,
                               number_hiddens = 20,
                               data = None,
                               visible_activation_function = AFct.Sigmoid,
                               hidden_activation_function = AFct.Sigmoid,
                               cost_function = CFct.CrossEntropyError,
                               initial_weights = 'AUTO',
                               initial_visible_bias = 'AUTO',
                               initial_hidden_bias = 'AUTO',
                               initial_visible_offsets = 0,
                               initial_hidden_offsets = 0,
                               dtype = numx.float64)
        self.perform_training(ae = ae,
                          data = data,
                          epsilon = 0.01,
                          momentum = 0.0,
                          update_visible_offsets = 0.0,
                          update_hidden_offsets = 0.0,
                          corruptor = None,
                          reg_L1Norm = 0.000,
                          reg_L2Norm = 0.000,
                          reg_sparseness = 0.0,
                          desired_sparseness = 0.1,
                          reg_contractive = 0.0,
                          reg_slowness = 0.0,
                          data_next = data_next,
                          restrict_gradient = None,
                          restriction_norm = 'Mat')
        # Centered
        ae = MODEL.AutoEncoder(number_visibles = 16,
                               number_hiddens = 20,
                               data = data,
                               visible_activation_function = AFct.Sigmoid,
                               hidden_activation_function = AFct.Sigmoid,
                               cost_function = CFct.CrossEntropyError,
                               initial_weights = 'AUTO',
                               initial_visible_bias = 'AUTO',
                               initial_hidden_bias = 'AUTO',
                               initial_visible_offsets = 'AUTO',
                               initial_hidden_offsets = 'AUTO',
                               dtype = numx.float64)
        self.perform_training(ae = ae,
                          data = data,
                          epsilon = 0.01,
                          momentum = 0.0,
                          update_visible_offsets = 0.01,
                          update_hidden_offsets = 0.01,
                          corruptor = None,
                          reg_L1Norm = 0.000,
                          reg_L2Norm = 0.000,
                          reg_sparseness = 0.0,
                          desired_sparseness = 0.1,
                          reg_contractive = 0.0,
                          reg_slowness = 0.0,
                          data_next = data_next,
                          restrict_gradient = None,
                          restriction_norm = 'Mat')
        # Momentum
        ae = MODEL.AutoEncoder(number_visibles = 16,
                               number_hiddens = 20,
                               data = None,
                               visible_activation_function = AFct.Sigmoid,
                               hidden_activation_function = AFct.Sigmoid,
                               cost_function = CFct.CrossEntropyError,
                               initial_weights = 'AUTO',
                               initial_visible_bias = 'AUTO',
                               initial_hidden_bias = 'AUTO',
                               initial_visible_offsets = 'AUTO',
                               initial_hidden_offsets = 'AUTO',
                               dtype = numx.float64)
        self.perform_training(ae = ae,
                          data = data,
                          epsilon = 0.01,
                          momentum = 0.9,
                          update_visible_offsets = 0.01,
                          update_hidden_offsets = 0.01,
                          corruptor = None,
                          reg_L1Norm = 0.000,
                          reg_L2Norm = 0.000,
                          reg_sparseness = 0.0,
                          desired_sparseness = 0.1,
                          reg_contractive = 0.0,
                          reg_slowness = 0.0,
                          data_next = data_next,
                          restrict_gradient = None,
                          restriction_norm = 'Mat')
        # L1 L2 Norm
        ae = MODEL.AutoEncoder(number_visibles = 16,
                               number_hiddens = 20,
                               data = data,
                               visible_activation_function = AFct.Sigmoid,
                               hidden_activation_function = AFct.Sigmoid,
                               cost_function = CFct.CrossEntropyError,
                               initial_weights = 'AUTO',
                               initial_visible_bias = 'AUTO',
                               initial_hidden_bias = 'AUTO',
                               initial_visible_offsets = 'AUTO',
                               initial_hidden_offsets = 'AUTO',
                               dtype = numx.float64)
        self.perform_training(ae = ae,
                          data = data,
                          epsilon = 0.01,
                          momentum = 0.0,
                          update_visible_offsets = 0.01,
                          update_hidden_offsets = 0.01,
                          corruptor = None,
                          reg_L1Norm = 0.0002,
                          reg_L2Norm = 0.0002,
                          reg_sparseness = 0.0,
                          desired_sparseness = 0.1,
                          reg_contractive = 0.0,
                          reg_slowness = 0.0,
                          data_next = data_next,
                          restrict_gradient = None,
                          restriction_norm = 'Mat')
        # Sparse
        ae = MODEL.AutoEncoder(number_visibles = 16,
                               number_hiddens = 20,
                               data = data,
                               visible_activation_function = AFct.Sigmoid,
                               hidden_activation_function = AFct.Sigmoid,
                               cost_function = CFct.CrossEntropyError,
                               initial_weights = 'AUTO',
                               initial_visible_bias = 'AUTO',
                               initial_hidden_bias = 'AUTO',
                               initial_visible_offsets = 'AUTO',
                               initial_hidden_offsets = 'AUTO',
                               dtype = numx.float64)
        self.perform_training(ae = ae,
                          data = data,
                          epsilon = 0.01,
                          momentum = 0.0,
                          update_visible_offsets = 0.01,
                          update_hidden_offsets = 0.01,
                          corruptor = None,
                          reg_L1Norm = 0.0,
                          reg_L2Norm = 0.0,
                          reg_sparseness = 0.1,
                          desired_sparseness = 0.1,
                          reg_contractive = 0.0,
                          reg_slowness = 0.0,
                          data_next = data_next,
                          restrict_gradient = None,
                          restriction_norm = 'Mat')
        # Contractive
        ae = MODEL.AutoEncoder(number_visibles = 16,
                               number_hiddens = 20,
                               data = data,
                               visible_activation_function = AFct.Sigmoid,
                               hidden_activation_function = AFct.Sigmoid,
                               cost_function = CFct.CrossEntropyError,
                               initial_weights = 'AUTO',
                               initial_visible_bias = 'AUTO',
                               initial_hidden_bias = 'AUTO',
                               initial_visible_offsets = 'AUTO',
                               initial_hidden_offsets = 'AUTO',
                               dtype = numx.float64)
        self.perform_training(ae = ae,
                          data = data,
                          epsilon = 0.01,
                          momentum = 0.0,
                          update_visible_offsets = 0.01,
                          update_hidden_offsets = 0.01,
                          corruptor = None,
                          reg_L1Norm = 0.0,
                          reg_L2Norm = 0.0,
                          reg_sparseness = 0.0,
                          desired_sparseness = 0.1,
                          reg_contractive = 0.1,
                          reg_slowness = 0.0,
                          data_next = data_next,
                          restrict_gradient = None,
                          restriction_norm = 'Mat')
        # Slowness
        ae = MODEL.AutoEncoder(number_visibles = 16,
                               number_hiddens = 20,
                               data = data,
                               visible_activation_function = AFct.Sigmoid,
                               hidden_activation_function = AFct.Sigmoid,
                               cost_function = CFct.CrossEntropyError,
                               initial_weights = 'AUTO',
                               initial_visible_bias = 'AUTO',
                               initial_hidden_bias = 'AUTO',
                               initial_visible_offsets = 'AUTO',
                               initial_hidden_offsets = 'AUTO',
                               dtype = numx.float64)
        self.perform_training(ae = ae,
                          data = data,
                          epsilon = 0.01,
                          momentum = 0.0,
                          update_visible_offsets = 0.01,
                          update_hidden_offsets = 0.01,
                          corruptor = None,
                          reg_L1Norm = 0.0,
                          reg_L2Norm = 0.0,
                          reg_sparseness = 0.0,
                          desired_sparseness = 0.0,
                          reg_contractive = 0.0,
                          reg_slowness = 0.1,
                          data_next = data_next,
                          restrict_gradient = None,
                          restriction_norm = 'Mat')
        # Restrict Mat
        ae = MODEL.AutoEncoder(number_visibles = 16,
                               number_hiddens = 20,
                               data = data,
                               visible_activation_function = AFct.Sigmoid,
                               hidden_activation_function = AFct.Sigmoid,
                               cost_function = CFct.CrossEntropyError,
                               initial_weights = 'AUTO',
                               initial_visible_bias = 'AUTO',
                               initial_hidden_bias = 'AUTO',
                               initial_visible_offsets = 'AUTO',
                               initial_hidden_offsets = 'AUTO',
                               dtype = numx.float64)
        self.perform_training(ae = ae,
                          data = data,
                          epsilon = 0.01,
                          momentum = 0.0,
                          update_visible_offsets = 0.01,
                          update_hidden_offsets = 0.01,
                          corruptor = None,
                          reg_L1Norm = 0.0,
                          reg_L2Norm = 0.0,
                          reg_sparseness = 0.0,
                          desired_sparseness = 0.0,
                          reg_contractive = 0.0,
                          reg_slowness = 0.0,
                          data_next = data_next,
                          restrict_gradient = 0.1,
                          restriction_norm = 'Mat')
        # Restrict rows
        ae = MODEL.AutoEncoder(number_visibles = 16,
                               number_hiddens = 20,
                               data = data,
                               visible_activation_function = AFct.Sigmoid,
                               hidden_activation_function = AFct.Sigmoid,
                               cost_function = CFct.CrossEntropyError,
                               initial_weights = 'AUTO',
                               initial_visible_bias = 'AUTO',
                               initial_hidden_bias = 'AUTO',
                               initial_visible_offsets = 'AUTO',
                               initial_hidden_offsets = 'AUTO',
                               dtype = numx.float64)
        self.perform_training(ae = ae,
                          data = data,
                          epsilon = 0.01,
                          momentum = 0.0,
                          update_visible_offsets = 0.01,
                          update_hidden_offsets = 0.01,
                          corruptor = None,
                          reg_L1Norm = 0.0,
                          reg_L2Norm = 0.0,
                          reg_sparseness = 0.0,
                          desired_sparseness = 0.0,
                          reg_contractive = 0.0,
                          reg_slowness = 0.0,
                          data_next = data_next,
                          restrict_gradient = 0.1,
                          restriction_norm = 'Rows')
        # Restrict Cols
        ae = MODEL.AutoEncoder(number_visibles = 16,
                               number_hiddens = 20,
                               data = data,
                               visible_activation_function = AFct.Sigmoid,
                               hidden_activation_function = AFct.Sigmoid,
                               cost_function = CFct.CrossEntropyError,
                               initial_weights = 'AUTO',
                               initial_visible_bias = 'AUTO',
                               initial_hidden_bias = 'AUTO',
                               initial_visible_offsets = 'AUTO',
                               initial_hidden_offsets = 'AUTO',
                               dtype = numx.float64)
        self.perform_training(ae = ae,
                          data = data,
                          epsilon = 0.01,
                          momentum = 0.0,
                          update_visible_offsets = 0.01,
                          update_hidden_offsets = 0.01,
                          corruptor = None,
                          reg_L1Norm = 0.0,
                          reg_L2Norm = 0.0,
                          reg_sparseness = 0.0,
                          desired_sparseness = 0.0,
                          reg_contractive = 0.0,
                          reg_slowness = 0.0,
                          data_next = data_next,
                          restrict_gradient = 0.1,
                          restriction_norm = 'Cols')
        print(' successfully passed!')
        sys.stdout.flush()


if __name__ is "__main__":
    unittest.main()
