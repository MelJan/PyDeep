''' Test module for WE models.

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
import pydeep.ae.model as TRAINER
import pydeep.base.activationfunction as AFct
import pydeep.base.costfunction as CFct

print("\n... pydeep.ae.model.py")

class Test_AE_Model(unittest.TestCase):

    def test_init(self):
        sys.stdout.write('Auto encoder -> Performing auto encoder initialization test ...')
        sys.stdout.flush()
        ae = MODEL.AutoEncoder(number_visibles = 10,
                               number_hiddens = 10,
                               data=None,
                               visible_activation_function=AFct.SoftPlus,
                               hidden_activation_function=AFct.SoftPlus,
                               cost_function=CFct.SquaredError,
                               initial_weights='AUTO',
                               initial_visible_bias='AUTO',
                               initial_hidden_bias='AUTO',
                               initial_visible_offsets='AUTO',
                               initial_hidden_offsets='AUTO')
        assert numx.all(ae.bv == 0.0)
        assert numx.all(ae.bh == 0.0)

        ae = MODEL.AutoEncoder(number_visibles = 10,
                               number_hiddens = 10,
                               data=None,
                               visible_activation_function=AFct.SoftPlus,
                               hidden_activation_function=AFct.SoftPlus,
                               cost_function=CFct.SquaredError,
                               initial_weights='AUTO',
                               initial_visible_bias='INVERSE_SIGMOID',
                               initial_hidden_bias='INVERSE_SIGMOID',
                               initial_visible_offsets='AUTO',
                               initial_hidden_offsets='AUTO')
        assert numx.all(ae.bv == 0.0)
        assert numx.all(ae.bh == 0.0)
        print(' successfully passed!')
        sys.stdout.flush()

    def check_all(self, data, epsilon, contractive, sparseness, desired_sparseness, data_next, slowness_penalty):
        ''' Checks several possible combinations.

        '''
        N = data.shape[1]
        M = 2*data.shape[1]
        weights = numx.random.randn(N,M)*0.1
        bv = numx.random.randn(1,N)*0.1
        bh = numx.random.randn(1,M)*0.1
        ov = numx.random.random((1,N))
        oh = numx.random.random((1,M))
        for loss in [CFct.SquaredError,CFct.CrossEntropyError]:
            for act_in in [AFct.Identity,AFct.SoftPlus,AFct.Sigmoid,AFct.HyperbolicTangent,AFct.RadialBasis()]:
                for act_out in [AFct.Identity,AFct.SoftPlus,AFct.Sigmoid,AFct.HyperbolicTangent,AFct.RadialBasis()]:
                    if (loss != CFct.CrossEntropyError  or (loss == CFct.CrossEntropyError and act_in == AFct.Sigmoid)):
                        ae   =   MODEL.AutoEncoder(number_visibles = N,
                                               number_hiddens = M,
                                               data=None,
                                               visible_activation_function=act_in,
                                               hidden_activation_function=act_out,
                                               cost_function=loss,
                                               initial_weights=weights,
                                               initial_visible_bias=bv,
                                               initial_hidden_bias=bh,
                                               initial_visible_offsets=0,
                                               initial_hidden_offsets=0)
                        w,b,c = ae.finit_differences(data, 0.001, sparseness, desired_sparseness, contractive,
                                                       slowness_penalty,data_next)
                        maxW = numx.max(numx.abs(w))
                        maxb =  numx.max(numx.abs(b))
                        maxc =  numx.max(numx.abs(c))
                        if  maxW > 0.0001 or maxb > 0.0001 or maxc > 0.0001  :
                            print("Gradient check failed for ae with: ")
                            print(" CENTERING ",loss," ",act_in," ",act_out)
                        assert numx.all(maxW < 0.0001)
                        assert numx.all(maxb < 0.0001)
                        assert numx.all(maxc < 0.0001)

                        ae   =   MODEL.AutoEncoder(number_visibles = N,
                                                   number_hiddens = M,
                                                   data=None,
                                                   visible_activation_function=act_in,
                                                   hidden_activation_function=act_out,
                                                   cost_function=loss,
                                                   initial_weights=weights,
                                                   initial_visible_bias=bv,
                                                   initial_hidden_bias=bh,
                                                   initial_visible_offsets=ov,
                                                   initial_hidden_offsets=oh)

                        w,b,c = ae.finit_differences(data, 0.001, sparseness, desired_sparseness, contractive,
                                                       slowness_penalty,data_next)
                        maxW = numx.max(numx.abs(w))
                        maxb =  numx.max(numx.abs(b))
                        maxc =  numx.max(numx.abs(c))
                        if  maxW > 0.0001 or maxb > 0.0001 or maxc > 0.0001  :
                            print("Gradient check failed for ae with: ")
                            print(" CENTERING ",loss," ",act_in," ",act_out)
                            print(maxW,'\t',maxb,'\t',maxc)
                        assert numx.all(maxW < 0.0001)
                        assert numx.all(maxb < 0.0001)
                        assert numx.all(maxc < 0.0001)

    def test_model_gradients(self):
        ''' Checks all possible combinations (act fct., penalties, etc.) by finite differences.

        '''
        sys.stdout.write('Auto encoder -> Performing finite differences check of all possible Auto encoder models ...')
        sys.stdout.flush()
        data = generate_bars_and_stripes_complete(5)[0:2]
        data_next = generate_bars_and_stripes_complete(5)[2:4]
        self.check_all(data = data, epsilon = 0.0001, contractive = 0.0, sparseness = 0.0, desired_sparseness = 0.0,
                       data_next = data_next, slowness_penalty = 0.0)
        self.check_all(data = data, epsilon = 0.0001, contractive = 0.1, sparseness = 0.0, desired_sparseness = 0.0,
                       data_next = data_next, slowness_penalty = 0.0)
        self.check_all(data = data, epsilon = 0.0001, contractive = 0.0, sparseness = 0.0, desired_sparseness = 0.0,
                       data_next = data_next, slowness_penalty = 0.1)
        self.check_all(data = data, epsilon = 0.0001, contractive = 0.0, sparseness = 0.1, desired_sparseness = 0.1,
                       data_next = data_next, slowness_penalty = 0.0)
        self.check_all(data = data, epsilon = 0.0001, contractive = 0.1, sparseness = 0.1, desired_sparseness = 0.1,
                       data_next = data_next, slowness_penalty = 0.1)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_ae_gradients(self):
        ''' Checks all possible combinations by finite differences.

        '''
        sys.stdout.write('Auto encoder -> Performing finite differences check of all possible normal auto encoder ...')
        sys.stdout.flush()
        data = generate_bars_and_stripes_complete(5)[0:2]
        data_next = generate_bars_and_stripes_complete(5)[2:4]
        self.check_all(data=data, epsilon=0.0001, contractive=0.0, sparseness=0.0, desired_sparseness=0.0,
                       data_next=data_next, slowness_penalty=0.0)
        print(' successfully passed!')
        sys.stdout.flush()


    def test_cae_gradients(self):
        ''' Checks all possible combinations by finite differences.

        '''
        sys.stdout.write('Auto encoder -> Performing finite differences check of all possible contractive auto encoder models ...')
        sys.stdout.flush()
        data = generate_bars_and_stripes_complete(5)[0:2]
        data_next = generate_bars_and_stripes_complete(5)[2:4]
        self.check_all(data=data, epsilon=0.0001, contractive=0.1, sparseness=0.0, desired_sparseness=0.0,
                       data_next=data_next, slowness_penalty=0.0)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_sae_gradients(self):
        ''' Checks all possible combinations  by finite differences.

        '''
        sys.stdout.write('Auto encoder -> Performing finite differences check of all possible sparse auto encoder models ...')
        sys.stdout.flush()
        data = generate_bars_and_stripes_complete(5)[0:2]
        data_next = generate_bars_and_stripes_complete(5)[2:4]
        self.check_all(data=data, epsilon=0.0001, contractive=0.0, sparseness=0.1, desired_sparseness=0.1,
                       data_next=data_next, slowness_penalty=0.0)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_slae_gradients(self):
        ''' Checks all possible combinations by finite differences.

        '''
        sys.stdout.write('Auto encoder -> Performing finite differences check of all possible slowness auto encoder models ...')
        sys.stdout.flush()
        data = generate_bars_and_stripes_complete(5)[0:2]
        data_next = generate_bars_and_stripes_complete(5)[2:4]
        self.check_all(data=data, epsilon=0.0001, contractive=0.0, sparseness=0.0, desired_sparseness=0.0,
                       data_next=data_next, slowness_penalty=0.1)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_allpenelties_gradients(self):
        ''' Checks all possible combinations (act fct., penalties, etc.) by finite differences.

        '''
        sys.stdout.write('Auto encoder -> Performing finite differences check of all possible contractive+sparse+slow auto encoder models ...')
        sys.stdout.flush()
        data = generate_bars_and_stripes_complete(5)[0:2]
        data_next = generate_bars_and_stripes_complete(5)[2:4]
        self.check_all(data=data, epsilon=0.0001, contractive=0.1, sparseness=0.1, desired_sparseness=0.1,
                       data_next=data_next, slowness_penalty=0.1)
        print(' successfully passed!')
        sys.stdout.flush()

if __name__ is "__main__":
    unittest.main()
