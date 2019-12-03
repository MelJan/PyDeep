''' Test module for sae models.

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

import pydeep.ae.sae as STACK
import pydeep.ae.model as MODEL
import pydeep.base.activationfunction as AFct
import pydeep.base.costfunction as CFct

print("\n... pydeep.ae.sae.py")


class Test_SAE_Model(unittest.TestCase):

    numx.random.seed(42)
    ae1 = MODEL.AutoEncoder(number_visibles=2,
                            number_hiddens=4,
                            data=None,
                            visible_activation_function=AFct.SoftPlus,
                            hidden_activation_function=AFct.SoftPlus,
                            cost_function=CFct.SquaredError,
                            initial_weights=0.1,
                            initial_visible_bias='AUTO',
                            initial_hidden_bias='AUTO',
                            initial_visible_offsets='AUTO',
                            initial_hidden_offsets='AUTO')
    ae2 = MODEL.AutoEncoder(number_visibles=4,
                            number_hiddens=2,
                            data=None,
                            visible_activation_function=AFct.SoftPlus,
                            hidden_activation_function=AFct.SoftPlus,
                            cost_function=CFct.SquaredError,
                            initial_weights=0.1,
                            initial_visible_bias='INVERSE_SIGMOID',
                            initial_hidden_bias='INVERSE_SIGMOID',
                            initial_visible_offsets='AUTO',
                            initial_hidden_offsets='AUTO')
    stack = STACK.SAE([ae1, ae2])

    forward_target = numx.array([[0.66395408, 0.65454106], [ 0.6444802,  0.62371954]])
    backward_target = numx.array([[0.70836512, 0.69824519], [0.69078643, 0.68180852]])
    rec_target = numx.array([[0.71608233, 0.70979876], [0.71634822, 0.71010236]])

    def test_forward(self):
        sys.stdout.write('Stacked auto encoder -> Performing forward backward prop test ...')
        sys.stdout.flush()

        assert numx.sum(numx.abs(Test_SAE_Model.stack.forward_propagate(numx.array([[1, 2], [3, 4]])) - Test_SAE_Model.forward_target)) < 0.000001

        print(' successfully passed!')
        sys.stdout.flush()

    def test_backward(self):
        sys.stdout.write('Stacked auto encoder -> Performing forward backward prop test ...')
        sys.stdout.flush()

        assert numx.sum(numx.abs(Test_SAE_Model.stack.backward_propagate(
            numx.array([[1, 2], [3, 4]])) - Test_SAE_Model.backward_target)) < 0.000001

        print(' successfully passed!')
        sys.stdout.flush()

    def test_reconstruct(self):
        sys.stdout.write('Stacked auto encoder -> Performing forward backward prop test ...')
        sys.stdout.flush()

        assert numx.sum(numx.abs(
            Test_SAE_Model.stack.backward_propagate(Test_SAE_Model.stack.forward_propagate(
                numx.array([[1, 2], [3, 4]]))) - Test_SAE_Model.rec_target)) < 0.000001
        assert numx.sum(numx.abs(Test_SAE_Model.stack.reconstruct(
            numx.array([[1, 2], [3, 4]])) - Test_SAE_Model.rec_target)) < 0.000001
        print(' successfully passed!')
        sys.stdout.flush()


if __name__ is "__main__":
    unittest.main()
