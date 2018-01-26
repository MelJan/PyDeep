''' Test module for dbn models.
    
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

'''
import numpy as numx
import unittest
import sys

import pydeep.rbm.dbn as STACK
import pydeep.rbm.model as MODEL

print("\n... pydeep.rbm.dbn.py")


class TestDBNModel(unittest.TestCase):

    numx.random.seed(42)
    rbm1 = MODEL.BinaryBinaryRBM(number_visibles=2,
                                 number_hiddens=4)
    rbm2 = MODEL.BinaryBinaryRBM(number_visibles=4,
                                 number_hiddens=2)
    stack = STACK.DBN([rbm1, rbm2])

    def test_forward_propagate(self):
        sys.stdout.write('Deep Believe Network -> Performing forward_propagate test ...')
        sys.stdout.flush()

        forward_target = numx.array([[0.54640997, 0.8437009], [0.45359003, 0.1562991]])

        assert numx.sum(numx.abs(
            TestDBNModel.stack.forward_propagate(numx.array([[1, 0], [0, 1]])) - forward_target)) < 0.000001

        numx.random.seed(42)
        forward_target = numx.array([[1, 1], [1, 0]])
        assert numx.sum(
            numx.abs(TestDBNModel.stack.forward_propagate(numx.array([[1, 0], [0, 1]]),
                                                          True) - forward_target)) < 0.000001
        print(' successfully passed!')
        sys.stdout.flush()

    def test_backward_propagate(self):
        sys.stdout.write('Deep Believe Network -> Performing backward_propagate test ...')
        sys.stdout.flush()

        backward_target = numx.array([[0.30266536, 0.52656316], [0.69733464, 0.47343684]])
        assert numx.sum(
            numx.abs(
                TestDBNModel.stack.backward_propagate(numx.array([[1, 0], [0, 1]])) - backward_target)) < 0.000001

        numx.random.seed(42)
        backward_target = numx.array([[0, 0], [1, 0]])
        assert numx.sum(
            numx.abs(TestDBNModel.stack.backward_propagate(numx.array([[1, 0], [0, 1]]),
                                                           True) - backward_target)) < 0.000001
        print(' successfully passed!')
        sys.stdout.flush()

    def test_reconstruct(self):
        sys.stdout.write('Deep Believe Network -> Performing reconstruct test ...')
        sys.stdout.flush()

        rec_target = numx.array([[0.57055553, 0.23073692], [0.42944447, 0.76926308]])
        assert numx.sum(
            numx.abs(TestDBNModel.stack.reconstruct(numx.array([[1, 0], [0, 1]])) - rec_target)) < 0.000001

        numx.random.seed(42)
        rec_target = numx.array([[1, 0], [0, 1]])
        assert numx.sum(
            numx.abs(TestDBNModel.stack.reconstruct(numx.array([[1, 0], [0, 1]]), True) - rec_target)) < 0.000001
        print(' successfully passed!')
        sys.stdout.flush()

    def test_reconstruct_sample_top_layer(self):
        sys.stdout.write('Deep Believe Network -> Performing reconstruct_sample_top_layer test ...')
        sys.stdout.flush()

        numx.random.seed(42)
        assert numx.sum(
            numx.abs(TestDBNModel.stack.reconstruct_sample_top_layer(input_data=numx.array([[1, 0], [0, 1]]),
                                                                     sampling_steps=10,
                                                                     sample_forward_backward=False) - numx.array(
                [[0.69733464, 0.47343684], [0.69733464, 0.47343684]]))) < 0.000001
        numx.random.seed(42)
        assert numx.sum(
            numx.abs(TestDBNModel.stack.reconstruct_sample_top_layer(input_data=numx.array([[1, 1], [1, 1]]),
                                                                     sampling_steps=10,
                                                                     sample_forward_backward=True) - numx.array(
                [[0, 0], [0, 0]]))) < 0.000001

        print(' successfully passed!')
        sys.stdout.flush()

    def test_sample_top_layer(self):
        sys.stdout.write('Deep Believe Network -> Performing sample_top_layer test ...')
        sys.stdout.flush()
        assert numx.sum(numx.abs(TestDBNModel.stack.sample_top_layer(1, sample=False)
                                 - numx.array([[0.81216745, 0.80223488], [0.05331971, 0.94985651]]))) < 0.000001
        numx.random.seed(42)
        assert numx.sum(
            numx.abs(
                TestDBNModel.stack.sample_top_layer(100, sample=True) - numx.array([[0, 0], [0, 1]]))) < 0.000001
        numx.random.seed(42)
        assert numx.sum(
            numx.abs(TestDBNModel.stack.sample_top_layer(100, initial_state=numx.array([[1, 1], [1, 1]]),
                                                         sample=True) - numx.array([[0, 0], [0, 1]]))) < 0.000001
        print(' successfully passed!')
        sys.stdout.flush()


if __name__ is "__main__":
    unittest.main()
