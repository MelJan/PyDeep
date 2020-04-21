''' Test module for FNN trainer methods.

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
import unittest
import numpy as numx
from pydeep.fnn.layer import FullConnLayer
import pydeep.fnn.trainer as TRAINER
import pydeep.fnn.model as MODEL
import pydeep.base.activationfunction as AFct
import pydeep.base.costfunction as CFct

import sys

print("\n... pydeep.fnn.trainer.py")

class Test_FNN_trainer(unittest.TestCase):

    epsilon = 0.00001

    def test___init__(self):
        sys.stdout.write('FNN_trainer -> Performing init test ... ')
        sys.stdout.flush()
        l1 = FullConnLayer(2*2,3*3)
        l2 = FullConnLayer(3*3,2*2)
        l3 = FullConnLayer(2*2,1*1)
        model = MODEL.Model([l1,l2,l3])
        trainer = TRAINER.GDTrainer(model)
        assert numx.all(trainer._old_grad[0][0].shape == (4,9))
        assert numx.all(trainer._old_grad[0][1].shape == (1,9))
        assert numx.all(trainer._old_grad[1][0].shape == (9,4))
        assert numx.all(trainer._old_grad[1][1].shape == (1,4))
        assert numx.all(trainer._old_grad[2][0].shape == (4,1))
        assert numx.all(trainer._old_grad[2][1].shape == (1,1))
        print('successfully passed!')
        sys.stdout.flush()

    def test_calculate_errors(self):
        sys.stdout.write('FNN_trainer -> Performing calculate_errors test ... ')
        sys.stdout.flush()
        # Already tested see See base.numpyextensions.compare_index_of_max()
        print('successfully passed!')
        sys.stdout.flush()

    def test_check_setup(self):
        sys.stdout.write('FNN_trainer -> Performing check_setup test ... ')
        sys.stdout.flush()
        numx.random.seed(42)
        l1 = FullConnLayer(2*2,3*3)
        l2 = FullConnLayer(3*3,2*2)
        l3 = FullConnLayer(2*2,1*1)
        model = MODEL.Model([l1,l2,l3])
        trainer = TRAINER.GDTrainer(model)
        res = trainer.check_setup(data = numx.arange(4).reshape(1,4),
                                  labels = [numx.arange(9).reshape(1,9),numx.arange(4).reshape(1,4),numx.arange(1).reshape(1,1)],
                                  epsilon = [0.01,0.01,0.01],
                                  momentum = [0.09,0.09,0.09],
                                  reg_L1Norm  = [0.0002,0.0002,0.0002],
                                  reg_L2Norm  = [0.0002,0.0002,0.0002],
                                  corruptor = None,
                                  reg_costs = [1.0,1.0,1.0],
                                  costs = [CFct.SquaredError,CFct.SquaredError,CFct.SquaredError],
                                  reg_sparseness = [0.1,0.1,0.1],
                                  desired_sparseness = [0.01,0.01,0.01],
                                  costs_sparseness = [CFct.SquaredError,CFct.SquaredError,CFct.SquaredError],
                                  update_offsets = [0.01,0.01,0.01],
                                  restrict_gradient = [0.01,0.01,0.01],
                                  restriction_norm = 'Mat')
        assert numx.all(res)
        print('successfully passed!')
        sys.stdout.flush()

    def test_FNN_convergence(self):
        sys.stdout.write('FNN_trainer -> Performing several convergences tests ... ')
        sys.stdout.flush()
        numx.random.seed(42)
        x = numx.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
        l = numx.array([[1,0],[0,1],[0,1],[1,0]]) # 1,0 = 0, 0,1 = 1 otherwise Softmax would not work
        for act_out in [AFct.SoftMax]:
            for act_in in [AFct.SoftSign,AFct.SoftPlus,AFct.Sigmoid,AFct.HyperbolicTangent]:
                for cost in [CFct.CrossEntropyError,CFct.SquaredError,CFct.NegLogLikelihood]:
                    l1 = FullConnLayer(input_dim = 2,
                                       output_dim = 5,
                                       activation_function=act_in,
                                       initial_weights='AUTO',
                                       initial_bias=0.0,
                                       initial_offset=0.5,
                                       connections=None)
                    l2= FullConnLayer(input_dim = 5,
                                       output_dim = 2,
                                       activation_function=act_out,
                                       initial_weights='AUTO',
                                       initial_bias=0.0,
                                       initial_offset=0.5,
                                       connections=None)
                    model = MODEL.Model([l1,l2])
                    trainer = TRAINER.ADAGDTrainer(model)
                    for _ in range(1000):
                        trainer.train(data = x,
                                        labels = [None,l],
                                        epsilon = [0.3,0.3],
                                        reg_L1Norm  = [0.000,0.000],
                                        reg_L2Norm  = [0.000,0.000],
                                        corruptor = None,
                                        reg_costs = [0.0,1.0],
                                        costs = [None,cost],
                                        reg_sparseness = [0.0,0.0],
                                        desired_sparseness = [0.0,0.0],
                                        costs_sparseness = [None,None],
                                        update_offsets = [0.1,0.1],
                                        restrict_gradient = 0.0,
                                        restriction_norm = 'Mat')
                    model.forward_propagate(x)
                    assert numx.all(trainer.calculate_errors(l) == 0)
                    l1 = FullConnLayer(input_dim = 2,
                                       output_dim = 5,
                                       activation_function=act_in,
                                       initial_weights='AUTO',
                                       initial_bias=0.0,
                                       initial_offset=0.5,
                                       connections=None)
                    l2= FullConnLayer(input_dim = 5,
                                       output_dim = 2,
                                       activation_function=act_out,
                                       initial_weights='AUTO',
                                       initial_bias=0.0,
                                       initial_offset=0.5,
                                       connections=None)
                    model = MODEL.Model([l1,l2])
                    trainer = TRAINER.ADAGDTrainer(model)
                    for _ in range(1000):
                        trainer.train(data = x,
                                        labels = [None,l],
                                        epsilon = [0.3,0.3],
                                        reg_L1Norm  = [0.000,0.000],
                                        reg_L2Norm  = [0.000,0.000],
                                        corruptor = None,
                                        reg_costs = [0.0,1.0],
                                        costs = [None,cost],
                                        reg_sparseness = [0.1,0.0],
                                        desired_sparseness = [0.1,0.0],
                                        costs_sparseness = [CFct.SquaredError,None],
                                        update_offsets = [0.1,0.1],
                                        restrict_gradient = 0.0,
                                        restriction_norm = 'Mat')
                    model.forward_propagate(x)
                    assert numx.all(trainer.calculate_errors(l) == 0)

                    l1 = FullConnLayer(input_dim = 2,
                                       output_dim = 5,
                                       activation_function=act_in,
                                       initial_weights='AUTO',
                                       initial_bias=0.0,
                                       initial_offset=0.5,
                                       connections=None)
                    l2= FullConnLayer(input_dim = 5,
                                       output_dim = 2,
                                       activation_function=act_out,
                                       initial_weights='AUTO',
                                       initial_bias=0.0,
                                       initial_offset=0.5,
                                       connections=None)
                    model = MODEL.Model([l1,l2])
                    trainer = TRAINER.GDTrainer(model)
                    for _ in range(2000):
                        trainer.train(data = x,
                                        labels = [None,l],
                                        epsilon = [0.3,0.3],
                                        momentum = [0.9,0.9],
                                        reg_L1Norm  = [0.000,0.000],
                                        reg_L2Norm  = [0.000,0.000],
                                        corruptor = None,
                                        reg_costs = [0.0,1.0],
                                        costs = [None,cost],
                                        reg_sparseness = [0.1,0.0],
                                        desired_sparseness = [0.1,0.0],
                                        costs_sparseness = [CFct.SquaredError,None],
                                        update_offsets = [0.1,0.1],
                                        restrict_gradient = [0.0,0.0],
                                        restriction_norm = 'Mat')
                    model.forward_propagate(x)
                    assert numx.all(trainer.calculate_errors(l) == 0)


        print('successfully passed!')


if __name__ is "__main__":
    unittest.main()
