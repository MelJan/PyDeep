''' Test module for FNN model methods.

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
import pydeep.base.activationfunction as AFct
import pydeep.fnn.trainer as TRAINER
import pydeep.fnn.model as MODEL
import pydeep.base.costfunction as CFct
from pydeep.fnn.model import Model
from pydeep.base.numpyextension import generate_2d_connection_matrix

import sys

print("\n... pydeep.fnn.model.py")

class Test_FNN_model(unittest.TestCase):

    epsilon = 0.00001

    def test___init__(self):
        sys.stdout.write('FNN_model -> Performing init test ... ')
        sys.stdout.flush()
        l1 = FullConnLayer(2*2,3*3)
        l2 = FullConnLayer(3*3,2*2)
        l3 = FullConnLayer(2*2,1*1)
        model = Model([l1,l2,l3])
        passed = True
        try:
            model.consistency_check()
        except:
            passed = False
        assert passed
        assert (model.num_layers == 3)
        model = Model()
        passed = True
        try:
            model.consistency_check()
        except:
            passed = False
        assert passed
        assert (model.num_layers == 0)
        print('successfully passed!')
        sys.stdout.flush()

    def test_calculate_cost(self):
        sys.stdout.write('FNN_trainer -> Performing calculate_cost test ... ')
        sys.stdout.flush()
        numx.random.seed(42)
        l1 = FullConnLayer(2*2,3*3)
        l2 = FullConnLayer(3*3,2*2)
        l3 = FullConnLayer(2*2,1*1)
        model = MODEL.Model([l1,l2,l3])
        trainer = TRAINER.GDTrainer(model)
        model.forward_propagate(numx.arange(4).reshape(1,4))
        cost = model.calculate_cost([numx.arange(9).reshape(1,9),numx.arange(4).reshape(1,4),
                                      numx.arange(1).reshape(1,1)],
                                      [CFct.SquaredError,CFct.SquaredError,CFct.SquaredError],
                                      [1.0,1.0,1.0],
                                      [1.0,1.0,1.0],
                                      [CFct.SquaredError,CFct.SquaredError,CFct.SquaredError],
                                      [1.0,1.0,1.0])
        assert numx.all(numx.abs(cost - 117.72346036) < self.epsilon)
        cost = model.calculate_cost([numx.arange(9).reshape(1,9),numx.arange(4).reshape(1,4),
                                      numx.arange(1).reshape(1,1)],
                                      [CFct.SquaredError,CFct.SquaredError,CFct.SquaredError],
                                      [1.0,1.0,1.0],
                                      [1.0,1.0,1.0],
                                      [None,None,None],
                                      [0.0,0.0,0.0])
        assert numx.all(numx.abs(cost - 108.42118343)<self.epsilon)
        cost = model.calculate_cost([None,None,numx.arange(1).reshape(1,1)],
                                      [None,None,CFct.SquaredError],
                                      [0.0,0.0,1.0],
                                      [0.0,0.0,1.0],
                                      [None,None,None],
                                      [0.0,0.0,0.0])
        assert numx.all(numx.abs(cost -  0.10778406)<self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test_consistency_check(self):
        sys.stdout.write('FNN_model -> Performing consistency_check test ... ')
        sys.stdout.flush()
        l1 = FullConnLayer(2*2,3*3)
        l2 = FullConnLayer(3*3,2*2)
        l3 = FullConnLayer(2*2,1*1)
        model = Model([l1,l2,l3])
        passed = True
        try:
            model.consistency_check()
        except:
            passed = False
        assert passed
        print('successfully passed!')
        sys.stdout.flush()

    def test_pop_layer(self):
        sys.stdout.write('FNN_model -> Performing pop_layer test ... ')
        sys.stdout.flush()
        numx.random.seed(42)
        l1 = FullConnLayer(2*2,3*3)
        l2 = FullConnLayer(3*3,2*2)
        l3 = FullConnLayer(2*2,1*1)
        model = Model([l1,l2,l3])
        model.pop_layer()
        assert numx.all(model.layers[0] == l1)
        assert numx.all(model.layers[1] == l2)
        assert numx.all(model.num_layers == 2 == len(model.layers))
        print('successfully passed!')
        sys.stdout.flush()

    def test_append_layer(self):
        sys.stdout.write('FNN_model -> Performing append_layer test ... ')
        sys.stdout.flush()
        numx.random.seed(42)
        l1 = FullConnLayer(2*2,3*3)
        l2 = FullConnLayer(3*3,2*2)
        l3 = FullConnLayer(2*2,1*1)
        model = Model([l1])
        model.append_layer(l2)
        model.append_layer(l3)
        assert numx.all(model.layers[0] == l1)
        assert numx.all(model.layers[1] == l2)
        assert numx.all(model.layers[2] == l3)
        assert numx.all(model.num_layers == 3 == len(model.layers))
        print('successfully passed!')
        sys.stdout.flush()

    def test_forward_propagate(self):
        sys.stdout.write('FNN_model -> Performing forward_propagate test ...')
        sys.stdout.flush()
        numx.random.seed(42)
        l1 = FullConnLayer(2*2,3*3)
        l2 = FullConnLayer(3*3,2*2)
        l3 = FullConnLayer(2*2,1*1)
        model = Model([l1,l2,l3])
        target = numx.array([[-0.45978757]])
        assert numx.all(numx.abs(target - model.forward_propagate(numx.array([0.1,3.0,5.0,9.0]))) < self.epsilon)
        assert numx.all(numx.abs(target - model.forward_propagate(numx.array([[0.1,3.0,5.0,9.0]]))) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test_finite_differences(self):
        numx.random.seed(42)
        self.normal()
        self.local()
        self.sparse()
        self.target()

    def check(self,data, delta, act1,act2,act3, reg_sparseness, desired_sparseness, cost_sparseness,reg_targets, desired_targets, cost_targets, full):

        connections = None
        if full is False:
            connections = generate_2d_connection_matrix(6,6,3,3,2,2,False)

        model1 = FullConnLayer(6*6,
                                4*4,
                                activation_function = act1,
                                initial_weights = 'AUTO',
                                initial_bias = 0.0,
                                initial_offset = 0.0,
                                connections=connections,
                                dtype = numx.float64)

        model2 = FullConnLayer(4*4,
                             5*5,
                             activation_function = act2,
                             initial_weights = 'AUTO',
                             initial_bias = 0.0,
                             initial_offset = 0.5,
                             dtype = numx.float64)

        model3 = FullConnLayer(5*5,
                             6*6,
                             activation_function = act3,
                             initial_weights = 'AUTO',
                             initial_bias = 0.0,
                             initial_offset = 0.5,
                             dtype = numx.float64)

        model = MODEL.Model([model1,model2,model3])

        trainer = TRAINER.GDTrainer(model)

        _, _, maxw, maxb = model.finit_differences(delta, data, desired_targets, cost_targets, reg_targets, desired_sparseness, cost_sparseness, reg_sparseness )
        return numx.max([maxw,maxb])

    def normal(self,delta = 1e-4):
        sys.stdout.write('FNN_layer -> Performing finite difference check for 3 layer normal FNN  ... ')
        sys.stdout.flush()
        data = numx.random.rand(1,6*6)
        label = numx.zeros((1,6*6))
        label[0,14] = 1.0
        acts1 = [AFct.Sigmoid,AFct.SoftMax,AFct.SoftSign,AFct.Identity,AFct.RadialBasis(),AFct.SoftPlus,AFct.HyperbolicTangent]
        acts2 = [AFct.Sigmoid,AFct.SoftMax,AFct.SoftSign,AFct.Identity,AFct.RadialBasis(),AFct.SoftPlus,AFct.HyperbolicTangent]
        acts3 = [AFct.Sigmoid,AFct.SoftMax,AFct.SoftSign,AFct.Identity,AFct.RadialBasis(),AFct.SoftPlus,AFct.HyperbolicTangent]
        costs = [CFct.NegLogLikelihood,CFct.CrossEntropyError,CFct.SquaredError]
        for c in costs:
            for a3 in acts3:
                for a2 in acts2:
                    for a1 in acts1:
                        if ((c != CFct.CrossEntropyError and c != CFct.NegLogLikelihood) or
                            (c == CFct.CrossEntropyError and (a3 == AFct.Sigmoid or a3 == AFct.SoftMax)) or
                            (c == CFct.NegLogLikelihood and a3 == AFct.SoftMax)):
                            res = self.check(data = data,
                                        delta = delta,
                                        act1 = a1,
                                        act2 = a2,
                                        act3 = a3,
                                        reg_sparseness = [0.0,0.0,0.0],
                                        desired_sparseness = [0.0,0.0,0],
                                        cost_sparseness = [0.0,0.0,0],
                                        reg_targets = [0.0,0.0,1],
                                        desired_targets = [0.0,0.0,label],
                                        cost_targets = [None,None,c],
                                        full = True)
                            if res > delta:
                                print("Failed!     ", res, a1 ,a2, a3, c)
                            assert numx.all(res < delta)
        print('successfully passed!')
        sys.stdout.flush()

    def local(self,delta = 1e-4):
        sys.stdout.write('FNN_layer -> Performing finite difference check for 3 layer local FNN  ... ')
        sys.stdout.flush()
        data = numx.random.rand(1,6*6)
        label = numx.zeros((1,6*6))
        label[0,14] = 1.0
        acts1 = [AFct.Sigmoid,AFct.SoftMax,AFct.SoftSign,AFct.Identity,AFct.RadialBasis(),AFct.SoftPlus,AFct.HyperbolicTangent]
        acts2 = [AFct.Sigmoid,AFct.SoftMax,AFct.SoftSign,AFct.Identity,AFct.RadialBasis(),AFct.SoftPlus,AFct.HyperbolicTangent]
        acts3 = [AFct.Sigmoid,AFct.SoftMax,AFct.SoftSign,AFct.Identity,AFct.RadialBasis(),AFct.SoftPlus,AFct.HyperbolicTangent]
        costs = [CFct.NegLogLikelihood,CFct.CrossEntropyError,CFct.SquaredError]
        for c in costs:
            for a3 in acts3:
                for a2 in acts2:
                    for a1 in acts1:
                        if ((c != CFct.CrossEntropyError and c != CFct.NegLogLikelihood) or
                            (c == CFct.CrossEntropyError and (a3 == AFct.Sigmoid or a3 == AFct.SoftMax)) or
                            (c == CFct.NegLogLikelihood and a3 == AFct.SoftMax)):
                            res = self.check(data = data,
                                        delta = delta,
                                        act1 = a1,
                                        act2 = a2,
                                        act3 = a3,
                                        reg_sparseness = [0.0,0.0,0.0],
                                        desired_sparseness = [0.0,0.0,0.0],
                                        cost_sparseness = [None,None,None],
                                        reg_targets = [0.0,0.0,1],
                                        desired_targets = [0.0,0.0,label],
                                        cost_targets = [None,None,c],
                                        full = True)
                            if res > delta:
                                print("Failed!     ", res, a1 ,a2, a3, c)
                            assert numx.all(res < delta)
        print('successfully passed!')
        sys.stdout.flush()

    def sparse(self, delta = 1e-4):
        sys.stdout.write('FNN_layer -> Performing finite difference check for 3 layer sparse FNN  ... ')
        sys.stdout.flush()
        data = numx.random.rand(1,6*6)
        label = numx.zeros((1,6*6))
        label[0,14] = 1.0
        acts1 = [AFct.Sigmoid,AFct.SoftMax,AFct.SoftSign,AFct.Identity,AFct.RadialBasis(),AFct.SoftPlus,AFct.HyperbolicTangent]
        acts2 = [AFct.Sigmoid,AFct.SoftMax,AFct.SoftSign,AFct.Identity,AFct.RadialBasis(),AFct.SoftPlus,AFct.HyperbolicTangent]
        acts3 = [AFct.Sigmoid,AFct.SoftMax,AFct.SoftSign,AFct.Identity,AFct.RadialBasis(),AFct.SoftPlus,AFct.HyperbolicTangent]
        costs = [CFct.NegLogLikelihood,CFct.CrossEntropyError,CFct.SquaredError]
        for c in costs:
            for a3 in acts3:
                for a2 in acts2:
                    for a1 in acts1:
                        if ((c != CFct.CrossEntropyError and c != CFct.NegLogLikelihood) or
                            (c == CFct.CrossEntropyError and (a3 == AFct.Sigmoid or a3 == AFct.SoftMax)) or
                            (c == CFct.NegLogLikelihood and a3 == AFct.SoftMax)):
                            res = self.check(data = data,
                                        delta = delta,
                                        act1 = a1,
                                        act2 = a2,
                                        act3 = a3,
                                        reg_sparseness = [0.3,0.1,0.0],
                                        desired_sparseness = [0.01,0.1,0],
                                        cost_sparseness = [CFct.SquaredError,CFct.SquaredError,0],
                                        reg_targets = [0.0,0.0,1],
                                        desired_targets = [0.0,0.0,label],
                                        cost_targets = [None,None,c],
                                        full = True)
                            if res > delta:
                                print("Failed!     ", res, a1 ,a2, a3, c)
                            assert numx.all(res < delta)
        print('successfully passed!')
        sys.stdout.flush()

    def target(self,delta = 1e-4):
        sys.stdout.write('FNN_layer -> Performing finite difference check for 3 layer FNN all layers with targets ... ')
        sys.stdout.flush()
        delta = 1e-4
        data = numx.random.rand(1,6*6)
        label = numx.zeros((1,6*6))
        label[0,14] = 1.0

        target1 = numx.zeros((1,4*4))
        target1[0,3] = 1.0
        target2 = numx.zeros((1,5*5))
        target2[0,6] = 1.0

        acts1 = [AFct.Sigmoid,AFct.SoftMax,AFct.SoftSign,AFct.Identity,AFct.RadialBasis(),AFct.SoftPlus,AFct.HyperbolicTangent]
        acts2 = [AFct.Sigmoid,AFct.SoftMax,AFct.SoftSign,AFct.Identity,AFct.RadialBasis(),AFct.SoftPlus,AFct.HyperbolicTangent]
        acts3 = [AFct.Sigmoid,AFct.SoftMax,AFct.SoftSign,AFct.Identity,AFct.RadialBasis(),AFct.SoftPlus,AFct.HyperbolicTangent]
        costs = [CFct.NegLogLikelihood,CFct.CrossEntropyError,CFct.SquaredError]
        for c in costs:
            for a3 in acts3:
                for a2 in acts2:
                    for a1 in acts1:
                        if ((c != CFct.CrossEntropyError and c != CFct.NegLogLikelihood) or
                            (c == CFct.CrossEntropyError and (a3 == AFct.Sigmoid or a3 == AFct.SoftMax)) or
                            (c == CFct.NegLogLikelihood and a3 == AFct.SoftMax)):

                            if a1 == AFct.SoftMax:
                                res = self.check(data = data,
                                            delta = delta,
                                            act1 = a1,
                                            act2 = a2,
                                            act3 = a3,
                                            reg_sparseness = [0.0,0.0,0.0],
                                            desired_sparseness = [0.0,0.0,0],
                                            cost_sparseness = [0.0,0.0,0],
                                            reg_targets = [0.3,0.1,1.0],
                                            desired_targets = [target1,target2,label],
                                            cost_targets = [CFct.NegLogLikelihood,CFct.SquaredError,c],
                                            full = True)
                                if res > delta:
                                    print("Failed!     ", res, a1 ,a2, a3, c)
                                assert numx.all(res < delta)
                                res =self.check(data = data,
                                            delta = delta,
                                            act1 = a1,
                                            act2 = a2,
                                            act3 = a3,
                                            reg_sparseness = [0.0,0.0,0.0],
                                            desired_sparseness = [0.0,0.0,0],
                                            cost_sparseness = [0.0,0.0,0],
                                            reg_targets = [0.3,0.1,1.0],
                                            desired_targets = [target1,target2,label],
                                            cost_targets = [CFct.CrossEntropyError,CFct.SquaredError,c],
                                            full = True)
                                if res > delta:
                                    print("Failed!     ", res, a1 ,a2, a3, c)
                                assert numx.all(res < delta)
                                res = self.check(data = data,
                                            delta = delta,
                                            act1 = a1,
                                            act2 = a2,
                                            act3 = a3,
                                            reg_sparseness = [0.0,0.0,0.0],
                                            desired_sparseness = [0.0,0.0,0],
                                            cost_sparseness = [0.0,0.0,0],
                                            reg_targets = [0.3,0.1,1.0],
                                            desired_targets = [target1,target2,label],
                                            cost_targets = [CFct.SquaredError,CFct.SquaredError,c],
                                            full = True)
                                if res > delta:
                                    print("Failed!     ", res, a1 ,a2, a3, c)
                                assert numx.all(res < delta)
                            elif a1 == AFct.Sigmoid:
                                res = self.check(data = data,
                                            delta = delta,
                                            act1 = a1,
                                            act2 = a2,
                                            act3 = a3,
                                            reg_sparseness = [0.0,0.0,0.0],
                                            desired_sparseness = [0.0,0.0,0],
                                            cost_sparseness = [0.0,0.0,0],
                                            reg_targets = [0.3,0.1,1.0],
                                            desired_targets = [target1,target2,label],
                                            cost_targets = [CFct.CrossEntropyError,CFct.SquaredError,c],
                                            full = True)
                                if res > delta:
                                    print("Failed!     ", res, a1 ,a2, a3, c)
                                assert numx.all(res < delta)
                                res = self.check(data = data,
                                            delta = delta,
                                            act1 = a1,
                                            act2 = a2,
                                            act3 = a3,
                                            reg_sparseness = [0.0,0.0,0.0],
                                            desired_sparseness = [0.0,0.0,0],
                                            cost_sparseness = [0.0,0.0,0],
                                            reg_targets = [0.3,0.1,1.0],
                                            desired_targets = [target1,target2,label],
                                            cost_targets = [CFct.SquaredError,CFct.SquaredError,c],
                                            full = True)
                                if res > delta:
                                    print("Failed!     ", res, a1 ,a2, a3, c)
                                assert numx.all(res < delta)
                            else:
                                res = self.check(data = data,
                                            delta = delta,
                                            act1 = a1,
                                            act2 = a2,
                                            act3 = a3,
                                            reg_sparseness = [0.0,0.0,0.0],
                                            desired_sparseness = [0.0,0.0,0],
                                            cost_sparseness = [0.0,0.0,0],
                                            reg_targets = [0.3,0.1,1.0],
                                            desired_targets = [target1,target2,label],
                                            cost_targets = [CFct.SquaredError,CFct.SquaredError,c],
                                            full = True)
                                if res > delta:
                                    print("Failed!     ", res, a1 ,a2, a3, c)
                                assert numx.all(res < delta)
        print('successfully passed!')
        sys.stdout.flush()

if __name__ is "__main__":
    unittest.main()
