''' Test module for FNN layer methods.

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
import pydeep.base.costfunction as CFct
from pydeep.base.numpyextension import generate_2d_connection_matrix

import sys

print("\n... pydeep.fnn.layer.py")

class Test_FNN_layer(unittest.TestCase):

    epsilon = 0.00001

    def test___init__(self):
        sys.stdout.write('FNN_layer -> Performing init test ... ')
        sys.stdout.flush()
        l = FullConnLayer(input_dim = 9,
                          output_dim = 4,
                          activation_function = AFct.Sigmoid,
                          initial_weights ='AUTO',
                          initial_bias ='AUTO',
                          initial_offset ='AUTO',
                          connections = generate_2d_connection_matrix(3,3,2,2,1,1,False))
        assert numx.all(l.weights.shape == (9,4))
        assert numx.all(l.bias.shape == (1,4))
        assert numx.all(l.offset.shape == (1,9))
        assert numx.all(l.connections.shape == (9,4))
        assert numx.all(l.activation_function == AFct.Sigmoid)
        l = FullConnLayer(input_dim = 9,
                          output_dim = 4,
                          activation_function = AFct.Rectifier,
                          initial_weights =1.0,
                          initial_bias =2.0,
                          initial_offset =3.0,
                          connections =None)
        assert numx.all(l.weights.shape == (9,4))
        assert numx.all(numx.abs(l.bias - numx.ones((1,4))*2.0) < self.epsilon)
        assert numx.all(numx.abs(l.offset - numx.ones((1,9))*3.0) < self.epsilon)
        assert numx.all(l.connections is None)
        assert numx.all(l.activation_function == AFct.Rectifier)
        l = FullConnLayer(input_dim = 9,
                          output_dim = 4,
                          activation_function = AFct.SoftMax,
                          initial_weights = numx.ones((9,4))*1.0,
                          initial_bias = numx.ones((1,4))*2.0,
                          initial_offset = numx.ones((1,9))*3.0,
                          connections =None)
        assert numx.all(numx.abs(l.weights - numx.ones((9,4))*1.0) < self.epsilon)
        assert numx.all(numx.abs(l.bias - numx.ones((1,4))*2.0) < self.epsilon)
        assert numx.all(numx.abs(l.offset - numx.ones((1,9))*3.0) < self.epsilon)
        assert numx.all(l.activation_function == AFct.SoftMax)
        print('successfully passed!')
        sys.stdout.flush()

    def test_get_parameters(self):
        sys.stdout.write('FNN_layer -> Performing get_parameters test ... ')
        sys.stdout.flush()
        l = FullConnLayer(input_dim = 9,
                          output_dim = 4,
                          activation_function = AFct.Sigmoid,
                          initial_weights ='AUTO',
                          initial_bias ='AUTO',
                          initial_offset ='AUTO',
                          connections = generate_2d_connection_matrix(3,3,2,2,1,1,False))
        w, b = l.get_parameters()
        assert numx.all(l.weights.shape == (9,4))
        assert numx.all(l.bias.shape == (1,4))
        assert numx.all(numx.abs(l.weights - w) < self.epsilon)
        assert numx.all(numx.abs(l.bias - b) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test_update_parameters(self):
        sys.stdout.write('FNN_layer -> Performing update_parameters test ... ')
        sys.stdout.flush()
        l = FullConnLayer(input_dim = 9,
                          output_dim = 4,
                          activation_function = AFct.Sigmoid,
                          initial_weights =0,
                          initial_bias =0,
                          initial_offset =0,
                          connections = generate_2d_connection_matrix(3,3,2,2,1,1,False))
        assert numx.all(numx.abs(l.weights - numx.zeros((9,4))) < self.epsilon)
        assert numx.all(numx.abs(l.bias - numx.zeros((1,4))) < self.epsilon)
        l.update_parameters([-numx.ones((9,4)),-numx.ones((1,4))])
        assert numx.all(numx.abs(l.weights - numx.ones((9,4))) < self.epsilon)
        assert numx.all(numx.abs(l.bias - numx.ones((1,4))) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test_update_offsets(self):
        sys.stdout.write('FNN_layer -> Performing update_offsets test ... ')
        sys.stdout.flush()
        l = FullConnLayer(input_dim = 9,
                          output_dim = 4,
                          activation_function = AFct.Sigmoid,
                          initial_weights =numx.ones((9,4)),
                          initial_bias =0,
                          initial_offset =1.0,
                          connections = None)
        l.update_offsets(0.0,-numx.ones((1,9)))
        assert numx.all(numx.abs(l.offset - numx.ones((1,9))) < self.epsilon)
        assert numx.all(numx.abs(l.bias - numx.zeros((1,4))) < self.epsilon)
        l.update_offsets(1.0,-numx.ones((1,9)))
        assert numx.all(numx.abs(l.offset + numx.ones((1,9))) < self.epsilon)
        assert numx.all(numx.abs(l.bias - numx.ones((1,4))*-18) < self.epsilon)
        l.update_offsets(0.5,numx.ones((1,9)))
        assert numx.all(numx.abs(l.offset - numx.zeros((1,9))) < self.epsilon)
        assert numx.all(numx.abs(l.bias - numx.ones((1,4))*-9) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test_forward_propagate(self):
        sys.stdout.write('FNN_layer -> Performing forward_propagate test ... ')
        sys.stdout.flush()
        l = FullConnLayer(input_dim = 9,
                          output_dim = 4,
                          activation_function = AFct.Identity,
                          initial_weights =0.001*numx.arange(9*4).reshape(9,4),
                          initial_bias = 0.0,
                          initial_offset =0.5,
                          connections = None)
        res = l.forward_propagate(numx.arange(9).reshape(1,9))
        target = numx.array([[ 0.744 ,  0.7755 , 0.807 ,  0.8385]])
        assert numx.all(numx.abs(res - target) < self.epsilon)
        l = FullConnLayer(input_dim = 9,
                          output_dim = 4,
                          activation_function = AFct.SoftMax,
                          initial_weights =0.001*numx.arange(9*4).reshape(9,4),
                          initial_bias = 0.0,
                          initial_offset =0.5,
                          connections = None)
        res = l.forward_propagate(numx.arange(9).reshape(1,9))
        target = numx.array([[ 0.23831441 , 0.2459408 ,  0.25381124 , 0.26193355]])
        assert numx.all(numx.abs(res - target) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test_backward_propagate(self):
        sys.stdout.write('FNN_layer -> Performing backward_propagate test ... ')
        sys.stdout.flush()
        l = FullConnLayer(input_dim = 9,
                          output_dim = 4,
                          activation_function = AFct.Identity,
                          initial_weights =0.001*numx.arange(9*4).reshape(9,4),
                          initial_bias = 0.0,
                          initial_offset =0.5,
                          connections = None)
        l.forward_propagate(numx.arange(9).reshape(1,9))
        l._get_deltas(numx.arange(4).reshape(1,4),None,None,0.0,0.0,None,0.0)
        res = l._backward_propagate()
        target = numx.array([[ 0.014,0.038,0.062,0.086,0.11,0.134,0.158,0.182,0.206]])
        assert numx.all(numx.abs(res - target) < self.epsilon)
        l = FullConnLayer(input_dim = 9,
                          output_dim = 4,
                          activation_function = AFct.SoftMax,
                          initial_weights =0.001*numx.arange(9*4).reshape(9,4),
                          initial_bias = 0.0,
                          initial_offset =0.5,
                          connections = None)
        l.forward_propagate(numx.arange(9).reshape(1,9))
        l._get_deltas(numx.arange(4).reshape(1,4),None,None,0.0,0.0,None,0.0)
        res = l._backward_propagate()
        target = numx.array([[ 0.00124895,0.00124895,0.00124895,0.00124895,0.00124895,
                               0.00124895,0.00124895,0.00124895,0.00124895]])
        assert numx.all(numx.abs(res - target) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test_calculate_gradient(self):
        sys.stdout.write('FNN_layer -> Performing calculate_gradient test ... ')
        sys.stdout.flush()
        l = FullConnLayer(input_dim = 9,
                          output_dim = 4,
                          activation_function = AFct.Identity,
                          initial_weights =0.001*numx.arange(9*4).reshape(9,4),
                          initial_bias = 0.0,
                          initial_offset =0.5,
                          connections = None)
        l.forward_propagate(numx.arange(9).reshape(1,9))
        l._get_deltas(numx.arange(4).reshape(1,4),None,None,0.0,0.0,None,0.0)
        l._backward_propagate()
        dw,db = l._calculate_gradient()
        targetW = numx.array([[0.,-0.5,-1.,-1.5],[0.,0.5,1.,1.5], [0.,1.5,3., 4.5],
                              [0.,2.5,5.,7.5],[0.,3.5,7.,10.5], [0.,4.5,9.,13.5],
                              [0.,5.5,11.,16.5], [0.,6.5,13.,19.5], [0.,7.5,15.,22.5]])
        assert numx.all(numx.abs(dw - targetW) < self.epsilon)
        targetb = numx.array([0.,1.,2.,3.])
        assert numx.all(numx.abs(db - targetb) < self.epsilon)

        l = FullConnLayer(input_dim = 9,
                          output_dim = 4,
                          activation_function = AFct.SoftMax,
                          initial_weights =0.001*numx.arange(9*4).reshape(9,4),
                          initial_bias = 0.0,
                          initial_offset =0.5,
                          connections = None)
        l.forward_propagate(numx.arange(9).reshape(1,9))
        l._get_deltas(numx.arange(4).reshape(1,4),None,None,0.0,0.0,None,0.0)
        l._backward_propagate()
        dw,db = l._calculate_gradient()
        targetW = numx.array( [[ 0.1834263,0.0663258,-0.05845731,-0.1912948 ],
                               [-0.1834263,-0.0663258,0.05845731,0.1912948 ],
                               [-0.55027891,-0.19897739,0.17537192,0.57388439],
                               [-0.91713152,-0.33162899,0.29228653,0.95647398],
                               [-1.28398412,-0.46428059,0.40920114,1.33906357],
                               [-1.65083673,-0.59693218,0.52611575,1.72165316],
                               [-2.01768934,-0.72958378,0.64303037,2.10424275],
                               [-2.38454194,-0.86223538,0.75994498,2.48683234],
                               [-2.75139455,-0.99488697,0.87685959,2.86942193]])
        targetb = numx.array([-0.36685261,-0.1326516,0.11691461,0.38258959])
        assert numx.all(numx.abs(dw - targetW) < self.epsilon)
        assert numx.all(numx.abs(db - targetb) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test_get_deltas(self):
        sys.stdout.write('FNN_layer -> Performing get_deltas test ... ')
        sys.stdout.flush()
        l = FullConnLayer(input_dim = 9,
                          output_dim = 4,
                          activation_function = AFct.Sigmoid,
                          initial_weights =0.01*numx.arange(9*4).reshape(9,4),
                          initial_bias = 0.0,
                          initial_offset =0.5,
                          connections = None)
        l.forward_propagate(1.0*numx.arange(9).reshape(1,9))
        d = l._get_deltas(1.0*numx.arange(4).reshape(1,4),None,None,0.0,0.0,None,0.0)
        targetd = numx.array([[ 0., 0.00042823,0.00062518,0.00068448]])
        assert numx.all(numx.abs(d - targetd) < self.epsilon)
        d = l._get_deltas(None,1.0*numx.arange(4).reshape(1,4),CFct.SquaredError,1.0,0.0,None,0.0)
        targetd = numx.array([[  5.86251700e-04,-1.83457004e-07 , -3.12685448e-04 , -4.56375237e-04]])
        assert numx.all(numx.abs(d - targetd) < self.epsilon)
        d = l._get_deltas(1.0*numx.arange(4).reshape(1,4),None,None,0.0,0.01,CFct.SquaredError,1.0)
        targetd = numx.array([[ 0.00058039,0.00085199,0.00093454,0.00091031]])
        assert numx.all(numx.abs(d - targetd) < self.epsilon)
        d = l._get_deltas(1.0*numx.arange(4).reshape(1,4),1.0*numx.arange(4).reshape(1,4),CFct.SquaredError,1.0,0.0,None,0.0)
        targetd = numx.array([[ 0.00058625 , 0.00042804 , 0.00031249 , 0.00022811]])
        assert numx.all(numx.abs(d - targetd) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

if __name__ is "__main__":
    unittest.main()
