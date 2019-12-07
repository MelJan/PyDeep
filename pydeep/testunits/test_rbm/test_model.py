''' Test module for RBM models.

    :Version:
        1.1.0

    :Date:
        12.04.2017

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
import copy
import unittest
import sys

from pydeep.misc.toyproblems import generate_bars_and_stripes_complete
import pydeep.rbm.model as Model

print("\n... pydeep.rbm.model.py")


class TestBinaryBinaryRBM(unittest.TestCase):
    # Known RBM
    bbrbmData = generate_bars_and_stripes_complete(2)
    bbrbmData = numx.vstack((bbrbmData[0], bbrbmData, bbrbmData[5]))
    bbrbmw = numx.array([[0.12179488, 2.95950177, 0.33513356, 35.05380642],
                         [0.20318085, -28.62372894, 26.52611278, 28.41793445],
                         [-0.19105386, -28.58530584, -26.52747507, 28.78447320],
                         [0.08953740, -59.82556859, -0.06665933, -27.71723459]])
    bbrbmbv = numx.array([[-19.24399659, -13.26258696, 13.25909850, 43.74408543]])
    bbrbmbh = numx.array([[-0.11155958, 57.02097584, -0.13331758, -32.25991501]])
    bbrbm = Model.BinaryBinaryRBM(4, 4, bbrbmData, bbrbmw, bbrbmbv, bbrbmbh, 0.0, 0.0)

    bbrbmTruelogZ = 59.6749019726
    bbrbmTrueLL = -1.7328699078
    bbrbmBestLLPossible = -1.732867951

    epsilon = 0.00001

    def test___init__(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing init test ...')
        sys.stdout.flush()
        assert numx.all(self.bbrbm.bv_base == numx.array([[0, 0, 0, 0]]))
        print(' successfully passed!')
        sys.stdout.flush()

    def test__add_visible_units(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing add_visible_units test ...')
        sys.stdout.flush()
        localmodel = copy.deepcopy(self.bbrbm)
        localmodel._add_visible_units(2, 3)
        assert numx.all(localmodel.bv_base == numx.array([[0, 0, 0, 0, 0, 0]]))
        print(' successfully passed!')
        sys.stdout.flush()

    def test__remove_visible_units(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing remove_visible_units test ...')
        sys.stdout.flush()
        localmodel = copy.deepcopy(self.bbrbm)
        localmodel._remove_visible_units([0, 2])
        assert numx.all(localmodel.bv_base == numx.array([[0, 0]]))
        print(' successfully passed!')
        sys.stdout.flush()

    def test__calculate_weight_gradient(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing calculate_weight_gradient test ...')
        sys.stdout.flush()
        deltaW = self.bbrbm._calculate_weight_gradient(numx.array([[1, 1, 1, 0], [0, 1, 0, 1]]),
                                                       numx.array([[0, 1, 0, 1], [0, 1, 1, 0]]))
        target = numx.array([[0., 1., 0., 1.], [0., 2., 1., 1.], [0., 1., 0., 1.], [0., 1., 1., 0.]])
        assert numx.all(target == deltaW)
        print(' successfully passed!')
        sys.stdout.flush()

    def test__calculate_visible_bias_gradient(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing calculate_visible_bias_gradient test ...')
        sys.stdout.flush()
        deltaBv = self.bbrbm._calculate_visible_bias_gradient(numx.array([[1, 1, 1, 0], [0, 1, 0, 1]]))
        target = numx.array([[1., 2., 1., 1.]])
        assert numx.all(target == deltaBv)
        print(' successfully passed!')
        sys.stdout.flush()

    def test__calculate_hidden_bias_gradient(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing calculate_hidden_bias_gradient test ...')
        sys.stdout.flush()
        deltaBh = self.bbrbm._calculate_hidden_bias_gradient(numx.array([[0, 1, 0, 1], [0, 1, 1, 0]]))
        target = numx.array([[0., 2., 1., 1.]])
        assert numx.all(target == deltaBh)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_calculate_gradients(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing calculate_gradients test ...')
        sys.stdout.flush()
        deltaW = self.bbrbm._calculate_weight_gradient(numx.array([[1, 1, 1, 0], [0, 1, 0, 1]]),
                                                       numx.array([[0, 1, 0, 1], [0, 1, 1, 0]]))
        deltaBv = self.bbrbm._calculate_visible_bias_gradient(numx.array([[1, 1, 1, 0], [0, 1, 0, 1]]))
        deltaBh = self.bbrbm._calculate_hidden_bias_gradient(numx.array([[0, 1, 0, 1], [0, 1, 1, 0]]))
        deltas = self.bbrbm.calculate_gradients(numx.array([[1, 1, 1, 0], [0, 1, 0, 1]]),
                                                numx.array([[0, 1, 0, 1], [0, 1, 1, 0]]))
        assert numx.all(deltaW == deltas[0])
        assert numx.all(deltaBv == deltas[1])
        assert numx.all(deltaBh == deltas[2])
        print(' successfully passed!')
        sys.stdout.flush()

    def test_sample_v(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing sample_v test ...')
        sys.stdout.flush()
        assert numx.all(self.bbrbm.sample_v(numx.ones((10000, 4))) == 1.0)
        assert numx.all(self.bbrbm.sample_v(numx.zeros((10000, 4))) == 0.0)
        numx.random.seed(42)
        samples = self.bbrbm.sample_v(numx.ones((10000, 4)) * 0.5)
        assert numx.sum(samples != 0.0) + numx.sum(samples != 1.0) == 40000
        assert numx.abs(numx.sum(samples) / 40000.0 - 0.5) < 0.01
        print(' successfully passed!')
        sys.stdout.flush()

    def test_sample_h(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing sample_h test ...')
        sys.stdout.flush()
        assert numx.all(self.bbrbm.sample_h(numx.ones((10000, 4))) == 1.0)
        assert numx.all(self.bbrbm.sample_h(numx.zeros((10000, 4))) == 0.0)
        numx.random.seed(42)
        samples = self.bbrbm.sample_h(numx.ones((10000, 4)) * 0.5)
        assert numx.sum(samples != 0.0) + numx.sum(samples != 1.0) == 40000
        assert numx.abs(numx.sum(samples) / 40000.0 - 0.5) < 0.01
        print(' successfully passed!')
        sys.stdout.flush()

    def test_probability_v_given_h(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing probability_v_given_h test ...')
        sys.stdout.flush()
        probs = self.bbrbm.probability_v_given_h(self.bbrbmData)
        target = numx.array([[4.38973669e-09, 1.73832475e-06, 9.99998256e-01, 1.00000000e+00],
                             [4.38973669e-09, 1.73832475e-06, 9.99998256e-01, 1.00000000e+00],
                             [6.93234181e-09, 9.99998583e-01, 1.42771991e-06, 1.00000000e+00],
                             [9.99999993e-01, 1.41499740e-06, 9.99998571e-01, 0.00000000e+00],
                             [9.56375764e-08, 0.00000000e+00, 1.82363957e-07, 1.13445207e-07],
                             [9.99999903e-01, 1.00000000e+00, 9.99999817e-01, 9.99999883e-01],
                             [9.99999996e-01, 9.99998259e-01, 1.74236911e-06, 0.00000000e+00],
                             [9.99999996e-01, 9.99998259e-01, 1.74236911e-06, 0.00000000e+00]])
        assert numx.all(numx.abs(probs - target) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_probability_h_given_v(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing probability_h_given_v test ...')
        sys.stdout.flush()
        probs = self.bbrbm.probability_h_given_v(self.bbrbmData)
        target = numx.array([[4.72138994e-01, 1.00000000e+00, 4.66719883e-01, 9.76996262e-15],
                             [4.72138994e-01, 1.00000000e+00, 4.66719883e-01, 9.76996262e-15],
                             [4.54918124e-01, 1.00000000e+00, 3.68904907e-12, 1.00000000e+00],
                             [5.45166211e-01, 2.24265051e-14, 1.00000000e+00, 1.97064587e-14],
                             [5.53152448e-01, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00],
                             [4.46931620e-01, 2.33146835e-14, 2.46841436e-12, 2.83661983e-14],
                             [5.27945768e-01, 0.00000000e+00, 5.33398782e-01, 1.00000000e+00],
                             [5.27945768e-01, 0.00000000e+00, 5.33398782e-01, 1.00000000e+00]])
        assert numx.all(numx.abs(probs - target) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_energy(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing energy test ...')
        sys.stdout.flush()
        energies = self.bbrbm.energy(self.bbrbmData, self.bbrbmData)
        target = numx.array([[0.], [0.], [32.49137574], [32.50603837], [0.93641873], [0.91694445], [0.03276686], [0.03276686]])
        assert numx.all(numx.abs(energies - target) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_unnormalized_log_probability_v(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing unnormalized_log_probability_v test ...')
        sys.stdout.flush()
        probs = self.bbrbm.unnormalized_log_probability_v(self.bbrbmData)
        target = numx.array(
            [[58.28860656], [58.28860656], [57.59545755], [57.59545757], [57.59545753], [57.59545756], [58.28860656],
             [58.28860656]])
        assert numx.all(numx.abs(probs - target) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_unnormalized_log_probability_h(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing unnormalized_log_probability_h test ...')
        sys.stdout.flush()
        probs = self.bbrbm.unnormalized_log_probability_h(self.bbrbmData)
        target = numx.array(
            [[57.00318742], [57.00318742], [56.98879586], [56.98864114], [56.90941665], [56.90945961], [57.00333938],
             [57.00333938]])
        assert numx.all(numx.abs(probs - target) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_log_probability_v(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing log_probability_v test ...')
        sys.stdout.flush()
        probs = self.bbrbm.log_probability_v(self.bbrbmTruelogZ, self.bbrbmData)
        target = numx.array(
            [[-1.38629541], [-1.38629541], [-2.07944442], [-2.07944441], [-2.07944444], [-2.07944441], [-1.38629541],
             [-1.38629541]])
        assert numx.all(numx.abs(probs - target) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_log_probability_h(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing log_probability_h test ...')
        sys.stdout.flush()
        probs = self.bbrbm.log_probability_h(self.bbrbmTruelogZ, self.bbrbmData)
        target = numx.array(
            [[-2.67171456], [-2.67171456], [-2.68610611], [-2.68626083], [-2.76548532], [-2.76544237], [-2.67156259],
             [-2.67156259]])
        assert numx.all(numx.abs(probs - target) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_log_probability_v_h(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing log_probability_v_h test ...')
        sys.stdout.flush()
        h = self.bbrbm.probability_h_given_v(self.bbrbmData)
        probs = self.bbrbm.log_probability_v_h(self.bbrbmTruelogZ, self.bbrbmData, h)
        target = numx.array(
            [[-2.76881973], [-2.76881973], [-2.76852132], [-2.76850605], [-2.76693057], [-2.76694846], [-2.76879441],
             [-2.76879441]])
        assert numx.all(numx.abs(probs - target) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test__base_log_partition(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing base_log_partition test ...')
        sys.stdout.flush()
        localmodel = copy.deepcopy(self.bbrbm)
        localmodel.bv_base += 1.0
        assert numx.abs(localmodel._base_log_partition(False) - 5.54517744448) < self.epsilon
        assert numx.abs(localmodel._base_log_partition(True) - 8.02563547231) < self.epsilon
        print(' successfully passed!')
        sys.stdout.flush()

    def test__getbasebias(self):
        sys.stdout.write('BinaryBinaryRBM -> Performing getbasebias test ...')
        sys.stdout.flush()
        # zero base bias when data mean is 0.5
        localmodel = copy.deepcopy(self.bbrbm)
        localmodel._data_mean = localmodel._data_mean / 2.0
        assert numx.all(numx.abs(localmodel._getbasebias() + 1.09861229) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()


class Test_GaussianBinaryRBM(unittest.TestCase):
    gbrbmData = numx.array([[0.57459607, 0.57689834], [-0.60788602, -0.57941004]])
    gbrbmw = numx.array([[0.12179488, 2.95950177],
                         [0.20318085, -28.62372894, ]])
    gbrbmbv = numx.array([[-19.24399659, -13.26258696]])
    gbrbmbh = numx.array([[-0.11155958, 57.02097584]])
    gbrbm = Model.GaussianBinaryRBM(2, 2, gbrbmData, gbrbmw, gbrbmbv, gbrbmbh, 1.0, 0.0, 0.0)

    gbrbmTruelogZ = 59.6749019726
    gbrbmTrueLL = -1.7328699078
    gbrbmBestLLPossible = -1.732867951

    epsilon = 0.00001

    def test___init__(self):
        sys.stdout.write('GaussianBinaryRBM -> Performing init test ...')
        sys.stdout.flush()
        localmodel = Model.GaussianBinaryRBM(number_visibles=2,
                                             number_hiddens=2,
                                             data=None,
                                             initial_weights='AUTO',
                                             initial_visible_bias='AUTO',
                                             initial_hidden_bias='AUTO',
                                             initial_sigma='AUTO',
                                             initial_visible_offsets='AUTO',
                                             initial_hidden_offsets='AUTO')
        assert numx.all(localmodel.bv == numx.array([[0, 0]]))
        localmodel = Model.GaussianBinaryRBM(number_visibles=2,
                                             number_hiddens=2,
                                             data=None,
                                             initial_weights='AUTO',
                                             initial_visible_bias='INVERSE_SIGMOID',
                                             initial_hidden_bias='AUTO',
                                             initial_sigma='AUTO',
                                             initial_visible_offsets='AUTO',
                                             initial_hidden_offsets='AUTO')
        assert numx.all(localmodel.bv == numx.array([[0, 0]]))
        localmodel = Model.GaussianBinaryRBM(number_visibles=2,
                                             number_hiddens=2,
                                             data=numx.array([[0.5, 0.1], [0.9, 0.4]]),
                                             initial_weights='AUTO',
                                             initial_visible_bias='AUTO',
                                             initial_hidden_bias='AUTO',
                                             initial_sigma='AUTO',
                                             initial_visible_offsets='AUTO',
                                             initial_hidden_offsets='AUTO')
        assert numx.all(localmodel.bv == numx.mean(numx.array([[0.5, 0.1], [0.9, 0.4]]), axis=0))
        localmodel = Model.GaussianBinaryRBM(number_visibles=2,
                                             number_hiddens=2,
                                             data=numx.array([[0.5, 0.1], [0.9, 0.4]]),
                                             initial_weights='AUTO',
                                             initial_visible_bias='INVERSE_SIGMOID',
                                             initial_hidden_bias='AUTO',
                                             initial_sigma='AUTO',
                                             initial_visible_offsets='AUTO',
                                             initial_hidden_offsets='AUTO')
        assert numx.all(localmodel.bv == numx.mean(numx.array([[0.5, 0.1], [0.9, 0.4]]), axis=0))
        assert numx.all(localmodel._data_mean == numx.mean(numx.array([[0.5, 0.1], [0.9, 0.4]]), axis=0))
        assert numx.all(localmodel._data_std == numx.clip(numx.std(numx.array([[0.5, 0.1], [0.9, 0.4]]), axis=0), 0.001,
                                                          numx.finfo(localmodel.dtype).max))
        assert numx.all(localmodel.sigma == numx.clip(numx.std(numx.array([[0.5, 0.1], [0.9, 0.4]]), axis=0), 0.001,
                                                      numx.finfo(localmodel.dtype).max))
        assert numx.all(
            localmodel._data_mean == numx.clip(numx.mean(numx.array([[0.5, 0.1], [0.9, 0.4]]), axis=0), 0.001,
                                               numx.finfo(localmodel.dtype).max))
        print(' successfully passed!')
        sys.stdout.flush()

    def test__add_visible_units(self):
        sys.stdout.write('GaussianBinaryRBM -> Performing add_visible_units test ...')
        sys.stdout.flush()
        localmodel = Model.GaussianBinaryRBM(number_visibles=1,
                                             number_hiddens=2,
                                             data=self.gbrbmData[:, 0].reshape(2, 1),
                                             initial_weights='AUTO',
                                             initial_visible_bias='AUTO',
                                             initial_hidden_bias='AUTO',
                                             initial_sigma='AUTO',
                                             initial_visible_offsets='AUTO',
                                             initial_hidden_offsets='AUTO')
        localmodel._add_visible_units(num_new_visibles=1,
                                      position=1,
                                      initial_weights='AUTO',
                                      initial_bias='AUTO',
                                      initial_sigmas='AUTO',
                                      initial_offsets='AUTO',
                                      data=self.gbrbmData[:, 1].reshape(2, 1))
        assert numx.all(localmodel.bv == numx.mean(self.gbrbmData, axis=0).reshape(1, self.gbrbmData.shape[1]))
        assert numx.all(localmodel.ov == 0.0)
        assert numx.all(numx.abs(
            localmodel.bv_base - numx.mean(self.gbrbmData, axis=0).reshape(1, self.gbrbmData.shape[1])) < self.epsilon)
        assert numx.all(numx.abs(
            localmodel.sigma - numx.clip(numx.std(self.gbrbmData, axis=0).reshape(1, self.gbrbmData.shape[1]), 0.001,
                                         numx.finfo(localmodel.dtype).max)) < self.epsilon)

        localmodel = Model.GaussianBinaryRBM(number_visibles=1,
                                             number_hiddens=2,
                                             data=self.gbrbmData[:, 0].reshape(2, 1),
                                             initial_weights='AUTO',
                                             initial_visible_bias='AUTO',
                                             initial_hidden_bias='AUTO',
                                             initial_sigma='AUTO',
                                             initial_visible_offsets='AUTO',
                                             initial_hidden_offsets='AUTO')
        localmodel._add_visible_units(num_new_visibles=1,
                                      position=1,
                                      initial_weights='AUTO',
                                      initial_bias='AUTO',
                                      initial_sigmas=1.0,
                                      initial_offsets=0.0,
                                      data=None)
        assert localmodel.bv[0, 0] == numx.mean(self.gbrbmData, axis=0)[0]
        assert localmodel.bv[0, 1] == 0.0
        assert numx.all(localmodel.ov == numx.zeros((1, 2)))
        assert localmodel.bv_base[0, 0] == numx.mean(self.gbrbmData, axis=0)[0]
        assert localmodel.bv_base[0, 1] == 0.0
        assert localmodel.sigma[0, 0] == numx.std(self.gbrbmData, axis=0)[0]
        assert localmodel.sigma[0, 1] == 1.0
        print(' successfully passed!')
        sys.stdout.flush()

    def test__add_hidden_units(self):
        sys.stdout.write('GaussianBinaryRBM -> Performing add_hidden_units test ...')
        sys.stdout.flush()
        localmodel = Model.GaussianBinaryRBM(number_visibles=1,
                                             number_hiddens=2,
                                             data=self.gbrbmData[:, 0].reshape(2, 1),
                                             initial_weights='AUTO',
                                             initial_visible_bias='AUTO',
                                             initial_hidden_bias='AUTO',
                                             initial_sigma='AUTO',
                                             initial_visible_offsets='AUTO',
                                             initial_hidden_offsets='AUTO')
        localmodel._add_hidden_units(num_new_hiddens=1,
                                     position=1,
                                     initial_weights='AUTO',
                                     initial_bias='AUTO',
                                     initial_offsets=0.0)
        assert localmodel.bh[0, 1] == 0.0
        assert numx.all(localmodel.ov == numx.zeros((1, 2)))
        print(' successfully passed!')
        sys.stdout.flush()

    def test__remove_visible_units(self):
        sys.stdout.write('GaussianBinaryRBM -> Performing remove_visible_units test ...')
        sys.stdout.flush()
        localmodel = Model.GaussianBinaryRBM(number_visibles=1,
                                             number_hiddens=2,
                                             data=self.gbrbmData[:, 0].reshape(2, 1),
                                             initial_weights='AUTO',
                                             initial_visible_bias='AUTO',
                                             initial_hidden_bias='AUTO',
                                             initial_sigma='AUTO',
                                             initial_visible_offsets='AUTO',
                                             initial_hidden_offsets='AUTO')
        localmodel._remove_hidden_units([0])
        assert localmodel.bh.shape != [1, 1]
        assert localmodel.bv_base.shape != [1, 1]
        print(' successfully passed!')
        sys.stdout.flush()

    def test__calculate_weight_gradient(self):
        sys.stdout.write('GaussianBinaryRBM -> Performing calculate_weight_gradient test ...')
        sys.stdout.flush()
        deltaW = self.gbrbm._calculate_weight_gradient(numx.array([[0.98, -0.56], [-0.3, 0.8]]),
                                                       numx.array([[0, 1], [1, 1]]))
        target = numx.array([[-0.3, 0.68], [0.8, 0.24]])
        assert numx.all(numx.abs(target - deltaW < self.epsilon))
        print(' successfully passed!')
        sys.stdout.flush()

    def test__calculate_visible_bias_gradient(self):
        sys.stdout.write('GaussianBinaryRBM -> Performing calculate_visible_bias_gradient test ...')
        sys.stdout.flush()
        deltaB = self.gbrbm._calculate_visible_bias_gradient(numx.array([[0.98, -0.56], [-0.3, 0.8]]))
        target = numx.array([[39.16799318, 26.76517392]])
        assert numx.all(numx.abs(target - deltaB < self.epsilon))
        print(' successfully passed!')
        sys.stdout.flush()

    def test_sample_v(self):
        sys.stdout.write('GaussianBinaryRBM -> Performing sample_v test ...')
        sys.stdout.flush()
        numx.random.seed(42)
        samples = self.gbrbm.sample_v(numx.zeros((10000, 2)))
        assert numx.abs(numx.mean(samples)) < 0.01
        assert numx.abs(1.0 - numx.std(samples)) < 0.01
        print(' successfully passed!')
        sys.stdout.flush()

    def test_probability_v_given_h(self):
        sys.stdout.write('GaussianBinaryRBM -> Performing probability_v_given_h test ...')
        sys.stdout.flush()
        probs = self.gbrbm.probability_v_given_h(numx.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
        target = numx.array([[-19.24399659, -13.26258696], [-19.12220171, -13.05940611],
                             [-16.28449482, -41.8863159], [-16.16269994, -41.68313505]])
        assert numx.all(numx.abs(target - probs < self.epsilon))
        print(' successfully passed!')
        sys.stdout.flush()

    def test_probability_h_given_v(self):
        sys.stdout.write('GaussianBinaryRBM -> Performing probability_h_given_v test ...')
        sys.stdout.flush()
        probs = self.gbrbm.probability_h_given_v(numx.array(
            [[-19.24399659, -13.26258696], [-19.12220171, -13.05940611], [-16.28449482, -41.8863159],
             [-16.16269994, -41.68313505]]))
        target = numx.array([[5.76548674e-03, 1], [6.09624677e-03, 1], [2.47805937e-05, 1], [2.62109132e-05, 1]])
        assert numx.all(numx.abs(target - probs < self.epsilon))
        print(' successfully passed!')
        sys.stdout.flush()

    def test_energy(self):
        sys.stdout.write('GaussianBinaryRBM -> Performing energy test ...')
        sys.stdout.flush()
        energy = self.gbrbm.energy(numx.array([[0.98, -0.56], [-0.3, 0.8]]), numx.array([[0, 1], [1, 1]]))
        target = numx.array([[209.23230099], [245.06709061]])
        assert numx.all(numx.abs(target - energy < self.epsilon))
        print(' successfully passed!')
        sys.stdout.flush()

    def test_unnormalized_log_probability_v(self):
        sys.stdout.write('GaussianBinaryRBM -> Performing unnormalized_log_probability_v test ...')
        sys.stdout.flush()
        probs = self.gbrbm.unnormalized_log_probability_v(numx.array([[0.98, -0.56], [-0.3, 0.8]]))
        target = numx.array([[-208.59074139], [-244.38114066]])
        assert numx.all(numx.abs(target - probs) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_unnormalized_log_probability_h(self):
        sys.stdout.write('GaussianBinaryRBM -> Performing unnormalized_log_probability_h test ...')
        sys.stdout.flush()
        probs = self.gbrbm.unnormalized_log_probability_h(numx.array([[0, 1], [1, 1]]))
        target = numx.array([[795.5691597], [784.99179299]])
        assert numx.all(numx.abs(target - probs) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test__base_log_partition(self):
        sys.stdout.write('GaussianBinaryRBM -> Performing base_log_partition test ...')
        sys.stdout.flush()
        base = self.gbrbm._base_log_partition()
        target = 58.4243290171
        assert numx.all(numx.abs(target - base) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()


class Test_GaussianBinaryVarianceRBM(unittest.TestCase):
    gbrbmData = numx.array([[0.57459607, 0.57689834], [-0.60788602, -0.57941004]])
    gbrbmw = numx.array([[0.12179488, 2.95950177],
                         [0.20318085, -28.62372894, ]])
    gbrbmbv = numx.array([[-19.24399659, -13.26258696]])
    gbrbmbh = numx.array([[-0.11155958, 57.02097584]])
    gbrbm = Model.GaussianBinaryVarianceRBM(2, 2, gbrbmData, gbrbmw, gbrbmbv, gbrbmbh, 1.0, 0.0, 0.0)

    gbrbmTruelogZ = 59.6749019726
    gbrbmTrueLL = -1.7328699078
    gbrbmBestLLPossible = -1.732867951

    epsilon = 0.00001

    def test___init__(self):
        sys.stdout.write('GaussianBinaryVarianceRBM -> Performing init test ...')
        sys.stdout.flush()
        print(' successfully passed!')
        sys.stdout.flush()
        pass

    def test__calculate_sigma_gradient(self):
        sys.stdout.write('GaussianBinaryVarianceRBM -> Performing calculate_sigma_gradient test ...')
        sys.stdout.flush()
        deltaSigma = self.gbrbm._calculate_sigma_gradient(numx.array([[0.98, -0.56], [-0.3, 0.8]]),
                                                          numx.array([[0, 1], [1, 1]]))
        target = numx.array([[763.9331994, 372.52636802]])
        assert numx.all(numx.abs(target - deltaSigma) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_get_parameters(self):
        sys.stdout.write('GaussianBinaryVarianceRBM -> Performing get_parameters test ...')
        sys.stdout.flush()
        assert len(self.gbrbm.get_parameters()) == 4
        print(' successfully passed!')
        sys.stdout.flush()

    def test_calculate_gradients(self):
        sys.stdout.write('GaussianBinaryVarianceRBM -> Performing calculate_gradients test ...')
        sys.stdout.flush()
        assert len(
            self.gbrbm.calculate_gradients(numx.array([[0.98, -0.56], [-0.3, 0.8]]), numx.array([[0, 1], [1, 1]]))) == 4
        print(' successfully passed!')
        sys.stdout.flush()


class Test_BinaryBinaryLabelRBM(unittest.TestCase):
    epsilon = 0.00001

    def test___init__(self):
        sys.stdout.write('BinaryBinaryLabelRBM -> Performing init test ...')
        numx.random.seed(42)
        data_dim = 12
        label_dim = 8
        input_dim = data_dim + label_dim
        hidden_dim = 10
        batchsize = 3
        rbm = Model.BinaryBinaryLabelRBM(data_dim, label_dim, hidden_dim)
        assert numx.all(rbm.w.shape == (input_dim, hidden_dim))
        assert numx.all(rbm.bv.shape == (1, input_dim))
        assert numx.all(rbm.bh.shape == (1, hidden_dim))
        assert numx.all(rbm.data_dim == data_dim)
        assert numx.all(rbm.label_dim == label_dim)
        assert numx.all(rbm.input_dim == input_dim)
        assert numx.all(rbm.output_dim == hidden_dim)

        print(' successfully passed!')
        sys.stdout.flush()
        pass

    def test_sample(self):
        sys.stdout.write('BinaryBinaryLabelRBM -> Performing sample test ...')
        numx.random.seed(42)
        data_dim = 12
        label_dim = 8
        input_dim = data_dim + label_dim
        hidden_dim = 10
        batchsize = 3
        rbm = Model.BinaryBinaryLabelRBM(data_dim, label_dim, hidden_dim)

        data = numx.random.rand(batchsize, input_dim)
        h = rbm.probability_h_given_v(data)
        assert numx.all(h.shape == (batchsize, hidden_dim))
        h = rbm.sample_h(h)
        assert numx.all(h.shape == (batchsize, hidden_dim))
        v = rbm.probability_v_given_h(h)
        assert numx.all(v.shape == (batchsize, input_dim))
        v = rbm.sample_v(v)
        assert numx.all(v.shape == (batchsize, input_dim))
        assert numx.all(numx.abs(numx.sum(v[:, data_dim:]) - batchsize) < self.epsilon)
        assert numx.all(v[:, 0:data_dim] >= 0.0)
        assert numx.all(v[:, 0:data_dim] <= 1.0)
        sys.stdout.flush()
        print(' successfully passed!')
        sys.stdout.flush()
        pass


class Test_GaussianBinaryLabelRBM(unittest.TestCase):
    epsilon = 0.00001

    def test___init__(self):
        sys.stdout.write('GaussianBinaryLabelRBM -> Performing init test ...')
        numx.random.seed(42)
        data_dim = 12
        label_dim = 8
        input_dim = data_dim + label_dim
        hidden_dim = 10
        batchsize = 3
        rbm = Model.GaussianBinaryLabelRBM(data_dim, label_dim, hidden_dim)
        assert numx.all(rbm.w.shape == (input_dim, hidden_dim))
        assert numx.all(rbm.bv.shape == (1, input_dim))
        assert numx.all(rbm.bh.shape == (1, hidden_dim))
        assert numx.all(rbm.data_dim == data_dim)
        assert numx.all(rbm.label_dim == label_dim)
        assert numx.all(rbm.input_dim == input_dim)
        assert numx.all(rbm.output_dim == hidden_dim)

        print(' successfully passed!')
        sys.stdout.flush()
        pass

    def test_sample(self):
        sys.stdout.write('GaussianBinaryLabelRBM -> Performing sample test ...')
        numx.random.seed(42)
        data_dim = 12
        label_dim = 8
        input_dim = data_dim + label_dim
        hidden_dim = 10
        batchsize = 3
        rbm = Model.GaussianBinaryLabelRBM(data_dim, label_dim, hidden_dim)

        data = numx.random.rand(batchsize, input_dim)
        h = rbm.probability_h_given_v(data)
        assert numx.all(h.shape == (batchsize, hidden_dim))
        h = rbm.sample_h(h)
        assert numx.all(h.shape == (batchsize, hidden_dim))
        v = rbm.probability_v_given_h(h)
        assert numx.all(v.shape == (batchsize, input_dim))
        v = rbm.sample_v(v)
        assert numx.all(v.shape == (batchsize, input_dim))
        assert numx.all(numx.abs(numx.sum(v[:, data_dim:]) - batchsize) < self.epsilon)
        assert numx.all(v[:, 0:data_dim] != 0.0)
        assert numx.all(v[:, 0:data_dim] != 1.0)
        sys.stdout.flush()
        print(' successfully passed!')
        sys.stdout.flush()
        pass


class Test_BinaryRectRBM(unittest.TestCase):
    epsilon = 0.00001

    def test___init__(self):
        sys.stdout.write('BinaryRectRBM -> Performing init test ...')
        numx.random.seed(42)
        input_dim = 8
        hidden_dim = 10
        batchsize = 3
        rbm = Model.BinaryRectRBM(input_dim, hidden_dim)
        assert numx.all(rbm.w.shape == (input_dim, hidden_dim))
        assert numx.all(rbm.bv.shape == (1, input_dim))
        assert numx.all(rbm.bh.shape == (1, hidden_dim))
        assert numx.all(rbm.input_dim == input_dim)
        assert numx.all(rbm.output_dim == hidden_dim)
        print(' successfully passed!')
        sys.stdout.flush()
        pass

    def test_sample(self):
        sys.stdout.write('BinaryRectRBM -> Performing sample test ...')
        numx.random.seed(420)
        input_dim = 8
        hidden_dim = 10
        batchsize = 3
        rbm = Model.BinaryRectRBM(input_dim, hidden_dim)

        data = numx.random.rand(batchsize, input_dim)
        h = rbm.probability_h_given_v(data)
        assert numx.all(h >= 0.0)
        assert numx.all(h.shape == (batchsize, hidden_dim))
        h = rbm.sample_h(h)
        assert numx.all(h >= 0.0)
        assert numx.all(h <= rbm.max_act)
        assert numx.all(h.shape == (batchsize, hidden_dim))
        v = rbm.probability_v_given_h(h)
        assert numx.all(v > 0.0)
        assert numx.all(v < 1.0)
        assert numx.all(v.shape == (batchsize, input_dim))
        v = rbm.sample_v(v)
        assert numx.all(v >= 0.0)
        assert numx.all(v <= 1.0)
        assert numx.all(v.shape == (batchsize, input_dim))
        print(' successfully passed!')
        sys.stdout.flush()
        pass


class Test_RectBinaryRBM(unittest.TestCase):
    epsilon = 0.00001

    def test___init__(self):
        sys.stdout.write('RectBinaryRBM -> Performing init test ...')

        numx.random.seed(42)
        input_dim = 8
        hidden_dim = 10
        batchsize = 3
        rbm = Model.RectBinaryRBM(input_dim, hidden_dim)
        assert numx.all(rbm.w.shape == (input_dim, hidden_dim))
        assert numx.all(rbm.bv.shape == (1, input_dim))
        assert numx.all(rbm.bh.shape == (1, hidden_dim))
        assert numx.all(rbm.input_dim == input_dim)
        assert numx.all(rbm.output_dim == hidden_dim)
        sys.stdout.flush()
        print(' successfully passed!')
        sys.stdout.flush()
        pass

    def test_sample(self):
        sys.stdout.write('RectBinaryRBM -> Performing sample test ...')
        numx.random.seed(420)
        input_dim = 8
        hidden_dim = 10
        batchsize = 3
        rbm = Model.RectBinaryRBM(input_dim, hidden_dim)

        data = numx.random.rand(batchsize, input_dim)
        h = rbm.probability_h_given_v(data)
        assert numx.all(h >= 0.0)
        assert numx.all(h <= 1.0)
        assert numx.all(h.shape == (batchsize, hidden_dim))
        h = rbm.sample_h(h)
        assert numx.all(h >= 0.0)
        assert numx.all(h <= 1.0)
        assert numx.all(h.shape == (batchsize, hidden_dim))
        v = rbm.probability_v_given_h(h)
        assert numx.all(v >= 0.0)
        assert numx.all(v.shape == (batchsize, input_dim))
        v = rbm.sample_v(v)
        assert numx.all(v >= 0.0)
        assert numx.all(v <= rbm.max_act)
        assert numx.all(v.shape == (batchsize, input_dim))
        sys.stdout.flush()
        print(' successfully passed!')
        sys.stdout.flush()
        pass


class Test_RectBinaryRBM(unittest.TestCase):
    epsilon = 0.00001

    def test___init__(self):
        sys.stdout.write('RectRectRBM -> Performing init test ...')

        numx.random.seed(42)
        input_dim = 8
        hidden_dim = 10
        batchsize = 3
        rbm = Model.RectRectRBM(input_dim, hidden_dim)
        assert numx.all(rbm.w.shape == (input_dim, hidden_dim))
        assert numx.all(rbm.bv.shape == (1, input_dim))
        assert numx.all(rbm.bh.shape == (1, hidden_dim))
        assert numx.all(rbm.input_dim == input_dim)
        assert numx.all(rbm.output_dim == hidden_dim)
        print(' successfully passed!')
        sys.stdout.flush()
        pass

    def test_sample(self):
        sys.stdout.write('RectRectRBM -> Performing sample test ...')
        numx.random.seed(420)
        input_dim = 8
        hidden_dim = 10
        batchsize = 3
        rbm = Model.RectRectRBM(input_dim, hidden_dim)

        data = numx.random.rand(batchsize, input_dim)
        h = rbm.probability_h_given_v(data)
        assert numx.all(h >= 0.0)
        assert numx.all(h.shape == (batchsize, hidden_dim))
        h = rbm.sample_h(h)
        assert numx.all(h >= 0.0)
        assert numx.all(h <= rbm.max_act)
        assert numx.all(h.shape == (batchsize, hidden_dim))
        v = rbm.probability_v_given_h(h)
        assert numx.all(v >= 0.0)
        assert numx.all(v.shape == (batchsize, input_dim))
        v = rbm.sample_v(v)
        assert numx.all(v >= 0.0)
        assert numx.all(v <= rbm.max_act)
        assert numx.all(v.shape == (batchsize, input_dim))
        sys.stdout.flush()
        print(' successfully passed!')
        sys.stdout.flush()
        pass


class Test_GaussianRectRBM(unittest.TestCase):
    epsilon = 0.00001

    def test___init__(self):
        sys.stdout.write('GaussianRectRBM -> Performing init test ...')

        numx.random.seed(42)
        input_dim = 8
        hidden_dim = 10
        batchsize = 3
        rbm = Model.GaussianRectRBM(input_dim, hidden_dim)
        assert numx.all(rbm.w.shape == (input_dim, hidden_dim))
        assert numx.all(rbm.bv.shape == (1, input_dim))
        assert numx.all(rbm.bh.shape == (1, hidden_dim))
        assert numx.all(rbm.input_dim == input_dim)
        assert numx.all(rbm.output_dim == hidden_dim)
        print(' successfully passed!')
        sys.stdout.flush()
        pass

    def test_sample(self):
        sys.stdout.write('GaussianRectRBM -> Performing sample test ...')
        numx.random.seed(420)
        input_dim = 8
        hidden_dim = 10
        batchsize = 3
        rbm = Model.GaussianRectRBM(input_dim, hidden_dim)

        data = numx.random.rand(batchsize, input_dim)
        h = rbm.probability_h_given_v(data)
        assert numx.all(h >= 0.0)
        assert numx.all(h.shape == (batchsize, hidden_dim))
        h = rbm.sample_h(h)
        assert numx.all(h >= 0.0)
        assert numx.all(h <= rbm.max_act)
        assert numx.all(h.shape == (batchsize, hidden_dim))
        v = rbm.probability_v_given_h(h)
        assert numx.all(v.shape == (batchsize, input_dim))
        v = rbm.sample_v(v)
        assert numx.all(v.shape == (batchsize, input_dim))
        sys.stdout.flush()
        print(' successfully passed!')
        sys.stdout.flush()
        pass


class Test_GaussianRectVarianceRBM(unittest.TestCase):
    epsilon = 0.00001

    def test___init__(self):
        sys.stdout.write('GaussianRectVarianceRBM -> Performing init test ...')

        numx.random.seed(42)
        input_dim = 8
        hidden_dim = 10
        batchsize = 3
        rbm = Model.GaussianRectVarianceRBM(input_dim, hidden_dim)
        assert numx.all(rbm.w.shape == (input_dim, hidden_dim))
        assert numx.all(rbm.bv.shape == (1, input_dim))
        assert numx.all(rbm.bh.shape == (1, hidden_dim))
        assert numx.all(rbm.input_dim == input_dim)
        assert numx.all(rbm.output_dim == hidden_dim)
        print(' successfully passed!')
        sys.stdout.flush()
        pass

    def test_sample(self):
        sys.stdout.write('GaussianRectVarianceRBM -> Performing sample test ...')
        numx.random.seed(420)
        input_dim = 8
        hidden_dim = 10
        batchsize = 3
        rbm = Model.GaussianRectVarianceRBM(input_dim, hidden_dim)

        data = numx.random.rand(batchsize, input_dim)
        h = rbm.probability_h_given_v(data)
        assert numx.all(h >= 0.0)
        assert numx.all(h.shape == (batchsize, hidden_dim))
        h = rbm.sample_h(h)
        assert numx.all(h >= 0.0)
        assert numx.all(h <= rbm.max_act)
        assert numx.all(h.shape == (batchsize, hidden_dim))
        v = rbm.probability_v_given_h(h)
        assert numx.all(v.shape == (batchsize, input_dim))
        v = rbm.sample_v(v)
        assert numx.all(v.shape == (batchsize, input_dim))
        sys.stdout.flush()
        print(' successfully passed!')
        sys.stdout.flush()
        pass

    def test__calculate_sigma_gradient(self):
        sys.stdout.write('GaussianRectVarianceRBM -> Performing calculate_sigma_gradient test ...')
        numx.random.seed(42)
        input_dim = 2
        hidden_dim = 2
        rbm = Model.GaussianRectVarianceRBM(input_dim, hidden_dim)
        deltaSigma = rbm._calculate_sigma_gradient(numx.array([[0.98, -0.56], [-0.3, 0.8]]),
                                                   numx.array([[0, 1], [1, 1]]))
        target = numx.array([[-0.63545491, -0.07162506]])
        assert numx.all(numx.abs(target - deltaSigma) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_get_parameters(self):
        sys.stdout.write('GaussianRectVarianceRBM -> Performing get_parameters test ...')
        numx.random.seed(42)
        input_dim = 2
        hidden_dim = 2
        rbm = Model.GaussianRectVarianceRBM(input_dim, hidden_dim)
        assert len(rbm.get_parameters()) == 4
        print(' successfully passed!')
        sys.stdout.flush()

    def test_calculate_gradients(self):
        sys.stdout.write('GaussianRectVarianceRBM -> Performing calculate_gradients test ...')
        numx.random.seed(42)
        input_dim = 2
        hidden_dim = 2
        rbm = Model.GaussianRectVarianceRBM(input_dim, hidden_dim)
        assert len(
            rbm.calculate_gradients(numx.array([[0.98, -0.56], [-0.3, 0.8]]), numx.array([[0, 1], [1, 1]]))) == 4
        print(' successfully passed!')
        sys.stdout.flush()


if __name__ is "__main__":
    unittest.main()
