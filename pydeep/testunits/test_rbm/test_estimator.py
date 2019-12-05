''' Test module for RBM estimators.

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

from pydeep.misc.toyproblems import generate_bars_and_stripes_complete
import pydeep.rbm.model as Model
import pydeep.rbm.estimator as Estimator

print("\n... pydeep.rbm.estimator.py")


class TestEstimator(unittest.TestCase):
    # Known model
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

    def test_reconstruction_error(self):
        sys.stdout.write('RBM Estimator -> Performing reconstruction_error test ...')
        sys.stdout.flush()
        numx.random.seed(42)
        rec = Estimator.reconstruction_error(self.bbrbm,
                                             self.bbrbmData,
                                             k=1,
                                             beta=1.0,
                                             use_states=True,
                                             absolut_error=False)
        assert numx.all(numx.abs(rec) < self.epsilon)
        rec = Estimator.reconstruction_error(self.bbrbm,
                                             self.bbrbmData,
                                             k=1,
                                             beta=1.0,
                                             use_states=False,
                                             absolut_error=False)
        assert numx.all(numx.abs(rec) < self.epsilon)
        rec = Estimator.reconstruction_error(self.bbrbm,
                                             self.bbrbmData,
                                             k=1,
                                             beta=1.0,
                                             use_states=True,
                                             absolut_error=True)
        assert numx.all(numx.abs(rec) < self.epsilon)
        rec = Estimator.reconstruction_error(self.bbrbm,
                                             self.bbrbmData,
                                             k=1,
                                             beta=1.0,
                                             use_states=False,
                                             absolut_error=True)
        assert numx.all(numx.abs(rec) < self.epsilon)
        rec = Estimator.reconstruction_error(self.bbrbm,
                                             self.bbrbmData,
                                             k=10,
                                             beta=1.0,
                                             use_states=False,
                                             absolut_error=False)
        assert numx.all(numx.abs(rec) < self.epsilon)
        # Test List
        testList = []
        for i in range(self.bbrbmData.shape[0]):
            testList.append(self.bbrbmData[i].reshape(1, 4))
        rec = Estimator.reconstruction_error(self.bbrbm,
                                             testList,
                                             k=10,
                                             beta=1.0,
                                             use_states=False,
                                             absolut_error=False)
        assert numx.all(numx.abs(rec) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_log_likelihood_v(self):
        sys.stdout.write('RBM Estimator -> Performing log_likelihood_v test ...')
        sys.stdout.flush()
        numx.random.seed(42)
        ll = numx.mean(Estimator.log_likelihood_v(self.bbrbm, self.bbrbmTruelogZ, self.bbrbmData, 1.0))
        assert numx.all(numx.abs(ll - self.bbrbmTrueLL) < self.epsilon)
        # Test List
        testList = []
        for i in range(self.bbrbmData.shape[0]):
            testList.append(self.bbrbmData[i].reshape(1, 4))
        ll = numx.mean(Estimator.log_likelihood_v(self.bbrbm, self.bbrbmTruelogZ, testList, 1.0))
        assert numx.all(numx.abs(ll - self.bbrbmTrueLL) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_log_likelihood_h(self):
        sys.stdout.write('RBM Estimator -> Performing log_likelihood_h test ...')
        sys.stdout.flush()
        numx.random.seed(42)
        hdata = numx.float64(self.bbrbm.probability_h_given_v(self.bbrbmData) < 0.5)
        ll = numx.mean(Estimator.log_likelihood_h(self.bbrbm, self.bbrbmTruelogZ, hdata, 1.0))
        assert numx.all(numx.abs(ll + 9.55929166739) < self.epsilon)
        # Test List
        testList = []
        for i in range(hdata.shape[0]):
            testList.append(hdata[i].reshape(1, 4))
        ll = numx.mean(Estimator.log_likelihood_v(self.bbrbm, self.bbrbmTruelogZ, testList, 1.0))
        assert numx.all(numx.abs(ll + 9.55929166739) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_partition_function_factorize_v(self):
        sys.stdout.write('RBM Estimator -> Performing partition_function_factorize_v test ...')
        sys.stdout.flush()
        LogZ = Estimator.partition_function_factorize_v(self.bbrbm, beta=None, batchsize_exponent='AUTO', status=False)
        assert numx.all(numx.abs(LogZ - self.bbrbmTruelogZ) < self.epsilon)
        LogZ = Estimator.partition_function_factorize_v(self.bbrbm, beta=None, batchsize_exponent=0, status=False)
        assert numx.all(numx.abs(LogZ - self.bbrbmTruelogZ) < self.epsilon)
        LogZ = Estimator.partition_function_factorize_v(self.bbrbm, beta=None, batchsize_exponent=3, status=False)
        assert numx.all(numx.abs(LogZ - self.bbrbmTruelogZ) < self.epsilon)
        LogZ = Estimator.partition_function_factorize_v(self.bbrbm, beta=None, batchsize_exponent=555, status=False)
        assert numx.all(numx.abs(LogZ - self.bbrbmTruelogZ) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_partition_function_factorize_h(self):
        sys.stdout.write('RBM Estimator -> Performing partition_function_factorize_v test ...')
        sys.stdout.flush()
        LogZ = Estimator.partition_function_factorize_h(self.bbrbm, beta=None, batchsize_exponent='AUTO', status=False)
        assert numx.all(numx.abs(LogZ - self.bbrbmTruelogZ) < self.epsilon)
        LogZ = Estimator.partition_function_factorize_h(self.bbrbm, beta=None, batchsize_exponent=0, status=False)
        assert numx.all(numx.abs(LogZ - self.bbrbmTruelogZ) < self.epsilon)
        LogZ = Estimator.partition_function_factorize_h(self.bbrbm, beta=None, batchsize_exponent=3, status=False)
        assert numx.all(numx.abs(LogZ - self.bbrbmTruelogZ) < self.epsilon)
        LogZ = Estimator.partition_function_factorize_h(self.bbrbm, beta=None, batchsize_exponent=555, status=False)
        assert numx.all(numx.abs(LogZ - self.bbrbmTruelogZ) < self.epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_annealed_importance_sampling(self):
        sys.stdout.write('RBM Estimator -> Performing annealed_importance_sampling test ...')
        sys.stdout.flush()
        numx.random.seed(42)
        LogZ = Estimator.annealed_importance_sampling(self.bbrbm, num_chains=100, k=1, betas=100, status=False)
        assert numx.all(numx.abs(LogZ[0] - self.bbrbmTruelogZ) < 0.5)
        LogZ = Estimator.annealed_importance_sampling(self.bbrbm, num_chains=100, k=1, betas=1000, status=False)
        assert numx.all(numx.abs(LogZ[0] - self.bbrbmTruelogZ) < 0.05)
        LogZ = Estimator.annealed_importance_sampling(self.bbrbm, num_chains=100, k=1, betas=10000, status=False)
        assert numx.all(numx.abs(LogZ[0] - self.bbrbmTruelogZ) < 0.005)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_reverse_annealed_importance_sampling(self):
        sys.stdout.write('RBM Estimator -> Performing reverse_annealed_importance_sampling test ...')
        sys.stdout.flush()
        numx.random.seed(42)
        LogZ = Estimator.reverse_annealed_importance_sampling(self.bbrbm, num_chains=100, k=1, betas=100, status=False)
        assert numx.all(numx.abs(LogZ[0] - self.bbrbmTruelogZ) < 0.5)
        LogZ = Estimator.reverse_annealed_importance_sampling(self.bbrbm, num_chains=100, k=1, betas=1000, status=False)
        assert numx.all(numx.abs(LogZ[0] - self.bbrbmTruelogZ) < 0.05)
        LogZ = Estimator.reverse_annealed_importance_sampling(self.bbrbm, num_chains=100, k=1, betas=10000, status=False)
        assert numx.all(numx.abs(LogZ[0] - self.bbrbmTruelogZ) < 0.005)
        print(' successfully passed!')
        sys.stdout.flush()


if __name__ is "__main__":
    unittest.main()
