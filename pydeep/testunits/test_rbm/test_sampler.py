''' Test module for RBM sampler.

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
import pydeep.rbm.sampler as Sampler

print("\n... pydeep.rbm.sampler.py")


class TestSampler(unittest.TestCase):
    bbrbmData = generate_bars_and_stripes_complete(2)
    bbrbmData = numx.vstack((bbrbmData[0], bbrbmData, bbrbmData[5]))
    bbrbmw = numx.array([[0.12179488, 2.95950177, 0.33513356, 35.05380642],
                         [0.20318085, -28.62372894, 26.52611278, 28.41793445],
                         [-0.19105386, -28.58530584, -26.52747507, 28.78447320],
                         [0.08953740, -59.82556859, -0.06665933, -27.71723459]])
    bbrbmbv = numx.array([[-19.24399659, -13.26258696, 13.25909850, 43.74408543]])
    bbrbmbh = numx.array([[-0.11155958, 57.02097584, -0.13331758, -32.25991501]])
    bbrbm = Model.BinaryBinaryRBM(4, 4, bbrbmData, bbrbmw, bbrbmbv, bbrbmbh, 0.0, 0.0)
    epsilon = 0.05
    num_samples = 2000.0

    @classmethod
    def execute_sampler(cls, sampler, num_samples):
        dictC = {'[0. 0. 0. 0.]': 0,  # 2
                 '[1. 1. 1. 1.]': 0,  # 2
                 '[0. 0. 1. 1.]': 0,  # 1
                 '[1. 1. 0. 0.]': 0,  # 1
                 '[1. 0. 1. 0.]': 0,  # 1
                 '[0. 1. 0. 1.]': 0,  # 1
                 '[0. 1. 1. 0.]': 0,
                 '[1. 0. 0. 1.]': 0,
                 '[0. 0. 0. 1.]': 0,
                 '[0. 0. 1. 0.]': 0,
                 '[0. 1. 0. 0.]': 0,
                 '[1. 0. 0. 0.]': 0,
                 '[0. 1. 1. 1.]': 0,
                 '[1. 1. 1. 0.]': 0,
                 '[1. 0. 1. 1.]': 0,
                 '[1. 1. 0. 1.]': 0}
        for _ in range(numx.int32(num_samples)):
            if isinstance(sampler, Sampler.GibbsSampler):
                # Start form random since model is rather deterministic
                samples = sampler.sample(numx.random.rand(1, 4), 1, ret_states=True)
            else:
                if isinstance(sampler, Sampler.PersistentGibbsSampler):
                    # Start form random since model is rather deterministic
                    sampler.chains = numx.random.rand(1, 4)
                    samples = sampler.sample(1, 1, ret_states=True)
                else:
                    samples = sampler.sample(1, 1, ret_states=True)

            dictC[str(samples[0])] += 1
        probCD1 = dictC['[0. 0. 0. 0.]'] / num_samples
        probCD2 = dictC['[1. 1. 1. 1.]'] / num_samples
        probCS1 = dictC['[0. 0. 1. 1.]'] / num_samples
        probCS2 = dictC['[1. 1. 0. 0.]'] / num_samples
        probCS3 = dictC['[1. 0. 1. 0.]'] / num_samples
        probCS4 = dictC['[0. 1. 0. 1.]'] / num_samples
        sumProbs = probCD1 + probCD2 + probCS1 + probCS2 + probCS3 + probCS4
        return [probCD1, probCD2, probCS1, probCS2, probCS3, probCS4, sumProbs]

    def test_Gibbs_sampler(self):
        sys.stdout.write('RBM Sampler -> Performing GibbsSampler test ... ')
        sys.stdout.flush()
        numx.random.seed(42)
        sampler = Sampler.GibbsSampler(self.bbrbm)
        probCD1, probCD2, probCS1, probCS2, probCS3, probCS4, sumProbs = self.execute_sampler(sampler, self.num_samples)
        assert numx.all(numx.abs(1.0 / 4.0 - probCD1) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 4.0 - probCD2) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS1) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS2) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS3) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS4) < self.epsilon)
        assert numx.all(numx.abs(1.0 - sumProbs) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test_Persistent_Gibbs_sampler(self):
        sys.stdout.write('RBM Sampler -> Performing PersistentGibbsSampler test ... ')
        sys.stdout.flush()
        numx.random.seed(42)
        sampler = Sampler.PersistentGibbsSampler(self.bbrbm, 1)
        probCD1, probCD2, probCS1, probCS2, probCS3, probCS4, sumProbs = self.execute_sampler(sampler, self.num_samples)
        assert numx.all(numx.abs(1.0 / 4.0 - probCD1) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 4.0 - probCD2) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS1) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS2) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS3) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS4) < self.epsilon)
        assert numx.all(numx.abs(1.0 - sumProbs) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test_Parallel_Tempering_sampler(self):
        sys.stdout.write('RBM Sampler -> Performing ParallelTemperingSampler test ... ')
        sys.stdout.flush()
        numx.random.seed(42)
        sampler = Sampler.ParallelTemperingSampler(self.bbrbm, 10)
        probCD1, probCD2, probCS1, probCS2, probCS3, probCS4, sumProbs = self.execute_sampler(sampler, self.num_samples)
        assert numx.all(numx.abs(1.0 / 4.0 - probCD1) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 4.0 - probCD2) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS1) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS2) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS3) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS4) < self.epsilon)
        assert numx.all(numx.abs(1.0 - sumProbs) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test_Independent_Parallel_Tempering_sampler(self):
        sys.stdout.write('RBM Sampler -> Performing IndependentParallelTemperingSampler test ... ')
        sys.stdout.flush()
        numx.random.seed(42)
        sampler = Sampler.IndependentParallelTemperingSampler(self.bbrbm, 10, 10)
        probCD1, probCD2, probCS1, probCS2, probCS3, probCS4, sumProbs = self.execute_sampler(sampler, self.num_samples)
        assert numx.all(numx.abs(1.0 / 4.0 - probCD1) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 4.0 - probCD2) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS1) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS2) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS3) < self.epsilon)
        assert numx.all(numx.abs(1.0 / 8.0 - probCS4) < self.epsilon)
        assert numx.all(numx.abs(1.0 - sumProbs) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()


if __name__ is '__main__':
    unittest.main()
