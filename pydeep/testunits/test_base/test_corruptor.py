""" Test module for corruptor.

    :Version:
        1.1.0

    :Date:
        13.03.2017

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

"""
import unittest
import sys
from pydeep.base.corruptor import *
import numpy as numx

print("\n... pydeep.base.corruptor.py")

epsilon = 0.000001


class TestCorruptor(unittest.TestCase):

    def test_Corruptor_Identity(self):
        sys.stdout.write('Corruptor -> Performing Identity test ...')
        sys.stdout.flush()
        corr = Identity()
        x = numx.array([1.1, 0.9, 9.0, 4.0])
        assert numx.all(numx.abs(x - corr.corrupt(x)) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_Additive_Gauss_noise(self):
        sys.stdout.write('Corruptor -> Performing Additive_Gauss_noise test ...')
        sys.stdout.flush()
        numx.random.seed(42)
        corr = AdditiveGaussNoise(1.0, 4.0)
        x = numx.array([1.1, 0.9, 9.0, 4.0])
        assert numx.all(
            numx.abs(numx.array([4.08685661, 1.3469428, 12.59075415, 11.09211943]) - corr.corrupt(x)) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_Multi_Gauss_noise(self):
        sys.stdout.write('Corruptor -> Performing Multi_Gauss_noise test ...')
        sys.stdout.flush()
        numx.random.seed(42)
        corr = MultiGaussNoise(1.0, 4.0)
        x = numx.array([1.1, 0.9, 9.0, 4.0])
        target = corr.corrupt(x)
        assert numx.all(numx.abs(numx.array([3.28554227, 0.40224852, 32.31678737, 28.3684777]) - target) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_Dropout(self):
        sys.stdout.write('Corruptor -> Performing Dropout test ...')
        sys.stdout.flush()
        numx.random.seed(42)
        corr = Dropout(0.5)
        x = numx.array([1.1, 0.9, 9.0, 4.0])
        assert numx.all(numx.abs(numx.array([0., 1.8, 18., 8.]) - corr.corrupt(x)) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_RandomPermutation(self):
        sys.stdout.write('Corruptor -> Performing RandomPermutation test ...')
        sys.stdout.flush()
        numx.random.seed(42)
        corr = RandomPermutation(0.4)
        x = numx.array([[0, 18, 16, 88, 9, 77, 44, 1, 2, 5], [0, 12, 4, 19, 17, 56, 5, 3, 2, 1]])
        target = numx.array([[18, 0, 16, 88, 9, 2, 44, 1, 77, 5], [2, 56, 4, 19, 17, 12, 5, 3, 0, 1]])
        assert numx.all(numx.abs(target - corr.corrupt(x)) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_Sampling_Binary(self):
        sys.stdout.write('Corruptor -> Performing Sampling_Binary test ...')
        sys.stdout.flush()
        numx.random.seed(42)
        corr = SamplingBinary()
        x = numx.array([1.1, 0.9, 9.0, 4.0])
        assert numx.all(numx.array([True, False, True, True]) == corr.corrupt(x))
        print(' successfully passed!')
        sys.stdout.flush()

    def test_KeepKWinner(self):
        sys.stdout.write('Corruptor -> Performing KeepKWinner test ...')
        sys.stdout.flush()
        numx.random.seed(42)
        numx.random.seed(42)
        data = numx.arange(1, 21)
        numx.random.shuffle(data)
        data = data.reshape(4, 5)
        k = 2
        target = numx.array([[0, 18, 16, 0, 9], [0, 12, 0, 19, 17], [14, 0, 0, 20, 0], [13, 0, 11, 0, 0]])
        cor = KeepKWinner(k, 0)
        assert numx.all(target == cor.corrupt(data))
        target = numx.array([[0, 18, 16, 0, 0], [0, 0, 0, 19, 17], [14, 0, 0, 20, 0], [13, 0, 0, 15, 0]])
        cor = KeepKWinner(k, 1)
        assert numx.all(target == cor.corrupt(data))
        print(' successfully passed!')
        sys.stdout.flush()

    def test_KWinnerTakesAll(self):
        sys.stdout.write('Corruptor -> Performing KWinnerTakesAll test ...')
        sys.stdout.flush()
        numx.random.seed(42)
        data = numx.arange(1, 21)
        numx.random.shuffle(data)
        data = data.reshape(4, 5)
        k = 2
        target = numx.array([[0., 1., 1., 0., 1.], [0., 1., 0., 1., 1.], [1., 0., 0., 1., 0.], [1., 0., 1., 0., 0.]])
        cor = KWinnerTakesAll(k, 0)
        x = cor.corrupt(data)
        assert numx.all(target == x)
        target = numx.array([[0., 1., 1., 0., 0.], [0., 0., 0., 1., 1.], [1., 0., 0., 1., 0.], [1., 0., 0., 1., 0.]])
        cor = KWinnerTakesAll(k, 1)
        x = cor.corrupt(data)
        assert numx.all(target == x)
        print(' successfully passed!')
        sys.stdout.flush()


if __name__ is "__main__":
    unittest.main()
