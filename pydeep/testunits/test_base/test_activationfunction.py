""" Test module for activation functions.

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
from pydeep.base.activationfunction import *
import numpy as numx

print("\n... pydeep.base.activationfunction.py")

epsilon = 0.000001


class TestActivationFunction(unittest.TestCase):

    def test_Identity(self):
        sys.stdout.write('Activationfunction -> Performing Identity() test ...')
        sys.stdout.flush()
        assert numx.all(numx.abs(4.0 - Identity().f(4.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - Identity().f(0.0)) < epsilon)
        assert numx.all(numx.abs(-4.0 - Identity().f(-4.0)) < epsilon)

        assert numx.all(numx.abs(4.0 - Identity().g(4.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - Identity().g(0.0)) < epsilon)
        assert numx.all(numx.abs(-4.0 - Identity().g(-4.0)) < epsilon)

        assert numx.all(numx.abs(1.0 - Identity().df(4.0)) < epsilon)
        assert numx.all(numx.abs(1.0 - Identity().df(0.0)) < epsilon)
        assert numx.all(numx.abs(1.0 - Identity().df(-4.0)) < epsilon)

        assert numx.all(numx.abs(0.0 - Identity().ddf(4.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - Identity().ddf(0.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - Identity().ddf(-4.0)) < epsilon)

        assert numx.all(numx.abs(1.0 - Identity().dg(4.0)) < epsilon)
        assert numx.all(numx.abs(1.0 - Identity().dg(0.0)) < epsilon)
        assert numx.all(numx.abs(1.0 - Identity().dg(-4.0)) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_Sigmoid(self):
        sys.stdout.write('Activationfunction -> Performing Sigmoid() test ...')
        sys.stdout.flush()
        assert numx.all(numx.abs(0.982013790038 - Sigmoid().f(4.0)) < epsilon)
        assert numx.all(numx.abs(0.5 - Sigmoid().f(0.0)) < epsilon)
        assert numx.all(numx.abs(0.0179862099621 - Sigmoid().f(-4.0)) < epsilon)

        assert numx.all(numx.abs(-1.09861228867 - Sigmoid().g(0.25)) < epsilon)
        assert numx.all(numx.abs(0.0 - Sigmoid().g(0.5)) < epsilon)
        assert numx.all(numx.abs(1.09861228867 - Sigmoid().g(0.75)) < epsilon)

        assert numx.all(numx.abs(0.0176627062133 - Sigmoid().df(4.0)) < epsilon)
        assert numx.all(numx.abs(0.25 - Sigmoid().df(0.0)) < epsilon)
        assert numx.all(numx.abs(0.0176627062133 - Sigmoid().df(-4.0)) < epsilon)

        assert numx.all(numx.abs(-0.0170273359284 - Sigmoid().ddf(4.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - Sigmoid().ddf(0.0)) < epsilon)
        assert numx.all(numx.abs(0.0170273359284 - Sigmoid().ddf(-4.0)) < epsilon)

        assert numx.all(numx.abs(4.0 - Sigmoid().dg(0.5)) < epsilon)
        assert numx.all(numx.abs(-1.0 / 0.75 - Sigmoid().dg(-0.5)) < epsilon)
        assert numx.all(numx.abs(5.33333333333 - Sigmoid().dg(0.75)) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_HyperbolicTangent(self):
        sys.stdout.write('Activationfunction -> Performing HyperbolicTangent() test ...')
        sys.stdout.flush()
        assert numx.all(numx.abs(0.999329299739 - HyperbolicTangent().f(4.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - HyperbolicTangent().f(0.0)) < epsilon)
        assert numx.all(numx.abs(-0.999329299739 - HyperbolicTangent().f(-4.0)) < epsilon)

        assert numx.all(numx.abs(-0.972955074528 - HyperbolicTangent().g(-0.75)) < epsilon)
        assert numx.all(numx.abs(0.0 - HyperbolicTangent().g(0.0)) < epsilon)
        assert numx.all(numx.abs(0.972955074528 - HyperbolicTangent().g(0.75)) < epsilon)

        assert numx.all(numx.abs(0.00134095068303 - HyperbolicTangent().df(4.0)) < epsilon)
        assert numx.all(numx.abs(1.0 - HyperbolicTangent().df(0.0)) < epsilon)
        assert numx.all(numx.abs(0.00134095068303 - HyperbolicTangent().df(-4.0)) < epsilon)

        assert numx.all(numx.abs(-0.00268010261411 - HyperbolicTangent().ddf(4.0)) < epsilon)
        assert numx.all(numx.abs(-.0 - HyperbolicTangent().ddf(0.0)) < epsilon)
        assert numx.all(numx.abs(0.00268010261411 - HyperbolicTangent().ddf(-4.0)) < epsilon)

        assert numx.all(numx.abs(2.28571428571 - HyperbolicTangent().dg(-0.75)) < epsilon)
        assert numx.all(numx.abs(1.0 - HyperbolicTangent().dg(0.0)) < epsilon)
        assert numx.all(numx.abs(2.28571428571 - HyperbolicTangent().dg(0.75)) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_SoftSign(self):
        sys.stdout.write('Activationfunction -> Performing SoftSign() test ...')
        sys.stdout.flush()
        assert numx.all(numx.abs(0.8 - SoftSign().f(4.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - SoftSign().f(0.0)) < epsilon)
        assert numx.all(numx.abs(-0.8 - SoftSign().f(-4.0)) < epsilon)

        assert numx.all(numx.abs(0.04 - SoftSign().df(4.0)) < epsilon)
        assert numx.all(numx.abs(1.0 - SoftSign().df(0.0)) < epsilon)
        assert numx.all(numx.abs(0.04 - SoftSign().df(-4.0)) < epsilon)

        assert numx.all(numx.abs(-0.016 - SoftSign().ddf(4.0)) < epsilon)
        assert numx.all(numx.abs(-1.99999400001 - SoftSign().ddf(0.000001)) < epsilon)
        assert numx.all(numx.abs(0.016 - SoftSign().ddf(-4.0)) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_Step(self):
        sys.stdout.write('Activationfunction -> Performing Step() test ...')
        sys.stdout.flush()
        assert numx.all(numx.abs(0.0 - Step().f(-1.0)) < epsilon)
        assert numx.all(numx.abs(1.0 - Step().f(+1.0)) < epsilon)

        assert numx.all(numx.abs(0.0 - Step().df(0.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - Step().df(0.0)) < epsilon)

        assert numx.all(numx.abs(0.0 - Step().ddf(0.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - Step().ddf(0.0)) < epsilon)

        print(' successfully passed!')
        sys.stdout.flush()

    def test_Rectifier(self):
        sys.stdout.write('Activationfunction -> Performing Rectifier() test ...')
        sys.stdout.flush()
        assert numx.all(numx.abs(0.0 - Rectifier().f(-0.0001)) < epsilon)
        assert numx.all(numx.abs(0.0 - Rectifier().f(-4.0)) < epsilon)
        assert numx.all(numx.abs(0.0001 - Rectifier().f(+0.0001)) < epsilon)
        assert numx.all(numx.abs(4.0 - Rectifier().f(4.0)) < epsilon)

        assert numx.all(numx.abs(0.0 - Rectifier().df(-0.0001)) < epsilon)
        assert numx.all(numx.abs(0.0 - Rectifier().df(-4.0)) < epsilon)
        assert numx.all(numx.abs(1.0 - Rectifier().df(+0.0001)) < epsilon)
        assert numx.all(numx.abs(1.0 - Rectifier().df(4.0)) < epsilon)

        assert numx.all(numx.abs(0.0 - Rectifier().ddf(-0.0001)) < epsilon)
        assert numx.all(numx.abs(0.0 - Rectifier().ddf(-4.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - Rectifier().ddf(+0.0001)) < epsilon)
        assert numx.all(numx.abs(0.0 - Rectifier().ddf(4.0)) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_SoftPlus(self):
        sys.stdout.write('Activationfunction -> Performing SoftPlus() test ...')
        sys.stdout.flush()

        assert numx.all(numx.abs(4.01814992792 - SoftPlus().f(4.0)) < epsilon)
        assert numx.all(numx.abs(0.69314718056 - SoftPlus().f(0.0)) < epsilon)
        assert numx.all(numx.abs(0.0181499279178 - SoftPlus().f(-4.0)) < epsilon)

        assert numx.all(numx.abs(3.98151455317 - SoftPlus().g(4.0)) < epsilon)
        assert numx.all(numx.abs(-9.21029037156 - SoftPlus().g(0.0001)) < epsilon)

        assert numx.all(numx.abs(1.01865736036 - SoftPlus().dg(4.0)) < epsilon)
        assert numx.all(numx.abs(10000.5000083 - SoftPlus().dg(0.0001)) < epsilon)

        assert numx.all(numx.abs(0.982013790038 - SoftPlus().df(4.0)) < epsilon)
        assert numx.all(numx.abs(0.5 - SoftPlus().df(0.0)) < epsilon)
        assert numx.all(numx.abs(0.0179862099621 - SoftPlus().df(-4.0)) < epsilon)

        assert numx.all(numx.abs(0.0176627062133 - SoftPlus().ddf(4.0)) < epsilon)
        assert numx.all(numx.abs(0.25 - SoftPlus().ddf(0.0)) < epsilon)
        assert numx.all(numx.abs(0.0176627062133 - SoftPlus().ddf(-4.0)) < epsilon)

        print(' successfully passed!')
        sys.stdout.flush()

    def test_SoftMax(self):
        sys.stdout.write('Activationfunction -> Performing SoftMax() test ...')
        sys.stdout.flush()

        assert numx.all(numx.abs(
            numx.array([[0.28001309, 0.34200877, 0.37797814]]) - SoftMax().f(numx.array([[0.2, 0.4, 0.5]]))) < epsilon)
        assert numx.all(numx.abs(
            numx.array([[0.33333333, 0.33333333, 0.33333333]]) - SoftMax().f(numx.array([[0.5, 0.5, 0.5]]))) < epsilon)
        assert numx.all(numx.abs(numx.array([[0.57244661, 0.21484557, 0.21270782]]) - SoftMax().f(
            numx.array([[0.99, 0.01, 0.0]]))) < epsilon)

        assert numx.all(
            numx.abs(numx.array([[[0.16, -0.08], [-0.08, 0.24]]]) - SoftMax().df(numx.array([[0.2, 0.4]]))) < epsilon)
        assert numx.all(
            numx.abs(numx.array([[[0.25, -0.25], [-0.25, 0.25]]]) - SoftMax().df(numx.array([[0.5, 0.5]]))) < epsilon)
        assert numx.all(
            numx.abs(numx.array([[[0.0099, 0.], [0., 0.]]]) - SoftMax().df(numx.array([[0.99, 0.0]]))) < epsilon)

        print(' successfully passed!')
        sys.stdout.flush()

    def test_Sinus(self):
        sys.stdout.write('Activationfunction -> Performing Sinus() test ...')
        sys.stdout.flush()

        assert numx.all(numx.abs(0.0 - Sinus().f(0.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - Sinus().f(numx.pi)) < epsilon)
        assert numx.all(numx.abs(0.0 - Sinus().f(2 * numx.pi)) < epsilon)

        assert numx.all(numx.abs(1.0 - Sinus().f(numx.pi / 2)) < epsilon)
        assert numx.all(numx.abs(-1.0 - Sinus().f(3 * (numx.pi / 2))) < epsilon)

        assert numx.all(numx.abs(0.0 - Sinus().ddf(0.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - Sinus().ddf(numx.pi)) < epsilon)
        assert numx.all(numx.abs(0.0 - Sinus().ddf(2 * numx.pi)) < epsilon)

        assert numx.all(numx.abs(-1.0 - Sinus().ddf(numx.pi / 2)) < epsilon)
        assert numx.all(numx.abs(1.0 - Sinus().ddf(3 * (numx.pi / 2))) < epsilon)

        assert numx.all(numx.abs(1.0 - Sinus().df(0.0)) < epsilon)
        assert numx.all(numx.abs(-1.0 - Sinus().df(numx.pi)) < epsilon)
        assert numx.all(numx.abs(1.0 - Sinus().df(2 * numx.pi)) < epsilon)

        assert numx.all(numx.abs(0.0 - Sinus().df(numx.pi / 2)) < epsilon)
        assert numx.all(numx.abs(0.0 - Sinus().df(3 * (numx.pi / 2))) < epsilon)

        print(' successfully passed!')
        sys.stdout.flush()

    def test_RestrictedRectifier(self):
        sys.stdout.write('Activationfunction -> Performing test_RestrictedRectifier() test ...')
        sys.stdout.flush()
        assert numx.all(numx.abs(0.0 - RestrictedRectifier().f(-0.0001)) < epsilon)
        assert numx.all(numx.abs(0.0 - RestrictedRectifier().f(-4.0)) < epsilon)
        assert numx.all(numx.abs(0.0001 - RestrictedRectifier().f(+0.0001)) < epsilon)
        assert numx.all(numx.abs(1.0 - RestrictedRectifier().f(4.0)) < epsilon)

        assert numx.all(numx.abs(0.0 - RestrictedRectifier().df(-0.0001)) < epsilon)
        assert numx.all(numx.abs(0.0 - RestrictedRectifier().df(-4.0)) < epsilon)
        assert numx.all(numx.abs(1.0 - RestrictedRectifier().df(+0.0001)) < epsilon)
        assert numx.all(numx.abs(0.0 - RestrictedRectifier().df(4.0)) < epsilon)

        assert numx.all(numx.abs(0.0 - RestrictedRectifier().ddf(-0.0001)) < epsilon)
        assert numx.all(numx.abs(0.0 - RestrictedRectifier().ddf(-4.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - RestrictedRectifier().ddf(+0.0001)) < epsilon)
        assert numx.all(numx.abs(0.0 - RestrictedRectifier().ddf(4.0)) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_RadialBasis(self):
        sys.stdout.write('Activationfunction -> Performing RadialBasis() test ...')
        sys.stdout.flush()

        assert numx.all(numx.abs(0.0183156388887 - RadialBasis(0.0, 1.0).f(2.0)) < epsilon)
        assert numx.all(numx.abs(1.0 - RadialBasis(0.0, 1.0).f(0.0)) < epsilon)
        assert numx.all(numx.abs(0.0183156388887 - RadialBasis(0.0, 1.0).f(-2.0)) < epsilon)
        assert numx.all(numx.abs(0.513417119033 - RadialBasis(1.0, 1.5).f(2.0)) < epsilon)
        assert numx.all(numx.abs(0.513417119033 - RadialBasis(1.0, 1.5).f(0.0)) < epsilon)
        assert numx.all(numx.abs(0.00247875217667 - RadialBasis(1.0, 1.5).f(-2.0)) < epsilon)

        assert numx.all(numx.abs(-0.0732625555549 - RadialBasis(0.0, 1.0).df(2.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - RadialBasis(0.0, 1.0).df(0.0)) < epsilon)
        assert numx.all(numx.abs(0.0732625555549 - RadialBasis(0.0, 1.0).df(-2.0)) < epsilon)
        assert numx.all(numx.abs(-0.68455615871 - RadialBasis(1.0, 1.5).df(2.0)) < epsilon)
        assert numx.all(numx.abs(0.68455615871 - RadialBasis(1.0, 1.5).df(0.0)) < epsilon)
        assert numx.all(numx.abs(0.00991500870667 - RadialBasis(1.0, 1.5).df(-2.0)) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_LeakyRestricted(self):
        sys.stdout.write('Activationfunction -> Performing test_LeakyRectifier() test ...')
        sys.stdout.flush()
        rect = LeakyRectifier(0.01, 0.5)
        assert numx.all(numx.abs(-2.0*rect.negativeSlope - rect.f(-2.0)) < epsilon)
        assert numx.all(numx.abs(0.0*rect.positiveSlope  - rect.f(0.0)) < epsilon)
        assert numx.all(numx.abs(2.0*rect.positiveSlope - rect.f(2.0)) < epsilon)

        assert numx.all(numx.abs(rect.negativeSlope - rect.df(-2.0)) < epsilon)
        assert numx.all(numx.abs(rect.positiveSlope  - rect.df(0.0)) < epsilon)
        assert numx.all(numx.abs(rect.positiveSlope - rect.df(2.0)) < epsilon)

        print(' successfully passed!')
        sys.stdout.flush()

    def test_SigmoidWeightedLinear(self):
        sys.stdout.write('Activationfunction -> Performing test_SigmoidWeightedLinear() test ...')
        sys.stdout.flush()
        sig = SigmoidWeightedLinear()

        assert numx.all(numx.abs(-0.26894142137- sig.f(-1.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - sig.f(0.0)) < epsilon)
        assert numx.all(numx.abs(0.73105857863 - sig.f(1.0)) < epsilon)

        assert numx.all(numx.abs(0.0723294881285 - sig.df(-1.0)) < epsilon)
        assert numx.all(numx.abs(0.5 - sig.df(0.0)) < epsilon)
        assert numx.all(numx.abs(0.927670511871 - sig.df(1.0)) < epsilon)

        print(' successfully passed!')
        sys.stdout.flush()

    def test_ExponentialLinear(self):
        sys.stdout.write('Activationfunction -> Performing test_ExponentialLinear() test ...')
        sys.stdout.flush()
        expLin = ExponentialLinear()

        assert numx.all(numx.abs(-0.632120558829 - expLin.f(-1.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - expLin.f(0.0)) < epsilon)
        assert numx.all(numx.abs(2.0 - expLin.f(2.0)) < epsilon)

        assert numx.all(numx.abs(0.367879441171 - expLin.df(-1.0)) < epsilon)
        assert numx.all(numx.abs(1.0 - expLin.df(0.0)) < epsilon)
        assert numx.all(numx.abs(1.0 - expLin.df(2.0)) < epsilon)

        print(' successfully passed!')
        sys.stdout.flush()

    def test_KWinnerTakeAll(self):
        sys.stdout.write('Activationfunction -> Performing KWinnerTakeAll() test ...')
        sys.stdout.flush()

        act = KWinnerTakeAll(k=2,axis = 1,activation_function=Identity())

        data = numx.array([[0.28001309, 0.34200877, 0.37797814], [3, 1, 6]])
        target = numx.array([[0.0, 0.34200877, 0.37797814], [3, 0, 6]])
        assert numx.all(numx.abs(target - act.f(data) < epsilon))

        target = numx.array([[0, 1, 1], [1, 0, 1]])
        assert numx.all(numx.abs(target - act.df(data) < epsilon))

        act = KWinnerTakeAll(k=1,axis = 0,activation_function=Identity())

        data = numx.array([[0.28001309, 0.34200877, 0.37797814], [3, 0, -1]])
        target = numx.array([[0, 0.34200877, 0.37797814], [3, 0, 0]])

        assert numx.all(numx.abs(target - act.f(data) < epsilon))

        target = numx.array([[0, 1, 1], [1, 0, 0]])
        assert numx.all(numx.abs(target - act.df(data) < epsilon))


        act = KWinnerTakeAll(k=2,axis = 1,activation_function=Sigmoid())

        data = numx.array([[0.28001309, 0.34200877, 0.37797814], [3, 1, 6]])
        target = numx.array([[0.0, Sigmoid.f(0.34200877), Sigmoid.f(0.37797814)], [Sigmoid.f(3), 0, Sigmoid.f(6)]])
        assert numx.all(numx.abs(target - act.f(data) < epsilon))

        target = numx.array([[0.0, Sigmoid.df(0.34200877), Sigmoid.df(0.37797814)], [Sigmoid.df(3), 0, Sigmoid.df(6)]])
        assert numx.all(numx.abs(target - act.df(data) < epsilon))

        print(' successfully passed!')
        sys.stdout.flush()

if __name__ is "__main__":
    unittest.main()
