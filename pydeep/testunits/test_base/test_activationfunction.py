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

print "\n... pydeep.base.activationfunction.py"

epsilon = 0.000001


class TestActivationFunction(unittest.TestCase):

    def test_Identity(self):
        print('Activationfunction -> Performing Identity() test ...')
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
        print('successfully passed!')
        sys.stdout.flush()

    def test_Sigmoid(self):
        print('Activationfunction -> Performing Sigmoid() test ...')
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
        print('successfully passed!')
        sys.stdout.flush()

    def test_TangentsHyperbolicus(self):
        print('Activationfunction -> Performing TangentsHyperbolicus() test ...')
        sys.stdout.flush()
        assert numx.all(numx.abs(0.999329299739 - TangentsHyperbolicus().f(4.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - TangentsHyperbolicus().f(0.0)) < epsilon)
        assert numx.all(numx.abs(-0.999329299739 - TangentsHyperbolicus().f(-4.0)) < epsilon)

        assert numx.all(numx.abs(-0.972955074528 - TangentsHyperbolicus().g(-0.75)) < epsilon)
        assert numx.all(numx.abs(0.0 - TangentsHyperbolicus().g(0.0)) < epsilon)
        assert numx.all(numx.abs(0.972955074528 - TangentsHyperbolicus().g(0.75)) < epsilon)

        assert numx.all(numx.abs(0.00134095068303 - TangentsHyperbolicus().df(4.0)) < epsilon)
        assert numx.all(numx.abs(1.0 - TangentsHyperbolicus().df(0.0)) < epsilon)
        assert numx.all(numx.abs(0.00134095068303 - TangentsHyperbolicus().df(-4.0)) < epsilon)

        assert numx.all(numx.abs(-0.00268010261411 - TangentsHyperbolicus().ddf(4.0)) < epsilon)
        assert numx.all(numx.abs(-.0 - TangentsHyperbolicus().ddf(0.0)) < epsilon)
        assert numx.all(numx.abs(0.00268010261411 - TangentsHyperbolicus().ddf(-4.0)) < epsilon)

        assert numx.all(numx.abs(2.28571428571 - TangentsHyperbolicus().dg(-0.75)) < epsilon)
        assert numx.all(numx.abs(1.0 - TangentsHyperbolicus().dg(0.0)) < epsilon)
        assert numx.all(numx.abs(2.28571428571 - TangentsHyperbolicus().dg(0.75)) < epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test_SoftSign(self):
        print('Activationfunction -> Performing SoftSign() test ...')
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
        print('successfully passed!')
        sys.stdout.flush()

    def test_Step(self):
        print('Activationfunction -> Performing Step() test ...')
        sys.stdout.flush()
        assert numx.all(numx.abs(0.0 - Step().f(-1.0)) < epsilon)
        assert numx.all(numx.abs(1.0 - Step().f(+1.0)) < epsilon)

        assert numx.all(numx.abs(0.0 - Step().df(0.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - Step().df(0.0)) < epsilon)

        assert numx.all(numx.abs(0.0 - Step().ddf(0.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - Step().ddf(0.0)) < epsilon)

        print('successfully passed!')
        sys.stdout.flush()

    def test_Rectifier(self):
        print('Activationfunction -> Performing Rectifier() test ...')
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
        print('successfully passed!')
        sys.stdout.flush()

    def test_SoftPlus(self):
        print('Activationfunction -> Performing SoftPlus() test ...')
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

        print('successfully passed!')
        sys.stdout.flush()

    def test_SoftMax(self):
        print('Activationfunction -> Performing SoftMax() test ...')
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

        print('successfully passed!')
        sys.stdout.flush()

    def test_Sinus(self):
        print('Activationfunction -> Performing Sinus() test ...')
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

        print('successfully passed!')
        sys.stdout.flush()

    def test_RectifierRestricted(self):
        print('Activationfunction -> Performing test_RestrictedRectifier() test ...')
        sys.stdout.flush()
        assert numx.all(numx.abs(0.0 - RectifierRestricted().f(-0.0001)) < epsilon)
        assert numx.all(numx.abs(0.0 - RectifierRestricted().f(-4.0)) < epsilon)
        assert numx.all(numx.abs(0.0001 - RectifierRestricted().f(+0.0001)) < epsilon)
        assert numx.all(numx.abs(1.0 - RectifierRestricted().f(4.0)) < epsilon)

        assert numx.all(numx.abs(0.0 - RectifierRestricted().df(-0.0001)) < epsilon)
        assert numx.all(numx.abs(0.0 - RectifierRestricted().df(-4.0)) < epsilon)
        assert numx.all(numx.abs(1.0 - RectifierRestricted().df(+0.0001)) < epsilon)
        assert numx.all(numx.abs(0.0 - RectifierRestricted().df(4.0)) < epsilon)

        assert numx.all(numx.abs(0.0 - RectifierRestricted().ddf(-0.0001)) < epsilon)
        assert numx.all(numx.abs(0.0 - RectifierRestricted().ddf(-4.0)) < epsilon)
        assert numx.all(numx.abs(0.0 - RectifierRestricted().ddf(+0.0001)) < epsilon)
        assert numx.all(numx.abs(0.0 - RectifierRestricted().ddf(4.0)) < epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test_RadialBasis(self):
        print('Activationfunction -> Performing RadialBasis() test ...')
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
        print('successfully passed!')
        sys.stdout.flush()


if __name__ is "__main__":
    unittest.main()
