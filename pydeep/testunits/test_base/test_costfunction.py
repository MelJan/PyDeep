""" Test module for cost functions.

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
from pydeep.base.costfunction import *
import numpy as numx
import sys

print("\n... pydeep.base.costfunction.py")

epsilon = 0.0001


class TestCostFunction(unittest.TestCase):

    def test_MeanSquaredError(self):
        sys.stdout.write('CostFunction -> Performing MeanSquaredError test ...')
        sys.stdout.flush()
        x = numx.array([[0.0, -92.0, 42.0, 87.0]])
        t = numx.array([[0.00000001, 2, 4.0, 86.9999]])
        assert numx.all(numx.abs(5140.00000001 - SquaredError.f(x, t)) < epsilon)
        res = [[-1.00000000e-08, -9.40000000e+01, 3.80000000e+01, 1.00000000e-04]]
        assert numx.all(numx.abs(res - SquaredError.df(x, t)) < epsilon)

        x = numx.array([[0, 1, 2, 3]])
        t = numx.array([[0, 1, 2, 3]])
        assert numx.all(numx.abs(0.0 - SquaredError.f(x, t)) < epsilon)
        assert numx.all(numx.abs(0.0 - SquaredError.df(x, t)) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_MeanAbsoluteError(self):
        sys.stdout.write('CostFunction -> Performing MeanAbsoluteError test ...')
        sys.stdout.flush()
        x = numx.array([[0.0, -92.0, 42.0, 87.0]])
        t = numx.array([[0.00000001, 2, 4.0, 86.9999]])
        assert numx.all(numx.abs(132.00010001 - AbsoluteError.f(x, t)) < epsilon)
        res = [[-1., -1., 1., 1.]]
        assert numx.all(numx.abs(res - AbsoluteError.df(x, t)) < epsilon)

        x = numx.array([[0, 1, 2, 3]])
        t = numx.array([[0, 1, 2, 3]])
        assert numx.all(numx.abs(0.0 - AbsoluteError.f(x, t)) < epsilon)
        assert numx.all(numx.abs(0.0 - AbsoluteError.df(x, t)) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_CrossEntropyError(self):
        sys.stdout.write('CostFunction -> Performing CrossEntropyError test ...')
        sys.stdout.flush()
        x = numx.array([[0.5, 0.01, 0.99]])
        t = numx.array([[0.0, 0.0, 1.0]])
        assert numx.all(numx.abs(0.713247852267 - CrossEntropyError.f(x, t)) < epsilon)
        res = numx.array([[2., 1.01010101, -1.01010101]])
        assert numx.all(numx.abs(res - CrossEntropyError.df(x, t)) < epsilon)

        x = numx.array([[0.5, 0.01, 0.99]])
        t = numx.array([[1.0, 1.0, 0.0]])
        assert numx.all(numx.abs(9.90348755254 - CrossEntropyError.f(x, t)) < epsilon)
        res = numx.array([[-2., -100., 100.]])
        assert numx.all(numx.abs(res - CrossEntropyError.df(x, t)) < epsilon)
        x = numx.array([[0.0000000001, 0.9999999999]])
        t = numx.array([[0.0, 1.0]])
        assert numx.all(numx.abs(2.00000016558e-10 - CrossEntropyError.f(x, t)) < epsilon)

        x = numx.array([[0.0000000001, 0.9999999999]])
        t = numx.array([[0.0, 1.0]])
        res = numx.array([[1., -1.]])
        assert numx.all(numx.abs(res - CrossEntropyError.df(x, t)) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()

    def test_NegLogLikelihood(self):
        sys.stdout.write('CostFunction -> Performing NegLogLikelihood test ...')
        sys.stdout.flush()
        x = numx.array([[0.5, 0.01, 0.99]])
        t = numx.array([[0.0, 0.0, 1.0]])
        assert numx.all(numx.abs(0.01005034 - NegLogLikelihood.f(x, t)) < epsilon)
        res = numx.array([[0., 0., -1.01010101]])
        assert numx.all(numx.abs(res - NegLogLikelihood.df(x, t)) < epsilon)

        x = numx.array([[0.5, 0.01, 0.99]])
        t = numx.array([[1.0, 0.0, 0.0]])
        assert numx.all(numx.abs(0.69314718 - NegLogLikelihood.f(x, t)) < epsilon)
        res = numx.array([[-2., 0., 0.]])
        assert numx.all(numx.abs(res - NegLogLikelihood.df(x, t)) < epsilon)
        print(' successfully passed!')
        sys.stdout.flush()


if __name__ is "__main__":
    unittest.main()
