""" Different kind of cost functions and their derivatives.
    
    :Implemented:
        - Squared error
        - Absolute error
        - Cross entropy
        - Negative Log-likelihood
      
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
import numpy as numx

MIN_VALUE = 1e-10


class SquaredError(object):
    """ Mean Squared error.
    """

    @classmethod
    def f(cls, x, t):
        """ Calculates the Squared Error value for a given input x and target t.

        :param x: Input data.
        :type x: scalar or numpy array

        :param t: Target vales
        :type t: scalar or numpy array

        :return: Value of the cost function for x and t.
        :rtype: scalar or numpy array with the same shape as x and t.
        """
        return numx.sum((x - t) ** 2, axis=1) * 0.5

    @classmethod
    def df(cls, x, t):
        """ Calculates the derivative of the Squared Error value for a given input x and target t.

        :param x: Input data.
        :type x: scalar or numpy array

        :param t: Target vales.
        :type t: scalar or numpy array

        :return: Value of the derivative of the cost function for x and t.
        :rtype: scalar or numpy array with the same shape as x and t.
        """
        return x - t


class AbsoluteError(object):
    """ Absolute error.
    """

    @classmethod
    def f(cls, x, t):
        """ Calculates the absolute error value for a given input x and target t.

        :param x: Input data.
        :type x: scalar or numpy array

        :param t: Target vales
        :type t: scalar or numpy array

        :return: Value of the cost function for x and t.
        :rtype: scalar or numpy array with the same shape as x and t.
        """
        return numx.sum(numx.abs(x - t), axis=1)

    @classmethod
    def df(cls, x, t):
        """ Calculates the derivative of the absolute error value for a given input x and target t.

        :param x: Input data.
        :type x: scalar or numpy array

        :param t: Target vales.
        :type t: scalar or numpy array

        :return: Value of the derivative of the cost function for x and t.
        :rtype: scalar or numpy array with the same shape as x and t.
        """
        return numx.sign(x - t)


class CrossEntropyError(object):
    """ Cross entropy functions.
    """

    @classmethod
    def f(cls, x, t):
        """ Calculates the cross entropy value for a given input x and target t.

        :param x: Input data.
        :type x: scalar or numpy array

        :param t: Target vales
        :type t: scalar or numpy array

        :return: Value of the cost function for x and t.
        :rtype: scalar or numpy array with the same shape as x and t.
        """
        x = numx.clip(x, a_min=MIN_VALUE, a_max=1.0 - MIN_VALUE)
        return -numx.sum(numx.log(x) * t + numx.log(1.0 - x) * (1.0 - t), axis=1)

    @classmethod
    def df(cls, x, t):
        """ Calculates the derivative of the cross entropy value for a given input x and target t.

        :param x: Input data.
        :type x: scalar or numpy array

        :param t: Target vales.
        :type t: scalar or numpy array

        :return: Value of the derivative of the cost function for x and t.
        :rtype: scalar or numpy array with the same shape as x and t.
        """
        x = numx.clip(x, a_min=MIN_VALUE, a_max=1.0 - MIN_VALUE)
        return -t / x + (1.0 - t) / (1.0 - x)


class NegLogLikelihood(object):
    """ Negative log likelihood function.
    """

    @classmethod
    def f(cls, x, t):
        """ Calculates the negative log-likelihood value for a given input x and target t.

        :param x: Input data.
        :type x: scalar or numpy array

        :param t: Target vales
        :type t: scalar or numpy array

        :return: Value of the cost function for x and t.
        :rtype: scalar or numpy array with the same shape as x and t.
        """
        return -numx.sum(numx.log(x) * t, axis=1)

    @classmethod
    def df(cls, x, t):
        """ Calculates the derivative of the negative log-likelihood value for a given input x and target t.

        :param x: Input data.
        :type x: scalar or numpy array

        :param t: Target vales.
        :type t: scalar or numpy array

        :return: Value of the derivative of the cost function for x and t.
        :rtype: scalar or numpy array with the same shape as x and t.
        """
        x = numx.clip(x, a_min=MIN_VALUE, a_max=1.0 - MIN_VALUE)
        return -t / x
