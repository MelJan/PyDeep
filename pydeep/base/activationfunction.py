""" Different kind of non linear activation functions and their derivatives.

    :Implemented:

    # Unbounded
        # Linear
            - Identity
        # Piecewise-linear
            - Rectifier
            - RestrictedRectifier (hard bounded)
            - LeakyRectifier
        # Soft-linear
            - ExponentialLinear
            - SigmoidWeightedLinear
            - SoftPlus
    # Bounded
        # Step
            - Step
        # Soft-Step
            - Sigmoid
            - SoftSign
            - HyperbolicTangent
            - SoftMax
            - K-Winner takes all
        # Symmetric, periodic
            - Radial Basis function
            - Sinus

    :Info:
        http://en.wikipedia.org/wiki/Activation_function

    :Version:
        1.1.1

    :Date:
        16.01.2018

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2018 Jan Melchior

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
from pydeep.base.numpyextension import log_sum_exp


# Unbounded

# Linear


class Identity(object):
    """ Identity function.

        :Info: http://www.wolframalpha.com/input/?i=line
    """

    @classmethod
    def f(cls, x):
        """ Calculates the identity function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the identity function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return x

    @classmethod
    def g(cls, y):
        """ Calculates the inverse identity function value for a given input y.

        :param y: Input data.
        :type y: scalar or numpy array.

        :return: Value of the inverse identity function for y.
        :rtype: scalar or numpy array with the same shape as y.
        """
        return y

    @classmethod
    def df(cls, x):
        """ Calculates the derivative of the identity function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the derivative of the identity function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        if numx.isscalar(x):
            return 1.0
        else:
            return numx.ones(x.shape)

    @classmethod
    def ddf(cls, x):
        """ Calculates the second derivative of the identity function value for a given input x.

        :param x: Inout data.
        :type x: scalar or numpy array.

        :return: Value of the second derivative of the identity function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        if numx.isscalar(x):
            return 0.0
        else:
            return numx.zeros(x.shape)

    @classmethod
    def dg(cls, y):
        """ Calculates the derivative of the inverse identity function value for a given input y.

        :param y: Input data.
        :type y: scalar or numpy array.

        :return: Value of the derivative of the inverse identity function for y.
        :rtype: scalar or numpy array with the same shape as y.
        """
        if numx.isscalar(y):
            return 1.0
        else:
            return numx.ones(y.shape)


# Piecewise-linear


class Rectifier(object):
    """ Rectifier activation function function.

        :Info: http://www.wolframalpha.com/input/?i=max%280%2Cx%29&dataset=&asynchronous=false&equal=Submit
    """

    @classmethod
    def f(cls, x):
        """ Calculates the Rectifier function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the Rectifier function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return numx.maximum(0.0, x)

    @classmethod
    def df(cls, x):
        """ Calculates the derivative of the Rectifier function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the derivative of the Rectifier function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return numx.float64(x > 0.0)

    @classmethod
    def ddf(cls, x):
        """ Calculates the second derivative of the Rectifier function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the 2nd derivative of the Rectifier function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return 0.0


class RestrictedRectifier(Rectifier):
    """ Restricted Rectifier activation function function.

        :Info: http://www.wolframalpha.com/input/?i=max%280%2Cx%29&dataset=&asynchronous=false&equal=Submit
    """

    def __init__(self, restriction=1.0):
        """ Constructor.

        :param restriction: Restriction value / upper limit value.
        :type restriction: float.
        """
        self.restriction = restriction

    def f(self, x):
        """ Calculates the Restricted Rectifier function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the Restricted Rectifier function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return numx.minimum(numx.maximum(0.0, x), self.restriction)

    def df(self, x):
        """ Calculates the derivative of the Restricted Rectifier function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Value of the derivative of the Restricted Rectifier function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return numx.float64(x > 0.0) * numx.float64(x < self.restriction)


class LeakyRectifier(Rectifier):
    """ Leaky Rectifier activation function function.

        :Info: https://en.wikipedia.org/wiki/Activation_function
    """

    def __init__(self, negativeSlope=0.01, positiveSlope=1.0):
        """ Constructor.

        :param negativeSlope: Slope when x < 0
        :type negativeSlope: scalar

        :param positiveSlope: Slope when x >= 0
        :type positiveSlope: scalar

        """
        self.negativeSlope = negativeSlope
        self.positiveSlope = positiveSlope

    def f(self, x):
        """ Calculates the Leaky Rectifier function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the Leaky Rectifier function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return x*((x < 0)*(self.negativeSlope-self.positiveSlope) + self.positiveSlope)

    def df(self, x):
        """ Calculates the derivative of the Leaky Rectifier function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Value of the derivative of the Leaky Rectifier function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return (x < 0)*(self.negativeSlope-self.positiveSlope) + self.positiveSlope


# Soft-linear


class ExponentialLinear(object):
    """ Exponential Linear activation function function.

        :Info: https://en.wikipedia.org/wiki/Activation_function
    """

    def __init__(self, alpha=1.0):
        """ Constructor.

        :param alpha: scaling factor
        :type alpha: scalar

        """
        self.alpha = alpha

    def f(self, x):
        """ Calculates the Exponential Linear function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the Exponential Linear function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        decision = x > 0.0
        return x*decision+(1.0-decision)*self.alpha*(numx.exp(x)-1.0)

    def df(self, x):
        """ Calculates the derivative of the Exponential Linear function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the derivative of the Exponential Linear function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        decision = x > 0.0
        return decision+(1.0-decision)*self.alpha*numx.exp(x)


class SigmoidWeightedLinear(object):
    """ Sigmoid weighted linear units (also named Swish)

        :Info: https://arxiv.org/pdf/1702.03118v1.pdf and for Swish: https://arxiv.org/pdf/1710.05941.pdf
    """

    def __init__(self, beta=1.0):
        """ Constructor.

        :param beta: scaling factor
        :type beta: scalar

        """
        self.beta = beta

    def f(self, x):
        """ Calculates the Sigmoid weighted linear function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the Sigmoid weighted linear function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return x*Sigmoid.f(self.beta*x)

    def df(self, x):
        """ Calculates the derivative of the Sigmoid weighted linear function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the derivative of the Sigmoid weighted linear function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        sig = Sigmoid.f(self.beta*x)
        return sig*(1.0+x*(1.0-sig))


class SoftPlus(object):
    """ Soft Plus function.

        :Info: http://www.wolframalpha.com/input/?i=log%28exp%28x%29%2B1%29
    """

    @classmethod
    def f(cls, x):
        """ Calculates the SoftPlus function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the SoftPlus function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return numx.log(1.0 + numx.exp(x))

    @classmethod
    def g(cls, y):
        """ Calculates the inverse SoftPlus function value for a given input y.

        :param y: Input data.
        :type y: scalar or numpy array.

        :return: Value of the inverse SoftPlus function for y.
        :rtype: scalar or numpy array with the same shape as y.
        """
        return numx.log(numx.exp(y) - 1.0)

    @classmethod
    def df(cls, x):
        """ Calculates the derivative of the SoftPlus function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the derivative of the SoftPlus function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return 1.0 / (1.0 + numx.exp(-x))

    @classmethod
    def ddf(cls, x):
        """ Calculates the second derivative of the SoftPlus function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Value of the 2nd derivative of the SoftPlus function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        exp_x = numx.exp(x)
        return exp_x / ((1.0 + exp_x) ** 2)

    @classmethod
    def dg(cls, y):
        """ Calculates the derivative of the inverse SoftPlus function value for a given input y.

        :param y: Input data.
        :type y: scalar or numpy array.

        :return: Value of the derivative of the inverse SoftPlus function for x.
        :rtype: scalar or numpy array with the same shape as y.
        """
        return 1.0 / (1.0 - numx.exp(-y))


# Bounded

# Step


class Step(object):
    """ Step activation function function.
    """

    @classmethod
    def f(cls, x):
        """ Calculates the step function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the step function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return numx.float64(x > 0)

    @classmethod
    def df(cls, x):
        """ Calculates the derivative of the step function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the derivative of the step function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return 0.0

    @classmethod
    def ddf(cls, x):
        """ Calculates the second derivative of the step function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the derivative of the Step function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return 0.0

# Soft-step


class Sigmoid(object):
    """ Sigmoid function.
          
        :Info: http://www.wolframalpha.com/input/?i=sigmoid
    """

    @classmethod
    def f(cls, x):
        """ Calculates the Sigmoid function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the Sigmoid function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return 0.5 + 0.5 * numx.tanh(0.5 * x)

    @classmethod
    def g(cls, y):
        """ Calculates the inverse Sigmoid function value for a given input y.

        :param y: Input data.
        :type y: scalar or numpy array.

        :return: Value of the inverse Sigmoid function for y.
        :rtype: scalar or numpy array with the same shape as y.
        """
        return 2.0 * numx.arctanh(2.0 * y - 1.0)

    @classmethod
    def df(cls, x):
        """ Calculates the derivative of the Sigmoid function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the derivative of the Sigmoid function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        # expx = numx.exp(-x)
        # return numx.exp(-x)/((1+numx.exp(-x))**2)
        sig = cls.f(x)
        return sig * (1.0 - sig)

    @classmethod
    def ddf(cls, x):
        """ Calculates the second derivative of the Sigmoid function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the second derivative of the Sigmoid function for x.
        :rtype:  scalar or numpy array with the same shape as x.
        """
        sig = cls.f(x)
        return sig - 3 * (sig ** 2) + 2 * (sig ** 3)

    @classmethod
    def dg(cls, y):
        """ Calculates the derivative of the inverse Sigmoid function value for a given input y.

        :param y: Input data.
        :type y: scalar or numpy array.

        :return: Value of the derivative of the inverse Sigmoid function for y.
        :rtype: scalar or numpy array with the same shape as y.
        """
        return 1.0 / (y - y ** 2)


class SoftSign(object):
    """ SoftSign function.
          
        :Info: http://www.wolframalpha.com/input/?i=x%2F%281%2Babs%28x%29%29
    """

    @classmethod
    def f(cls, x):
        """ Calculates the SoftSign function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the SoftSign function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return x / (1.0 + numx.abs(x))

    @classmethod
    def df(cls, x):
        """ Calculates the derivative of the SoftSign function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the SoftSign function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return 1.0 / ((1.0 + numx.abs(x)) ** 2)

    @classmethod
    def ddf(cls, x):
        """ Calculates the second derivative of the SoftSign function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the 2nd derivative of the SoftSign function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        absx = numx.abs(x)
        return -(2.0 * x) / (absx * (1 + absx) ** 3)


class HyperbolicTangent(object):
    """ HyperbolicTangent function.

        :Info: http://www.wolframalpha.com/input/?i=tanh
    """

    @classmethod
    def f(cls, x):
        """ Calculates the Hyperbolic Tangent function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the Hyperbolic Tangent function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return numx.tanh(x)

    @classmethod
    def g(cls, y):
        """ Calculates the inverse Hyperbolic Tangent function value for a given input y.

        :param y: Input data.
        :type y: scalar or numpy array.

        :return: alue of the inverse Hyperbolic Tangent function for y.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return 0.5 * (numx.log(1.0 + y) - numx.log(1.0 - y))

    @classmethod
    def df(cls, x):
        """ Calculates the derivative of the Hyperbolic Tangent function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the derivative of the Hyperbolic Tangent function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        tanh = cls.f(x)
        return 1.0 - tanh ** 2

    @classmethod
    def ddf(cls, x):
        """ Calculates the second derivative of the Hyperbolic Tangent function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the second derivative of the Hyperbolic Tangent function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        tanh = cls.f(x)
        return -2 * tanh * (1 - (tanh ** 2))

    @classmethod
    def dg(cls, y):
        """ Calculates the derivative of the inverse Hyperbolic Tangent function value for a given input y.

        :param y: Input data.
        :type y: scalar or numpy array.

        :return: Value the derivative of the inverse Hyperbolic Tangent function for x.
        :rtype: scalar or numpy array with the same shape as y.
        """
        return numx.exp(-numx.log((1.0 - y ** 2)))


class SoftMax(object):
    """ Soft Max function.

        :Info: https://en.wikipedia.org/wiki/Activation_function
    """

    @classmethod
    def f(cls, x):
        """ Calculates the function value of the SoftMax function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Value of the SoftMax function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return numx.exp(x - log_sum_exp(x, axis=1).reshape(x.shape[0], 1))

    @classmethod
    def df(cls, x):
        """ Calculates the derivative of the SoftMax function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Value of the derivative of the SoftMax function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        result = x[0] * numx.eye(x.shape[1], x.shape[1]) - numx.dot(x[0].reshape(x.shape[1], 1), x[0]
                                                                    .reshape(1, x.shape[1])
                                                                    ).reshape(1, x.shape[1], x.shape[1])
        for i in range(1, x.shape[0], 1):
            result = numx.vstack(
                (result, x[i] * numx.eye(x.shape[1], x.shape[1]) - numx.dot(x[i].reshape(x.shape[1], 1),
                                                                            x[i].reshape(1, x.shape[1])
                                                                            ).reshape(1, x.shape[1], x.shape[1])))
        ''' WITHOUT LOOP BUT MUCH MOR MEMORY CONSUMPTION
        result = x.reshape((1, 100*100))
        result = numx.tile(result, (100, 1))
        result_t = result.T
        result_t = numx.array_split(result_t, 100)
        result_t = numx.hstack(result_t)
        result *= (numx.tile( numx.eye(100), (1, 100)) - result_t)
        result *= numx.tile(y.reshape((1, 100*100)), (100, 1))
        result = numx.sum(result, axis=0)
        '''
        return result

# Symmetric, periodic

class RadialBasis(object):
    """ Radial Basis function.
    
        :Info: http://www.wolframalpha.com/input/?i=Gaussian
    """

    def __init__(self, mean=0.0, variance=1.0):
        """ Constructor.

        :param mean: Mean of the function.
        :type mean: scalar or numpy array

        :param variance: Variance of the function.
        :type variance: scalar or numpy array
        """
        self.mean = mean
        self.variance = variance

    def f(self, x):
        """ Calculates the Radial Basis function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Value of the Radial Basis function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        activation = x - self.mean
        return numx.exp(-(activation ** 2) / self.variance)

    def df(self, x):
        """ Calculates the derivative of the Radial Basis function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Value of the derivative of the Radial Basis function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return (self.f(x) * 2 * (self.mean - x)) / self.variance

    def ddf(self, x):
        """ Calculates the second derivative of the Radial Basis function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Value of the second derivative of the Radial Basis function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        activation = ((x - self.mean) ** 2) / self.variance
        return 2.0 / self.variance * numx.exp(-activation) * (2 * activation - 1.0)


class Sinus(object):
    """ Sinus function.

        :Info: http://www.wolframalpha.com/input/?i=sin(x)
    """

    @classmethod
    def f(cls, x):
        """ Calculates the function value of the Sinus function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Value of the Sinus function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return numx.sin(x)

    @classmethod
    def df(cls, x):
        """ Calculates the derivative of the Sinus function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Value of the derivative of the Sinus function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return numx.cos(x)

    @classmethod
    def ddf(cls, x):
        """ Calculates the second derivative of the Sinus function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Value of the second derivative of the Sinus function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return -numx.sin(x)


class KWinnerTakeAll(object):
    """ K Winner take all activation function.

        :WARNING: The derivative gets already calcluated in the forward pass. 
                  Thus, for the same data-point the order should always be forward_pass, backward_pass!

    """

    def __init__(self, k, axis=1, activation_function=Identity()):
        """ Constructor.

        :param k: Number of active units.
        :type k: int

        :param axis: Axis to compute the maximum.
        :type axis: int

        :param k: activation_function
        :type k: Instance of an activation function

        """
        self.k = k
        self.axis = axis
        self.activation_function = activation_function
        self._temp_derivative = None

    def f(self, x):
        """ Calculates the K-max function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Value of the Kmax function for x.
        :rtype: scalar or numpy array with the same shape as x.

        """
        act = self.activation_function.f(numx.atleast_2d(x))
        winner = None
        if self.axis == 0:
            winner = numx.float64(act >= numx.atleast_2d(numx.sort(act, axis=self.axis)[-self.k, :]))
        else:
            winner = numx.float64(act.T >= numx.atleast_2d(numx.sort(act, axis=self.axis)[:, -self.k])).T
        self._temp_derivative = winner * self.activation_function.df(x)
        return act * winner

    def df(self, x):
        """ Calculates the derivative of the KWTA function.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Derivative of the KWTA function
        :rtype: scalar or numpy array with the same shape as x.

        """
        return self._temp_derivative