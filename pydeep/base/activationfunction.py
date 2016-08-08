''' Different kind of non linear activation functions and their derivatives.
    
    :Implemented:
        - Identity
        - Sigmoid
        - Tangents Hyperbolicus
        - SoftSign
        - Step function.
        - Rectifier
        - Softplus  
        - Radial Basis function
        - SoftMax
      
    :Info:
        http://en.wikipedia.org/wiki/Activation_function       
        
    :Version:
        1.0

    :Date:
        08.08.2016

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
import numpy as numx
import pydeep.base.numpyextension as numxExt

class ActivationFunction(object):
    ''' Interface for activation functions.
          
    '''

    def __init__(self):
        ''' Dummy Constructor

        '''
        raise NotImplementedError("This is an abstract class no objects allowed")

    @classmethod
    def f(cls, x):
        ''' Calculates the function value for a given input x.

            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        raise NotImplementedError("Cannot use abstract class!")

    @classmethod
    def g(cls, y):
        ''' Calculates the inverse function value for a given input y.
        
            :Parameters:
                y: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the inverse function for y.
           -type: scalar or numpy array with the same shape as y.
              
        '''
        raise NotImplementedError("Cannot use abstract class!")

    @classmethod 
    def df(cls, x):
        ''' Calculates the derivative of the  function value 
            for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the derivative of the function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        raise NotImplementedError("Cannot use abstract class!")        

    @classmethod 
    def ddf(cls, x):
        ''' Calculates the second derivative of the  function value 
            for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the second derivative of the function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        raise NotImplementedError("Cannot use abstract class!")

    @classmethod
    def dg(cls, y):
        ''' Calculates the derivative of the inverse function value 
            for a given input y.
        
            :Parameters:
                y: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the derivative of the inverse function for y.
           -type: scalar or numpy array with the same shape as y.
              
        '''
        raise NotImplementedError("Cannot use abstract class!") 

class Identity(ActivationFunction):
    ''' Identity function.

        :Info: http://www.wolframalpha.com/input/?i=line
          
    '''

    @classmethod
    def f(cls, x):
        ''' Calculates the identity function value for a given input x.
          
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the identity function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        return x

    @classmethod
    def g(cls, y):
        ''' Calculates the inverse identity function value for a 
            given input y.
        
            :Parameters:
                y: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the inverse identity function for y.
           -type: scalar or numpy array with the same shape as y.
              
        '''
        return y

    @classmethod 
    def df(cls, x):
        ''' Calculates the derivative of the identity function value 
            for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the derivative of the identity function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        if numx.isscalar(x):
            return 1.0
        else:
            return numx.ones(x.shape)
    
    @classmethod 
    def ddf(cls, x):
        ''' Calculates the second derivative of the identity function value 
            for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the second derivative of the identity function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        if numx.isscalar(x):
            return 0.0
        else:
            return numx.zeros(x.shape)

    @classmethod
    def dg(cls, y):
        ''' Calculates the derivative of the inverse identity function value
            for a given input y.
        
            :Parameters:
                y: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the derivative of the inverse identity function for y.
           -type: scalar or numpy array with the same shape as y.
              
        '''
        if numx.isscalar(y):
            return 1.0
        else:
            return numx.ones(y.shape)

class Sigmoid(ActivationFunction):
    ''' Sigmoid function.
          
        :Info: http://www.wolframalpha.com/input/?i=sigmoid
          
    '''
    
    @classmethod
    def f(cls, x):
        ''' Calculates the Sigmoid function value for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the Sigmoid function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        return 0.5 + 0.5 * numx.tanh(0.5 * x)

    @classmethod
    def g(cls, y):
        ''' Calculates the inverse Sigmoid function value for a given input y.
        
            :Parameters:
                y: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the inverse Sigmoid function for y.
           -type: scalar or numpy array with the same shape as y.
              
        '''
        return 2.0 * numx.arctanh(2.0 * y - 1.0)

    @classmethod
    def df(cls, x):
        ''' Calculates the derivative of the Sigmoid function value 
            for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the derivative of the Sigmoid function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        sig = Sigmoid.f(x)
        return sig*(1.0-sig)
    
    @classmethod 
    def ddf(cls, x):
        ''' Calculates the second derivative of the identity function value 
            for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the second derivative of the identity function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        sig = Sigmoid.f(x)
        return sig-3*(sig**2)+2*(sig**3)

    @classmethod
    def dg(cls, y):
        ''' Calculates the derivative of the inverse Sigmoid function 
            value for a given input y.
        
            :Parameters:
                y: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the derivative of the inverse Sigmoid function for y.
           -type: scalar or numpy array with the same shape as y.
              
        '''
        return 1.0/(y-y**2)

class TangentsHyperbolicus(ActivationFunction):
    ''' Tangents hyperbolicus function.
    
        :Info: http://www.wolframalpha.com/input/?i=tanh
    
    '''

    @classmethod
    def f(cls, x):
        ''' Calculates the tangents hyperbolicus function value 
            for a given input x.
            
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the tangents hyperbolicus function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        return numx.tanh(x)

    @classmethod
    def g(cls, y):
        ''' Calculates the inverse tangents hyperbolicus function 
            value for a given input y.
        
            :Parameters:
                y: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the inverse tangents hyperbolicus function for y.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        return 0.5*(numx.log(1.0+y)-numx.log(1.0-y))

    @classmethod
    def df(cls, x):
        ''' Calculates the derivative of the tangents hyperbolicus 
            function value for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the derivative of the tangents hyperbolicus 
            function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        tanh = TangentsHyperbolicus.f(x)
        return 1.0-tanh**2
    
    @classmethod
    def ddf(cls, x):
        ''' Calculates the second derivative of the tangents hyperbolicus 
            function value for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the second derivative of the tangents hyperbolicus 
            function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        tanh = TangentsHyperbolicus.f(x)
        return -2*tanh*(1-(tanh**2))

    @classmethod
    def dg(cls, y):
        ''' Calculates the derivative of the inverse tangents 
            hyperbolicus function value for a given input y.
        
            :Parameters:
                y: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value the derivative of the inverse tangents hyperbolicus 
            function for x.
           -type: scalar or numpy array with the same shape as y.
              
        '''
        return numx.exp(-numx.log((1.0-y**2)))

class SoftSign(ActivationFunction):
    ''' SoftSign function.
          
        :Info: http://www.wolframalpha.com/input/?i=x%2F%281%2Babs%28x%29%29
          
    '''

    @classmethod
    def f(cls, x):
        ''' Calculates the SoftSign function value for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the SoftSign function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        return x/(1.0+numx.abs(x))

    @classmethod
    def df(cls, x):
        ''' Calculates the derivative of the SoftSign function value 
            for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the SoftSign function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        return 1.0/((1.0+numx.abs(x))**2)

    @classmethod
    def ddf(cls, x):
        ''' Calculates the second derivative of the SoftSign function value 
            for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the 2nd derivative of the SoftSign function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        absx = numx.abs(x)
        return  -(2.0*x)/(absx*(1+absx)**3)

class Step(ActivationFunction):
    ''' Step activation function function.

    '''

    @classmethod
    def f(cls, x):
        ''' Calculates the step function value for a given input x.

            :Parameters:
                x: Input data.
                  -type: scalar or numpy array

        :Returns:
            Value of the step function for x.
           -type: scalar or numpy array with the same shape as x.

        '''
        return numx.float64(x>0)

    @classmethod
    def df(cls, x):
        ''' Calculates the derivative of the step function value
            for a given input x.

            :Parameters:
                x: Input data.
                  -type: scalar or numpy array

        :Returns:
            Value of the derivative of the step function for x.
           -type: scalar or numpy array with the same shape as x.

        '''
        return 0.0

    @classmethod
    def ddf(cls, x):
        ''' Calculates the second derivative of the Step function value
            for a given input x.

            :Parameters:
                x: Input data.
                  -type: scalar or numpy array

        :Returns:
            Value of the derivative of the Step function for x.
           -type: scalar or numpy array with the same shape as x.

        '''
        return 0.0

class Rectifier(ActivationFunction):
    ''' Rectifier activation function function.
          
        :Info: http://www.wolframalpha.com/input/?i=max%280%2Cx%29&dataset=&asynchronous=false&equal=Submit
          
    '''

    @classmethod
    def f(cls, x):
        ''' Calculates the Rectifier function value for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the Rectifier function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        return numx.maximum(0.0 , x)

    @classmethod
    def df(cls, x):
        ''' Calculates the derivative of the Rectifier function value
            for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the derivative of the Rectifier function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        return numx.float64(x > 0.0)

    @classmethod
    def ddf(cls, x):
        ''' Calculates the second derivative of the Rectifier function value
            for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the 2nd derivative of the Rectifier function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        return 0.0

class SoftPlus(ActivationFunction):
    ''' Soft Plus function.

        :Info: http://www.wolframalpha.com/input/?i=log%28exp%28x%29%2B1%29

    '''

    @classmethod
    def f(cls, x):
        ''' Calculates the SoftPlus function value for a given input x.

            :Parameters:
                x: Input data.
                  -type: scalar or numpy array

        :Returns:
            Value of the SoftPlus function for x.
           -type: scalar or numpy array with the same shape as x.

        '''
        return numx.log(1.0+numx.exp(x))

    @classmethod
    def g(cls, y):
        ''' Calculates the inverse SoftPlus function value for a
            given input y.

            :Parameters:
                y: Input data.
                  -type: scalar or numpy array

        :Returns:
            Value of the inverse SoftPlus function for y.
           -type: scalar or numpy array with the same shape as y.

        '''
        return numx.log(numx.exp(y)-1.0)

    @classmethod
    def df(cls, x):
        ''' Calculates the derivative of the SoftPlus function value
            for a given input x.

            :Parameters:
                x: Input data.
                  -type: scalar or numpy array

        :Returns:
            Value of the derivative of the SoftPlus function for x.
           -type: scalar or numpy array with the same shape as x.

        '''
        return 1.0/(1.0+numx.exp(-x))

    @classmethod
    def ddf(cls, x):
        ''' Calculates the second derivative of the SoftPlus function value
            for a given input x.

            :Parameters:
                x: Input data.
                  -type: scalar or numpy array

        :Returns:
            Value of the 2nd derivative of the SoftPlus function for x.
           -type: scalar or numpy array with the same shape as x.

        '''
        exp_x = numx.exp(x)
        return  exp_x/((1.0+exp_x)**2)

    @classmethod
    def dg(cls, y):
        ''' Calculates the derivative of the inverse SoftPlus function
            value for a given input y.

            :Parameters:
                y: Input data.
                  -type: scalar or numpy array

        :Returns:
            Value of the derivative of the inverse SoftPlus function for x.
           -type: scalar or numpy array with the same shape as y.

        '''
        return 1.0/(1.0-numx.exp(-y))
   
class RadialBasis(ActivationFunction):
    ''' Radial Basis function.
    
        :Info: http://www.wolframalpha.com/input/?i=Gaussian
         
    '''
    
    @classmethod  
    def f(cls, x, mean= 0.0, variance = 1.0):
        ''' Calculates the Radial Basis function value 
            for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
                  
                mean: Mean of the function.
                  -type: scalar or numpy array
                  
                variance: Variance of the function.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the Radial Basis function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        activation = x-mean
        return numx.exp(-(activation**2)/(variance))

    @classmethod
    def df(cls, x, mean= 0.0, variance = 1.0):
        ''' Calculates the derivative of the Radial Basis function value for a 
            given input x.
        
            :Parameters:
                x:        Input data.
                         -type: scalar or numpy array

                mean:     Mean of the function.
                         -type: scalar or numpy array
                  
                variance: Variance of the function.
                         -type: scalar or numpy array

        :Returns:
            Value of the derivative of the Radial Basis function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        return (RadialBasis.f(x, mean, variance)*2*(mean-x))/variance

    @classmethod
    def ddf(cls, x, mean= 0.0, variance = 1.0):
        ''' Calculates the second derivative of the Radial Basis function value for a 
            given input x.
        
            :Parameters:
                x:        Input data.
                         -type: scalar or numpy array

                mean:     Mean of the function.
                         -type: scalar or numpy array
                  
                variance: Variance of the function.
                         -type: scalar or numpy array

        :Returns:
            Value of the second derivative of the Radial Basis function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        activation = ((x-mean)**2)/(variance)
        return 2.0/variance*numx.exp(-activation)*(2*activation-1.0)

class SoftMax(ActivationFunction):
    ''' Soft Max function.
          
    '''

    @classmethod
    def f(cls, x):
        ''' Calculates the SoftMax function value 
            for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: numpy array
            
        :Returns:
            Value of the SoftMax function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        return numx.exp(x-numxExt.log_sum_exp(x, axis = 1).reshape(x.shape[0],1))

    @classmethod
    def df(cls, x):
        ''' Calculates the derivative of the SoftMax 
            function value for a given input x.
            Not that this is the Jacobian Matrix, one for each datapoint.
        
            :Parameters:
                x: Input data.
                  -type: numpy array
            
        :Returns:
            Value of the SoftMax function for x.
           -type: scalar or numpy array with the same shape as [x.shape[0],x.shape[1],x.shape[1]].
              
        '''

        result = x[0]*numx.eye(x.shape[1],x.shape[1])-numx.dot(x[0].reshape(x.shape[1],1),x[0].reshape(1,x.shape[1])).reshape(1,x.shape[1],x.shape[1])
        for i in range(1,x.shape[0],1):
            result = numx.vstack((result,x[i]*numx.eye(x.shape[1],x.shape[1])-numx.dot(x[i].reshape(x.shape[1],1),x[i].reshape(1,x.shape[1])).reshape(1,x.shape[1],x.shape[1]))) 
        return result
        '''
        result = x.reshape((1, 100*100))
        result = numx.tile(result, (100, 1))
        result_t = result.T
        result_t = numx.array_split(result_t, 100)
        result_t = numx.hstack(result_t)
        result *= (numx.tile( numx.eye(100), (1, 100)) - result_t)
        result *= numx.tile(y.reshape((1, 100*100)), (100, 1))
        result = numx.sum(result, axis=0)
        '''
