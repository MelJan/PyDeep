''' Different kind of non linear activation functions and their derivatives.
    
    :Implemented:
        - Identity
        - Sigmoid
      
    :Info:
        http://en.wikipedia.org/wiki/Activation_function       
        
    :Version:
        1.0

    :Date:
        10.08.2016

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