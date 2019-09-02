""" This module contains models for the particular case of an 3 layer binary DBM.

    :Version:
        1.0.0

    :Date:
        02.009.2019

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2019 Jan Melchior

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
import pydeep.base.numpyextension as npExt
from pydeep.base.activationfunction import Sigmoid


class BinaryBinaryDBM(object):
    
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, offset_typ, data, dtype = numx.float64):
        ''' Initializes the network
        
        :Parameters:
            input_dim:    Number of input dimensions.
                         -type: int

            hidden1_dim:  Number of hidden dimensions for the first hidden layer.
                         -type: int
                 
            hidden2_dim:  Number of hidden dimensions for the first hidden layer.
                         -type: int
                         
            offset_typ:   Typs of offset values used for specific initialization
                          'DDD' -> Centering, 'AAA'-> Enhanced gradient,'MMM' -> Model mean centering
                         -type: string (3 chars)
                         
        ''' 
        # Set used data type
        self.dtype = dtype
        
        # Set dimensions
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        
        # Initialize weights
        self.W1 = numx.random.randn(input_dim, hidden1_dim) * 0.01
        self.W2 = numx.random.randn(hidden1_dim, hidden2_dim) * 0.01

        # Initialize offsets
        self.o1 = numx.zeros((1,input_dim)) 
        self.o2 = numx.zeros((1,hidden1_dim)) 
        self.o3 = numx.zeros((1,hidden2_dim))

        self.b1 = numx.zeros((1,input_dim)) 
        self.b2 = numx.zeros((1,hidden1_dim)) 
        self.b3 = numx.zeros((1,hidden2_dim))

        if data is not None:
            datamean = numx.mean(data, axis = 0).reshape(1,input_dim)
            if offset_typ[0] is '0':
                self.b1 = Sigmoid.g(numx.clip(datamean,0.001,0.999))
            if offset_typ[0] is 'D':
                self.o1 = numx.copy(datamean)
                self.b1 = Sigmoid.g(numx.clip(self.o1,0.001,0.999))
            if offset_typ[0] is 'A':
                self.o1 = (datamean + 0.5)/2.0
                self.b1 = Sigmoid.g(numx.clip(self.o1,0.001,0.999))
            if offset_typ[0] is 'M':
                self.o1 += 0.5
        else:
            if offset_typ[0] != '0':
                self.o1 += 0.5

        if offset_typ[1] != '0':
            self.o2 += 0.5
            
        if offset_typ[2] != '0':
            self.o3 += 0.5

        
    def energy(self,x,h1,h2):
        ''' Computes the energy for x, h1 and h2.
        
        :Parameters:
            x:    Input layer states.
                 -type: numpy array [batch size, input dim]

            h1:   First layer states.
                 -type: numpy array [batch size, hidden1 dim]
                 
            h2:   Second layer states.
                 -type: numpy array [batch size, hidden2 dim]
                  
        :Returns:
            Energy for x, h1 and h2.
           -type: numpy array [batch size, 1]
            
        ''' 
        # centered variables
        xtemp = x-self.o1
        h1temp = h1-self.o2
        h2temp = h2-self.o3
        # Caluclate energy
        return - numx.dot(xtemp, self.b1.T)\
                - numx.dot(h1temp, self.b2.T) \
                - numx.dot(h2temp, self.b3.T) \
                - numx.sum(numx.dot(xtemp, self.W1) * h1temp,axis=1).reshape(h1temp.shape[0], 1)\
                - numx.sum(numx.dot(h1temp, self.W2) * h2temp,axis=1).reshape(h2temp.shape[0], 1)

    def unnormalized_log_probability_x(self,x):
        ''' Computes the unnormalized log probabilities of x.
        
        :Parameters:
            x:    Input layer states.
                 -type: numpy array [batch size, input dim]
                  
        :Returns:
            Unnormalized log probability of x.
           -type: numpy array [batch size, 1]
            
        '''  
        # Generate all possibel binary codes for h1 and h2
        all_h1 = npExt.generate_binary_code(self.W2.shape[0])
        all_h2 = npExt.generate_binary_code(self.W2.shape[1])
        # Center variables
        xtemp = x-self.o1
        h1temp = all_h1-self.o2
        h2temp = all_h2-self.o3
        # Bias term
        bias = numx.dot(xtemp, self.b1.T)
        # Both quadratic terms
        part1 = numx.exp(numx.dot(numx.dot(xtemp, self.W1)+self.b2, h1temp.T))
        part2 = numx.exp(numx.dot(numx.dot(h1temp, self.W2)+self.b3, h2temp.T))
        # Dot product of all combination of all quadratic terms + bias
        return bias+numx.log(numx.sum(numx.dot(part1,part2), axis = 1).reshape(x.shape[0],1))

    def unnormalized_log_probability_h1(self,h1):
        ''' Computes the unnormalized log probabilities of h1.
        
        :Parameters:
            h1:    First hidden layer states.
                  -type: numpy array [batch size, hidden1 dim]
                  
        :Returns:
            Unnormalized log probability of h1.
           -type: numpy array [batch size, 1]
            
        '''  
        # Centered
        temp = h1 - self.o2
        # Bias term
        bias = numx.dot(temp, self.b2.T).reshape(temp.shape[0], 1)
        # Value for h1 via factorization over x 
        activation = numx.dot(temp, self.W1.T) + self.b1
        factorx = numx.sum(
                           numx.log(
                                    numx.exp(activation*(1.0 - self.o1))
                                    + numx.exp(-activation*self.o1)
                                    ) 
                           , axis=1).reshape(temp.shape[0], 1)   
        # Value for h1 via factorization over h2
        activation = numx.dot(temp, self.W2) + self.b3  
        factorh2 = numx.sum(
                            numx.log(
                                     numx.exp(activation*(1.0 - self.o3))
                                     + numx.exp(-activation*self.o3)
                                     ) 
                            , axis=1).reshape(temp.shape[0], 1)  
        return bias + factorx + factorh2
