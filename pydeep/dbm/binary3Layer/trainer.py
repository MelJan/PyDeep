""" This module contains trainers for the particular case of an 3 layer binary DBM.

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
from pydeep.base.activationfunction import Sigmoid
import pydeep.rbm.model as RBM_MODEL
import pydeep.rbm.sampler as RBM_SAMPLER

class CD(object):
    
    def __init__(self, model, batch_size):
        
        # Set batch size
        self.batch_size = batch_size
        
        # Store model
        self.model = model
        
        self.rbm = RBM_MODEL.BinaryBinaryRBM(number_visibles = model.input_dim+model.hidden2_dim, 
                                             number_hiddens = model.hidden1_dim, 
                                             data=None, 
                                             initial_weights=numx.vstack((model.W1,model.W2.T)), 
                                             initial_visible_bias=numx.hstack((model.b1,model.b3)), 
                                             initial_hidden_bias=model.b2, 
                                             initial_visible_offsets=numx.hstack((model.o1,model.o3)), 
                                             initial_hidden_offsets=model.o2)
        
        self.sampler = RBM_SAMPLER.GibbsSampler(self.rbm)

    def train(self, data, epsilon, k=[3,1], offset_typ = 'DDD',meanfield = False):
        
        #positive phase
        id1 = numx.dot(data-self.model.o1,self.model.W1)
        d3 = numx.copy(self.model.o3)
        d2 = numx.copy(self.model.o2)
        #for _ in range(k[0]):  
        if meanfield == False:
            for _ in range(k[0]):
                d2 = Sigmoid.f( id1 + numx.dot(d3-self.model.o3,self.model.W2.T) + self.model.b2)
                d2 = self.model.dtype(d2 > numx.random.random(d2.shape))
                d3 = Sigmoid.f(numx.dot(d2-self.model.o2,self.model.W2) + self.model.b3)
                d3 = self.model.dtype(d3 > numx.random.random(d3.shape))
        else:
            if meanfield == True:
                for _ in range(k[0]): 
                    d2 = Sigmoid.f( id1 + numx.dot(d3-self.model.o3,self.model.W2.T) + self.model.b2)
                    d3 = Sigmoid.f(numx.dot(d2-self.model.o2,self.model.W2) + self.model.b3)
            else:
                d2_new = Sigmoid.f( id1 + numx.dot(d3-self.model.o3,self.model.W2.T) + self.model.b2)
                d3_new = Sigmoid.f(numx.dot(d2_new-self.model.o2,self.model.W2) + self.model.b3)
                while numx.max(numx.abs(d2_new-d2)) > meanfield or numx.max(numx.abs(d3_new-d3)) > meanfield: 
                    d2 = d2_new
                    d3 = d3_new
                    d2_new = Sigmoid.f( id1 + numx.dot(d3_new-self.model.o3,self.model.W2.T) + self.model.b2)
                    d3_new = Sigmoid.f(numx.dot(d2_new-self.model.o2,self.model.W2) + self.model.b3)
                d2 = d2_new
                d3 = d3_new

        self.sampler.model = RBM_MODEL.BinaryBinaryRBM(number_visibles = self.model.input_dim+self.model.hidden2_dim, 
                                             number_hiddens = self.model.hidden1_dim, 
                                             data=None, 
                                             initial_weights=numx.vstack((self.model.W1,self.model.W2.T)), 
                                             initial_visible_bias=numx.hstack((self.model.b1,self.model.b3)), 
                                             initial_hidden_bias=self.model.b2, 
                                             initial_visible_offsets=numx.hstack((self.model.o1,self.model.o3)), 
                                             initial_hidden_offsets=self.model.o2)
        if isinstance(self.sampler,RBM_SAMPLER.GibbsSampler):
            sample = self.sampler.sample(numx.hstack((data,d3)))
        else:
            sample = self.sampler.sample(self.batch_size, k[1])
        self.m2 = self.sampler.model.probability_h_given_v(sample)
        self.m1 = sample[:,0:self.model.input_dim]
        self.m3 = sample[:,self.model.input_dim:]
 
        # Estimate new means
        new_o1 = 0
        if offset_typ[0] is 'D':
            new_o1 = data.mean(axis=0)
        if offset_typ[0] is 'A':
            new_o1 = (self.m1.mean(axis=0)+data.mean(axis=0))/2.0
        if offset_typ[0] is 'M':
            new_o1 = self.m1.mean(axis=0)

        new_o2 = 0
        if offset_typ[1] is 'D':
            new_o2 = d2.mean(axis=0)
        if offset_typ[1] is 'A':
            new_o2 = (self.m2.mean(axis=0)+d2.mean(axis=0))/2.0
        if offset_typ[1] is 'M':
            new_o2 = self.m2.mean(axis=0)

        new_o3 = 0
        if offset_typ[2] is 'D':
            new_o3 = d3.mean(axis=0)
        if offset_typ[2] is 'A':
            new_o3 = (self.m3.mean(axis=0)+d3.mean(axis=0))/2.0
        if offset_typ[2] is 'M':
            new_o3 = self.m3.mean(axis=0)
             
        # Reparameterize
        self.model.b1 += epsilon[6]*numx.dot(new_o2-self.model.o2,self.model.W1.T)
        self.model.b2 += epsilon[5]*numx.dot(new_o1-self.model.o1,self.model.W1) + epsilon[7]*numx.dot(new_o3-self.model.o3,self.model.W2.T)
        self.model.b3 += epsilon[7]*numx.dot(new_o2-self.model.o2,self.model.W2)

        # Shift means
        self.model.o1 = (1.0-epsilon[5])*self.model.o1 + epsilon[5]*new_o1
        self.model.o2 = (1.0-epsilon[6])*self.model.o2 + epsilon[6]*new_o2
        self.model.o3 = (1.0-epsilon[7])*self.model.o3 + epsilon[7]*new_o3

        # Calculate gradients
        dW1 = (numx.dot((data-self.model.o1).T,d2-self.model.o2)-numx.dot((self.m1-self.model.o1).T,self.m2-self.model.o2))
        dW2 = (numx.dot((d2-self.model.o2).T,d3-self.model.o3)-numx.dot((self.m2-self.model.o2).T,self.m3-self.model.o3))
        
        db1 = (numx.sum(data-self.m1,axis = 0)).reshape(1,self.model.input_dim)
        db2 = (numx.sum(d2-self.m2,axis = 0)).reshape(1,self.model.hidden1_dim)
        db3 = (numx.sum(d3-self.m3,axis = 0)).reshape(1,self.model.hidden2_dim)

        # Update Model
        self.model.W1 += epsilon[0]/self.batch_size*dW1
        self.model.W2 += epsilon[1]/self.batch_size*dW2
        
        self.model.b1 += epsilon[2]/self.batch_size*db1
        self.model.b2 += epsilon[3]/self.batch_size*db2
        self.model.b3 += epsilon[4]/self.batch_size*db3
        
        
class PCD(CD):
    
    def __init__(self, model, batch_size):
        
        # Call constructor of superclass
        super(PCD, self).__init__(model = model, batch_size = batch_size)
        
        self.sampler = RBM_SAMPLER.PersistentGibbsSampler(self.rbm,self.batch_size)
        
class PT(CD):
    
    def __init__(self, model, batch_size, num_chains=3, betas=None):
        
        # Call constructor of superclass
        super(PT, self).__init__(model = model, batch_size = batch_size)
        
        self.sampler = RBM_SAMPLER.ParallelTemperingSampler(self.rbm,num_chains,betas)
        
class IPT(CD):
    
    def __init__(self, model, batch_size, num_chains=3, betas=None):
        
        # Call constructor of superclass
        super(IPT, self).__init__(model = model, batch_size = batch_size)
        
        self.sampler = RBM_SAMPLER.IndependentParallelTemperingSampler(self.rbm,self.batch_size,num_chains,betas)
        
        
class PCD_check(object):
    
    def __init__(self, model, batch_size):
        
        # Set batch size
        self.batch_size = batch_size
        
        # Store model
        self.model = model
        
        rbm = RBM_MODEL.BinaryBinaryRBM(number_visibles = model.input_dim+model.hidden2_dim, 
                                             number_hiddens = model.hidden1_dim, 
                                             data=None, 
                                             initial_weights=numx.vstack((model.W1,model.W2.T)), 
                                             initial_visible_bias=numx.hstack((model.b1,model.b3)), 
                                             initial_hidden_bias=model.b2, 
                                             initial_visible_offsets=numx.hstack((model.o1,model.o3)), 
                                             initial_hidden_offsets=model.o2)
        
        # Initializee Markov chains
        self.m1 = model.o1+numx.zeros((batch_size,model.input_dim))
        self.m2 = model.o2+numx.zeros((batch_size,model.hidden1_dim))
        self.m3 = model.o3+numx.zeros((batch_size,model.hidden2_dim))

    def train(self, data, epsilon, k=[3,1], offset_typ = 'DDD',meanfield = False):
        
        #positive phase
        id1 = numx.dot(data-self.model.o1,self.model.W1)
        d3 = numx.copy(self.model.o3)
        d2 = 0.0
        #for _ in range(k[0]):  
        if meanfield == False:
            for _ in range(k[0]): 
                d3 = self.model.dtype(d3 > numx.random.random(d3.shape))
                d2 = Sigmoid.f( id1 + numx.dot(d3-self.model.o3,self.model.W2.T) + self.model.b2)
                d2 = self.model.dtype(d2 > numx.random.random(d2.shape))
                d3 = Sigmoid.f(numx.dot(d2-self.model.o2,self.model.W2) + self.model.b3)
        else:
            if meanfield == True:
                for _ in range(k[0]): 
                    d2 = Sigmoid.f( id1 + numx.dot(d3-self.model.o3,self.model.W2.T) + self.model.b2)
                    d3 = Sigmoid.f(numx.dot(d2-self.model.o2,self.model.W2) + self.model.b3)
            else:
                d2_new = Sigmoid.f( id1 + numx.dot(d3-self.model.o3,self.model.W2.T) + self.model.b2)
                d3_new = Sigmoid.f(numx.dot(d2_new-self.model.o2,self.model.W2) + self.model.b3)
                while numx.max(numx.abs(d2_new-d2)) > meanfield or numx.max(numx.abs(d3_new-d3)) > meanfield: 
                    d2 = d2_new
                    d3 = d3_new
                    d2_new = Sigmoid.f( id1 + numx.dot(d3_new-self.model.o3,self.model.W2.T) + self.model.b2)
                    d3_new = Sigmoid.f(numx.dot(d2_new-self.model.o2,self.model.W2) + self.model.b3)
                d2 = d2_new
                d3 = d3_new
                
        #negative phase
        for _ in range(k[1]):  
            self.m2 = Sigmoid.f(numx.dot(self.m1-self.model.o1,self.model.W1) + numx.dot(self.m3-self.model.o3,self.model.W2.T) + self.model.b2)
            self.m2 = self.model.dtype(self.m2 > numx.random.random(self.m2.shape))
            self.m1 = Sigmoid.f(numx.dot(self.m2-self.model.o2,self.model.W1.T) + self.model.b1)
            self.m1 = self.model.dtype(self.m1 > numx.random.random(self.m1.shape))
            self.m3 = Sigmoid.f(numx.dot(self.m2-self.model.o2,self.model.W2) + self.model.b3)
            self.m3 = self.model.dtype(self.m3 > numx.random.random(self.m3.shape))
            
        # Estimate new means
        new_o1 = 0
        if offset_typ[0] is 'D':
            new_o1 = data.mean(axis=0)
        if offset_typ[0] is 'A':
            new_o1 = (self.m1.mean(axis=0)+data.mean(axis=0))/2.0
        if offset_typ[0] is 'M':
            new_o1 = self.m1.mean(axis=0)

        new_o2 = 0
        if offset_typ[1] is 'D':
            new_o2 = d2.mean(axis=0)
        if offset_typ[1] is 'A':
            new_o2 = (self.m2.mean(axis=0)+d2.mean(axis=0))/2.0
        if offset_typ[1] is 'M':
            new_o2 = self.m2.mean(axis=0)

        new_o3 = 0
        if offset_typ[2] is 'D':
            new_o3 = d3.mean(axis=0)
        if offset_typ[2] is 'A':
            new_o3 = (self.m3.mean(axis=0)+d3.mean(axis=0))/2.0
        if offset_typ[2] is 'M':
            new_o3 = self.m3.mean(axis=0)
             
        # Reparameterize
        self.model.b1 += epsilon[6]*numx.dot(new_o2-self.model.o2,self.model.W1.T)
        self.model.b2 += epsilon[5]*numx.dot(new_o1-self.model.o1,self.model.W1) + epsilon[7]*numx.dot(new_o3-self.model.o3,self.model.W2.T)
        self.model.b3 += epsilon[6]*numx.dot(new_o2-self.model.o2,self.model.W2)

        # Shift means
        self.model.o1 = (1.0-epsilon[5])*self.model.o1 + epsilon[5]*new_o1
        self.model.o2 = (1.0-epsilon[6])*self.model.o2 + epsilon[6]*new_o2
        self.model.o3 = (1.0-epsilon[7])*self.model.o3 + epsilon[7]*new_o3

        # Calculate gradients
        dW1 = (numx.dot((data-self.model.o1).T,d2-self.model.o2)-numx.dot((self.m1-self.model.o1).T,self.m2-self.model.o2))
        dW2 = (numx.dot((d2-self.model.o2).T,d3-self.model.o3)-numx.dot((self.m2-self.model.o2).T,self.m3-self.model.o3))
        
        db1 = (numx.sum(data-self.m1,axis = 0)).reshape(1,self.model.input_dim)
        db2 = (numx.sum(d2-self.m2,axis = 0)).reshape(1,self.model.hidden1_dim)
        db3 = (numx.sum(d3-self.m3,axis = 0)).reshape(1,self.model.hidden2_dim)

        # Update Model
        self.model.W1 += epsilon[0]/self.batch_size*dW1
        self.model.W2 += epsilon[1]/self.batch_size*dW2
        
        self.model.b1 += epsilon[2]/self.batch_size*db1
        self.model.b2 += epsilon[3]/self.batch_size*db2
        self.model.b3 += epsilon[4]/self.batch_size*db3
