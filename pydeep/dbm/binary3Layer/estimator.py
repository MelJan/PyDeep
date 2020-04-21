""" This module contains estimators for the particular case of an 3 layer binary DBM.

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
import pydeep.rbm.model as RBM_MODEL
import pydeep.rbm.estimator as RBM_ESTIMATOR


def _partition_function_exact_check(model, batchsize_exponent='AUTO'):
    ''' Computes the true partition function for the given model by factoring
        over the visible and hidden2 units.

        This is just proof of concept, use _partition_function_exact() instead,
        it is heaps faster!

    :Parameters:
        model:              The model
                           -type: Valid DBM model

        batchsize_exponent: 2^batchsize_exponent will be the batch size.
                           -type: int

    :Returns:
        Log Partition function for the model.
       -type: float

    '''
    bit_length = model.W1.shape[1]
    if batchsize_exponent is 'AUTO' or batchsize_exponent > 20:
        batchsize_exponent = numx.min([model.W1.shape[1], 12])
    batchSize = numx.power(2, batchsize_exponent)
    num_combinations = numx.power(2, bit_length)
    num_batches = num_combinations // batchSize
    bitCombinations = numx.zeros((batchSize, model.W1.shape[1]))
    log_prob_vv_all = numx.zeros(num_combinations)

    for batch in range(1, num_batches + 1):
        # Generate current batch
        bitCombinations = npExt.generate_binary_code(bit_length,
                                                     batchsize_exponent,
                                                     batch - 1)
        # calculate LL
        log_prob_vv_all[(batch - 1) * batchSize:batch * batchSize] = model.unnormalized_log_probability_h1(bitCombinations).reshape(
                                                bitCombinations.shape[0])
    # return the log_sum of values
    return npExt.log_sum_exp(log_prob_vv_all)

def partition_function_exact(model, batchsize_exponent='AUTO'):
    ''' Computes the true partition function for the given model by factoring
        over the visible and hidden2 units.

    :Parameters:
        model:              The model
                           -type: Valid DBM model

        batchsize_exponent: 2^batchsize_exponent will be the batch size.
                           -type: int

    :Returns:
        Log Partition function for the model.
       -type: float

    '''
    # We transform the DBM to an RBM with restricted connections.
    rbm = RBM_MODEL.BinaryBinaryRBM(number_visibles = model.input_dim+model.hidden2_dim,
                                    number_hiddens = model.hidden1_dim,
                                    data=None,
                                    initial_weights=numx.vstack((model.W1,model.W2.T)),
                                    initial_visible_bias=numx.hstack((model.b1,model.b3)),
                                    initial_hidden_bias=model.b2,
                                    initial_visible_offsets=numx.hstack((model.o1,model.o3)),
                                    initial_hidden_offsets=model.o2)
    return RBM_ESTIMATOR.partition_function_factorize_h(rbm)

def partition_function_AIS(model, num_chains = 100, k = 1, betas = 10000, status = False):
    ''' Approximates the partition function for the given model using annealed
        importance sampling.

        :Parameters:
            model:      The model.
                       -type: Valid RBM model.

            num_chains: Number of AIS runs.
                       -type: int

            k:          Number of Gibbs sampling steps.
                       -type: int

            beta:       Number or a list of inverse temperatures to sample from.
                       -type: int, numpy array [num_betas]

            status:     If true prints the progress on console.
                       -type: bool


        :Returns:
            Mean estimated log partition function.
           -type: float
            Mean +3std estimated log partition function.
           -type: float
            Mean -3std estimated log partition function.
           -type: float

    '''
    # We transform the DBM to an RBM with restricted connections.
    rbm = RBM_MODEL.BinaryBinaryRBM(number_visibles = model.input_dim+model.hidden2_dim,
                                    number_hiddens = model.hidden1_dim,
                                    data=None,
                                    initial_weights=numx.vstack((model.W1,model.W2.T)),
                                    initial_visible_bias=numx.hstack((model.b1,model.b3)),
                                    initial_hidden_bias=model.b2,
                                    initial_visible_offsets=numx.hstack((model.o1,model.o3)),
                                    initial_hidden_offsets=model.o2)
    # Run AIS for the transformed DBM
    return RBM_ESTIMATOR.annealed_importance_sampling(model = rbm,
                                                  num_chains =  num_chains,
                                                  k = k, betas= betas,
                                                  status = status)

def _LL_exact_check(model, x, lnZ):
    ''' Computes the exact log likelihood for x by summing over all possible
        states for h1, h2. Only possible for small hidden layers!

        This is just proof of concept, use LL_exact() instead, it is heaps faster!

    :Parameters:
        model:  The model
               -type: Valid DBM model

        x:      Input states.
               -type: numpy array [batch size, input dim]

        lnZ:    Logarithm of the patition function.
               -type: float

    :Returns:
        Exact log likelihood for x.
       -type: numpy array [batch size, 1]

    '''
    # Generate all binary codes
    all_h1 = npExt.generate_binary_code(model.W2.shape[0])
    all_h2 = npExt.generate_binary_code(model.W2.shape[1])
    result = numx.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(all_h1.shape[0]):
            for k in range(all_h2.shape[0]):
                result[i] += numx.exp(
                                      -model.energy(
                                                   x[i].reshape(1,x.shape[1]),
                                                   all_h1[j].reshape(1,all_h1.shape[1]),
                                                   all_h2[k].reshape(1,all_h2.shape[1]),
                                                   )
                                      )
    return numx.log(result) - lnZ

def LL_exact(model, x, lnZ):
    ''' Computes the exact log likelihood for x by summing over all possible
        states for h1, h2. Only possible for small hidden layers!

    :Parameters:
        model:  The model
               -type: Valid DBM model

        x:     Input states.
              -type: numpy array [batch size, input dim]

        lnZ:   Logarithm of the patition function.
              -type: float

    :Returns:
        Exact log likelihood for x.
       -type: numpy array [batch size, 1]

    '''
    return model.unnormalized_log_probability_x(x)- lnZ

def _LL_lower_bound_check(model, x, lnZ, conv_thres= 0.0001, max_iter=100000):
    ''' Computes the log likelihood lower bound for x by approximating h1, h2
        by Mean field estimates.
        .. seealso:: AISTATS 2009: Deep Bolzmann machines
             http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS09_SalakhutdinovH.pdf

    :Parameters:
        model:       The model
                    -type: Valid DBM model

        x:           Input states.
                    -type: numpy array [batch size, input dim]

        lnZ:         Logarithm of the patition function.
                    -type: float

        conv_thres:  Convergence threshold for the mean field approximation
                    -type: float

        max_iter:    If convergence threshold not reached, maximal number of sampling steps
                    -type: int

    :Returns:
        Log likelihood lower bound for x.
       -type: numpy array [batch size, 1]

    '''
    # Pre calc activation from x since it is constant
    id1 = numx.dot(x-model.o1,model.W1)
    # Initialize mu3 with its mean
    d3 = numx.zeros((x.shape[0],model.hidden2_dim))
    d2 = numx.zeros((x.shape[0],model.hidden1_dim))
    # While convergence of max number of iterations not reached,
    # run mean field estimation
    for i in range(x.shape[0]):
        d3_temp = numx.copy(model.o3)
        d2_temp = 0.0
        d2_new = Sigmoid.f( id1[i,:] + numx.dot(d3_temp-model.o3,model.W2.T) + model.b2)
        d3_new = Sigmoid.f(numx.dot(d2_new-model.o2,model.W2) + model.b3)
        while numx.max(numx.abs(d2_new-d2_temp )) > conv_thres or numx.max(numx.abs(d3_new-d3_temp )) > conv_thres:
            d2_temp  = d2_new
            d3_temp  = d3_new
            d2_new = Sigmoid.f( id1[i,:]  + numx.dot(d3_new-model.o3,model.W2.T) + model.b2)
            d3_new = Sigmoid.f(numx.dot(d2_new-model.o2,model.W2) + model.b3)
        d2[i] = numx.clip(d2_new,0.0000000000000001,0.9999999999999999).reshape(1,model.hidden1_dim)
        d3[i] = numx.clip(d3_new,0.0000000000000001,0.9999999999999999).reshape(1,model.hidden2_dim)
    # Return ernegy of states + the entropy of h1.h2 due to the mean field approximation
    return -model.energy(x,d2,d3) -lnZ - numx.sum(d2*numx.log(d2)+(1.0-d2)*numx.log(1.0-d2),axis = 1).reshape(x.shape[0], 1) - numx.sum(d3*numx.log(d3)+(1.0-d3)*numx.log(1.0-d3),axis = 1).reshape(x.shape[0], 1)



def LL_lower_bound(model, x, lnZ, conv_thres= 0.0001, max_iter=1000):
    ''' Computes the log likelihood lower bound for x by approximating h1
        by Mean field estimates. The same as LL_lower_bound, but where h2 has been factorized.
        .. seealso:: AISTATS 2009: Deep Bolzmann machines
             http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS09_SalakhutdinovH.pdf

    :Parameters:
        model:       The model
                    -type: Valid DBM model

        x:           Input states.
                    -type: numpy array [batch size, input dim]

        lnZ:         Logarithm of the patition function.
                    -type: float

        conv_thres:  Convergence threshold for the mean field approximation
                    -type: float

        max_iter:    If convergence threshold not reached, maximal number of sampling steps
                    -type: int

    :Returns:
        Log likelihood lower bound for x.
       -type: numpy array [batch size, 1]

    '''
    # Pre calc activation from x since it is constant
    id1 = numx.dot(x-model.o1,model.W1)
    # Initialize mu3 with its mean
    d2 = numx.zeros((x.shape[0],model.hidden1_dim))
    # While convergence of max number of iterations not reached,
    # run mean field estimation
    for i in range(x.shape[0]):
        d3_temp = numx.copy(model.o3)
        d2_temp = 0.0
        d2_new = Sigmoid.f( id1[i,:] + numx.dot(d3_temp-model.o3,model.W2.T) + model.b2)
        d3_new = Sigmoid.f(numx.dot(d2_new-model.o2,model.W2) + model.b3)
        while numx.max(numx.abs(d2_new-d2_temp )) > conv_thres:
            d2_temp  = d2_new
            d3_temp  = d3_new
            d2_new = Sigmoid.f( id1[i,:]  + numx.dot(d3_new-model.o3,model.W2.T) + model.b2)
            d3_new = Sigmoid.f(numx.dot(d2_new-model.o2,model.W2) + model.b3)
        d2[i] = numx.clip(d2_new,0.0000000000000001,0.9999999999999999).reshape(1,model.hidden1_dim)

    # Foactorize over h2
    xtemp = x-model.o1
    h1temp = d2-model.o2
    e2 = numx.sum(numx.log(numx.exp(-(numx.dot(h1temp, model.W2)+model.b3)*(model.o3))+numx.exp((numx.dot(h1temp, model.W2)+model.b3)*(1.0-model.o3))), axis = 1).reshape(x.shape[0],1)
    e1 =  numx.dot(xtemp, model.b1.T)\
        + numx.dot(h1temp, model.b2.T) \
        + numx.sum(numx.dot(xtemp, model.W1) * h1temp ,axis=1).reshape(h1temp.shape[0], 1) + e2
    # Return energy of states + the entropy of h1 due to the mean field approximation
    return e1-lnZ - numx.sum(d2*numx.log(d2) + (1.0-d2)*numx.log(1.0-d2) ,axis = 1).reshape(x.shape[0], 1)
