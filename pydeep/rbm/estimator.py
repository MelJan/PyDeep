''' This module provides methods for estimating the model performance 
    (running on the CPU). Provided performance measures are for example 
    the reconstruction error (RE) and the log-likelihood (LL). For estimating   
    the LL we need to know the value of the partition function Z. If at least
    one layer is binary it is possible to calculate the value by factorizing 
    over the binary values. Since it involves calculating all possible binary
    states, it is only possible for small models i.e. less than 25 
    (e.g. ~2^25 = 33554432 states). For bigger models we can estimate the 
    partition function using annealed importance sampling (AIS).
    
    :Implemented:
        - kth order reconstruction error
        - Log likelihood for visible data.
        - Log likelihood for hidden data.   
        - True partition by factorization over the visible units.
        - True partition by factorization over the hidden units.
        - Annealed importance sampling to approximated the partition function.
      
    :Info:
        For the derivations see:
        http://www.ini.rub.de/data/documents/tns/masterthesis_janmelchior.pdf
        
    :Version:
        1.0

    :Date:
        06.06.2016

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
import pydeep.base.numpyextension as npExt
  
def reconstruction_error(model, 
                         data, 
                         k=1, 
                         beta=None, 
                         use_states=False, 
                         absolut_error = False):
    ''' This function calculates the reconstruction errors for a given model 
        and data.         
        
    :Parameters:
        model          The model
                      -type: Valid RBM model
                     
        data:          The data as 2D array or 3D array.
                      -type: numpy array [num samples, num dimensions] or 
                             numpy array [num batches, num samples in batch,
                                          num dimensions]
                     
        k:             Number of Gibbs sampling steps.
                      -type: int
        
        beta:          Inverse temperature(s) for the models energy.
                      -type: None, float or numpy array [batchsize,1]
                     
        use_states:    If false (default) the probabilities are used as 
                       reconstruction, if true states are sampled.
                      -type: bool

        absolut_error: If false (default) the squared error is used, the 
                       absolute error otherwise 
                      -type: bool
        
    :Returns:
        Reconstruction errors of the data.
       -type: nump array [num samples]
             
    ''' 
    # Data is seperated into batches
    if isinstance(data,list):
        result = []
        # for each batch sample k times
        for batch in data:
            vis_probs = batch
            if use_states:
                for _ in xrange(0, k):
                    hid_probs = model.probability_h_given_v(vis_probs, beta)
                    hid_probs = model.sample_h(hid_probs)
                    vis_probs = model.probability_v_given_h(hid_probs, beta)  
                    vis_probs = model.sample_v(vis_probs) 
            else:
                hid_probs = model.probability_h_given_v(vis_probs, beta)
                vis_probs = model.probability_v_given_h(hid_probs, beta)  
            if absolut_error is False:
                diff = numx.mean((batch - vis_probs) ** 2,axis = 1)
            else:
                diff = numx.mean(numx.abs(batch - vis_probs),axis = 1)
            result.append(diff)
        return result
    else:
        # Data is given in one batch
        vis_probs = data
        if use_states:
            for _ in xrange(0, k):
                hid_probs = model.probability_h_given_v(vis_probs, beta)
                hid_probs = model.sample_h(hid_probs)
                vis_probs = model.probability_v_given_h(hid_probs, beta)  
                vis_probs = model.sample_v(vis_probs) 
        else:
            hid_probs = model.probability_h_given_v(vis_probs, beta)
            vis_probs = model.probability_v_given_h(hid_probs, beta) 
        if absolut_error is False:
            return numx.mean((data-vis_probs) ** 2,axis = 1)
        else:
            return numx.mean(numx.abs(data-vis_probs),axis = 1)

def log_likelihood_v(model,
                     logZ, 
                     data, 
                     beta=None):
    ''' Computes the log-likelihood (LL) for a given model and visible data 
        given its log partition function.
        
        :Info: logZ needs to be the partition function for the same beta 
               (i.e. beta = 1.0)!
        
        :Parameters:
            model: The model.
                  -type: Valid RBM model.
            
            logZ:  The logarithm of the partition function.
                  -type: float
            
            data:  The visible data as 2D array [num samples, num input dim] 
                   or 3D type numpy array [num batches, num samples in batch,
                                           num input dim]
                   
            beta:  Inverse temperature(s) for the models energy.
                  -type: None, float, numpy array [batchsize,1]
            
        :Returns:
            The log-likelihood for each sample.
           -type: numpy array [num samples]
            
    '''
    ll = []
    if isinstance(data,list):
        for batch in data:
            ll.append(model.log_probability_v(logZ, batch, beta))         
        return ll
    else:
        return model.log_probability_v(logZ, data, beta)

def log_likelihood_h(model, 
                     logZ, 
                     data, 
                     beta=None):
    ''' Computes the log-likelihood (LL) for a given model and hidden data 
        given its log partition function.
        
        :Info: logZ needs to be the partition function for the same beta!
               (i.e. beta = 1.0)!
               
        :Parameters:
            model: The model.
                  -type: Valid RBM model.
            
            logZ:  The logarithm of the partition function.
                  -type: float
            
            data:  The hidden data as 2D array [num samples, num output dim]
                  -type: numpy array [num batches,num samples in batch, 
                                      num output dim]
                   
            beta:  Inverse temperature(s) for the models energy.
                  -type: None, float, numpy array [batchsize,1]
            
        :Returns:
            The log-likelihood for each sample.
           -type: numpy array [num samples]
            
    '''
    ll = []
    if isinstance(data,list):
        for batch in data:
            ll.append(model.log_probability_h(logZ, batch, beta))         
        return ll
    else:
        return model.log_probability_v(logZ, data, beta)

def partition_function_factorize_v(model, 
                                   beta=None, 
                                   batchsize_exponent='AUTO', 
                                   status=False):
    ''' Computes the true partition function for the given model by factoring 
        over the visible units.
       
    :Info:
        Exponential increase of computations by the number of visible units.
        (16 usually ~ 20 seconds)
        
    :Parameters:
        model:              The model.
                           -type: Valid RBM model.
        
        beta:               Inverse temperature(s) for the models energy.
                           -type: None, float, numpy array [batchsize,1]
        
        batchsize_exponent: 2^batchsize_exponent will be the batch size.
                           -type: int
        
        status:             If true prints the progress to the console.
                           -type: bool
    
    :Returns:
        Log Partition function for the model.
       -type: float
        
    '''    
    if status is True:
        print "Calculating the partition function by factoring over v: "
        print '%3.2f' % (0.0), '%'
        
    bit_length = model.input_dim
    if batchsize_exponent is 'AUTO' or batchsize_exponent > 20:
        batchsize_exponent = numx.min([model.input_dim, 12])
    batchSize = numx.power(2, batchsize_exponent)
    num_combinations = numx.power(2, bit_length)

    num_batches = num_combinations / batchSize
    bitCombinations = numx.zeros((batchSize, model.input_dim))
    log_prob_vv_all = numx.zeros(num_combinations)
    
    for batch in range(1, num_batches + 1):
        # Generate current batch
        bitCombinations = npExt.generate_binary_code(bit_length, 
                                                     batchsize_exponent, 
                                                     batch - 1)

        # calculate LL
        log_prob_vv_all[(batch - 1) * batchSize:batch * batchSize] = model.\
        unnormalized_log_probability_v(bitCombinations, beta).reshape(
                                                    bitCombinations.shape[0])
        # print status if wanted    
        if status is True:
            print '%3.2f' %(100*numx.double(batch
                                            )/numx.double(num_batches)),'%'
    
    # return the log_sum of values
    return npExt.log_sum_exp(log_prob_vv_all)

def partition_function_factorize_h(model, 
                                   beta=None, 
                                   batchsize_exponent='AUTO', 
                                   status=False):
    ''' Computes the true partition function for the given model by factoring 
        over the hidden units.
       
    :Info:
        Exponential increase of computations by the number of hidden units.
        (16 usually ~ 20 seconds)
        
    :Parameters:
        model:              The model.
                           -type: Valid RBM model.
        
        beta:               Inverse temperature(s) for the models energy.
                           -type: None, float, numpy array [batchsize,1]
        
        batchsize_exponent: 2^batchsize_exponent will be the batch size.
                           -type: int
        
        status:             If true prints the progress to the console.
                           -type: bool
    
    :Returns:
        Log Partition function for the model.
       -type: float
        
    '''  
    if status is True:
        print "Calculating the partition function by factoring over h: "
        print '%3.2f' % (0.0), '%'
        
    bit_length = model.output_dim
    if batchsize_exponent is 'AUTO' or batchsize_exponent > 20:
        batchsize_exponent = numx.min([model.output_dim, 12])
    batchSize = numx.power(2, batchsize_exponent)
    num_combinations = numx.power(2, bit_length)
    
    num_batches = num_combinations / batchSize
    bitCombinations = numx.zeros((batchSize, model.output_dim))
    log_prob_vv_all = numx.zeros(num_combinations)

    for batch in range(1, num_batches + 1):
        # Generate current batch
        bitCombinations = npExt.generate_binary_code(bit_length, 
                                                     batchsize_exponent, 
                                                     batch - 1)
        
        # calculate LL
        log_prob_vv_all[(batch - 1) * batchSize:batch * batchSize] = model.\
        unnormalized_log_probability_h(bitCombinations, beta).reshape(
                                                    bitCombinations.shape[0])
        
        # print status if wanted
        if status is True:
            print '%3.2f' %(100*numx.double(batch
                                            )/numx.double(num_batches)),'%'
            
    # return the log_sum of values
    return npExt.log_sum_exp(log_prob_vv_all)

def annealed_importance_sampling(model, 
                                 num_chains=100, 
                                 k=1, 
                                 betas=10000,
                                 status=False):
    ''' Approximates the partition function for the given model using annealed
        importance sampling.

    :See: Accurate and Conservative Estimates of MRF Log-likelihood using Reverse Annealing
          http://arxiv.org/pdf/1412.8566.pdf
    
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
    # Setup temerpatures if not given
    if numx.isscalar(betas):
        betas = numx.linspace(0.0, 1.0, betas)

    # Sample the first time from the base model
    v = model.probability_v_given_h(numx.zeros((num_chains, model.output_dim))
                                    , betas[0], True)
    v = model.sample_v(v, betas[0], True)

    # Calculate the unnormalized probabilties of v
    lnPvSum = -model.unnormalized_log_probability_v(v, betas[0], True)

    if status is True:
        t = 1
        print "Calculating the partition function using AIS: "
        print '%3.2f' % (0.0), '%'
        print '%3.2f' % (100.0*numx.double(t)/numx.double(betas.shape[0])), '%'

    for beta in betas[1:betas.shape[0]-1]:

        if status is True:
            t += 1
            print '%3.2f' % (100.0*numx.double(t)/numx.double(betas.shape[0])), '%'
        # Calculate the unnormalized probabilties of v
        lnPvSum += model.unnormalized_log_probability_v(v, beta, True)
        
        # Sample k times from the intermidate distribution
        for _ in range(0, k):   
            h = model.sample_h(model.probability_h_given_v(v, beta, True),beta, True)
            v = model.sample_v(model.probability_v_given_h(h, beta, True),beta, True)

        # Calculate the unnormalized probabilties of v
        lnPvSum -= model.unnormalized_log_probability_v(v, beta, True)

    # Calculate the unnormalized probabilties of v
    lnPvSum += model.unnormalized_log_probability_v(v, betas[betas.shape[0]-1]
                                                    , True)

    lnPvSum = numx.float128(lnPvSum)

    # Calculate an estimate of logZ . 
    logZ = npExt.log_sum_exp(lnPvSum) - numx.log(num_chains)  

    # Calculate +/- 3 standard deviations
    lnPvMean = numx.mean(lnPvSum) 
    lnPvStd = numx.log(
                       numx.std(
                                numx.exp (lnPvSum-lnPvMean)
                                )
                       ) + lnPvMean - numx.log(num_chains)/2.0   
    lnPvStd = numx.vstack((numx.log(3.0)+lnPvStd,logZ))

    # Calculate partition function of base distribution
    baseLogZ = model._base_log_partition(True)
    
    # Add the base partition function
    logZ = logZ + baseLogZ
    logZ_up = npExt.log_sum_exp(lnPvStd) + baseLogZ
    logZ_down = npExt.log_diff_exp(lnPvStd) + baseLogZ

    if status is True:
        print '%3.2f' % (100.0), '%'

    return logZ, logZ_up, logZ_down


def reverse_annealed_importance_sampling(model,
                                        num_chains=100,
                                        k=1,
                                        betas=10000,
                                        status=False,
                                        data=None):
    ''' Approximates the partition function for the given model using reverse annealed
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

    # Setup temerpatures if not given
    if numx.isscalar(betas):
        betas = numx.linspace(0.0, 1.0, betas)

    if data is None:
        data = numx.zeros((num_chains, model.output_dim))
    else:
        data = model.sample_h(model.probability_h_given_v(numx.random.permutation(data)[0:num_chains]))

    # Sample the first time from the true model model
    v = model.probability_v_given_h(data, betas[betas.shape[0]-1], True)
    v = model.sample_v(v, betas[betas.shape[0]-1],True)

    # Calculate the unnormalized probabilties of v
    lnPvSum = model.unnormalized_log_probability_v(v, betas[betas.shape[0]-1], True)


    # Setup temerpatures if not given
    if status is True:
        t = 1
        print "Calculating the partition function using AIS: "
        print '%3.2f' % (0.0), '%'
        print '%3.2f' % (100.0*numx.double(t)/numx.double(betas.shape[0])), '%'

    for beta in reversed(betas[1:betas.shape[0]-1]):

        if status is True:
            t += 1
            print '%3.2f' % (100.0*numx.double(t)/numx.double(betas.shape[0])), '%'

        # Calculate the unnormalized probabilties of v
        lnPvSum -= model.unnormalized_log_probability_v(v, beta, True)

        # Sample k times from the intermidate distribution
        for _ in range(0, k):
            h = model.sample_h(model.probability_h_given_v(v, beta, True),beta, True)
            v = model.sample_v(model.probability_v_given_h(h, beta, True),beta, True)

        # Calculate the unnormalized probabilties of v
        lnPvSum += model.unnormalized_log_probability_v(v, beta, True)


    # Calculate the unnormalized probabilties of v
    lnPvSum -= model.unnormalized_log_probability_v(v, betas[0]
                                                    , True)

    lnPvSum = numx.float128(lnPvSum)

    # Calculate an estimate of logZ .
    logZ = npExt.log_sum_exp(lnPvSum) - numx.log(num_chains)

    # Calculate +/- 3 standard deviations
    lnPvMean = numx.mean(lnPvSum)
    lnPvStd = numx.log(
                       numx.std(
                                numx.exp (lnPvSum-lnPvMean)
                                )
                       ) + lnPvMean - numx.log(num_chains)/2.0
    lnPvStd = numx.vstack((numx.log(3.0)+lnPvStd,logZ))

    # Calculate partition function of base distribution
    baseLogZ = model._base_log_partition(True)

    # Add the base partition function
    logZ = logZ + baseLogZ
    logZ_up = npExt.log_sum_exp(lnPvStd) + baseLogZ
    logZ_down = npExt.log_diff_exp(lnPvStd) + baseLogZ

    if status is True:
        print '%3.2f' % (100.0), '%'

    return logZ, logZ_up, logZ_down