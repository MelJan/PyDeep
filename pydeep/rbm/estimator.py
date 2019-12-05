""" This module provides methods for estimating the model performance
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
        - Reverse annealed importance sampling to approximated the partition function.

    :Info:
        For the derivations .. seealso::
        https://www.ini.rub.de/PEOPLE/wiskott/Reprints/Melchior-2012-MasterThesis-RBMs.pdf

    :Version:
        1.1.0

    :Date:
        04.04.2017

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
import pydeep.base.numpyextension as numxext


def reconstruction_error(model,
                         data,
                         k=1,
                         beta=None,
                         use_states=False,
                         absolut_error=False):
    """ This function calculates the reconstruction errors for a given model and data.

    :param model: The model.
    :type model: Valid RBM model

    :param data: The data as 2D array or 3D array.
    :type data: numpy array [num samples, num dimensions] or
                numpy array [num batches, num samples in batch, num dimensions]

    :param k: Number of Gibbs sampling steps.
    :type k: int

    :param beta: Temperature(s) for the models energy.
    :type beta: None, float or numpy array [batchsize,1]

    :param use_states: If false (default) the probabilities are used as reconstruction, if true states are sampled.
    :type use_states: bool

    :param absolut_error: If false (default) the squared error is used, the absolute error otherwise.
    :type absolut_error: boll

    :return: Reconstruction errors of the data.
    :rtype: nump array [num samples]
    """
    # Data is seperated into batches
    if isinstance(data, list):
        result = []
        # for each batch sample k times
        for batch in data:
            vis_probs = batch
            if use_states:
                for _ in range(0, k):
                    hid_probs = model.probability_h_given_v(vis_probs, beta)
                    hid_probs = model.sample_h(hid_probs)
                    vis_probs = model.probability_v_given_h(hid_probs, beta)
                    vis_probs = model.sample_v(vis_probs)
            else:
                hid_probs = model.probability_h_given_v(vis_probs, beta)
                vis_probs = model.probability_v_given_h(hid_probs, beta)
            if absolut_error is False:
                diff = numx.mean((batch - vis_probs) ** 2, axis=1)
            else:
                diff = numx.mean(numx.abs(batch - vis_probs), axis=1)
            result.append(diff)
        return result
    else:
        # Data is given in one batch
        vis_probs = data
        if use_states:
            for _ in range(0, k):
                hid_probs = model.probability_h_given_v(vis_probs, beta)
                hid_probs = model.sample_h(hid_probs)
                vis_probs = model.probability_v_given_h(hid_probs, beta)
                vis_probs = model.sample_v(vis_probs)
        else:
            hid_probs = model.probability_h_given_v(vis_probs, beta)
            vis_probs = model.probability_v_given_h(hid_probs, beta)
        if absolut_error is False:
            return numx.mean((data - vis_probs) ** 2, axis=1)
        else:
            return numx.mean(numx.abs(data - vis_probs), axis=1)


def log_likelihood_v(model,
                     logz,
                     data,
                     beta=None):
    """ Computes the log-likelihood (LL) for a given model and visible data given its log partition function.

        :Info: logz needs to be the partition function for the same beta (i.e. beta = 1.0)!

    :param model: The model.
    :type model: Valid RBM model.

    :param logz: The logarithm of the partition function.
    :type logz: float

    :param data: The visible data.
    :type data: 2D array [num samples, num input dim] or
                3D type numpy array [num batches, num samples in batch, num input dim]

    :param beta: Inverse temperature(s) for the models energy.
    :type beta: None, float, numpy array [batchsize,1]

    :return: The log-likelihood for each sample.
    :rtype: numpy array [num samples]
    """
    ll = []
    if isinstance(data, list):
        for batch in data:
            ll.append(model.log_probability_v(logz, batch, beta))
        return ll
    else:
        return model.log_probability_v(logz, data, beta)


def log_likelihood_h(model,
                     logz,
                     data,
                     beta=None):
    """ Computes the log-likelihood (LL) for a given model and hidden data given its log partition function.

        :Info: logz needs to be the partition function for the same beta (i.e. beta = 1.0)!

    :param model: The model.
    :type model: Valid RBM model.

    :param logz: The logarithm of the partition function.
    :type logz: float

    :param data: The hidden data.
    :type data: 2D array [num samples, num output dim] or
                3D type numpy array [num batches, num samples in batch, num output dim]

    :param beta: Inverse temperature(s) for the models energy.
    :type beta: None, float, numpy array [batchsize,1]

    :return: The log-likelihood for each sample.
    :rtype: numpy array [num samples]
    """
    ll = []
    if isinstance(data, list):
        for batch in data:
            ll.append(model.log_probability_h(logz, batch, beta))
        return ll
    else:
        return model.log_probability_v(logz, data, beta)


def partition_function_factorize_v(model,
                                   beta=None,
                                   batchsize_exponent='AUTO',
                                   status=False):
    """ Computes the true partition function for the given model by factoring over the visible units.

        :Info: Exponential increase of computations by the number of visible units. (16 usually ~ 20 seconds)

    :param model: The model.
    :type model: Valid RBM model.

    :param beta: Inverse temperature(s) for the models energy.
    :type beta: None, float, numpy array [batchsize,1]

    :param batchsize_exponent: 2^batchsize_exponent will be the batch size.
    :type batchsize_exponent: int

    :param status: If true prints the progress to the console.
    :type status: bool

    :return: Log Partition function for the model.
    :rtype: float
    """
    if status is True:
        print("Calculating the partition function by factoring over v: ")
        print('%3.2f%%' % 0.0)

    bit_length = model.input_dim
    if batchsize_exponent is 'AUTO' or batchsize_exponent > 20:
        batchsize_exponent = numx.min([model.input_dim, 12])
    batchsize = numx.power(2, batchsize_exponent)
    num_combinations = numx.power(2, bit_length)

    num_batches = num_combinations // batchsize
    log_prob_vv_all = numx.zeros(num_combinations)

    for batch in range(1, num_batches + 1):
        # Generate current batch
        bitcombinations = numxext.generate_binary_code(bit_length, batchsize_exponent, batch - 1)

        # calculate LL
        log_prob_vv_all[(batch - 1) * batchsize:batch * batchsize] = model.unnormalized_log_probability_v(
            bitcombinations, beta).reshape(bitcombinations.shape[0])
        # print status if wanted
        if status is True:
            print('%3.2f%%' % (100 * numx.double(batch) / numx.double(num_batches)))

    # return the log_sum of values
    return numxext.log_sum_exp(log_prob_vv_all)


def partition_function_factorize_h(model,
                                   beta=None,
                                   batchsize_exponent='AUTO',
                                   status=False):
    """ Computes the true partition function for the given model by factoring over the hidden units.

        :Info: Exponential increase of computations by the number of visible units. (16 usually ~ 20 seconds)

    :param model: The model.
    :type model: Valid RBM model.

    :param beta: Inverse temperature(s) for the models energy.
    :type beta: None, float, numpy array [batchsize,1]

    :param batchsize_exponent: 2^batchsize_exponent will be the batch size.
    :type batchsize_exponent: int

    :param status: If true prints the progress to the console.
    :type status: bool

    :return: Log Partition function for the model.
    :rtype: float
    """
    if status is True:
        print("Calculating the partition function by factoring over h: ")
        print('%3.2f%%' % 0.0)

    bit_length = model.output_dim
    if batchsize_exponent is 'AUTO' or batchsize_exponent > 20:
        batchsize_exponent = numx.min([model.output_dim, 12])
    batchsize = numx.power(2, batchsize_exponent)
    num_combinations = numx.power(2, bit_length)

    num_batches = num_combinations // batchsize
    log_prob_vv_all = numx.zeros(num_combinations)

    for batch in range(1, num_batches + 1):
        # Generate current batch
        bitcombinations = numxext.generate_binary_code(bit_length, batchsize_exponent, batch - 1)

        # calculate LL
        log_prob_vv_all[(batch - 1) * batchsize:batch * batchsize] = model.unnormalized_log_probability_h(
            bitcombinations, beta).reshape(bitcombinations.shape[0])

        # print status if wanted
        if status is True:
            print('%3.2f%%' % (100 * numx.double(batch) / numx.double(num_batches)))

    # return the log_sum of values
    return numxext.log_sum_exp(log_prob_vv_all)


def annealed_importance_sampling(model,
                                 num_chains=100,
                                 k=1,
                                 betas=10000,
                                 status=False):
    """ Approximates the partition function for the given model using annealed importance sampling.

    .. seealso:: Accurate and Conservative Estimates of MRF Log-likelihood using Reverse Annealing \
                 http://arxiv.org/pdf/1412.8566.pdf

    :param model: The model.
    :type model: Valid RBM model.

    :param num_chains: Number of AIS runs.
    :type num_chains: int

    :param k: Number of Gibbs sampling steps.
    :type k: int

    :param betas: Number or a list of inverse temperatures to sample from.
    :type betas: int, numpy array [num_betas]

    :param status: If true prints the progress on console.
    :type status: bool

    :return: | Mean estimated log partition function,
             | Mean +3std estimated log partition function,
             | Mean -3std estimated log partition function.
    :rtype: float
    """
    # Setup temerpatures if not given
    if numx.isscalar(betas):
        betas = numx.linspace(0.0, 1.0, betas)

    # Sample the first time from the base model
    v = model.probability_v_given_h(numx.zeros((num_chains, model.output_dim)), betas[0], True)
    v = model.sample_v(v, betas[0], True)

    # Calculate the unnormalized probabilties of v
    lnpvsum = -model.unnormalized_log_probability_v(v, betas[0], True)

    if status is True:
        t = 1
        print("Calculating the partition function using AIS: ")
        print('%3.2f%%' % 0.0)
        print('%3.2f%%' % (100.0 * numx.double(t) / numx.double(betas.shape[0])))

    for beta in betas[1:betas.shape[0] - 1]:

        if status is True:
            t += 1
            print('%3.2f%%' % (100.0 * numx.double(t) / numx.double(betas.shape[0])))
        # Calculate the unnormalized probabilties of v
        lnpvsum += model.unnormalized_log_probability_v(v, beta, True)

        # Sample k times from the intermidate distribution
        for _ in range(0, k):
            h = model.sample_h(model.probability_h_given_v(v, beta, True), beta, True)
            v = model.sample_v(model.probability_v_given_h(h, beta, True), beta, True)

        # Calculate the unnormalized probabilties of v
        lnpvsum -= model.unnormalized_log_probability_v(v, beta, True)

    # Calculate the unnormalized probabilties of v
    lnpvsum += model.unnormalized_log_probability_v(v, betas[betas.shape[0] - 1], True)

    lnpvsum = numx.longdouble(lnpvsum)

    # Calculate an estimate of logz .
    logz = numxext.log_sum_exp(lnpvsum) - numx.log(num_chains)

    # Calculate +/- 3 standard deviations
    lnpvmean = numx.mean(lnpvsum)
    lnpvstd = numx.log(numx.std(numx.exp(lnpvsum - lnpvmean))) + lnpvmean - numx.log(num_chains) / 2.0
    lnpvstd = numx.vstack((numx.log(3.0) + lnpvstd, logz))

    # Calculate partition function of base distribution
    baselogz = model._base_log_partition(True)

    # Add the base partition function
    logz = logz + baselogz
    logz_up = numxext.log_sum_exp(lnpvstd) + baselogz
    logz_down = numxext.log_diff_exp(lnpvstd) + baselogz

    if status is True:
        print('%3.2f%%' % 100.0)

    return logz, logz_up, logz_down


def reverse_annealed_importance_sampling(model,
                                         num_chains=100,
                                         k=1,
                                         betas=10000,
                                         status=False,
                                         data=None):
    """ Approximates the partition function for the given model using reverse annealed importance sampling.

    .. seealso:: Accurate and Conservative Estimates of MRF Log-likelihood using Reverse Annealing \
                 http://arxiv.org/pdf/1412.8566.pdf

    :param model: The model.
    :type model: Valid RBM model.

    :param num_chains: Number of AIS runs.
    :type num_chains: int

    :param k: Number of Gibbs sampling steps.
    :type k: int

    :param betas: Number or a list of inverse temperatures to sample from.
    :type betas: int, numpy array [num_betas]

    :param status: If true prints the progress on console.
    :type status: bool

    :param data: If data is given, initial sampling is started from data samples.
    :type data: numpy array

    :return: | Mean estimated log partition function,
             | Mean +3std estimated log partition function,
             | Mean -3std estimated log partition function.
    :rtype: float
    """
    # Setup temerpatures if not given
    if numx.isscalar(betas):
        betas = numx.linspace(0.0, 1.0, betas)

    if data is None:
        data = numx.zeros((num_chains, model.output_dim))
    else:
        data = model.sample_h(model.probability_h_given_v(numx.random.permutation(data)[0:num_chains]))

    # Sample the first time from the true model model
    v = model.probability_v_given_h(data, betas[betas.shape[0] - 1], True)
    v = model.sample_v(v, betas[betas.shape[0] - 1], True)

    # Calculate the unnormalized probabilties of v
    lnpvsum = model.unnormalized_log_probability_v(v, betas[betas.shape[0] - 1], True)

    # Setup temerpatures if not given
    if status is True:
        t = 1
        print("Calculating the partition function using AIS: ")
        print('%3.2f%%' % (0.0))
        print('%3.2f%%' % (100.0 * numx.double(t) / numx.double(betas.shape[0])))

    for beta in reversed(betas[1:betas.shape[0] - 1]):

        if status is True:
            t += 1
            print('%3.2f%%' % (100.0 * numx.double(t) / numx.double(betas.shape[0])))

        # Calculate the unnormalized probabilties of v
        lnpvsum -= model.unnormalized_log_probability_v(v, beta, True)

        # Sample k times from the intermidate distribution
        for _ in range(0, k):
            h = model.sample_h(model.probability_h_given_v(v, beta, True), beta, True)
            v = model.sample_v(model.probability_v_given_h(h, beta, True), beta, True)

        # Calculate the unnormalized probabilties of v
        lnpvsum += model.unnormalized_log_probability_v(v, beta, True)

    # Calculate the unnormalized probabilties of v
    lnpvsum -= model.unnormalized_log_probability_v(v, betas[0], True)

    lnpvsum = numx.longdouble(lnpvsum)

    # Calculate an estimate of logz .
    logz = numxext.log_sum_exp(lnpvsum) - numx.log(num_chains)

    # Calculate +/- 3 standard deviations
    lnpvmean = numx.mean(lnpvsum)
    lnpvstd = numx.log(numx.std(numx.exp(lnpvsum - lnpvmean))) + lnpvmean - numx.log(num_chains) / 2.0
    lnpvstd = numx.vstack((numx.log(3.0) + lnpvstd, logz))

    # Calculate partition function of base distribution
    baselogz = model._base_log_partition(True)

    # Add the base partition function
    logz = logz + baselogz
    logz_up = numxext.log_sum_exp(lnpvstd) + baselogz
    logz_down = numxext.log_diff_exp(lnpvstd) + baselogz

    if status is True:
        print('%3.2f%%' % 100.0)

    return logz, logz_up, logz_down
