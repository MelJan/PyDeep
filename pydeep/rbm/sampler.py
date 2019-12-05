""" This module provides different sampling algorithms for RBMs running on CPU.
    The structure is kept modular to simplify the understanding of the code and
    the mathematics. In addition the modularity helps to create other kind of
    sampling algorithms by inheritance.

    :Implemented:
        - Gibbs Sampling
        - Persistent Gibbs Sampling
        - Parallel Tempering Sampling
        - Independent Parallel Tempering Sampling

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

class GibbsSampler(object):
    """ Implementation of k-step Gibbs-sampling for bipartite graphs.

    """

    def __init__(self, model):
        """ Initializes the sampler with the model.

        :param model: The model to sample from.
        :type model: Valid model class like BinaryBinary-RBM.
        """
        # Set the model
        if not hasattr(model, 'probability_h_given_v'):
            raise ValueError("The model needs to implement the function probability_h_given_v!")
        if not hasattr(model, 'probability_v_given_h'):
            raise ValueError("The model needs to implement the function probability_v_given_h!")
        if not hasattr(model, 'sample_h'):
            raise ValueError("The model needs to implement the function sample_h!")
        if not hasattr(model, 'sample_v'):
            raise ValueError("The model needs to implement the function sample_v!")
        self.model = model

    def sample(self,
               vis_states,
               k=1,
               betas=None,
               ret_states=True):
        """ Performs k steps Gibbs-sampling starting from given visible data.

        :param vis_states: The initial visible states to sample from.
        :type vis_states: numpy array [num samples, input dimension]

        :param k: The number of Gibbs sampling steps.
        :type k: int

        :param betas: Inverse temperature to sample from.(energy based models)
        :type betas: None, float, numpy array [num_betas,1]

        :param ret_states: If False returns the visible probabilities instead of the states.
        :type ret_states: bool

        :return: The visible samples of the Markov chains.
        :rtype: numpy array [num samples, input dimension]
        """
        # Sample hidden states
        hid = self.model.probability_h_given_v(vis_states, betas)
        hid = self.model.sample_h(hid, betas)

        # sample further
        return self.sample_from_h(hid, k, betas, ret_states)

    def sample_from_h(self,
                      hid_states,
                      k=1,
                      betas=None,
                      ret_states=True):
        """ Performs k steps Gibbs-sampling starting from given hidden states.

        :param hid_states: The initial hidden states to sample from.
        :type hid_states: numpy array [num samples, output dimension]

        :param k: The number of Gibbs sampling steps.
        :type k: int

        :param betas: Inverse temperature to sample from.
        :type betas: (energy based models)

        :param ret_states: If False returns the visible probabilities instead of the states.
        :type ret_states: bool

        :return: The visible samples of the Markov chains.
        :rtype: numpy array [num samples, input dimension]
        """
        # Sample k times
        vis = self.model.probability_v_given_h(hid_states, betas)
        for _ in range(k - 1):
            vis = self.model.sample_v(vis, betas)
            hid = self.model.probability_h_given_v(vis, betas)
            hid = self.model.sample_h(hid, betas)
            vis = self.model.probability_v_given_h(hid, betas)

            # Return states or probs
        if ret_states:
            return self.model.sample_v(vis, betas)
        else:
            return vis


class PersistentGibbsSampler(object):
    """ Implementation of k-step persistent Gibbs sampling.

    """

    def __init__(self, model, num_chains):
        """ Initializes the sampler with the model.

        :param model: The model to sample from.
        :type model: Valid model class.

        :param num_chains: The number of Markov chains.
                           .. Note:: Optimal performance is achieved if the number of
                           samples and the number of chains equal the batch_size.
        :type num_chains: int
        """
        # Check and set the model
        if not hasattr(model, 'probability_h_given_v'):
            raise ValueError("The model needs to implement the function probability_h_given_v!")
        if not hasattr(model, 'probability_v_given_h'):
            raise ValueError("The model needs to implement the function probability_v_given_h!")
        if not hasattr(model, 'sample_h'):
            raise ValueError("The model needs to implement the function sample_h!")
        if not hasattr(model, 'sample_v'):
            raise ValueError("The model needs to implement the function sample_v!")
        if not hasattr(model, 'input_dim'):
            raise ValueError("The model needs to implement the parameter input_dim!")
        self.model = model

        # Initialize persistent Markov chains to Gaussian random samples.
        if numx.isscalar(num_chains):
            self.chains = model.sample_v(numx.random.randn(num_chains, model.input_dim) * 0.01)
        else:
            raise ValueError("Number of chains needs to be an integer or None.")

    def sample(self,
               num_samples,
               k=1,
               betas=None,
               ret_states=True):
        """ Performs k steps persistent Gibbs-sampling.

        :param num_samples: The number of samples to generate.
                            .. Note:: Optimal performance is achieved if the number of samples and the number of chains
                            equal the batch_size.
        :type num_samples: int, numpy array

        :param k: The number of Gibbs sampling steps.
        :type k: int

        :param betas: Inverse temperature to sample from.(energy based models)
        :type betas: None, float, numpy array [num_betas,1]

        :param ret_states: If False returns the visible probabilities instead of the states.
        :type ret_states: bool

        :return: The visible samples of the Markov chains.
        :rtype: numpy array [num samples, input dimension]
        """
        # Sample k times
        for _ in range(k):
            hid = self.model.probability_h_given_v(self.chains, betas)
            hid = self.model.sample_h(hid, betas)
            vis = self.model.probability_v_given_h(hid, betas)
            self.chains = self.model.sample_v(vis, betas)
        if ret_states:
            samples = self.chains
        else:
            samples = vis

        if num_samples == self.chains.shape[0]:
            return samples
        else:
            # If more samples than chains,
            repeats = numx.int32(num_samples / self.chains.shape[0])

            for _ in range(repeats):

                # Sample k times
                for u in range(k):
                    hid = self.model.probability_h_given_v(self.chains, betas)
                    hid = self.model.sample_h(hid, betas)
                    vis = self.model.probability_v_given_h(hid, betas)
                    self.chains = self.model.sample_v(vis, betas)
                if ret_states:
                    samples = numx.vstack([samples, self.chains])
                else:
                    samples = numx.vstack([samples, vis])
            return samples[0:num_samples, :]


class ParallelTemperingSampler(object):
    """ Implementation of k-step parallel tempering sampling.

    """

    def __init__(self,
                 model,
                 num_chains=3,
                 betas=None):
        """ Initializes the sampler with the model.

        :param model: The model to sample from.
        :type model: Valid model Class.

        :param num_chains: The number of Markov chains.
        :type num_chains: int

        :param betas: Array of inverse temperatures to sample from, its dimensionality needs to equal the number of \
                      chains or if None is given the inverse temperatures are initialized linearly from 0.0 to 1.0 in \
                      'num_chains' steps.
        :type betas: int, None
        """
        # Check and set the model
        if not hasattr(model, 'probability_h_given_v'):
            raise ValueError("The model needs to implement the function probability_h_given_v!")
        if not hasattr(model, 'probability_v_given_h'):
            raise ValueError("The model needs to implement the function probability_v_given_h!")
        if not hasattr(model, 'sample_h'):
            raise ValueError("The model needs to implement the function sample_h!")
        if not hasattr(model, 'sample_v'):
            raise ValueError("The model needs to implement the function sample_v!")
        if not hasattr(model, 'energy'):
            raise ValueError("The model needs to implement the function energy!")
        if not hasattr(model, 'input_dim'):
            raise ValueError("The model needs to implement the parameter input_dim!")

        self.model = model

        # Initialize persistent Markov chains to Gaussian random samples.
        if numx.isscalar(num_chains):
            self.chains = model.sample_v(numx.random.randn(num_chains, model.input_dim) * 0.01)

            # Sets the beta values
        if betas is None:
            self.betas = numx.linspace(0.0, 1.0, num_chains).reshape(num_chains, 1)
        else:
            self.betas = betas.reshape(numx.array(betas).shape[0], 1)
            if self.betas.shape[0] != num_chains:
                raise ValueError("The number of betas and Markov chains must be equivalent!")

    def sample(self,
               num_samples,
               k=1,
               ret_states=True):
        """ Performs k steps parallel tempering sampling.

        :param num_samples: The number of samples to generate.
                            .. Note:: Optimal performance is achieved if the number of samples and the number of chains \
                            equal the batch_size.
        :type num_samples: int, numpy array

        :param k: The number of Gibbs sampling steps.
        :type k: int

        :param ret_states: If False returns the visible probabilities instead of the states.
        :type ret_states: bool

        :return: The visible samples of the Markov chains.
        :rtype: numpy array [num samples, input dimension]
        """
        # Initialize persistent Markov chains to Gaussian random samples.
        samples = numx.empty((num_samples, self.model.input_dim))

        # Generate a sample for each given data sample
        for b in range(0, num_samples):

            # Perform k steps of Gibbs sampling
            hid = self.model.probability_h_given_v(self.chains, self.betas)
            hid = self.model.sample_h(hid, self.betas)
            for _ in range(k - 1):
                vis = self.model.probability_v_given_h(hid, self.betas)
                vis = self.model.sample_v(vis, self.betas)
                hid = self.model.probability_h_given_v(vis, self.betas)
                hid = self.model.sample_h(hid, self.betas)
            self.chains = self.model.probability_v_given_h(hid, self.betas)

            # Use states for calculations
            if ret_states:
                self.chains = self.model.sample_v(self.chains, self.betas)

                # Calculate the energies for the samples and their hidden activity
            self._swap_chains(self.chains, hid, self.model, self.betas)

            # Take sample from inverse temperature 1.0
            samples[b, :] = numx.copy(self.chains[self.betas.shape[0] - 1, :])

            # If we used probs for calculations set the chains to states now
            if not ret_states:
                self.chains = self.model.sample_v(self.chains, self.betas)

        return samples

    @classmethod
    def _swap_chains(cls,
                     chains,
                     hid_states,
                     model,
                     betas):
        """ Swaps the samples between the Markov chains according to the Metropolis Hastings Ratio.

        :param chains: Chains with visible data.
        :type chains: [num samples, input dimension]

        :param hid_states: Hidden states.
        :type hid_states: [num samples, output dimension]

        :param model: The model to sample from.
        :type model: Valid RBM Class.

        :param betas: Array of inverse temperatures to sample from, its dimensionality needs to equal the number of \
                      chains or if None is given the inverse temperatures are initialized linearly from 0.0 to 1.0 in \
                      'num_chains' steps.
        :type betas: int, None
        """
        # If we have a binary binary RBM the swap calculation gets a bit efficient
        # This always works if the energy is scaled by a factor that can be pulled out
        # beta*E(x,h), which is not the case for Gaussian binary RBM since it would
        # lead to infinite variance.
        if model._fast_PT:
            energies = model.energy(chains, hid_states)

            # even neighbor swapping
            for t in range(0, betas.shape[0] - 1, 2):

                # Calculate swap probability
                pswap = numx.exp((energies[t + 1, 0] - energies[t, 0])
                                 * (betas[t + 1, 0] - betas[t, 0]))

                # Probability higher then a random number
                if pswap > numx.random.rand():
                    # swap sample neighbors using advance indexing
                    chains[[t, t + 1], :] = chains[[t + 1, t], :]
                    energies[[t, t + 1], :] = energies[[t + 1, t], :]
                    hid_states[[t, t + 1], :] = hid_states[[t + 1, t], :]

            # odd neighbor swapping
            for t in range(1, betas.shape[0] - 1, 2):

                # Calculate swap probability
                pswap = numx.exp((energies[t + 1, 0] - energies[t, 0]) * (betas[t + 1, 0] - betas[t, 0]))

                # Probability higher then a random number
                if pswap > numx.random.rand():
                    # swap sample neighbors using advance indexing
                    chains[[t, t + 1], :] = chains[[t + 1, t], :]
                    hid_states[[t, t + 1], :] = hid_states[[t + 1, t], :]
        else:
            energies = model.energy(chains, hid_states, betas)

            chains_swap = numx.copy(chains)
            hid_states_swap = numx.copy(hid_states)
            for t in range(0, betas.shape[0] - 1, 2):
                chains_swap[[t, t + 1], :] = chains_swap[[t + 1, t], :]
                hid_states_swap[[t, t + 1], :] = hid_states_swap[[t + 1, t], :]
            energies_swap = model.energy(chains_swap, hid_states_swap, betas)

            # even neighbor swapping
            for t in range(0, betas.shape[0] - 1, 2):

                # Calculate swap probability
                pswap = numx.exp((energies[t + 1, 0] - energies_swap[t, 0] - energies_swap[t + 1, 0] + energies[t, 0]))
                # Probability higher then a random number
                if pswap > numx.random.rand():
                    # swap sample neighbors using advance indexing
                    chains[[t, t + 1], :] = chains[[t + 1, t], :]
                    hid_states[[t, t + 1], :] = hid_states[[t + 1, t], :]

            energies = model.energy(chains, hid_states, betas)
            chains_swap = numx.copy(chains)
            hid_states_swap = numx.copy(hid_states)
            for t in range(1, betas.shape[0] - 1, 2):
                chains_swap[[t, t + 1], :] = chains_swap[[t + 1, t], :]
                hid_states_swap[[t, t + 1], :] = hid_states_swap[[t + 1, t], :]
            energies_swap = model.energy(chains_swap, hid_states_swap, betas)

            # odd neighbor swapping
            for t in range(1, betas.shape[0] - 1, 2):

                pswap = numx.exp((energies[t + 1, 0] - energies_swap[t, 0] - energies_swap[t + 1, 0] + energies[t, 0]))
                # Probability higher then a random number
                if pswap > numx.random.rand():
                    # swap sample neighbors using advance indexing
                    chains[[t, t + 1], :] = chains[[t + 1, t], :]
                    hid_states[[t, t + 1], :] = hid_states[[t + 1, t], :]


class IndependentParallelTemperingSampler(object):
    """ Implementation of k-step independent parallel tempering sampling. IPT runs an PT instance for each sample in \
        parallel. This speeds up the sampling but also decreases the mixing rate.

    """

    def __init__(self,
                 model,
                 num_samples,
                 num_chains=3,
                 betas=None):
        """ Initializes the sampler with the model.

        :param model: The model to sample from.
        :type model: Valid model Class.

        :param num_samples: The number of samples to generate.
                            .. Note:: Optimal performance (ATLAS,MKL) is achieved if the number of samples equals the \
                            batchsize.
        :type num_samples:

        :param num_chains: The number of Markov chains.
        :type num_chains: int

        :param betas: Array of inverse temperatures to sample from, its dimensionality needs to equal the number of \
                      chains or if None is given the inverse temperatures are initialized linearly from 0.0 to 1.0 \
                      in 'num_chains' steps.
        :type betas: int, None
        """
        if not model._fast_PT:
            raise  NotImplementedError("Only more efficient for Binary RBMs")

        # Check and set the model
        if not hasattr(model, 'probability_h_given_v'):
            raise ValueError("The model needs to implement the function probability_h_given_v!")
        if not hasattr(model, 'probability_v_given_h'):
            raise ValueError("The model needs to implement the function probability_v_given_h!")
        if not hasattr(model, 'sample_h'):
            raise ValueError("The model needs to implement the function sample_h!")
        if not hasattr(model, 'sample_v'):
            raise ValueError("The model needs to implement the function sample_v!")
        if not hasattr(model, 'energy'):
            raise ValueError("The model needs to implement the function energy!")
        if not hasattr(model, 'input_dim'):
            raise ValueError("The model needs to implement the parameter input_dim!")
        self.model = model

        # Initialize persistent Markov chains to Gaussian random samples.
        self.num_samples = num_samples
        self.chains = model.sample_v(numx.random.randn(num_chains * self.num_samples, model.input_dim) * 0.01)

        # Sets the beta values
        self.num_betas = num_chains
        if betas is None:
            self.betas = numx.linspace(0.0, 1.0, num_chains).reshape(num_chains, 1)
        else:
            self.betas = self.betas.reshape(numx.array(betas).shape[0], 1)
            if self.betas.shape[0] != num_chains:
                raise ValueError("The number of betas and Markov chains must be equivalent!")

        # Repeat betas batchsize times
        self.betas = numx.tile(self.betas.T, self.num_samples).T.reshape(num_chains * self.num_samples, 1)

        # Indices of the chains on temperature beta = 1.0
        self.select_indices = numx.arange(self.num_betas - 1, self.num_samples * self.num_betas, self.num_betas)

    def sample(self,
               num_samples='AUTO',
               k=1,
               ret_states=True):
        """ Performs k steps independent parallel tempering sampling.

        :param num_samples: The number of samples to generate.
                            .. Note:: Optimal performance is achieved if the number of samples and the number of chains \
                            equal the batch_size. -> AUTO
        :type num_samples: int or 'AUTO'

        :param k: The number of Gibbs sampling steps.
        :type k: int

        :param ret_states: If False returns the visible probabilities instead of the states.
        :type ret_states: bool

        :return: The visible samples of the Markov chains.
        :rtype: numpy array [num samples, input dimension]
        """
        if not numx.isscalar(num_samples):
            num_samples = num_samples.shape[0]

        # Perform k steps of Gibbs sampling
        hid = self.model.probability_h_given_v(self.chains, self.betas)
        hid = self.model.sample_h(hid, self.betas)
        for _ in range(k - 1):
            vis = self.model.probability_v_given_h(hid, self.betas)
            vis = self.model.sample_v(vis, self.betas)
            hid = self.model.probability_h_given_v(vis, self.betas)
            hid = self.model.sample_h(hid, self.betas)
        self.chains = self.model.probability_v_given_h(hid, self.betas)

        # Use the states
        if ret_states:
            self.chains = self.model.sample_v(self.chains, self.betas)

            # Calculate the energies for the samples and their hidden activity
        self._swap_chains(self.chains, self.num_samples, hid, self.model, self.betas)

        samples = self.chains[self.select_indices]

        # If we return probs set the chains to states
        if not ret_states:
            self.chains = self.model.sample_v(self.chains, self.betas)

        repeats = numx.int32(num_samples / self.select_indices.shape[0])
        for _ in range(repeats):
            # Perform k steps of Gibbs sampling
            hid = self.model.probability_h_given_v(self.chains, self.betas)
            hid = self.model.sample_h(hid, self.betas)
            for u in range(k - 1):
                vis = self.model.probability_v_given_h(hid, self.betas)
                vis = self.model.sample_v(vis, self.betas)
                hid = self.model.probability_h_given_v(vis, self.betas)
                hid = self.model.sample_h(hid, self.betas)
            self.chains = self.model.probability_v_given_h(hid, self.betas)

            # Use the states
            if ret_states:
                self.chains = self.model.sample_v(self.chains, self.betas)

                # Calculate the energies for the samples and their hidden activity
            self._swap_chains(self.chains, self.num_samples, hid, self.model, self.betas)

            samples = numx.vstack([samples, self.chains[self.select_indices]])

            # If we return probs set the chains to states
            if not ret_states:
                self.chains = self.model.sample_v(self.chains, self.betas)

        return samples[0:num_samples, :]

    @classmethod
    def _swap_chains(cls,
                     chains,
                     num_chains,
                     hid_states,
                     model,
                     betas):
        """ Swaps the samples between the Markov chains according to the Metropolis Hastings Ratio.

        :param chains: Chains with visible data.
        :type chains: [num samples*num_chains, input dimension]

        :param hid_states: Hidden states.
        :type hid_states: [num samples*num_chains, output dimension]

        :param model: The model to sample from.
        :type model: Valid RBM Class.

        :param betas: Array of inverse temperatures to sample from, its dimensionality needs to equal the number of \
                      chains or if None is given the inverse temperatures are initialized linearly from 0.0 to 1.0 in \
                      'num_chains' steps.
        :type betas: int, None
        """
        # Calculate the energies for the samples and their hidden activity
        energies = model.energy(chains, hid_states)
        num_betas = chains.shape[0] // num_chains

        # for each batch
        for m in range(0, num_chains, 1):

            # for each temperature even
            for b in range(0, num_betas - 1, 2):

                t = m * num_betas + b

                # Calculate swap probability
                pswap = numx.exp((energies[t + 1, 0] - energies[t, 0])
                                 * (betas[t + 1, 0] - betas[t, 0]))

                # Probability higher then a random number
                if pswap > numx.random.rand():
                    # swap sample neighbors using advanced indexing
                    chains[[t, t + 1], :] = chains[[t + 1, t], :]
                    energies[[t, t + 1], :] = energies[[t + 1, t], :]
                    hid_states[[t, t + 1], :] = hid_states[[t + 1, t], :]

            # for each temperature odd
            for b in range(1, num_betas - 1, 2):

                t = m * num_betas + b

                # Calculate swap probability
                pswap = numx.exp((energies[t + 1, 0] - energies[t, 0]) * (betas[t + 1, 0] - betas[t, 0]))

                # Probability higher then a random number
                if pswap > numx.random.rand():
                    # swap sample neighbors using advanced indexing
                    chains[[t, t + 1], :] = chains[[t + 1, t], :]
                    energies[[t, t + 1], :] = energies[[t + 1, t], :]
                    hid_states[[t, t + 1], :] = hid_states[[t + 1, t], :]
