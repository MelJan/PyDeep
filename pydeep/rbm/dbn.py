""" Helper class for deep believe networks.

    :Version:
        1.1.0

    :Date:
        06.04.2017

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
from pydeep.base.basicstructure import StackOfBipartiteGraphs


class DBN(StackOfBipartiteGraphs):
    """ Deep believe network.
    """

    def __init__(self, list_of_rbms):
        """ Initializes the network with rbms.

        :param list_of_rbms: List of rbms.
        :type list_of_rbms: list
        """
        super(DBN, self).__init__(list_of_layers=list_of_rbms)

    def forward_propagate(self,
                          input_data,
                          sample=False):
        """ Propagates the data through the network.

        :param input_data: Input data
        :type input_data: numpy array [batchsize x input dim]

        :param sample: If true the states are sampled, otherwise the probabilities are used.
        :type sample: bool

        :return: Output of the network.
        :rtype: numpy array [batchsize x output dim]
        """
        if input_data.shape[1] != self.input_dim:
            raise Exception("Input dimensionality has to match dbn.input_dim!")
        self.states[0] = input_data
        for l in range(len(self._layers)):
            self.states[l + 1] = self._layers[l].probability_h_given_v(self.states[l])
            if sample is True:
                self.states[l + 1] = self._layers[l].sample_h(self.states[l + 1])
        return self.states[len(self._layers)]

    def backward_propagate(self,
                           output_data,
                           sample=False):
        """ Propagates the output back through the input.

        :param output_data: Output data.
        :type output_data: numpy array [batchsize x output dim]

        :param sample: If true the states are sampled, otherwise the probabilities are used.
        :type sample: bool

        :return: Input of the network.
        :rtype: numpy array [batchsize x input dim]
        """
        if output_data.shape[1] != self.output_dim:
            raise Exception("Output dimensionality has to match dbn.output_dim!")
        self.states[len(self._layers)] = output_data
        for l in range(len(self._layers), 0, -1):
            self.states[l - 1] = self._layers[l - 1].probability_v_given_h(self.states[l])
            if sample is True:
                self.states[l - 1] = self._layers[l - 1].sample_v(self.states[l - 1])
        return self.states[0]

    def reconstruct(self,
                    input_data,
                    sample=False):
        """ Reconstructs the data by propagating the data to the output and back to the input.

        :param input_data: Input data.
        :type input_data: numpy array [batchsize x input dim]

        :param sample: If true the states are sampled, otherwise the probabilities are used.
        :type sample: bool

        :return: Output of the network.
        :rtype: numpy array [batchsize x output dim]
        """
        return self.backward_propagate(self.forward_propagate(input_data, sample), sample)

    def sample_top_layer(self,
                         sampling_steps=100,
                         initial_state=None,
                         sample=True):
        """ Samples the top most layer, if initial_state is None the current state is used otherwise sampling is \
            started from the given initial state

        :param sampling_steps: Number of Sampling steps.
        :type sampling_steps: int

        :param initial_state: Output data
        :type initial_state: numpy array [batchsize x output dim]

        :param sample: If true the states are sampled, otherwise the probabilities are used (Mean field estimate).
        :type sample: bool

        :return: Output of the network.
        :rtype: numpy array [batchsize x output dim]
        """
        if initial_state is not None:
            self.states[len(self._layers)] = initial_state
        for s in range(sampling_steps):
            self.states[len(self._layers) - 1] = self._layers[len(self._layers) - 1].probability_v_given_h(
                self.states[len(self._layers)])
            if sample is True:
                self.states[len(self._layers) - 1] = self._layers[len(self._layers) - 1].sample_v(
                    self.states[len(self._layers) - 1])
            self.states[len(self._layers)] = self._layers[len(self._layers) - 1].probability_h_given_v(
                self.states[len(self._layers) - 1])
            if sample is True:
                self.states[len(self._layers)] = self._layers[len(self._layers) - 1].sample_h(
                    self.states[len(self._layers)])
        return self.states[len(self._layers)]

    def reconstruct_sample_top_layer(self,
                                     input_data,
                                     sampling_steps=100,
                                     sample_forward_backward=False):
        """ Reconstructs data by propagating the data forward, sampling the top most layer and propagating the result \
            backward.

        :param input_data: Input data
        :type input_data: numpy array [batchsize x input dim]

        :param sampling_steps: Number of Sampling steps.
        :type sampling_steps: int

        :param sample_forward_backward: If true the states for the forward and backward phase are sampled.
        :type sample_forward_backward: bool

        :return: reconstruction of the network.
        :rtype: numpy array [batchsize x output dim]
        """
        self.forward_propagate(input_data, sample_forward_backward)
        self.sample_top_layer(sampling_steps, None, True)
        self.backward_propagate(self.states[len(self.states) - 1], sample_forward_backward)
        return self.states[0]

