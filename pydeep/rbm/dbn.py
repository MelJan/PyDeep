""" Helper class for deep believe networks.

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

import pydeep.ae.sae as sae


class DBN(sae.SAE):
    """ Deep believe network.
    """

    def __init__(self, rbms):
        """ Constructore tales a list of RBMs.

        :param rbms: List of Restricted Boltzmann machines.
        :type rbms: list of RBMs
        """
        super(DBN, self).__init__(model_list=rbms)

    def forward_propagate(self, input_data, sample=False):
        """ Propagates the data through the network.

        :param input_data: Input data
        :type input_data: numpy array [batchsize x input dim]

        :param sample: If true the states are sampled, otherwise the probabilities are used.
        :type sample: bool

        :return: Output of the network.
        :rtype: numpy array [batchsize x output dim]
        """
        if input_data.shape[1] != self._input_dim:
            raise Exception("Input dimensionality has to match dbn.input_dim!")
        self._states[0] = input_data
        for l in range(len(self._layers)):
            self._states[l + 1] = self._layers[l].probability_h_given_v(self._states[l])
            if sample is True:
                self._states[l + 1] = self._layers[l].sample_h(self._states[l + 1])
        return self._states[len(self._layers)]

    def backward_propagate(self, output, sample=False):
        """ Propagates the output back through the input.

        :param output: Output data.
        :type output: numpy array [batchsize x output dim]

        :param sample: If true the states are sampled, otherwise the probabilities are used.
        :type sample: bool

        :return: Input of the network.
        :rtype: numpy array [batchsize x input dim]
        """
        if output.shape[1] != self._output_dim:
            raise Exception("Output dimensionality has to match dbn.output_dim!")
        self._states[len(self._layers)] = output
        for l in range(len(self._layers), 0, -1):
            self._states[l - 1] = self._layers[l - 1].probability_v_given_h(self._states[l])
            if sample is True:
                self._states[l - 1] = self._layers[l - 1].sample_v(self._states[l - 1])
        return self._states[0]

    def reconstruct(self, input_data, sample=True):
        """ Reconstructs the data by propagating the data to the output and back to the input.

        :param input_data: Input data.
        :type input_data: numpy array [batchsize x input dim]

        :param sample: If true the states are sampled, otherwise the probabilities are used.
        :type sample: bool

        :return: Output of the network.
        :rtype: numpy array [batchsize x output dim]
        """
        return self.backward_propagate(self.forward_propagate(input_data, sample), sample)

    def sample_top_layer(self, sampling_steps=100, output=None, sample=True):
        """ Reconstructs the data by propagating the data to the output and back to the input.

        :param sampling_steps: Number of Sampling steps.
        :type sampling_steps: int

        :param output: Output data
        :type output: numpy array [batchsize x output dim]

        :param sample: If true the states are sampled, otherwise the probabilities are used.
        :type sample: bool

        :return: Output of the network.
        :rtype: numpy array [batchsize x output dim]
        """
        if output is not None:
            self._states[len(self._layers)] = output
        for s in range(sampling_steps):
            self._states[len(self._layers) - 1] = self._layers[len(self._layers) - 1].probability_v_given_h(
                self._states[len(self._layers)])
            if sample is True:
                self._states[len(self._layers) - 1] = self._layers[len(self._layers) - 1].sample_v(
                    self._states[len(self._layers) - 1])
            self._states[len(self._layers)] = self._layers[len(self._layers) - 1].probability_h_given_v(
                self._states[len(self._layers) - 1])
            if sample is True:
                self._states[len(self._layers)] = self._layers[len(self._layers) - 1].sample_h(
                    self._states[len(self._layers)])
        return self._states[len(self._layers)]
