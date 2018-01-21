""" Helper class for stacked auto encoder networks.

    :Version:
        1.1.0

    :Date:
        21.01.2018

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2018 Jan Melchior

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


class SAE(StackOfBipartiteGraphs):
    """ Stack of auto encoders.
    """

    def __init__(self, list_of_autoencoders):
        """ Initializes the network with auto encoders.

        :param list_of_autoencoders: List of auto-encoders
        :type list_of_autoencoders: list
        """
        super(SAE, self).__init__(list_of_layers=list_of_autoencoders)

    def forward_propagate(self, input_data):
        """ Propagates the data through the network.

        :param input_data: Input data.
        :type input_data: numpy array [batchsize x input dim]

        :return: Output of the network.
        :rtype: numpy array [batchsize x output dim]
        """
        if input_data.shape[1] != self.input_dim:
            raise Exception("Input dimensionality has to match dbn.input_dim!")
        self.states[0] = input_data
        for l in range(len(self._layers)):
            self.states[l + 1] = self._layers[l].encode(self.states[l])
        return self.states[len(self._layers)]

    def backward_propagate(self, output_data):
        """ Propagates the output back through the input.

        :param output_data: Output data.
        :type output_data: numpy array [batchsize x output dim]

        :return:  Input of the network.
        :rtype: numpy array [batchsize x input dim]
        """
        if output_data.shape[1] != self.output_dim:
            raise Exception("Output dimensionality has to match dbn.output_dim!")
        self.states[len(self._layers)] = output_data
        for l in range(len(self._layers), 0, -1):
            self.states[l - 1] = self._layers[l - 1].decode(self.states[l])
        return self.states[0]
