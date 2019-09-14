""" Example code for Hebbian descent. The default setting reproduces a single trial for Hebbian descent as shown in
    Figure 8 a) in https://arxiv.org/pdf/1905.10585.pdf.

    :Version:
        1.0.0

    :Date:
        14.09.2019

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
import numpy as np

import pydeep.fnn.layer as fnnlayer
import pydeep.fnn.model as fnnmodel
import pydeep.base.activationfunction as actFct
import pydeep.misc.visualization as vis


def train_Hebbian_descent_model(train_data,
                                train_label,
                                centered,
                                act,
                                epochs,
                                epsilon,
                                batch_size,
                                weightdecay):
    """ Performs a training trail for Hebbian descent and returns the model and absolut errors.

    :param train_data: Training data.
    :type train_data: numpy array

    :param train_label: Training label.
    :type train_label: numpy array

    :param centered: True if centering is used false otherwise
    :type centered: bool

    :param act: An activation function to be used
    :type act: pydeep.base.activationFunction object

    :param epochs: Numbe rof epochs to be used
    :type epochs: int

    :param epsilon: Learning rate to be used
    :type epsilon: float

    :param batch_size: batch2 siez to be used
    :type batch_size: int

    :param weightdecay: Weight decay to be used
    :type weightdecay: float

    :return: Results for gradient descent, hebbian descent
    :rtype: 4 numpy arrays
    """
    # Get input, ouput dims and num datapoints length
    input_dim = train_data.shape[1]
    output_dim = train_label.shape[1]
    num_pattern = train_data.shape[0]

    if train_data.shape[0] != train_label.shape[0]:
        raise Exception("Length of the input and output datasets must match")

    # If not centered set the offset parameters to zero
    mu = 0
    if centered:
        mu = np.mean(train_data, axis=0).reshape(1, input_dim)

    # Create a model
    model = fnnmodel.Model([fnnlayer.FullConnLayer(input_dim=input_dim,
                                                   output_dim=output_dim,
                                                   activation_function=act,
                                                   initial_weights='AUTO',
                                                   initial_bias=0.0,
                                                   initial_offset=mu,
                                                   connections=None,
                                                   dtype=np.float64)])

    # TLoop over epochs and datapoints
    for e in range(epochs):
        for b in range(0, num_pattern, batch_size):
            # Get the next data point
            input_point = np.copy(train_data[b:(b + batch_size), :].reshape(batch_size, train_data.shape[1]))
            output_point = np.copy(train_label[b:(b + batch_size), :].reshape(batch_size, train_label.shape[1]))

            # Calculate output
            z = np.dot(input_point - model.layers[0].offset, model.layers[0].weights) + \
                model.layers[0].bias
            h = model.layers[0].activation_function.f(z)

            # Calcualte difference
            deltas = (h - output_point)

            # Calculate updates
            update_b_new = np.sum(-deltas, axis=0)
            update_w_new = -np.dot((input_point - model.layers[0].offset).T, deltas)

            # Update model
            model.layers[0].weights += epsilon / batch_size * update_w_new - weightdecay * model.layers[0].weights
            model.layers[0].bias += epsilon / batch_size * update_b_new

    # Calculate mean absolute deviation
    err_train = np.abs(model.forward_propagate(train_data) - train_label)

    # Return model and errors
    return model, np.mean(err_train, axis=1)

# Set the random seed value
np.random.seed(42)

# Choose data and model size
num_patterns = 100
data_dim = 200

# Get dataset
train_data = np.float64(np.random.random_integers(0, 1, (num_patterns, data_dim)))
train_label = np.float64(np.random.random_integers(0, 1, (num_patterns, data_dim)))

# Choose activation function
act = actFct.Sigmoid()

# Train model, this performs a single trial as shown in Figure 8 a) in the Hebbina descent paper
mymodel, results = train_Hebbian_descent_model(train_data=train_data,
                                               train_label=train_label,
                                               centered=True,
                                               act=actFct.Sigmoid(),
                                               epochs=1,
                                               epsilon=0.2,
                                               batch_size=1,
                                               weightdecay=0.0)

# Display errors
vis.xlabel("Pattern index (" + str(num_patterns) + " = latest pattern)")
vis.ylabel("Mean Absolute Error")

# Plot results and std range
vis.plot(np.arange(0, num_patterns), results, linestyle="-", color='green', linewidth=0, marker="o", markersize=4)
vis.legend(['Hebbian descent'], loc=0)
vis.ylim(0, 1)
vis.show()
