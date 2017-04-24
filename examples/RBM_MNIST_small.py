''' Example using a small BB-RBMs on the MNIST handwritten digit database.

    :Version:
        1.1.0

    :Date:
        20.04.2017

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

'''
import numpy as numx
import pydeep.rbm.model as model
import pydeep.rbm.trainer as trainer
import pydeep.rbm.estimator as estimator

import pydeep.misc.io as io
import pydeep.misc.visualization as vis
import pydeep.misc.measuring as mea

# Set random seed (optional)
numx.random.seed(42)

# normal RBM
#update_offsets = 0.0
# centered RBM
update_offsets = 0.01

# Flipped/Inverse MNIST
#flipped = True
# Normal MNIST
flipped = False

# Input and hidden dimensionality
v1 = v2 = 28
h1 = h2 = 4

# Load data , get it from 'deeplearning.net/data/mnist/mnist.pkl.gz'
train_data = io.load_mnist("../../data/mnist.pkl.gz", True)[0]

# Flip the dataset if chosen
if flipped:
    train_data = 1-train_data

# Training paramters
batch_size = 100
epochs = 39

# Create centered or normal model
if update_offsets <= 0.0:
    rbm = model.BinaryBinaryRBM(number_visibles=v1 * v2,
                                number_hiddens=h1 * h2,
                                data=train_data,
                                initial_visible_offsets=0.0,
                                initial_hidden_offsets=0.0)
else:
    rbm = model.BinaryBinaryRBM(number_visibles=v1 * v2,
                                number_hiddens=h1 * h2,
                                data=train_data,
                                initial_visible_offsets='AUTO',
                                initial_hidden_offsets='AUTO')
# Create trainer
trainer = trainer.PCD(rbm, batch_size)

# Measuring time
measurer = mea.Stopwatch()

# Train model
print('Training')
print('Epoch\t\tRecon. Error\tLog likelihood \tExpected End-Time')
for epoch in range(1, epochs + 1):

    # Shuffle training samples (optional)
    train_data = numx.random.permutation(train_data)

    # Loop over all batches
    for b in range(0, train_data.shape[0], batch_size):
        batch = train_data[b:b + batch_size, :]
        trainer.train(data=batch,
                      epsilon=0.05,
                      update_visible_offsets=update_offsets,
                      update_hidden_offsets=update_offsets)

    # Calculate Log-Likelihood, reconstruction error and expected end time every 10th epoch
    if epoch % 10 == 0:
        logZ = estimator.partition_function_factorize_h(rbm)
        ll = numx.mean(estimator.log_likelihood_v(rbm, logZ, train_data))
        re = numx.mean(estimator.reconstruction_error(rbm, train_data))
        print('{}\t\t{:.4f}\t\t\t{:.4f}\t\t\t{}'.format(
            epoch, re, ll, measurer.get_expected_end_time(epoch, epochs)))
    else:
        print(epoch)

measurer.end()

# Print end/training time
print("End-time: \t{}".format(measurer.get_end_time()))
print("Training time:\t{}".format(measurer.get_interval()))

# Calculate true partition function
logZ = estimator.partition_function_factorize_h(rbm, batchsize_exponent=h1, status=False)
print("True Partition: {} (LL: {})".format(logZ, numx.mean(
    estimator.log_likelihood_v(rbm, logZ, train_data))))

# Approximate partition function by AIS (tends to overestimate)
logZ_approx_ = estimator.annealed_importance_sampling(rbm)[0]
print(
"AIS Partition: {} (LL: {})".format(logZ_approx_, numx.mean(
    estimator.log_likelihood_v(rbm, logZ_approx_, train_data))))

# Approximate partition function by reverse AIS (tends to underestimate)
logZ_approx_up = estimator.reverse_annealed_importance_sampling(rbm, data=train_data)[0]
print("reverse AIS Partition: {} (LL: {})".format(logZ_approx_up, numx.mean(
    estimator.log_likelihood_v(rbm, logZ_approx_up, train_data))))

# Reorder RBM features by average activity decreasingly
reordered_rbm = vis.reorder_filter_by_hidden_activation(rbm, train_data)

# Display RBM parameters
vis.imshow_standard_rbm_parameters(reordered_rbm, v1, v2, h1, h2)

# Sample some steps and show results
samples = vis.generate_samples(rbm, train_data[0:30], 30, 1, v1, v2, False, None)
vis.imshow_matrix(samples, 'Samples')

# Display results
vis.show()
