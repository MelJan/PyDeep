''' Example using a small BB-RBMs on the MNIST handwritten digit database.

    :Version:
        1.0

    :Date:
        29.08.2016

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
import pydeep.rbm.model as MODEL
import pydeep.rbm.trainer as TRAINER
import pydeep.rbm.estimator as ESTIMATOR

import pydeep.misc.statistics as STATISTICS
import pydeep.misc.io as IO
import pydeep.misc.visualization as VISUALIZATION
import pydeep.misc.measuring as MEASURE

# Set random seed (optional)
numx.random.seed(42)

# Input and hidden dimensionality
v1 = v2 = 28
h1 = h2 = 4

# Load data , get it from 'deeplearning.net/data/mnist/mnist.pkl.gz'
train_data = IO.load_MNIST("../../../data/mnist.pkl.gz", True)[0]

# Training paramters
batch_size = 100
epochs = 30

# Create trainer and model
rbm = MODEL.BinaryBinaryRBM(number_visibles=v1 * v2,
                            number_hiddens=h1 * h2,
                            data=train_data)
trainer = TRAINER.PCD(rbm, batch_size)

# Measuring time
measurer = MEASURE.Stopwatch()

# Train model
print('Training')
print('Epoch\t\tRecon. Error\tLog likelihood \tExpected End-Time')
for epoch in range(1, epochs + 1):
    train_data = numx.random.permutation(train_data)
    for b in range(0, train_data.shape[0], batch_size):
        batch = train_data[b:b + batch_size, :]
        trainer.train(data=batch, epsilon=0.1)

    # Calculate Log-Likelihood, reconstruction error and expected end time every 10th epoch
    if epoch % 10 == 0:
        Z = ESTIMATOR.partition_function_factorize_h(rbm)
        LL = numx.mean(ESTIMATOR.log_likelihood_v(rbm, Z, train_data))
        RE = numx.mean(ESTIMATOR.reconstruction_error(rbm, train_data))
        print('{}\t\t{:.4f}\t\t\t{:.4f}\t\t\t{}'.format(epoch, RE, LL, measurer.get_expected_end_time(epoch, epochs)))
    else:
        print(epoch)

measurer.end()

# Print end time
print("End-time: \t{}".format(measurer.get_end_time()))
print("Training time:\t{}".format(measurer.get_interval()))

# Calculate and approximate partition function
Z = ESTIMATOR.partition_function_factorize_h(rbm, batchsize_exponent=h1, status=False)

print("True Partition: {} (LL: {})".format(Z, numx.mean(ESTIMATOR.log_likelihood_v(rbm, Z, train_data))))

# Reorder RBM features by average activity decreasingly
reordered_rbm = STATISTICS.reorder_filter_by_hidden_activation(rbm, train_data)
# Display RBM parameters
VISUALIZATION.imshow_standard_rbm_parameters(reordered_rbm, v1, v2, h1, h2)
# Sample some steps and show results
samples = STATISTICS.generate_samples(rbm, train_data[0:30], 30, 1, v1, v2, False, None)
VISUALIZATION.imshow_matrix(samples, 'Samples')

VISUALIZATION.show()
