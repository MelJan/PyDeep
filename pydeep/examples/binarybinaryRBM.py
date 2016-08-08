''' Example using BB-RBMs on the MNIST handwritten digit database.

    :Version:
        1.0

    :Date:
        08.08.2016

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

# Model Parameters
h1 = 4
h2 = 4
v1 = 28
v2 = 28

# Load data and whiten it
train_data = IO.load_MNIST("../../../data/mnist.pkl.gz",True)[0]

# Training paramters
batch_size = 100
epochs = 10
k = 1
eps = 0.01
mom = 0.9
decay = 0.0
update_visible_mean = 0.0
update_hidden_mean = 0.01

# Create trainer and model
rbm = MODEL.BinaryBinaryRBM(number_visibles = v1*v2,
                            number_hiddens = h1*h2, 
                            data=train_data,
                            initial_weights='AUTO', 
                            initial_visible_bias='AUTO',
                            initial_hidden_bias='AUTO',
                            initial_visible_offsets='AUTO',
                            initial_hidden_offsets='AUTO')
trainer = TRAINER.PCD(rbm,batch_size)
measurer = MEASURE.Stopwatch()

# Train model
print 'Training'
print 'Epoch\tRecon. Error\tLog likelihood \tExpected End-Time'
for epoch in range(0,epochs) :
    train_data = numx.random.permutation(train_data)
    for b in range(0,train_data.shape[0],batch_size):
        batch = train_data[b:b+batch_size,:]
        trainer.train(data = batch,
                      num_epochs=1, 
                      epsilon=eps, 
                      k=k, 
                      momentum=mom,
                      update_visible_offsets=update_visible_mean, 
                      update_hidden_offsets=update_hidden_mean)

    print epoch
    if(epoch == 5):
        mom = 0.0
        
    # Calculate Log-Likelihood every 10th epoch
    if(epoch % 5 == 0):
        Z = ESTIMATOR.partition_function_factorize_h(rbm, 
                                                     batchsize_exponent=h1, 
                                                     status = False)
        LL = numx.mean(ESTIMATOR.log_likelihood_v(rbm,Z, train_data))
        RE = numx.mean(ESTIMATOR.reconstruction_error(rbm, train_data))
        print '%d\t\t%8.6f\t%8.4f\t' % (epoch, RE, LL),
        print measurer.get_expected_end_time(epoch+1, epochs),
        print

measurer.end()

# Plot Likelihood and partition function calculate with different methods
print
print 'End-time: \t', measurer.get_end_time()
print 'Training time:\t', measurer.get_interval()

# Calculate and approximate partition function
Z = ESTIMATOR.partition_function_factorize_h(rbm, batchsize_exponent=h1, status= False)

print ""
print "\n True Partition: ", Z," (True         LL: ",
print numx.mean(ESTIMATOR.log_likelihood_v(rbm, Z , train_data)), ")"


# Prepare results
rbmReordered = STATISTICS.reorder_filter_by_hidden_activation(rbm, train_data)
VISUALIZATION.imshow_standard_rbm_parameters(rbmReordered, v1,v2,h1, h2)
samples = STATISTICS.generate_samples(rbm, train_data[0:30], 30, 1, v1, v2, False, None)
VISUALIZATION.imshow_matrix(samples,'Samples')

VISUALIZATION.show()




