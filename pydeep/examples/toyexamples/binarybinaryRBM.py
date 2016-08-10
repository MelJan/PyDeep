''' Toy example using BB-RBMs on Bars and Stripes.

    :Version:
        1.0

    :Date:
        10.08.2016

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
import pydeep.rbm.model as MODEL
import pydeep.rbm.trainer as TRAINER
import pydeep.rbm.estimator as ESTIMATOR

import pydeep.misc.toyproblems as TOY_DATA
import pydeep.misc.visualization as VISUALIZATION
import pydeep.misc.statistics as STATISTICS
import numpy as numx

# Set random seed (optional)
numx.random.seed(42)

h1 = 2
h2 = 2
v1 = 2
v2 = 2

# Load data
data = TOY_DATA.generate_bars_and_stripes_complete(v1)
data = numx.vstack((data[0],data,data[5]))
#Create model
rbm = MODEL.BinaryBinaryRBM(number_visibles = v1*v2, 
                            number_hiddens = h1*h2, 
                            data=data, 
                            initial_weights='AUTO', 
                            initial_visible_bias='AUTO', 
                            initial_hidden_bias='AUTO', 
                            initial_visible_offsets='AUTO', 
                            initial_hidden_offsets='AUTO',
                            dtype=numx.float64)

# Setup trainer
batch_size = 8
trainer = TRAINER.PCD(rbm,8)
max_epochs = 10000
k = 5
epsilon = 0.1
momentum = 0.0
weight_decay = 0
update_visible_offsets=0.0
update_hidden_offsets=0.01
desired_sparseness=None
restrict_gradient = None
use_hidden_states = False
use_centered_gradient = False

#Train model
print 'Training'
print 'Epoch\tRE train \tLL train '
for epoch in range(1,max_epochs+1) :
    # Shuffle data points
    trainer.train(data=data,
                  num_epochs=1, 
                  epsilon=epsilon, 
                  k=k, 
                  momentum=momentum, 
                  regL1Norm=0.0, 
                  regL2Norm=0.0, 
                  desired_sparseness=desired_sparseness, 
                  update_visible_offsets=update_visible_offsets, 
                  update_hidden_offsets=update_hidden_offsets, 
                  restrict_gradient=restrict_gradient, 
                  restriction_norm='Cols', 
                  use_hidden_states=use_hidden_states,
                  use_centered_gradient=use_centered_gradient)
    # Decay learning rate
    epsilon *= 0.99975
    #print epsilon
    if epoch % 1000 == 0:
        # Calculate and print Log likelihood and reconstruction error
        RE = numx.mean(ESTIMATOR.reconstruction_error(rbm, data))
        Z = ESTIMATOR.partition_function_factorize_v(rbm,batchsize_exponent=v1*v2)
        LL = numx.mean(ESTIMATOR.log_likelihood_v(rbm, Z , data))
        print '%5d \t%0.5f \t%0.5f' % (epoch, RE, LL)

# Calculate partition function
Z = ESTIMATOR.partition_function_factorize_v(rbm, batchsize_exponent=v1*v2)

# Calculate and print LL
print ""
print "\n True Partition: ", Z," (True         LL: ",
print numx.mean(ESTIMATOR.log_likelihood_v(rbm, Z , data)), ")"
print "Best LL possible: ", ((data.shape[0]-4)*numx.log(1.0/data.shape[0])+4*numx.log(2.0/data.shape[0]))/data.shape[0]

# Display results
rbmReordered = STATISTICS.reorder_filter_by_hidden_activation(rbm, data)
VISUALIZATION.imshow_standard_rbm_parameters(rbmReordered, v1,v2,h1, h2)
samples = STATISTICS.generate_samples(rbm, data[0:8], 30, 1, v1, v2, False, None)
VISUALIZATION.imshow_matrix(samples,'Samples')

VISUALIZATION.show()
