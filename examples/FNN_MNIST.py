''' Toy example using FNN on MNIST.

    :Version:
        3.0

    :Date
        25.05.2019

    :Author:
        Jan Melchior

    :Contact:
        pydeep@gmail.com

    :License:

        Copyright (C) 2019  Jan Melchior

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

import pydeep.fnn.model as MODEL
import pydeep.fnn.layer as LAYER
import pydeep.fnn.trainer as TRAINER
import pydeep.base.activationfunction as ACT
import pydeep.base.costfunction as COST
import pydeep.base.corruptor as CORR
import pydeep.misc.io as IO
import pydeep.base.numpyextension as npExt


# Set random seed (optional)
numx.random.seed(42)


# Load data and whiten it
train_data,train_label,valid_data, valid_label,test_data, test_label = IO.load_mnist("mnist.pkl.gz",False)
train_data = numx.vstack((train_data,valid_data))
train_label = numx.hstack((train_label,valid_label)).T
train_label = npExt.get_binary_label(train_label)
test_label = npExt.get_binary_label(test_label)

# Create model
l1 = LAYER.FullConnLayer(input_dim = train_data.shape[1],
                         output_dim = 1000,
                         activation_function=ACT.ExponentialLinear(),
                         initial_weights='AUTO',
                         initial_bias=0.0,
                         initial_offset=numx.mean(train_data,axis = 0).reshape(1,train_data.shape[1]),
                         connections=None,
                         dtype=numx.float64)
l2 = LAYER.FullConnLayer(input_dim = 1000,
                         output_dim = train_label.shape[1],
                         activation_function=ACT.SoftMax(),
                         initial_weights='AUTO',
                         initial_bias=0.0,
                         initial_offset=0.0,
                         connections=None,
                         dtype=numx.float64)
model = MODEL.Model([l1,l2])

# Choose an Optimizer
trainer = TRAINER.ADAGDTrainer(model)
#trainer = TRAINER.GDTrainer(model)

# Train model
max_epochs =20
batch_size = 20
eps = 0.1
print 'Training'
for epoch in range(1, max_epochs + 1):
    train_data, train_label = npExt.shuffle_dataset(train_data, train_label)
    for b in range(0, train_data.shape[0], batch_size):
        trainer.train(data=train_data[b:b + batch_size, :],
                      labels=[None,train_label[b:b + batch_size, :]],
                      costs = [None,COST.CrossEntropyError()],
                      reg_costs = [0.0,1.0],
                      #momentum=[0.0]*model.num_layers,
                      epsilon = [eps]*model.num_layers,
                      update_offsets = [0.0]*model.num_layers,
                      corruptor = [CORR.Dropout(0.2),CORR.Dropout(0.5),None],
                      reg_L1Norm = [0.0]*model.num_layers,
                      reg_L2Norm = [0.0]*model.num_layers,
                      reg_sparseness  = [0.0]*model.num_layers,
                      desired_sparseness = [0.0]*model.num_layers,
                      costs_sparseness = [None]*model.num_layers,
                      restrict_gradient = [0.0]*model.num_layers,
                      restriction_norm = 'Mat')
    print epoch,'\t',eps,'\t',
    print numx.mean(npExt.compare_index_of_max(model.forward_propagate(train_data),train_label)),'\t',
    print numx.mean(npExt.compare_index_of_max(model.forward_propagate(test_data), test_label))
