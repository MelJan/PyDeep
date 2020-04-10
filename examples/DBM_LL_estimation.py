""" This module contains example code for the AIS and exact LL estimation of an 3 layer binary DBM.

    :Version:
        1.0.0

    :Date:
        02.009.2019

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
import numpy as numx
import pydeep.dbm.binary3Layer.model as MODEL
import pydeep.dbm.binary3Layer.trainer as TRAINER
import pydeep.dbm.binary3Layer.estimator as ESTIMATOR
from pydeep.base.activationfunction import Sigmoid
import pydeep.misc.toyproblems as TOY
import pydeep.misc.visualization as VIS

# Set the same seed value for all algorithms
numx.random.seed(42)

# Set dimensions
v11 = v12 = 2
v21 = v22 = 4
v31 = v32 = 2

# Generate data
train_set = TOY.generate_bars_and_stripes_complete(v11)

N = v11 * v12
M = v21 * v22
O = v31 * v32

# Training parameters
batch_size = train_set.shape[0]
epochs = 100000
k_pos = 3
k_neg = 5
epsilon = 0.005*numx.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])

offset_typ = 'DDD'
dbm = MODEL.BinaryBinaryDBM(N,M,O,offset_typ,train_set)

# Set the same seed value for all algorithms
numx.random.seed(42)

# Initialize parameters
dbm.W1 = numx.random.randn(N, M) * 0.01
dbm.W2 = numx.random.randn(M, O) * 0.01

dbm.o1 = numx.mean(train_set, axis = 0).reshape(1,N)
dbm.o2 = numx.zeros((1,M)) + 0.5
dbm.o3 = numx.zeros((1,O)) + 0.5

dbm.b1 = Sigmoid.g(numx.clip(dbm.o1,0.001,0.999))
dbm.b2 = Sigmoid.g(numx.clip(dbm.o2,0.001,0.999))
dbm.b3 = Sigmoid.g(numx.clip(dbm.o3,0.001,0.999))

# Initialize negative Markov chain
dbm.m1 = dbm.o1+numx.zeros((batch_size,N))
dbm.m2 = dbm.o2+numx.zeros((batch_size,M))
dbm.m3 = dbm.o3+numx.zeros((batch_size,O))

# Choose trainer CD, PCD, PT
trainer = TRAINER.PCD(dbm,batch_size)

# Set AIS betas / inv. temps for AIS
a = numx.linspace(0.0, 0.5, 100+1)
a = a[0:a.shape[0]-1]
b = numx.linspace(0.5, 0.9, 800+1)
b = b[0:b.shape[0]-1]
c = numx.linspace(0.9, 1.0, 2000)
betas = numx.hstack((a,b,c))

numx.random.seed(42)
# Start time measure and training
for epoch in range(0,epochs+1) :
    # update model
    for b in range(0,train_set.shape[0],batch_size):
        trainer.train(data =train_set[b:b + batch_size, :],
                                                 epsilon = epsilon,
                                                 k = [k_pos,k_neg],
                                                 offset_typ = offset_typ,
                                                 meanfield=False)
    # estimate every 10k epochs
    if epoch % 10000 == 0:

        print("Epoche: ",epoch)
        logZ, logZ_up, logZ_down = ESTIMATOR.partition_function_AIS(trainer.model, betas=betas)
        train_LL = numx.mean(ESTIMATOR.LL_lower_bound(trainer.model, train_set, logZ))
        print("AIS  LL: ",2**(v11+1)* train_LL)

        logZ = ESTIMATOR.partition_function_exact(trainer.model)
        train_LL = numx.mean(ESTIMATOR.LL_exact(trainer.model, train_set, logZ))
        print("True LL: ",2**(v11+1)* train_LL)
        print()

# Show weights
VIS.imshow_matrix(VIS.tile_matrix_rows(dbm.W1, v11,v12, v21,v22, border_size = 1,normalized = False), 'Weights 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(dbm.W1,dbm.W2), v11,v12, v31,v32, border_size = 1,normalized = False), 'Weights 2')

VIS.show()
