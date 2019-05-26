import pydeep.misc.visualization as VIS
import pydeep.misc.io as IO
import pydeep.base.numpyextension as npExt
from pydeep.dbm.unit_layer import *
from pydeep.dbm.weight_layer import *
from pydeep.dbm.model import *

# Set the same seed value for all algorithms
numx.random.seed(42)

# Load Data
train_data = IO.load_mnist("../../../data/mnist.pkl.gz", True)[0]

# Set dimensions Layer 1-3
v11 = v12 = 28
v21 = v22 = 10
v31 = v32 = 10
N = v11 * v12
M = v21 * v22
O = v31 * v32

# Create weight layers, which connect the unit layers
wl1 = Weight_layer(input_dim=N,
                   output_dim=M,
                   initial_weights=0.01,
                   dtype=numx.float64)
wl2 = Weight_layer(input_dim=M,
                   output_dim=O,
                   initial_weights=0.01,
                   dtype=numx.float64)

# Create three unit layers
l1 = Binary_layer(None,
                  wl1,
                  data=train_data,
                  initial_bias='AUTO',
                  initial_offsets='AUTO',
                  dtype=numx.float64)

l2 = Binary_layer(wl1,
                  wl2,
                  data=None,
                  initial_bias='AUTO',
                  initial_offsets='AUTO',
                  dtype=numx.float64)

l3 =Binary_layer(wl2,
                  None,
                  data=None,
                  initial_bias='AUTO',
                  initial_offsets='AUTO',
                  dtype=numx.float64)

# Initialize parameters
max_epochs =100
batch_size = 10

# Sampling Setps positive and negative phase
k_d = 3
k_m = 1

# Set individual learning rates
lr_W1 = 0.01
lr_W2 = 0.01
lr_b1 = 0.01
lr_b2 = 0.01
lr_b3 = 0.01
lr_o1 = 0.01
lr_o2 = 0.01
lr_o3 = 0.01

# Initialize negative Markov chain
x_m = numx.zeros((batch_size, v11 * v12)) + l1.offset
y_m = numx.zeros((batch_size, v21 * v22)) + l2.offset
z_m = numx.zeros((batch_size, v31 * v32)) + l3.offset
chain_m = [x_m, y_m, z_m]

# Reparameterize RBM such that the inital setting is the same for centereing and centered training
l1.bias += numx.dot(0.0 - l2.offset, wl1.weights.T)
l2.bias += numx.dot(0.0 - l1.offset, wl1.weights) + numx.dot(0.0 - l3.offset, wl2.weights.T)
l3.bias += numx.dot(0.0 - l2.offset, wl2.weights)

# Finally create model
model = DBM_model([l1, l2, l3])

# Loop over data and batches to traing th emodel
for epoch in range(0, max_epochs):
    rec_sum = 0
    for b in range(0, train_data.shape[0], batch_size):
        # Positive Phase

        # Initialize Markov chains with data or offsets
        x_d = train_data[b:b + batch_size, :]
        y_d = numx.zeros((batch_size, M)) + l2.offset
        z_d = numx.zeros((batch_size, O)) + l3.offset
        chain_d = [x_d, y_d, z_d]

        # Sample for k_d steps inplace, but clamp the data units
        model.sample(chain_d, k_d, [True, False, True], True)

        # Negative Phase

        # PCD, sample k_m steps without clamping
        model.sample(chain_m, k_m, [False, False, False], True)

        # Alternatively use CD sampling, by not starting from the last state in the MArkov chains but from the data.
        #chain_m[0] = x_d
        #chain_m = model.sample(chain_d, k_m, [False,False,False], False)

        # Update the model using the sampled states and learning rates
        model.update(chain_d, chain_m, lr_W1, lr_b1, lr_o1)

    # Print Norms of the Parameters
    print numx.mean(numxExt.get_norms(wl1.weights)), '\t', numx.mean(numxExt.get_norms(wl2.weights)), '\t',
    print numx.mean(numxExt.get_norms(l1.bias)), '\t', numx.mean(numxExt.get_norms(l2.bias)), '\t',
    print numx.mean(numxExt.get_norms(l3.bias)), '\t', numx.mean(l1.offset), '\t', numx.mean(
    l2.offset), '\t', numx.mean(l3.offset)


# Show weights
VIS.imshow_matrix(VIS.tile_matrix_rows(wl1.weights, v11, v12, v21, v22, border_size=1, normalized=False), 'Weights 1')
VIS.imshow_matrix(
    VIS.tile_matrix_rows(numx.dot(wl1.weights, wl2.weights), v11, v12, v31, v32, border_size=1, normalized=False),
    'Weights 2')

# # Samplesome steps
chain_m = [numx.zeros((batch_size, v11 * v12)) + l1.offset, numx.zeros((batch_size, v21 * v22)) + l2.offset, numx.zeros((batch_size, v31 * v32)) + l3.offset]
model.sample(chain_m, 100, [False, False, False], True)

VIS.imshow_matrix(VIS.tile_matrix_columns(chain_m[0], v11, v12, 1, batch_size, 1, False), 'Samples')

VIS.show()