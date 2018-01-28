====================================================
Documentation
====================================================

API documentation for PyDeep.

pydeep
----------------------------------------------------

.. automodule:: pydeep

ae
````````````````````````````````````````````````````

.. automodule:: pydeep.ae

model
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.ae.model

AutoEncoder
....................................................

.. autoclass:: pydeep.ae.model.AutoEncoder
   :members:
   :private-members:
   :special-members: __init__

sae
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.ae.sae

SAE
....................................................

.. autoclass:: pydeep.ae.sae.SAE
   :members:
   :private-members:
   :special-members: __init__

trainer
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.ae.trainer

GDTrainer
....................................................

.. autoclass:: pydeep.ae.trainer.GDTrainer
   :members:
   :private-members:
   :special-members: __init__

base
````````````````````````````````````````````````````

.. automodule:: pydeep.base

activationfunction
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.base.activationfunction

Identity
....................................................

.. autoclass:: pydeep.base.activationfunction.Identity
   :members:
   :private-members:
   :special-members: __init__

Rectifier
....................................................

.. autoclass:: pydeep.base.activationfunction.Rectifier
   :members:
   :private-members:
   :special-members: __init__

RestrictedRectifier
....................................................

.. autoclass:: pydeep.base.activationfunction.RestrictedRectifier
   :members:
   :private-members:
   :special-members: __init__

LeakyRectifier
....................................................

.. autoclass:: pydeep.base.activationfunction.LeakyRectifier
   :members:
   :private-members:
   :special-members: __init__

ExponentialLinear
....................................................

.. autoclass:: pydeep.base.activationfunction.ExponentialLinear
   :members:
   :private-members:
   :special-members: __init__

SigmoidWeightedLinear
....................................................

.. autoclass:: pydeep.base.activationfunction.SigmoidWeightedLinear
   :members:
   :private-members:
   :special-members: __init__

SoftPlus
....................................................

.. autoclass:: pydeep.base.activationfunction.SoftPlus
   :members:
   :private-members:
   :special-members: __init__

Step
....................................................

.. autoclass:: pydeep.base.activationfunction.Step
   :members:
   :private-members:
   :special-members: __init__

Sigmoid
....................................................

.. autoclass:: pydeep.base.activationfunction.Sigmoid
   :members:
   :private-members:
   :special-members: __init__

SoftSign
....................................................

.. autoclass:: pydeep.base.activationfunction.SoftSign
   :members:
   :private-members:
   :special-members: __init__

HyperbolicTangent
....................................................

.. autoclass:: pydeep.base.activationfunction.HyperbolicTangent
   :members:
   :private-members:
   :special-members: __init__

SoftMax
....................................................

.. autoclass:: pydeep.base.activationfunction.SoftMax
   :members:
   :private-members:
   :special-members: __init__

RadialBasis
....................................................

.. autoclass:: pydeep.base.activationfunction.RadialBasis
   :members:
   :private-members:
   :special-members: __init__

Sinus
....................................................

.. autoclass:: pydeep.base.activationfunction.Sinus
   :members:
   :private-members:
   :special-members: __init__

KWinnerTakeAll
....................................................

.. autoclass:: pydeep.base.activationfunction.KWinnerTakeAll
   :members:
   :private-members:
   :special-members: __init__

basicstructure
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.base.basicstructure

BipartiteGraph
....................................................

.. autoclass:: pydeep.base.basicstructure.BipartiteGraph
   :members:
   :private-members:
   :special-members: __init__

StackOfBipartiteGraphs
....................................................

.. autoclass:: pydeep.base.basicstructure.StackOfBipartiteGraphs
   :members:
   :private-members:
   :special-members: __init__

corruptor
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.base.corruptor

Identity
....................................................

.. autoclass:: pydeep.base.corruptor.Identity
   :members:
   :private-members:
   :special-members: __init__

AdditiveGaussNoise
....................................................

.. autoclass:: pydeep.base.corruptor.AdditiveGaussNoise
   :members:
   :private-members:
   :special-members: __init__

MultiGaussNoise
....................................................

.. autoclass:: pydeep.base.corruptor.MultiGaussNoise
   :members:
   :private-members:
   :special-members: __init__

SamplingBinary
....................................................

.. autoclass:: pydeep.base.corruptor.SamplingBinary
   :members:
   :private-members:
   :special-members: __init__

Dropout
....................................................

.. autoclass:: pydeep.base.corruptor.Dropout
   :members:
   :private-members:
   :special-members: __init__

RandomPermutation
....................................................

.. autoclass:: pydeep.base.corruptor.RandomPermutation
   :members:
   :private-members:
   :special-members: __init__

KeepKWinner
....................................................

.. autoclass:: pydeep.base.corruptor.KeepKWinner
   :members:
   :private-members:
   :special-members: __init__

KWinnerTakesAll
....................................................

.. autoclass:: pydeep.base.corruptor.KWinnerTakesAll
   :members:
   :private-members:
   :special-members: __init__

costfunction
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.base.costfunction

SquaredError
....................................................

.. autoclass:: pydeep.base.costfunction.SquaredError
   :members:
   :private-members:
   :special-members: __init__

AbsoluteError
....................................................

.. autoclass:: pydeep.base.costfunction.AbsoluteError
   :members:
   :private-members:
   :special-members: __init__

CrossEntropyError
....................................................

.. autoclass:: pydeep.base.costfunction.CrossEntropyError
   :members:
   :private-members:
   :special-members: __init__

NegLogLikelihood
....................................................

.. autoclass:: pydeep.base.costfunction.NegLogLikelihood
   :members:
   :private-members:
   :special-members: __init__

numpyextension
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.base.numpyextension

log_sum_exp
....................................................

.. automethod:: pydeep.base.numpyextension.log_sum_exp

log_diff_exp
....................................................

.. automethod:: pydeep.base.numpyextension.log_diff_exp

multinominal_batch_sampling
....................................................

.. automethod:: pydeep.base.numpyextension.multinominal_batch_sampling

get_norms
....................................................

.. automethod:: pydeep.base.numpyextension.get_norms

restrict_norms
....................................................

.. automethod:: pydeep.base.numpyextension.restrict_norms

resize_norms
....................................................

.. automethod:: pydeep.base.numpyextension.resize_norms

angle_between_vectors
....................................................

.. automethod:: pydeep.base.numpyextension.angle_between_vectors

get_2d_gauss_kernel
....................................................

.. automethod:: pydeep.base.numpyextension.get_2d_gauss_kernel

generate_binary_code
....................................................

.. automethod:: pydeep.base.numpyextension.generate_binary_code

get_binary_label
....................................................

.. automethod:: pydeep.base.numpyextension.get_binary_label

compare_index_of_max
....................................................

.. automethod:: pydeep.base.numpyextension.compare_index_of_max

shuffle_dataset
....................................................

.. automethod:: pydeep.base.numpyextension.shuffle_dataset

rotation_sequence
....................................................

.. automethod:: pydeep.base.numpyextension.rotation_sequence

generate_2d_connection_matrix
....................................................

.. automethod:: pydeep.base.numpyextension.generate_2d_connection_matrix

misc
````````````````````````````````````````````````````

.. automodule:: pydeep.misc

io
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.misc.io

save_object
....................................................

.. automethod:: pydeep.misc.io.save_object

save_image
....................................................

.. automethod:: pydeep.misc.io.save_image

load_object
....................................................

.. automethod:: pydeep.misc.io.load_object

load_image
....................................................

.. automethod:: pydeep.misc.io.load_image

download_file
....................................................

.. automethod:: pydeep.misc.io.download_file

load_mnist
....................................................

.. automethod:: pydeep.misc.io.load_mnist

load_caltech
....................................................

.. automethod:: pydeep.misc.io.load_caltech

load_cifar
....................................................

.. automethod:: pydeep.misc.io.load_cifar

load_natural_image_patches
....................................................

.. automethod:: pydeep.misc.io.load_natural_image_patches

load_olivetti_faces
....................................................

.. automethod:: pydeep.misc.io.load_olivetti_faces

measuring
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.misc.measuring

print_progress
....................................................

.. automethod:: pydeep.misc.measuring.print_progress

Stopwatch
....................................................

.. autoclass:: pydeep.misc.measuring.Stopwatch
   :members:
   :private-members:
   :special-members: __init__

sshthreadpool
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.misc.sshthreadpool

SSHConnection
....................................................

.. autoclass:: pydeep.misc.sshthreadpool.SSHConnection
   :members:
   :private-members:
   :special-members: __init__

SSHJob
....................................................

.. autoclass:: pydeep.misc.sshthreadpool.SSHJob
   :members:
   :private-members:
   :special-members: __init__

SSHPool
....................................................

.. autoclass:: pydeep.misc.sshthreadpool.SSHPool
   :members:
   :private-members:
   :special-members: __init__

toyproblems
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.misc.toyproblems

generate_2d_mixtures
....................................................

.. automethod:: pydeep.misc.toyproblems.generate_2d_mixtures

generate_bars_and_stripes
....................................................

.. automethod:: pydeep.misc.toyproblems.generate_bars_and_stripes

generate_bars_and_stripes_complete
....................................................

.. automethod:: pydeep.misc.toyproblems.generate_bars_and_stripes_complete

generate_shifting_bars
....................................................

.. automethod:: pydeep.misc.toyproblems.generate_shifting_bars

generate_shifting_bars_complete
....................................................

.. automethod:: pydeep.misc.toyproblems.generate_shifting_bars_complete

visualization
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.misc.visualization

tile_matrix_columns
....................................................

.. automethod:: pydeep.misc.visualization.tile_matrix_columns

tile_matrix_rows
....................................................

.. automethod:: pydeep.misc.visualization.tile_matrix_rows

imshow_matrix
....................................................

.. automethod:: pydeep.misc.visualization.imshow_matrix

imshow_plot
....................................................

.. automethod:: pydeep.misc.visualization.imshow_plot

imshow_histogram
....................................................

.. automethod:: pydeep.misc.visualization.imshow_histogram

plot_2d_weights
....................................................

.. automethod:: pydeep.misc.visualization.plot_2d_weights

plot_2d_data
....................................................

.. automethod:: pydeep.misc.visualization.plot_2d_data

plot_2d_contour
....................................................

.. automethod:: pydeep.misc.visualization.plot_2d_contour

imshow_standard_rbm_parameters
....................................................

.. automethod:: pydeep.misc.visualization.imshow_standard_rbm_parameters

hidden_activation
....................................................

.. automethod:: pydeep.misc.visualization.hidden_activation

reorder_filter_by_hidden_activation
....................................................

.. automethod:: pydeep.misc.visualization.reorder_filter_by_hidden_activation

generate_samples
....................................................

.. automethod:: pydeep.misc.visualization.generate_samples

imshow_filter_tuning_curve
....................................................

.. automethod:: pydeep.misc.visualization.imshow_filter_tuning_curve

imshow_filter_optimal_gratings
....................................................

.. automethod:: pydeep.misc.visualization.imshow_filter_optimal_gratings

imshow_filter_frequency_angle_histogram
....................................................

.. automethod:: pydeep.misc.visualization.imshow_filter_frequency_angle_histogram

filter_frequency_and_angle
....................................................

.. automethod:: pydeep.misc.visualization.filter_frequency_and_angle

filter_frequency_response
....................................................

.. automethod:: pydeep.misc.visualization.filter_frequency_response

filter_angle_response
....................................................

.. automethod:: pydeep.misc.visualization.filter_angle_response

calculate_amari_distance
....................................................

.. automethod:: pydeep.misc.visualization.calculate_amari_distance

preprocessing
````````````````````````````````````````````````````

.. automodule:: pydeep.preprocessing

binarize_data
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automethod:: pydeep.preprocessing.binarize_data

rescale_data
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automethod:: pydeep.preprocessing.rescale_data

remove_rows_means
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automethod:: pydeep.preprocessing.remove_rows_means

remove_cols_means
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automethod:: pydeep.preprocessing.remove_cols_means

STANDARIZER
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. autoclass:: pydeep.preprocessing.STANDARIZER
   :members:
   :private-members:
   :special-members: __init__

PCA
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. autoclass:: pydeep.preprocessing.PCA
   :members:
   :private-members:
   :special-members: __init__

ZCA
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. autoclass:: pydeep.preprocessing.ZCA
   :members:
   :private-members:
   :special-members: __init__

ICA
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. autoclass:: pydeep.preprocessing.ICA
   :members:
   :private-members:
   :special-members: __init__

rbm
````````````````````````````````````````````````````

.. automodule:: pydeep.rbm

dbn
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.rbm.dbn

DBN
....................................................

.. autoclass:: pydeep.rbm.dbn.DBN
   :members:
   :private-members:
   :special-members: __init__

estimator
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.rbm.estimator

reconstruction_error
....................................................

.. automethod:: pydeep.rbm.estimator.reconstruction_error

log_likelihood_v
....................................................

.. automethod:: pydeep.rbm.estimator.log_likelihood_v

log_likelihood_h
....................................................

.. automethod:: pydeep.rbm.estimator.log_likelihood_h

partition_function_factorize_v
....................................................

.. automethod:: pydeep.rbm.estimator.partition_function_factorize_v

partition_function_factorize_h
....................................................

.. automethod:: pydeep.rbm.estimator.partition_function_factorize_h

annealed_importance_sampling
....................................................

.. automethod:: pydeep.rbm.estimator.annealed_importance_sampling

reverse_annealed_importance_sampling
....................................................

.. automethod:: pydeep.rbm.estimator.reverse_annealed_importance_sampling

model
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.rbm.model

BinaryBinaryRBM
....................................................

.. autoclass:: pydeep.rbm.model.BinaryBinaryRBM
   :members:
   :private-members:
   :special-members: __init__

GaussianBinaryRBM
....................................................

.. autoclass:: pydeep.rbm.model.GaussianBinaryRBM
   :members:
   :private-members:
   :special-members: __init__

GaussianBinaryVarianceRBM
....................................................

.. autoclass:: pydeep.rbm.model.GaussianBinaryVarianceRBM
   :members:
   :private-members:
   :special-members: __init__

BinaryBinaryLabelRBM
....................................................

.. autoclass:: pydeep.rbm.model.BinaryBinaryLabelRBM
   :members:
   :private-members:
   :special-members: __init__

SoftMaxSigmoid
....................................................

.. autoclass:: pydeep.rbm.model.SoftMaxSigmoid
   :members:
   :private-members:
   :special-members: __init__

GaussianBinaryLabelRBM
....................................................

.. autoclass:: pydeep.rbm.model.GaussianBinaryLabelRBM
   :members:
   :private-members:
   :special-members: __init__

SoftMaxLinear
....................................................

.. autoclass:: pydeep.rbm.model.SoftMaxLinear
   :members:
   :private-members:
   :special-members: __init__

BinaryRectRBM
....................................................

.. autoclass:: pydeep.rbm.model.BinaryRectRBM
   :members:
   :private-members:
   :special-members: __init__

RectBinaryRBM
....................................................

.. autoclass:: pydeep.rbm.model.RectBinaryRBM
   :members:
   :private-members:
   :special-members: __init__

RectRectRBM
....................................................

.. autoclass:: pydeep.rbm.model.RectRectRBM
   :members:
   :private-members:
   :special-members: __init__

GaussianRectRBM
....................................................

.. autoclass:: pydeep.rbm.model.GaussianRectRBM
   :members:
   :private-members:
   :special-members: __init__

GaussianRectVarianceRBM
....................................................

.. autoclass:: pydeep.rbm.model.GaussianRectVarianceRBM
   :members:
   :private-members:
   :special-members: __init__

sampler
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.rbm.sampler

GibbsSampler
....................................................

.. autoclass:: pydeep.rbm.sampler.GibbsSampler
   :members:
   :private-members:
   :special-members: __init__

PersistentGibbsSampler
....................................................

.. autoclass:: pydeep.rbm.sampler.PersistentGibbsSampler
   :members:
   :private-members:
   :special-members: __init__

ParallelTemperingSampler
....................................................

.. autoclass:: pydeep.rbm.sampler.ParallelTemperingSampler
   :members:
   :private-members:
   :special-members: __init__

IndependentParallelTemperingSampler
....................................................

.. autoclass:: pydeep.rbm.sampler.IndependentParallelTemperingSampler
   :members:
   :private-members:
   :special-members: __init__

trainer
''''''''''''''''''''''''''''''''''''''''''''''''''''

.. automodule:: pydeep.rbm.trainer

CD
....................................................

.. autoclass:: pydeep.rbm.trainer.CD
   :members:
   :private-members:
   :special-members: __init__

PCD
....................................................

.. autoclass:: pydeep.rbm.trainer.PCD
   :members:
   :private-members:
   :special-members: __init__

PT
....................................................

.. autoclass:: pydeep.rbm.trainer.PT
   :members:
   :private-members:
   :special-members: __init__

IPT
....................................................

.. autoclass:: pydeep.rbm.trainer.IPT
   :members:
   :private-members:
   :special-members: __init__

GD
....................................................

.. autoclass:: pydeep.rbm.trainer.GD
   :members:
   :private-members:
   :special-members: __init__