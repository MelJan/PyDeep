.. _FNN_MNIST:

Feed Forward Neural Network on MNIST
==========================================================

Example for training a Feed Forward Neural Network on the MNIST handwritten digit dataset.


Results
***********

The code_ given below produces the following output that is quite similar to the results produced by an RBM.

.. code-block:: Python

   1 	0.1 	0.0337166666667 	0.0396
   2 	0.1 	0.023 	                0.0285
   3 	0.1 	0.0198666666667 	0.0276
   4 	0.1 	0.0154 	                0.0264
   5 	0.1 	0.01385 	        0.0239
   6 	0.1 	0.01255 	        0.0219
   7 	0.1 	0.012 	                0.0229
   8 	0.1 	0.00926666666667 	0.0207
   9 	0.1 	0.0117 	                0.0237
   10 	0.1 	0.00881666666667 	0.0214
   11 	0.1 	0.007 	                0.0191
   12 	0.1 	0.00778333333333 	0.0199
   13 	0.1 	0.0067 	                0.0183
   14 	0.1 	0.00666666666667 	0.0194
   15 	0.1 	0.00665 	        0.0197
   16 	0.1 	0.00583333333333 	0.0197
   17 	0.1 	0.00563333333333 	0.0193
   18 	0.1 	0.005 	                0.0181
   19 	0.1 	0.00471666666667 	0.0186
   20 	0.1 	0.00431666666667 	0.0191

Showing the Epoch / Learning Rate / Training Error / Test Error

See also `RBM_MNIST_big <RBM_MNIST_big.html#RBM_MNIST_big>`__.

.. _code:

Source code
***********

.. figure:: images/download_icon.png
   :scale: 20 %
   :target: https://github.com/MelJan/PyDeep/blob/master/examples/FNN_MNIST.py

.. literalinclude:: ../../examples/FNN_MNIST.py