.. _DBM_MNIST:

Deep Boltzmann machines on MNIST
==========================================================

Example for training a centered Deep Boltzmann machine on the MNIST handwritten digit dataset.

It allows to reproduce the results from the publication `How to Center Deep Boltzmann Machines. Melchior et al. JMLR 2016. <http://jmlr.org/papers/v17/14-237.html>`_.

Results
***********

The code_ given below produces the following output that is quite similar to the results produced by an RBM.


The learned filters of the first layer

.. figure:: images/DBM_WEIGHTS_1.png
   :scale: 75 %
   :align: center
   :alt: DBM filters of the first layer on MNIST

The learned filters of the second layer, linearly back projected

.. figure:: images/DBM_WEIGHTS_2.png
   :scale: 75 %
   :align: center
   :alt: DBM filters of the second layer on MNIST

Some generated samples

.. figure:: images/DBM_MNIST_SAMPLES.png
   :scale: 75 %
   :align: center
   :alt: AE filter on MNIST with contrastive penalty


See also `RBM_MNIST_big <RBM_MNIST_big.html#RBM_MNIST_big>`__.

.. _code:


Source code
***********

.. figure:: images/download_icon.png
   :scale: 20 %
   :target: https://github.com/MelJan/PyDeep/blob/master/examples/DBM_MNIST.py

.. literalinclude:: ../../examples/DBM_MNIST.py