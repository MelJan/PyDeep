.. _AE_MNIST:

Autoencoder on MNIST
==========================================================

Example for training a centered Autoencoder on the MNIST handwritten digit dataset with and without contractive penalty,
dropout, ...

It allows to reproduce the results from the publication `How to Center Deep Boltzmann Machines. Melchior et al. JMLR 2016. <http://jmlr.org/papers/v17/14-237.html>`_. Running the code as it is for example reproduces a single trial of the plot in Figure 9. (PCD-1) for $dd^b_s$.

Theory
***********

If you are new on Autoencoders visit `Autoencoder tutorial <http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity>`_ or watch the video course by Hugo Larochelle

.. raw:: html

    <div style="margin-top:10px;">
      <iframe width="560" height="315" src="http://www.youtube.com/watch?v=FzS3tMl4Nsc" frameborder="0" allowfullscreen></iframe>
    </div>

Results
***********

The code_ given below produces the following output that is quite similar to the results produced by an RBM.



See also `RBM_MNIST <RBM_MNIST.html#RBM_MNIST>`__.
.. _code:


Source code
***********

.. figure:: images/download_icon.png
   :scale: 20 %
   :target: https://github.com/MelJan/PyDeep/blob/master/examples/AE_mnist.py

.. literalinclude:: ../../examples/AE_mnist.py