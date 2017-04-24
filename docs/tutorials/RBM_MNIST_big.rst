Big centered binary RBM on MNIST
==========================================================

Example for training a centered binary restricted Boltzmann machine on the MNIST handwritten digit dataset.
The model has 500 hidden units, is trainer 200 epochs, and the Log-likelihood is evaluated using Annealed Importance Sampling.
and allows to reproduce the the results from the publication `How to Center Deep Boltzmann Machines. Melchior, J., Fischer, A., & Wiskott, L.. (2016). Journal of Machine Learning Research, 17(99), 1–61. <http://jmlr.org/papers/v17/14-237.html>`_
Running the code as it is reproduces a single trial of the plot in Figure 9. (PCD-1) for $dd^b_s$.

See also `RBM_MNIST_small <RBM_MNIST_small.html#RBM_MNIST_small>`__.

Theory
***********

For an analysis of advantage of centering in RBMs see `How to Center Deep Boltzmann Machines. Melchior, J., Fischer, A., & Wiskott, L.. (2016). Journal of Machine Learning Research, 17(99), 1–61. <http://jmlr.org/papers/v17/14-237.html>`_

If you are new on RBMs, have a look into my `master's theses <https://www.ini.rub.de/PEOPLE/wiskott/Reprints/Melchior-2012-MasterThesis-RBMs.pdf>`_

A good theoretical introduction is also given by `Course Material RBMs <https://www.ini.rub.de/PEOPLE/wiskott/Teaching/Material/index.html>`_ and in the following video.

.. raw:: html

    <div style="margin-top:10px;">
      <iframe width="560" height="315" src="http://www.youtube.com/embed/bMaITeXhOaE" frameborder="0" allowfullscreen></iframe>
    </div>

and

.. raw:: html

    <div style="margin-top:10px;">
      <iframe width="560" height="315" src="http://www.youtube.com/embed/nyk5XUklb5M" frameborder="0" allowfullscreen></iframe>
    </div>

Results
***********

The code_ given below produces the following output.

Learned filters of a centered binary RBM with 500 hidden units on the MNIST dataset.
The filters have been normalized such that the structure is more prominent.

.. figure:: images/BRBM_big_centered_weights.png
   :scale: 75 %
   :alt: weights centered

Sampling results for some examples. The first row shows training data and the following rows are the results after one Gibbs-sampling step starting from the previous row.

.. figure:: images/BRBM_big_centered_samples.png
   :scale: 75 %
   :alt: samples centered

The Log-Likelihood is calculated using annealed importance sampling estimation (optimistic) and reverse annealed importance sampling estimation (pessimistic).

.. code-block:: Python

   Training time:          0:49:51.186054
   AIS Partition:          951.21017149  (LL: -76.0479396244)
   reverse AIS Partition:  954.687597369 (LL: -79.525365503)

Source code
***********

.. figure:: images/download_icon.png
   :scale: 20 %
   :target: https://github.com/MelJan/PyDeep/blob/master/examples/RBM_MNIST_big.py

.. literalinclude:: ../../examples/RBM_MNIST_big.py