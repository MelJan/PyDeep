.. _ICA_Natural_Images:

Gaussian-binary restricted Boltzmann machine on a natural image patches
==========================================================

Example for a  Gaussian-binary restricted Boltzmann machine on a natural image patches.
The learned filters are similar to those of ICA, see also `ICA_natural_images <ICA_natural_images.html#ICA_natural_images>`__.

Theory
***********

If you are new on RBMs, have a look into my `master's theses <https://www.ini.rub.de/PEOPLE/wiskott/Reprints/Melchior-2012-MasterThesis-RBMs.pdf>`_

For an theoretical and empirical analysis of on GRBMs on natural image patches see `Gaussian-binary restricted Boltzmann machines for modeling natural image statistics. Melchior, J., Wang, N., & Wiskott, L.. (2017). PLOS ONE, 12(2), 1–24. <http://doi.org/10.1371/journal.pone.0171015>`_

Results
***********

The code_ given below produces the following output.

Visualization of the learned filters, which are very similar to those of ICA.

.. figure:: images/GRBM_weights_unnormalized.png
   :scale: 75 %
   :alt: GRBM weights unnormalized

The same filter normalized independently for a better visualization of the structure.

.. figure:: images/GRBM_weights_normalized.png
   :scale: 75 %
   :alt: GRBM weights normalized

Sampling results for some examples. The first row shows training data and the following rows are the results after one
   Gibbs-sampling step starting from the previous row.

.. figure:: images/GRBM_samples.png
   :scale: 75 %
   :alt: GRBM samples

The log-likelihood of training and test data

.. code-block:: Python

                Epoch	RE train	RE test 	LL train	LL test
   AIS:         200 	0.73291 	0.75427 	-268.34107 	-270.82759
   reverse AIS:         0.73291 	0.75427 	-268.34078 	-270.82731

.. _code:

Source code
***********

.. figure:: images/download_icon.png
   :scale: 20 %
   :target: https://github.com/MelJan/PyDeep/blob/master/examples/GRBM_natural_images.py

.. literalinclude:: ../../examples/GRBM_natural_images.py