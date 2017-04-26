.. _ICA_Natural_Images:

Independent Component Analysis on a natural image patches
==========================================================

Example for Independent Component Analysis (`ICA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_)
on natural image patches. The independent components (columns of the ICA projection matrix) of natural image patches are edge detector filters.

Theory
***********

If you are new on ICA and blind source separation, see first `ICA_2D_example <ICA_2D_example.html#ICA_2D_example>`__.

For a comparison of ICA and GRBMs on natural image patches see `Gaussian-binary restricted Boltzmann machines for modeling natural image statistics. Melchior et. al. PLOS ONE 2017 <http://doi.org/10.1371/journal.pone.0171015>`_.

Results
***********

The code_ given below produces the following output.

Visualization of 100 examples of the gray scale natural image dataset.

.. figure:: images/ICA_natural_images_data.png
   :scale: 75 %
   :alt: 100 gray scale natural image patch examples

The corresponding whitened image patches.

.. figure:: images/ICA_natural_images_data_whitened.png
   :scale: 75 %
   :alt: 100 gray scale natural image patch examples whitend

The learned filters/independent components learned from the whitened natural image patches.

.. figure:: images/ICA_natural_images_filter.png
   :scale: 75 %
   :alt: ICA filter on natural images

See also `GRBM_natural_images <GRBM_natural_images.html#GRBM_natural_images>`__.

.. _code:

Source code
***********

.. figure:: images/download_icon.png
   :scale: 20 %
   :target: https://github.com/MelJan/PyDeep/blob/master/examples/ICA_natural_images.py

.. literalinclude:: ../../examples/ICA_natural_images.py