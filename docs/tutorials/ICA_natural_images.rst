Independent Component Analysis on a natural image patches
==========================================================

Example for Independent Component Analysis (`ICA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_)
on natural image patches. The independent components (columns of the ICA projection matrix) of natural image patches are edge detector filters.

See ICA_2D_example for a theoretical introduction.

Visualization of 100 examples of the gray scale natural image dataset.

.. figure:: images/ICA_natural_images_data.png
   :scale: 75 %
   :alt: 100 gray scale natural image patch examples

The corresponding whitened image patches.

.. figure:: images/ICA_natural_images_data_whitened.png
   :scale: 75 %
   :alt: 100 gray scale natural image patch examples

The learned filters/independent components learned from the whitened natural image patches.

.. figure:: images/ICA_natural_images_filter.png
   :scale: 75 %
   :alt: 100 gray scale natural image patch examples

See `Gaussian-binary restricted Boltzmann machines for modeling natural image statistics. Melchior, J., Wang, N., & Wiskott, L.. (2017). PLOS ONE, 12(2), 1–24. <http://doi.org/10.1371/journal.pone.0171015>`_
for a analysis of ICA and GRBM.

.. _code:

Source code
***********

.. figure:: images/download_icon.png
   :scale: 20 %
   :target: https://github.com/MelJan/PyDeep/blob/master/examples/ICA_natural_images.py

.. literalinclude:: ../../examples/ICA_natural_images.py