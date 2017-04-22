Principal Component Analysis 2D example.
============================================

Example for Principal Component Analysis (`PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_) on a linear 2D mixture.

If you are new on PCA, a good theoretical introduction is given in the following video.

.. raw:: html

    <div style="margin-top:10px;">
      <iframe width="560" height="315" src="http://www.youtube.com/embed/9H-1FH1gn6w" frameborder="0" allowfullscreen></iframe>
    </div>

The code_ given below produces the following output.

The data is plotted with the extracted principal components.

.. figure:: images/PCA_2D.png
   :scale: 75 %
   :alt: Examples of PCA 2D

Data and extracted principal components can also be plotted in the projected space.

.. image:: images/PCA_2D_projected.png
   :scale: 75 %
   :alt: Examples of PCA 2D in projected space

The PCA class can also perform whitening and data and extracted principal components are plotted in the whitened space.

.. image:: images/PCA_2D_whitened.png
   :scale: 50 %
   :alt: Examples of PCA 2D in whitened space

For a real-world application see the eigenfaces example.

.. _code:

Source code
***********

.. figure:: images/download_icon.png
   :scale: 20 %
   :target: https://github.com/MelJan/PyDeep/blob/master/examples/PCA_2D.py

.. literalinclude:: ../../examples/PCA_2D.py