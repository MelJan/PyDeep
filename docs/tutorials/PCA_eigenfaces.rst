.. _PCA_eigenfaces:

Eigenfaces
===========

Example for Principal Component Analysis (PCA) on face images also known as `Eigenfaces <https://en.wikipedia.org/wiki/Eigenface>`_

Theory
***********

If you are new on PCA, first see `PCA_2D_example <PCA_2D_example.html#PCA_2D_example>`__.

Results
***********

The code_ given below produces the following output.

Some examples of the face images of the olivetti face dataset.

.. figure:: images/example_faces.png
   :scale: 75 %
   :align: center
   :alt: Examples of the face datset

The first 100 principal components extracted from the dataset. The components focus on characteristics like glasses, lighting direction, nose shape, ...

.. image:: images/components_faces.png
   :scale: 75 %
   :align: center
   :alt: Principal components of teh face dataset

The cumulative sum of the Eigenvalues show how 'compressable' the dataset is.

.. image:: images/eigenspectrum_faces.png
   :scale: 50 %
   :align: center
   :alt: Eigenspectrum of the face dataset

For example using only the first 50 eigenvectors retains 87,5 % of the variance of data and the reconstructed images look as follows.

.. image:: images/reconstruction50.png
   :scale: 75 %
   :align: center
   :alt: Reconstruction using 50 PCs

For 200 eigenvectors we retain 98,0 % of the variance of the data and the reconstructed images look as follows.

.. image:: images/reconstruction50.png
   :scale: 75 %
   :align: center
   :alt: Reconstruction using 200 PCs

Comparing the results with the original images shows that the data can be compressed to 50 dimensions with an acceptable error.

.. _code:

Source code
***********

.. figure:: images/download_icon.png
   :scale: 20 %
   :target: https://github.com/MelJan/PyDeep/blob/master/examples/PCA_eigenfaces.py

.. literalinclude:: ../../examples/PCA_eigenfaces.py