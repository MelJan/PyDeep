Installation
##################################


To install PyDeep, simply change to the PyDeep folder and run the setup script:

.. code-block:: bash

    python setup.py install

Dependencies
============

PyDeep has the following dependencies:

Hard dependencies:
----------------------------------------------------

- numpy

- scipy

Soft dependencies:
----------------------------------------------------

-matplotlib

- cPickle

- encryptedpickle

- paramiko

- mdp


Optimized backend:
============================================================

It is highly recommended to use an multi-threading optimized linear algebra backend such as

-  `Automatically Tuned Linear Algebra Software (ATLAS) <https://software.intel.com/en-us/intel-mkl/>`_

-  `Intel® Math Kernel Library (Intel® MKL)  http://math-atlas.sourceforge.net/>`_

MKL is inlcuded in `Enthought <https://www.enthought.com/>`_ which provides a free academic license.
