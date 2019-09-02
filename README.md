Documentation: http://pydeep.readthedocs.io/en/latest/index.html

Welcome
##################################

PyDeep is a machine learning / deep learning library with focus on unsupervised learning.
The library has a modular design, is well documented and purely written in Python/Numpy.
This allows you to understand, use, modify, and debug the code easily. Furthermore,
its extensive use of unittests assures a high level of reliability and correctness.

News
''''''''''''''''''''''''''''''''''''''''''''''''''''
- Exact log-lLikelihood and annealed importance sampling estimate for a 3 layer Deep Boltzmann machines added
- Feed Forward neural networks added
- Deep Boltzmann machines added

- Future: RBM/DBM in tensorFlow / PyTorch


Features index
''''''''''''''''''''''''''''''''''''''''''''''''''''

- Principal Component Analysis (PCA)

    * Zero Phase Component Analysis (ZCA)

- Independent Component Analysis (ICA)

- Feed Forward Neural Networks
    
    * Fully Connected Layer / Randomly Connected Layer
    
    * Backpropagation / AdaGrad
    
    * Dropout / Centering / Corruptors / L1 / L2 / Sparseness
    
    * Various Activation Functions
    
    * Various Loss Functions
    
    * Gradient restriction
    
    * Finite Differences Test

- Autoencoder

    * Centered denoising autoencoder including various noise functions

    * Centered contractive autoencoder

    * Centered sparse autoencoder

    * Centered slowness autoencoder

    * Several regularization methods like l1,l2 norm, Dropout, gradient clipping, ...

- Restricted Boltzmann machines

    * centered BinaryBinary RBM (BB-RBM)

    * centered GaussianBinary RBM (GB-RBM) with fixed variance

    * centered GaussianBinaryVariance RBM (GB-RBM) with trainable variance

    * centered BinaryBinaryLabel RBM (BBL-RBM)

    * centered GaussianBinaryLabel RBM (GBL-RBM)

    * centered BinaryRect RBM (BR-RBM)

    * centered RectBinary RBM (RB-RBM)

    * centered RectRect RBM (RR-RBM)

    * centered GaussianRect RBM (GR-RBM)

    * centered GaussianRectVariance RBM (GRV-RBM)

    * Sampling Algorithms for RBMs

        + Gibbs Sampling

        + Persistent Gibbs Sampling

        + Parallel Tempering Sampling

        + Independent Parallel Tempering Sampling

    * Training for RBMs

        + Exact gradient (GD)

        + Contrastive Divergence (CD)

        + Persistent Contrastive Divergence (PCD)

        + Independent Parallel Tempering Sampling

    * Log-likelihodd estimation for RBMs

        + Exact Partition function

        + Annealed Importance Sampling (AIS)

        + reverse Annealed Importance Sampling (AIS)

- Deep Boltzmann machines
    
    * Binary / SoftMax / Gaussian / Rectifier Units
    
    * Fully connected and convolving weight layer
    
    * CD / PCD / Mean field sampling
    
    * Exact and annealed importance sampling estimate of the partition function of a 3 layer Deep Boltzmann machines.
    
    * Estimation of the log-likelihood lower bound of a 3 layer Deep Boltzmann machines.

Scientific use

The library contains code I have written during my PhD research allowing you to reproduce
the results described in the following publications.

- `Gaussian-binary restricted Boltzmann machines for modeling natural image statistics. Melchior, J., Wang, N., & Wiskott, L.. (2017). PLOS ONE, 12(2), 1–24. <http://doi.org/10.1371/journal.pone.0171015>`_

- `How to Center Deep Boltzmann Machines. Melchior, J., Fischer, A., & Wiskott, L.. (2016). Journal of Machine Learning Research, 17(99), 1–61. <http://jmlr.org/papers/v17/14-237.html>`_

- `Gaussian-binary Restricted Boltzmann Machines on Modeling Natural Image statistics Wang, N., Melchior, J., & Wiskott, L.. (2014). (Vol. 1401.5900). arXiv.org e-Print archive. <http://arxiv.org/abs/1401.5900>`_

- `How to Center Binary Restricted Boltzmann Machines (Vol. 1311.1354). Melchior, J., Fischer, A., Wang, N., & Wiskott, L.. (2013). arXiv.org e-Print archive. <http://arxiv.org/pdf/1311.1354.pdf>`_

- `An Analysis of Gaussian-Binary Restricted Boltzmann Machines for Natural Images. Wang, N., Melchior, J., & Wiskott, L.. (2012). In Proc. 20th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, Apr 25–27, Bruges, Belgium (pp. 287–292). <https://www.ini.rub.de/PEOPLE/wiskott/Reprints/WangMelchiorEtAl-2012a-ProcESANN-RBMImages.pdf>`_

- `Learning Natural Image Statistics with Gaussian-Binary Restricted Boltzmann Machines. Melchior, J, 29.05.2012. Master’s thesis, Applied Computer Science, Univ. of Bochum, Germany. <https://www.ini.rub.de/PEOPLE/wiskott/Reprints/Melchior-2012-MasterThesis-RBMs.pdf>`_

IF you want to use PyDeep in your publication, you can cite it as follows.

.. code-block:: latex

   @misc{melchior2017pydeep,
         title={PyDeep},
         author={Melchior, Jan},
         year={2017},
         publisher={GitHub},
         howpublished={\url{https://github.com/MelJan/PyDeep.git}},
        }

Contact:

`Jan Melchior <https://www.ini.rub.de/the_institute/people/jan-melchior/>`_