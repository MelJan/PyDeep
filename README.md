# PyDeep 

PyDeep is a machine learning / deep learning library with focus on unsupervised learning especially Restricted Boltzmann machines.

So why another ML lib?

- First of all, the library will contain code I produced during my research that allows to reproduce the results in corresponding publications.
- If you simply want to use standard algorithms, use one of the big ML libs having GPU support, symbolic differentiation, etc. , this library is not meant to compete with those libraries!
- The focus is on well documented and modular code that allows you to understand the functionality and thus modify the code easily.

Feature list:
  Centered Binary-Binary RBMs
  CD and PCD sampling
  Centered gradient
  Calculate true partition function
  Calculate Log-likelihood
  Calculate reconstruction error
  Visualization, profiling, misc. tools

Being prepared for commit:
- Centered Gaussian-Binary RBMs
- Centered Gaussian-Binary RBMs with trainable variance
- Centered Binary-Rectifier RBMs
- Centered Rectifier-Binary RBMs
- Centered RectRect RBM (RR-RBM)
- Centered GaussianRectVariance RBM (GRV-RBM)
- Centered RBMs with additional label units
- Centered Softmax RBMs
- unit tests
- PCA
- ZCA
- Fast ICA
- Annealed inportant sampling
- Reverse annealed important sampling
- MDP wrapper
- Auto encoder


