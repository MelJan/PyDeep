# PyDeep 

PyDeep is a machine learning / deep learning library with focus on unsupervised learning and comprehensibility of the code.

So why another ML lib? 

- First of all, the library will contain code I produced during my research that allows to reproduce the results in corresponding publications.
- If you simply want to use standard algorithms, use one of the big ML libs having GPU support, symbolic differentiation, etc. , this library is not meant to compete with those libraries! 
- The focus is on well documented and modular code that allows you to understand the functionality and thus modify the code easily.
- Let's say you studied an algorithm theoretically and you now want play around with an implementation. Usually you have two choices.
Either you go with one of the big ML libs, which have the advantage of speed and correctness (i.e. GPU suppport and symbolic differentiation), but which are often very hard to understand, modify, and debug (often not at least because of missing in-code documentation) without actually getting a developer/contributor. 
Or you can use code snips where it is usually easy to understand the main aspects of an algorithm, but which are often slow, limited, and not checked for correctness.
PyDeep tries to fill this gap by providing well documented and modular code that has the flexibility to be modified to your needs.
