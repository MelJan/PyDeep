import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pydeep",
    version = "1.0.0",
    author = "Jan Melchior",
    author_email = "JanMelchior@gmx.de",
    description = ("Machine learning library in particular for Restricted Boltzmann machines."),
    license = "GNU",
    keywords = "Resricted Boltzmann machines, Machine learning",
    url = "https://github.com/MelJan/PyDeep",
    packages=['pydeep','pydeep.base','pydeep.misc','pydeep.rbm','pydeep.examples.toyexamples','pydeep.examples'],
    py_modules=[],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU License",
    ],
)