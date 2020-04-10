''' Setup file for PyDeep.

    :Version:
        1.1.0

    :Date:
        04.04.2017

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2017 Jan Melchior

        This file is part of the Python library PyDeep.

        PyDeep is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''

import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

packages = ['pydeep',
            'pydeep.ae',
            'pydeep.base',
            'pydeep.dbm',
            'pydeep.dbm.binary3Layer',
            'pydeep.fnn',
            'pydeep.misc',
            'pydeep.rbm',
            ]
            
setup(
    name="PyDeep",
    version="1.2.0",
    author="Jan Melchior",
    author_email="JanMelchior@gmx.de",
    description=("Machine learning library with focus on Restricted Boltzmann machines."),
    license="GNU",
    keywords="Machine learning, Deep Learning, Feed-Forward Neural Networks, Autoencoder, "
             "Restricted Boltzmann machines, Deep Boltzmann machines, Independent Component Analysis, "
             "Principal Component Analysis, Zero-Phase Component Analysis",
    url="https://github.com/MelJan/PyDeep",
    packages=packages,
    py_modules=[],
    setup_requires=['numpy','scipy','matplotlib'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU License",
    ],
)
