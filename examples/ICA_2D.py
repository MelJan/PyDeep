""" Example for the Independent Component Analysis on a 2D example.

    :Version:
        1.1.0

    :Date:
        22.04.2017

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

"""

# Import numpy, numpy extensions, ZCA, ICA, 2D linear mixture, and visualization module
import numpy as numx
import pydeep.base.numpyextension as numxext
from pydeep.preprocessing import ZCA, ICA
from pydeep.misc.toyproblems import generate_2d_mixtures
import pydeep.misc.visualization as vis

# Set the random seed
# (optional, if stochastic processes are involved we get the same results)
numx.random.seed(42)

# Create 2D linear mixture
data, mixing_matrix = generate_2d_mixtures(50000, 0, 3.0)

# ZCA
zca = ZCA(data.shape[1])
zca.train(data)
data_zca = zca.project(data)

# ICA
ica = ICA(data_zca.shape[1])
ica.train(data_zca, iterations=1000)
data_ica = ica.project(data_zca)

# For better visualization the principal components are rescaled
scale_factor = 3

# Display results, the matrices are normalized such that the
# column norm equals the scale factor.
vis.figure(0, figsize=[7, 7])
vis.title("Data and mixing matrix")
vis.plot_2d_data(data)
vis.plot_2d_weights(numxext.resize_norms(mixing_matrix,
                                         norm=scale_factor,
                                         axis=0))
vis.axis('equal')
vis.axis([-4, 4, -4, 4])

vis.figure(1, figsize=[7, 7])
vis.title("Data and mixing matrix in whitened space")
vis.plot_2d_data(data_zca)
vis.plot_2d_weights(numxext.resize_norms(scale_factor * zca.project(mixing_matrix.T).T,
                                         norm=scale_factor,
                                         axis=0))
vis.axis('equal')
vis.axis([-4, 4, -4, 4])

vis.figure(2, figsize=[7, 7])
vis.title("Data and ica estimation of the mixing matrix in whitened space")
vis.plot_2d_data(data_zca)
vis.plot_2d_weights(numxext.resize_norms(scale_factor * ica.projection_matrix,
                                         norm=scale_factor,
                                         axis=0))
vis.axis('equal')
vis.axis([-4, 4, -4, 4])

vis.figure(3, figsize=[7, 7])
vis.title("Data and ica estimation of the mixing matrix")
vis.plot_2d_data(data)
vis.plot_2d_weights(
    numxext.resize_norms(scale_factor * zca.unproject(ica.projection_matrix.T).T,
                         norm=scale_factor,
                         axis=0))
vis.axis('equal')
vis.axis([-4, 4, -4, 4])

# Show all windows.
vis.show()
