""" Example for the Independent component analysis on natural image patches.

    :Version:
        1.1.0

    :Date:
        08.04.2017

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
# Import PCA, numpy, input output functions, and visualization functions
import numpy as numx
from pydeep.preprocessing import ICA, ZCA
import pydeep.misc.io as io
import pydeep.misc.visualization as vis

# Set the random seed (optional, if stochastic processes are involved we always get the same results)
numx.random.seed(42)

# Load the data
data = io.load_natural_image_patches('../../data/NaturalImage.mat')

# Specify image width and height for displaying
width = height = 14

# Create a ZCA node to whiten the data and train it (you could also use PCA whitened=True)
zca = ZCA(input_dim=width * height)
zca.train(data=data)

# ZCA projects the whitened data back to the original space, thus does not perform a
# dimensionality reduction but a whitening in the original space
whitened_data = zca.project(data)

# Create a ZCA node and train it (you could also use PCA whitened=True)
ica = ICA(input_dim=width * height)
ica.train(data=whitened_data,
          iterations=100,
          convergence=1.0,
          status=True)

# Show eigenvectors of the covariance matrix
eigenvectors = vis.tile_matrix_rows(matrix=zca.projection_matrix,
                                    tile_width=width,
                                    tile_height=height,
                                    num_tiles_x=width,
                                    num_tiles_y=height,
                                    border_size=1,
                                    normalized=True)
vis.imshow_matrix(matrix=eigenvectors,
                  windowtitle='Eigenvectors of the covariance matrix')

# Show whitened images
images = vis.tile_matrix_rows(matrix=data[0:width*height].T,
                              tile_width=width,
                              tile_height=height,
                              num_tiles_x=width,
                              num_tiles_y=height,
                              border_size=1,
                              normalized=True)
vis.imshow_matrix(matrix=images,
                  windowtitle='Some image patches')

# Show some whitened images
images = vis.tile_matrix_rows(matrix=whitened_data[0:width*height].T,
                              tile_width=width,
                              tile_height=height,
                              num_tiles_x=width,
                              num_tiles_y=height,
                              border_size=1,
                              normalized=True)
vis.imshow_matrix(matrix=images,
                  windowtitle='Some whitened image patches')

# Plot the cumulative sum of teh Eigenvalues.
eigenvalue_sum = numx.cumsum(zca.eigen_values)
vis.imshow_plot(matrix=eigenvalue_sum,
                windowtitle="Cumulative sum of Eigenvalues")
vis.xlabel("Eigenvalue index")
vis.ylabel("Sum of Eigenvalues 0 to index")

# Show the ICA filters/bases
ica_filters = vis.tile_matrix_rows(matrix=ica.projection_matrix,
                                   tile_width=width,
                                   tile_height=height,
                                   num_tiles_x=width,
                                   num_tiles_y=height,
                                   border_size=1,
                                   normalized=True)
vis.imshow_matrix(matrix=ica_filters,
                  windowtitle='Filters learned by ICA')

# Show all windows.
vis.show()
