""" This module provides functions for displaying and visualize data.
    It extends the matplotlib.pyplot.

    :Implemented:
        - Tile a matrix rows
        - Tile a matrix columns
        - Show a matrix
        - Show plot
        - Show a histogram

        - Plot data
        - Plot 2D weights
        - Plot PDF-contours

        - Show RBM parameters
        - hidden_activation
        - reorder_filter_by_hidden_activation
        - generate_samples

        - filter_frequency_and_angle
        - filter_angle_response
        - calculate_amari_distance
        - Show the tuning curves
        - Show the optimal gratings
        - Show the frequency angle histogram

    :Version:
        1.1.0

    :Date:
        19.03.2017

    :Author:
        Jan Melchior, Nan Wang

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
import copy

import numpy as numx
from matplotlib.pyplot import *

from pydeep.preprocessing import rescale_data


def tile_matrix_columns(matrix,
                        tile_width,
                        tile_height,
                        num_tiles_x,
                        num_tiles_y,
                        border_size=1,
                        normalized=True):
    """ Creates a matrix with tiles from columns.

    :param matrix: Matrix to display.
    :type matrix: numpy array 2D

    :param tile_width: Tile width dimension.
    :type tile_width: int

    :param tile_height: Tile height dimension.
    :type tile_height: int

    :param num_tiles_x: Number of tiles horizontal.
    :type num_tiles_x: int

    :param num_tiles_y: Number of tiles vertical.
    :type num_tiles_y: int

    :param border_size: Size of the border.
    :type border_size: int

    :param normalized: If true each image gets normalized to be between 0..1.
    :type normalized: bool

    :return: Matrix showing the 2D patches.
    :rtype: 2D numpy array
    """
    return tile_matrix_rows(matrix.T, tile_width, tile_height, num_tiles_x,
                            num_tiles_y, border_size, normalized)


def tile_matrix_rows(matrix,
                     tile_width,
                     tile_height,
                     num_tiles_x,
                     num_tiles_y,
                     border_size=1,
                     normalized=True):
    """ Creates a matrix with tiles from rows.

    :param matrix: Matrix to display.
    :type matrix: numpy array 2D

    :param tile_width: Tile width dimension.
    :type tile_width: int

    :param tile_height: Tile height dimension.
    :type tile_height: int

    :param num_tiles_x: Number of tiles horizontal.
    :type num_tiles_x: int

    :param num_tiles_y: Number of tiles vertical.
    :type num_tiles_y: int

    :param border_size: Size of the border.
    :type border_size: int

    :param normalized: If true each image gets normalized to be between 0..1.
    :type normalized: bool

    :return: Matrix showing the 2D patches.
    :rtype: 2D numpy array
    """
    if normalized is True:
        result = np.max(rescale_data(matrix))
    else:
        result = np.max(matrix)
    result *= np.ones( (int(tile_width * num_tiles_x + (num_tiles_x - 1) * border_size),
                        int(tile_height * num_tiles_y + (num_tiles_y - 1) * border_size)))
    for x in range(int(num_tiles_x)):
        for y in range(int(num_tiles_y)):
            single_image = matrix[:, (x * num_tiles_y) + y].reshape(tile_width,
                                                                    tile_height)
            if normalized is True:
                result[x * (tile_width + border_size):x * (tile_width + border_size) + tile_width, y * (
                    tile_height + border_size):y * (tile_height + border_size) + tile_height] = rescale_data(
                    single_image)
            else:
                result[x * (tile_width + border_size):x * (tile_width + border_size) + tile_width, y * (
                    tile_height + border_size):y * (tile_height + border_size) + tile_height] = single_image
    return result


def imshow_matrix(matrix, windowtitle, interpolation='nearest'):
    """ Displays a matrix in gray-scale.

    :param matrix: Data to display
    :type matrix: numpy array

    :param windowtitle: Figure title
    :type windowtitle: string

    :param interpolation: Interpolation style
    :type interpolation: string
    """
    figure(windowtitle)#.suptitle()
    axis('off')
    gray()
    imshow(np.array(matrix, np.float64), interpolation=interpolation)


def imshow_plot(matrix, windowtitle):
    """ Plots the colums of a matrix.

    :param matrix: Data to plot
    :type matrix: numpy array

    :param windowtitle: Figure title
    :type windowtitle: string
    """
    figure().suptitle(windowtitle)
    gray()
    plot(np.array(matrix, np.float64))


def imshow_histogram(matrix,
                     windowtitle,
                     num_bins=10,
                     normed=False,
                     cumulative=False,
                     log_scale=False):
    """ Shows a image of the histogram.

    :param matrix: Data to display
    :type matrix: numpy array 2D

    :param windowtitle: Figure title
    :type windowtitle: string

    :param num_bins: Number of bins
    :type num_bins: int

    :param normed: If true histogram is being normed to 0..1
    :type normed: bool

    :param cumulative: Show cumulative histogram
    :type cumulative: bool

    :param log_scale: Use logarithm Y-scaling
    :type log_scale: bool
    """
    figure().suptitle(windowtitle)
    hist(matrix, bins=num_bins, normed=normed, cumulative=cumulative, log=log_scale)


def plot_2d_weights(weights,
                    bias=np.zeros((1, 2)),
                    scaling_factor=1.0,
                    color='random',
                    bias_color='random'):
    """

    :param weights: Weight matrix (weights per column).
    :type weights: numpy array [2,2]

    :param bias: Bias value.
    :type bias: numpy array [1,2]

    :param scaling_factor: If not 1.0 the weights will be scaled by this factor.
    :type scaling_factor: float

    :param color: Color for the weights.
    :type color: string

    :param bias_color: Color for the bias.
    :type bias_color: string
    """
    width = 0.02
    hw = 0.0
    if np.sqrt(bias[0, 0] * bias[0, 0] + bias[0, 1] * bias[0, 1]) > hw:
        if bias_color is 'random':
            colorrgb = [np.random.rand(), np.random.rand(), np.random.rand()]
            arrow(0.0, 0.0, bias[0, 0], bias[0, 1], color=colorrgb, width=width, length_includes_head=True,
                  head_width=hw)
        else:
            arrow(0.0, 0.0, bias[0, 0], bias[0, 1], color=bias_color, width=width, length_includes_head=True,
                  head_width=hw)

    if color is 'random':
        for c in range(weights.shape[1]):
            colorrgb = [np.random.rand(), np.random.rand(), np.random.rand()]
            if np.sqrt(weights[0, c] * weights[0, c] + weights[1, c] * weights[1, c]) > hw:
                arrow(bias[0, 0], bias[0, 1], scaling_factor * weights[0, c],
                      scaling_factor * weights[1, c], color=colorrgb, width=width,
                      length_includes_head=True, head_width=hw)
    else:
        for c in range(weights.shape[1]):
            if np.sqrt(weights[0, c] * weights[0, c] + weights[1, c] * weights[1, c]) > hw:
                arrow(bias[0, 0], bias[0, 1], scaling_factor * weights[0, c],
                      scaling_factor * weights[1, c], color=color, width=width, length_includes_head=True,
                      head_width=hw)


def plot_2d_data(data,
                 alpha=0.1,
                 color='navy',
                 point_size=5):
    """ Plots the data into the current figure.

    :param data: Data matrix (Datapoint x dimensions).
    :type data: numpy array

    :param alpha: ranspary value 0.0 = invisible, 1.0 = solid.
    :type alpha: float

    :param color: Color for the data points.
    :type color: string (color name)

    :param point_size: Size of the data points.
    :type point_size: int
    """
    scatter(data[:, 0], data[:, 1], s=point_size, c=color, marker='o', alpha=alpha, linewidth=0)


def plot_2d_contour(probability_function,
                    value_range=[-5.0, 5.0, -5.0, 5.0],
                    step_size=0.01,
                    levels=20,
                    stylev=None,
                    colormap='jet'):
    """ Plots the data into the current figure.

    :param probability_function: Probability function must take 2D array [number of datapoint x 2]
    :type probability_function: python method

    :param value_range: Min x, max x , min y, max y.
    :type value_range: list with four float entries

    :param step_size: Step size for evaluating the pdf.
    :type step_size: float

    :param levels: Number of contour lines or array of contour height.
    :type levels: int

    :param stylev: None as normal contour, 'filled' as filled contour, 'image' as contour image
    :type stylev: string or None

    :param colormap: Selected colormap .. seealso:: http://www.scipy.org/Cookbook/Matplotlib/.../Show_colormaps
    :type colormap: string
    """
    # Generate x,y coordinates
    # Suprisingly using the stepsize of
    # arange directly does not work ( np.arange(min_x,max_x,step_size) )
    x = np.arange(value_range[0] / step_size, (value_range[1] + step_size) / step_size, 1.0) * step_size
    y = np.arange(value_range[2] / step_size, (value_range[3] + step_size) / step_size, 1.0) * step_size

    # Get distance or range
    dist_x = x.shape[0]
    dist_y = y.shape[0]

    # Generate 2D coordinate grid
    tablei = np.indices((dist_x, dist_y), dtype=np.float64)

    # Modify data to certain range
    tablei[0, :] = tablei[0, :] * step_size + value_range[0]
    tablei[1, :] = tablei[1, :] * step_size + value_range[2]

    # Reshape the array having all possible combination in a 2D array
    data = tablei.reshape(2, dist_x * dist_y)

    # we need to flip the first and second dimension to get the
    # ordering of x,y
    data = np.vstack((data[1, :], data[0, :]))

    # Compute PDF value for all combinations
    z = probability_function(data.T).reshape(dist_x, dist_y)

    # Set colormap ans scaling behaviour
    set_cmap(colormap)

    # Plot contours
    if stylev is 'filled':
        contourf(x, y, z, levels=list(np.linspace(np.min(z), np.max(z), levels)), origin='lower')
    elif stylev is 'image':
        imshow(z, origin='lower', extent=value_range)
    else:
        contour(x, y, z, levels=list(np.linspace(np.min(z), np.max(z), levels)), origin='lower')


def imshow_standard_rbm_parameters(rbm,
                                   v1,
                                   v2,
                                   h1,
                                   h2,
                                   whitening=None,
                                   window_title=""):
    """ Saves the weights and biases of a given RBM at the given location.

    :param rbm: RBM which weights and biases should be saved.
    :type rbm: RBM object

    :param v1: Visible bias and the single weights will be saved as an image with size
    :type v1: int

    :param v2: Visible bias and the single weights will be saved as an image with size
    :type v2: int

    :param h1: Hidden bias and the image containing all weights will be saved as an image with size h1 x h2.
    :type h1: int

    :param h2: Hidden bias and the image containing all weights will be saved as an image with size h1 x h2.
    :type h2: int

    :param whitening: If the data is PCA whitened it is useful to dewhiten the filters to wee the structure!
    :type whitening: preprocessing object or None

    :param window_title: Title for this rbm.
    :type window_title: string
    """
    if whitening is not None:
        imshow_matrix(tile_matrix_rows(whitening.unproject(rbm.w.T).T, v1, v2, h1, h2, border_size=1, normalized=True),
                      window_title + ' Normalized Weights')
        imshow_matrix(tile_matrix_rows(whitening.unproject(rbm.w.T).T, v1, v2, h1, h2, border_size=1, normalized=False),
                      window_title + ' Unormalized Weights')
        imshow_matrix(whitening.unproject(rbm.bv).reshape(v1, v2), window_title + ' Visible Bias')
        imshow_matrix(whitening.unproject(rbm.ov).reshape(v1, v2), window_title + ' Visible Offset')
    else:
        imshow_matrix(tile_matrix_rows(rbm.w, v1, v2, h1, h2, border_size=1, normalized=True), window_title +
                      ' Normalized Weights')
        imshow_matrix(tile_matrix_rows(rbm.w, v1, v2, h1, h2, border_size=1, normalized=False), window_title +
                      ' Unormalized Weights')
        imshow_matrix(rbm.bv.reshape(v1, v2), window_title + ' Visible Bias')
        imshow_matrix(rbm.ov.reshape(v1, v2), window_title + ' Visible Offset')

    imshow_matrix(rbm.bh.reshape(h1, h2), window_title + ' Hidden Bias')
    imshow_matrix(rbm.oh.reshape(h1, h2), window_title + ' Hidden Offset')

    if hasattr(rbm, 'variance'):
        imshow_matrix(rbm.variance.reshape(v1, v2), window_title + ' Variance')


def hidden_activation(rbm, data, states=False):
    """ Calculates the hidden activation.

    :param rbm: RBM model object.
    :type rbm: RBM model object

    :param data: Data for the activation calculation.
    :type data: numpy array [num samples, dimensions]

    :param states: If True uses states rather then probabilities by rounding to 0 or 1.
    :type states: bool

    :return: hidden activation and the mean and standard deviation over the data.
    :rtype: numpy array, float, floa
    """
    activation = rbm.probability_h_given_v(data)
    if states:
        activation = numx.round(activation)
    return activation, numx.mean(activation, axis=0), numx.std(activation, axis=0)


def reorder_filter_by_hidden_activation(rbm, data):
    """ Reorders the weights by its activation over the data set in decreasing order.

    :param rbm: RBM model object.
    :type rbm: RBM model object

    :param data: Data for the activation calculation.
    :type data: numpy array [num samples, dimensions]

    :return: RBM with reordered weights.
    :rtype: RBM object.
    """
    probs = numx.sum(rbm.probability_h_given_v(data), axis=0)
    index = numx.argsort(probs, axis=0)
    rbm_ordered = copy.deepcopy(rbm)
    for i in range(probs.shape[0]):
        u = probs.shape[0] - i - 1
        rbm_ordered.w[:, u] = rbm.w[:, index[i]]
        rbm_ordered.bh[0, u] = rbm.bh[0, index[i]]
    return rbm_ordered


def generate_samples(rbm,
                     data,
                     iterations,
                     stepsize,
                     v1,
                     v2,
                     sample_states=False,
                     whitening=None):
    """ Generates samples from the given RBM model.

    :param rbm: RBM model.
    :type rbm: RBM model object.

    :param data: Data to start sampling from.
    :type data: numpy array [num samples, dimensions]

    :param iterations: Number of Gibbs sampling steps.
    :type iterations: int

    :param stepsize: After how many steps a sample should be plotted.
    :type stepsize: int

    :param v1: X-Axis of the reorder image patch.
    :type v1: int

    :param v2: Y-Axis of the reorder image patch.
    :type v2: int

    :param sample_states: If true returns the sates , probabilities otherwise.
    :type sample_states: bool

    :param whitening: If the data has been preprocessed it needs to be undone.
    :type whitening: preprocessing object or None

    :return: Matrix with image patches order along X-Axis and it's evolution in Y-Axis.
    :rtype: numpy array
    """
    result = data
    if whitening is not None:
        result = whitening.unproject(data)
    vis_states = data
    for i in range(1, iterations + 1):
        hid_probs = rbm.probability_h_given_v(vis_states)
        hid_states = rbm.sample_h(hid_probs)
        vis_probs = rbm.probability_v_given_h(hid_states)
        vis_states = rbm.sample_v(vis_probs)
        if i % stepsize == 0:
            if whitening is not None:
                if sample_states:
                    result = numx.vstack((result,
                                          whitening.unproject(vis_states)))
                else:
                    result = numx.vstack((result,
                                          whitening.unproject(vis_probs)))
            else:
                if sample_states:
                    result = numx.vstack((result, vis_states))
                else:
                    result = numx.vstack((result, vis_probs))
    return tile_matrix_rows(result.T, v1, v2, iterations / stepsize + 1, data.shape[0], border_size=1, normalized=False)


def imshow_filter_tuning_curve(filters, num_of_ang=40):
    """ Plot the tuning curves of the filter's changes in frequency and angles.

    :param filters: Filters to analyze.
    :type filters: numpy array

    :param num_of_ang: Number of orientations to check.
    :type num_of_ang: int
    """
    # something here
    input_dim = filters.shape[0]
    output_dim = filters.shape[1]
    max_wavelength = int(np.sqrt(input_dim))
    frq_rsp, _ = filter_frequency_response(filters, num_of_ang)
    ang_rsp, _ = filter_angle_response(filters, num_of_ang)
    figure().suptitle('Tuning curves')
    for plot_idx in range(output_dim):
        subplot(output_dim, 2, 2 * plot_idx + 1)
        plot(range(2, max_wavelength + 1), frq_rsp[:, plot_idx])
        subplot(output_dim, 2, 2 * plot_idx + 2)
        plot(np.array(range(0, num_of_ang)) * np.pi / num_of_ang,
             ang_rsp[:, plot_idx])


def imshow_filter_optimal_gratings(filters, opt_frq, opt_ang):
    """ Plot the filters and corresponding optimal gating pattern.

    :param filters: Filters to analyze.
    :type filters: numpy array

    :param opt_frq: Optimal frequencies.
    :type opt_frq: int

    :param opt_ang: Optimal frequencies.
    :type opt_ang: int
    """
    # something here
    input_dim = filters.shape[0]
    output_dim = filters.shape[1]
    max_wavelength = int(np.sqrt(input_dim))

    frqmatrix = np.tile(float(1) / opt_frq * np.pi * 2, (input_dim, 1))
    thetamatrix = np.tile(opt_ang, (input_dim, 1))
    vec_xy = np.array(range(0, input_dim))
    vec_x = np.floor_divide(vec_xy, max_wavelength)
    vec_y = vec_xy + 1 - vec_x * max_wavelength
    xmatrix = np.tile(vec_x, (output_dim, 1))
    ymatrix = np.tile(vec_y, (output_dim, 1))
    gratingmatrix = np.cos(frqmatrix * (np.sin(thetamatrix) * xmatrix.transpose() + np.cos(thetamatrix
                                                                                           ) * ymatrix.transpose()))
    combinedmatrix = np.concatenate((rescale_data(filters), rescale_data(gratingmatrix)), 1)
    imshow_matrix(tile_matrix_rows(combinedmatrix, max_wavelength, max_wavelength, 2, output_dim, border_size=1,
                                   normalized=False), 'optimal grating')


def imshow_filter_frequency_angle_histogram(opt_frq,
                                            opt_ang,
                                            max_wavelength=14):
    """ lots the histograms of the optimal frequencies and angles.

    :param opt_frq: Optimal frequencies.
    :type opt_frq: int

    :param opt_ang: Optimal angle.
    :type opt_ang: int

    :param max_wavelength: Maximal wavelength.
    :type max_wavelength: int
    """
    figure().suptitle('Filter Frequency histogram \t\t\t Filter Angle ' + 'histogram')
    subplot(1, 2, 1)
    hist(opt_frq, max_wavelength - 1, (2, 14), normed=1)
    ylim((0, 1))
    subplot(1, 2, 2)
    hist(opt_ang, 20, (0, np.pi), normed=1)
    ylim((0, 1))


def filter_frequency_and_angle(filters, num_of_angles=40):
    """ Analyze the filters by calculating the responses when gratings, i.e. sinusoidal functions, are input to them.

    :Info: Hyv/"arinen, A. et al. (2009) Natural image statistics, Page 144-146

    :param filters: Filters to analyze
    :type filters: numpy array

    :param num_of_angles: Number of angles steps to check
    :type num_of_angles: int

    :return: The optimal frequency (pixels/cycle) of the filters, the optimal orientation angle (rad) of the filters
    :rtype: numpy array, numpy array
    """
    rsp_max_ang, rsp_max_ang_idx = filter_frequency_response(filters, num_of_angles)
    opt_frq = rsp_max_ang.argmax(0) + 2
    opt_ang = numx.diag(rsp_max_ang_idx[opt_frq - 2][:]) * numx.pi / num_of_angles
    return opt_frq, opt_ang


def filter_frequency_response(filters, num_of_angles=40):
    """ Compute the response of filters w.r.t. different frequency.

    :param filters: Filters to analyze
    :type filters: numpy array

    :param num_of_angles: Number of angles steps to check
    :type num_of_angles: int

    :return: Frequency response as output_dim x max_wavelength-1 index of the
    :rtype: numpy array, numpy array
    """
    input_dim = filters.shape[0]  # input dimensionality, 196
    max_wavelength = int(numx.sqrt(filters.shape[0]))
    output_dim = filters.shape[1]
    frq_rsp = numx.zeros([max_wavelength - 1, output_dim])
    frq_rsp_ang_idx = numx.zeros([max_wavelength - 1, output_dim])

    vec_theta = numx.array(range(0, num_of_angles))
    vec_theta = vec_theta * numx.pi / num_of_angles
    sinmatrix = numx.tile(numx.sin(vec_theta), (input_dim, 1))
    cosmatrix = numx.tile(numx.cos(vec_theta), (input_dim, 1))

    vec_xy = numx.array(range(0, input_dim))
    vec_x = numx.floor_divide(vec_xy, max_wavelength)
    vec_y = vec_xy + 1 - vec_x * max_wavelength
    xmatrix = numx.tile(vec_x.transpose(), (num_of_angles, 1))
    ymatrix = numx.tile(vec_y.transpose(), (num_of_angles, 1))
    umatrix = sinmatrix.transpose() * xmatrix + cosmatrix.transpose() * ymatrix

    for frq_idx in range(2, max_wavelength + 1):
        alpha = float(1) / float(frq_idx)
        # sine gratings of all angles under a specific freq.
        gratingmatrix_sin = numx.sin(2 * numx.pi * alpha * umatrix)
        # cosine gratings of all angles under a specific freq.
        gratingmatrix_cos = numx.cos(2 * numx.pi * alpha * umatrix)
        rsp_fix_frq = (numx.dot(gratingmatrix_sin, filters) ** 2 + numx.dot(gratingmatrix_cos, filters) ** 2)
        frq_rsp[frq_idx - 2] = rsp_fix_frq.max(0)
        frq_rsp_ang_idx[frq_idx - 2] = rsp_fix_frq.argmax(0)

    return frq_rsp, frq_rsp_ang_idx


def filter_angle_response(filters, num_of_angles=40):
    """ Compute the angle response of the given filter.

    :param filters: Filters to analyze
    :type filters: numpy array

    :param num_of_angles: Number of angles steps to check
    :type num_of_angles: int

    :return: Angle response as output_dim x num_of_ang, index of angles
    :rtype: numpy array, numpy array
    """
    input_dim = filters.shape[0]
    max_wavelength = int(numx.sqrt(filters.shape[0]))
    output_dim = filters.shape[1]
    ang_rsp = numx.zeros([num_of_angles, output_dim])
    ang_rsp_frq_idx = numx.zeros([num_of_angles, output_dim])

    vec_frq = numx.array(range(2, max_wavelength + 1))
    vec_frq = float(1) / vec_frq
    frqmatrix = numx.tile(vec_frq * numx.pi * 2, (input_dim, 1))

    vec_xy = numx.array(range(0, input_dim))
    vec_x = numx.floor_divide(vec_xy, max_wavelength)
    vec_y = vec_xy + 1 - vec_x * max_wavelength
    xmatrix = numx.tile(vec_x.transpose(), (max_wavelength - 1, 1))
    ymatrix = numx.tile(vec_y.transpose(), (max_wavelength - 1, 1))

    for ang_idx in range(0, num_of_angles):
        theta = ang_idx * numx.pi / num_of_angles
        umatrix = numx.sin(theta) * xmatrix + numx.cos(theta) * ymatrix
        gratingmatrix_sin = numx.sin(frqmatrix.transpose() * umatrix)
        gratingmatrix_cos = numx.cos(frqmatrix.transpose() * umatrix)
        rsp_fix_ang = (numx.dot(gratingmatrix_sin, filters) ** 2
                       + numx.dot(gratingmatrix_cos, filters) ** 2)
        ang_rsp[ang_idx] = rsp_fix_ang.max(0)
        ang_rsp_frq_idx[ang_idx] = rsp_fix_ang.argmax(0)

    return ang_rsp, ang_rsp_frq_idx


def calculate_amari_distance(matrix_one,
                             matrix_two,
                             version=1):
    """ Calculate the Amari distance between two input matrices.

    :param matrix_one: the first matrix
    :type matrix_one: numpy array

    :param matrix_two: the second matrix
    :type matrix_two: numpy array

    :param version: Variant to use.
    :type version: int

    :return: The amari distance between two input matrices.
    :rtype: float
    """
    if matrix_one.shape != matrix_two.shape:
        return "Two matrices must have the same shape."
    product_matrix = numx.abs(numx.dot(matrix_one,
                                       numx.linalg.inv(matrix_two)))

    product_matrix_max_col = numx.array(product_matrix.max(0))
    product_matrix_max_row = numx.array(product_matrix.max(1))

    n = product_matrix.shape[0]

    if version != 1:
        """ Formula from Teh
        Here they refered to as "amari distance"
        The value is in [2*N-2N^2, 0].
        reference:
            Teh, Y. W.; Welling, M.; Osindero, S. & Hinton, G. E. Energy-based
            models for sparse overcomplete representations J MACH LEARN RES,
            2003, 4, 1235--1260
        """
        amari_distance = product_matrix / numx.tile(product_matrix_max_col, (n, 1))
        amari_distance += product_matrix / numx.tile(product_matrix_max_row, (n, 1)).T
        amari_distance = amari_distance.sum() - 2 * n * n
    else:
        """ Formula from ESLII
        Here they refered to as "amari error"
        The value is in [0, N-1].
        reference:
            Bach, F. R.; Jordan, M. I. Kernel Independent Component
            Analysis, J MACH LEARN RES, 2002, 3, 1--48
        """
        amari_distance = product_matrix / numx.tile(product_matrix_max_col, (n, 1))
        amari_distance += product_matrix / numx.tile(product_matrix_max_row, (n, 1)).T
        amari_distance = amari_distance.sum() / (2 * n) - 1
    return amari_distance
