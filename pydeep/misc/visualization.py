''' This module provides functions for displaying and visualize data. 
    It extends the matplotlib.pyplot. 

    :Implemented:
        - tile a matrix
        - Plot data, weights and PDF-contours
        - Show a matrix, histogram
        - Show RBM parameters
        - Show the tuning curves
        - Show the optimal gratings
        - Show the frequency angle histogram

    :Version:
        1.0

    :Date:
        06.06.2016

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2016 Jan Melchior

        This program is free software: you can redistribute it and/or modify
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
from matplotlib.pyplot import *
from pydeep.preprocessing import rescale_data 
import pydeep.misc.statistics as Statistics

def tile_matrix_columns(matrix, 
                        tile_width, 
                        tile_height, 
                        num_tiles_x, 
                        num_tiles_y, 
                        border_size=1, 
                        normalized = True):
    ''' Creates a matrix with tiles from columns.
        
    :Parameters:
        matrix:      Matrix to display.
                    -type: numpy array 2D
                     
        tile_width:  tile width dimension.
                    -type: int
                     
        tile_height: tile height  dimension. 
                    -type: int
                     
        num_tiles_x: Number of tiles horizontal.
                    -type: int        
        
        num_tiles_y: Number of tiles vertical.
                    -type: int
                     
        border_size: Size of the border.
                    -type: int
                     
        normalized:  If true each image gets normalized to be between 0..1.
                    -type: bool
                     
    :Returns:
        Image showing the 2D patches.
       -type: 2D numpy array 
        
    '''
    return tile_matrix_rows(matrix.T, tile_width, tile_height, num_tiles_x, 
                            num_tiles_y, border_size, normalized)

def tile_matrix_rows(matrix, 
                     tile_width, 
                     tile_height, 
                     num_tiles_x, 
                     num_tiles_y, 
                     border_size=1,
                     normalized = True):
    ''' Creates a matrix with tiles from rows.
        
    :Parameters:
        matrix:      Matrix to display.
                    -type: numpy array 2D
                     
        tile_width:  tile width dimension.
                    -type: int
                     
        tile_height: tile height  dimension. 
                    -type: int
                     
        num_tiles_x: Number of tiles horizontal.
                    -type: int        
        
        num_tiles_y: Number of tiles vertical.
                    -type: int
                     
        border_size: Size of the border.
                    -type: int
                     
        normalized:  If true each image gets normalized to be between 0..1.
                    -type: bool
                     
    :Returns:
        Image showing the 2D patches.
       -type: numpy array 2D 
        
    '''
    if(normalized==True):
        result = np.max(rescale_data(matrix))
    else:
        result = np.max(matrix)
    result *= np.ones((tile_width*num_tiles_x+(num_tiles_x-1)
                       *border_size, tile_height*num_tiles_y
                       +(num_tiles_y-1)*border_size))
    for x in xrange(num_tiles_x) : 
        for y in xrange(num_tiles_y) :
            single_image = matrix[:,(x*num_tiles_y)+y].reshape(tile_width,
                                                               tile_height)
            if(normalized==True):
                result[x*(tile_width + border_size):x*(tile_width 
                + border_size)+tile_width, y*(tile_height + border_size ):y
                *(tile_height + border_size )+tile_height] =  rescale_data(
                                                                single_image)
            else:
                result[x*(tile_width + border_size):x*(tile_width 
                + border_size)+tile_width, y*(tile_height + border_size ):y
                *(tile_height + border_size )+tile_height] =  single_image
    return result

def imshow_matrix(matrix, 
                  title,
                  interpolation='nearest'):
    """ Displays a matrix in gray-scale.
    
    :Parameters:
        matrix         Data to display
                      -type: numpy array
                       
        title:         Figure title
                      -type: string
        
        interpolation: Interpolation style
                      -type: string
        
    """
    figure().suptitle(title)       
    gray()    
    imshow(np.array(matrix,np.float64),interpolation=interpolation)
 
def imshow_histogram(matrix, 
                     title, 
                     num_bins = 10, 
                     normed = False, 
                     cumulative = False, 
                     log_scale = False):
    ''' Shows a image of the histogram.
    
    :Parameters:
        matrix:     Data to display
                   -type: numpy array 2D
        
        title:      Figure title
                   -type: string
        
        num_bins:   Number of bins
                   -type: int
        
        normed:     If true histogram is being normed to 0..1
                   -type: bool
        
        cumulative: Show cumulative histogram
                   -type: bool
        
        log_scale:  Use logarithm Y-scaling
                   -type: bool     
             
    '''
    figure().suptitle(title)
    hist(matrix, bins = num_bins, normed = normed,  cumulative = cumulative,
         log = log_scale)

def plot_2d_weights(weights, 
                    bias = np.zeros((1,2)), 
                    scaling_factor = 1.0, 
                    color = 'random' , 
                    bias_color = 'random'):
    ''' Plots the weights of an RBM,GMM or ICA into the current figure.
        
        :Parameters:
            weights:        Weight matrix (weights per column).
                           -type: numpy array [2,2]
                            
            bias:           Bias value.
                           -type: numpy array [1,2]
            
            color:          Color for the weights.
                           -type: string (color name)
            
            bias_color:     Color for the bias.
                           -type: string (color name)
            
            scaling_factor: If not 1.0 the weights will be scaled by this 
                            factor.
                           -type: float
                 
    ''' 
    width = 0.02
    hw = 0.0
    if np.sqrt(bias[0,0]*bias[0,0]+bias[0,1]*bias[0,1]) > hw:
        if bias_color == 'random':
            colorRGB = [np.random.rand(),np.random.rand(),np.random.rand()]
            arrow(0.0,0.0,bias[0,0],bias[0,1],color = colorRGB,
                  edgecolor=colorRGB,width=width,length_includes_head=True,
                  head_width = hw)
        else:
            arrow(0.0,0.0,bias[0,0],bias[0,1],color=bias_color,
                  edgecolor=bias_color,width=width,length_includes_head=True,
                  head_width = hw)

    if color == 'random':
        for c in range(weights.shape[1]):
            colorRGB = [np.random.rand(),np.random.rand(),np.random.rand()]
            if np.sqrt(weights[0,c]*weights[0,c]
                       +weights[1,c]*weights[1,c]) > hw:
                arrow(bias[0,0],bias[0,1],scaling_factor*weights[0,c],
                      scaling_factor*weights[1,c],color = colorRGB,
                      edgecolor=colorRGB,width=width,
                      length_includes_head=True,head_width = hw)
    else:
        for c in range(weights.shape[1]):
            if np.sqrt(weights[0,c]*weights[0,c]
                       +weights[1,c]*weights[1,c]) > hw:
                arrow(bias[0,0],bias[0,1],scaling_factor*weights[0,c],
                      scaling_factor*weights[1,c],color = color,
                      edgecolor=color,width=width,length_includes_head=True,
                      head_width = hw)
    
def plot_2d_data(data, 
                 alpha = 0.1, 
                 color = 'navy' , 
                 point_size = 5 ):
    ''' Plots the data into the current figure.
        
        :Parameters:
            data:        Data matrix (Datapoint x dimensions). 
                        -type: numpy array    
            
            alpha:       Transpary value 0.0 = invisible, 1.0 = solid.
                        -type: float 
                                
            color:       Color for the data points.
                        -type: string (color name)    
            
            point_size:  Size of the data points.
                        -type: int
                         
    ''' 
    scatter(data[:,0], data[:,1],s = point_size,c = color, marker = 'o', 
            alpha = alpha,linewidth = 0)
    
def plot_2d_contour(probability_function, 
                    value_range = [-5.0, 5.0 , -5.0,  5.0], 
                    step_size = 0.01, 
                    levels = 20,
                    style = None, 
                    colormap = 'jet'):
    ''' Plots the data into the current figure.
        
        :Parameters:
            probability_function: Probability function must take 2D array 
                                  [number of datapoint x 2] 
                                 -type: python method
                                  
            value_range:          Min x, max x , min y, max y .
                                 -type: list with four float entries 
            
            step_size:            Step size for evaluating the pdf.
                                 -type: float
            
            levels:               Number of contour lines or array of contour
                                  height.  
                                 -type: int
            
            style = None:         None as normal contour, 'filled' as filled 
                                  contour, 'image' as contour image
                                 -type: string or None
            
            colormap:             Selected colormap see: 
                                  http://www.scipy.org/Cookbook/Matplotlib/...
                                  ...Show_colormaps
                                 -type: string
                                  
    ''' 
    # Generate x,y coordinates
    # Suprisingly using the stepsize of 
    # arange directly does not work ( np.arange(min_x,max_x,step_size) )
    x = np.arange(value_range[0]/step_size, (value_range[1]+step_size)
                  /step_size, 1.0)* step_size
    y = np.arange(value_range[2]/step_size, (value_range[3]+step_size)
                  /step_size, 1.0)* step_size

    # Get distance or range
    dist_x = x.shape[0]
    dist_y = y.shape[0]

    # Generate 2D coordinate grid
    table = np.indices((dist_x,dist_y), dtype = np.float64)

    # Modify data to certain range 
    table[0,:] = table[0,:] * step_size + value_range[0]
    table[1,:] = table[1,:] * step_size + value_range[2]
    
    # Reshape the array having all possible combination in a 2D array
    data = table.reshape(2,dist_x*dist_y)
    
    # we need to flip the first and second dimension to get the 
    # ordering of x,y
    data = np.vstack((data[1,:],data[0,:]))
    
    # Compute PDF value for all combinations
    z = probability_function(data.T).reshape(dist_x,dist_y)
    
    # Set colormap ans scaling behaviour
    set_cmap(colormap)
            
    # Plot contours
    if style == 'filled':
        contourf(x,y,z, levels = list(np.linspace(np.min(z), 
                                      np.max(z),levels)) ,origin='lower')
    elif style == 'image':
        imshow(z, origin='lower', extent=value_range)  
    else:
        contour(x,y,z, levels = list(np.linspace(np.min(z), np.max(z),
                                                 levels)) ,origin='lower')
     
def imshow_standard_rbm_parameters(rbm, 
                                   v1, 
                                   v2, 
                                   h1, 
                                   h2, 
                                   whitening = None, 
                                   title = ""):
    ''' Saves the weights and biases of a given RBM at the given location.
    
    :Parameters:
        rbm:         RBM which weights and biases should be saved.
                    -type: RBM object
        
        v1:          Visible bias and the single weights will be saved as an 
                     image with size 
                     v1 x v2.
                    -type: int
        
        v2:          Visible bias and the single weights will be saved as an 
                     image with size 
                     v1 x v2.
                    -type: int
        
        h1:          Hidden bias and the image containing all weights will be
                     saved as an image with size h1 x h2.
                    -type: int        
        
        h2:          Hidden bias and the image containing all weights will be
                     saved as an image with size h1 x h2.
                    -type: int        
        
        whitening:   if the data is PCA whitened it is useful to dewhiten the
                     filters to wee the structure!
                    -type: preprocessing object or None
                    
        title:       Title for this rbm.
                    -type: string
    
    '''
    if whitening != None:
        imshow_matrix(tile_matrix_rows(whitening.unproject(rbm.w.T).T, v1,v2,
                                   h1, h2, border_size = 1,normalized = True), 
                                       title+' Normalized Weights')
        imshow_matrix(tile_matrix_rows(whitening.unproject(rbm.w.T).T, v1,v2,
                                   h1, h2, border_size = 1,normalized = False), 
                                       title+' Unormalized Weights')
        imshow_matrix(whitening.unproject(rbm.bv).reshape(v1,v2), title+' Visible Bias')
        imshow_matrix(whitening.unproject(rbm.ov).reshape(v1,v2), title+' Visible Offset')
    else:
        imshow_matrix(tile_matrix_rows(rbm.w, v1,v2, h1, h2, border_size = 1,
                                       normalized = True), title+
                                       ' Normalized Weights')
        imshow_matrix(tile_matrix_rows(rbm.w, v1,v2, h1, h2, border_size = 1,
                                       normalized = False), title+
                                       ' Unormalized Weights')
        imshow_matrix(rbm.bv.reshape(v1,v2), title+' Visible Bias')
        imshow_matrix(rbm.ov.reshape(v1,v2), title+' Visible Offset')


    

    
    imshow_matrix(rbm.bh.reshape(h1,h2), title+' Hidden Bias')
    imshow_matrix(rbm.oh.reshape(h1,h2), title+' Hidden Offset')

    if hasattr(rbm, 'variance'):
        imshow_matrix(rbm.variance.reshape(v1,v2), title+' Variance')

def imshow_filter_tuning_curve(filters, num_of_ang = 40):
    '''Plot the tuning curves of the filter's changes in frequency and angles.
    
    :Parameter:
        filters:    Filters to analyze.
                   -type: numpy array
                    
        num_of_ang: Number of orientations to check.
                   -type: int
        
    '''
    #something here
    input_dim = filters.shape[0]
    output_dim = filters.shape[1]
    max_wavelength = int(np.sqrt(input_dim))
    frq_rsp, _ = Statistics.filter_frequency_response(filters, num_of_ang)
    ang_rsp, _ = Statistics.filter_angle_response(filters, num_of_ang)
    figure().suptitle('Tuning curves')   
    for plot_idx in range(output_dim):
        subplot(output_dim, 2, 2*plot_idx + 1)
        plot(range(2,max_wavelength+1), frq_rsp[:,plot_idx] )
        subplot(output_dim, 2, 2*plot_idx + 2)
        plot(np.array(range(0,num_of_ang)) * np.pi / num_of_ang, 
             ang_rsp[:,plot_idx])
    
def imshow_filter_optimal_gratings(filters, opt_frq, opt_ang):
    '''Plot the filters and corresponding optimal gating pattern.
    
    :Parameter:
        filters: Filters to analyze.
                -type: numpy array
                    
        opt_frq: Optimal frequencies.
                -type: int
        
        opt_ang: Optimal frequencies.
                -type: int
          
    '''
    #something here
    input_dim = filters.shape[0]
    output_dim = filters.shape[1]
    max_wavelength = int(np.sqrt(input_dim))
    
    frqMatrix = np.tile( float(1) / opt_frq * np.pi * 2, (input_dim, 1) )
    thetaMatrix = np.tile(opt_ang, (input_dim,1))
    vec_xy = np.array(range(0, input_dim))
    vec_x = np.floor_divide(vec_xy, max_wavelength)
    vec_y = vec_xy + 1 - vec_x * max_wavelength
    xMatrix = np.tile(vec_x, (output_dim, 1))
    yMatrix = np.tile(vec_y, (output_dim, 1))
    gratingMatrix = np.cos(frqMatrix * (np.sin(thetaMatrix)  
                                        * xMatrix.transpose() 
                                        + np.cos(thetaMatrix)
                                        * yMatrix.transpose()))

    combinedMatrix = np.concatenate((rescale_data(filters),
                                     rescale_data(gratingMatrix)),1)
    imshow_matrix(tile_matrix_rows(combinedMatrix, max_wavelength, 
                                max_wavelength, 2, output_dim, border_size=1, 
                                normalized=False), 'optimal grating') 
    
def imshow_filter_frequency_angle_histogram(opt_frq, 
                                            opt_ang, 
                                            max_wavelength = 14, 
                                            num_of_angles = 40):
    '''Plots the histograms of the optimal frequencies and angles.
    
    :Parameter:
        opt_frq:        Optimal frequencies.
                       -type: int        

        opt_ang:        Optimal angle.
                       -type: int
                 
        max_wavelength: Maximal wavelength.
                       -type: int
                        
        num_of_angles:  Number of orientations to check.
                       -type: int
         
    '''
    figure().suptitle('Filter Frequency histogram \t\t\t Filter Angle '+ 
                      'histogram')  
    subplot(1,2,1)
    hist(opt_frq, max_wavelength-1, (2,14), normed=1)
    ylim((0,1))
    subplot(1,2,2)
    hist(opt_ang, 20, (0,np.pi), normed=1)
    ylim((0,1))
