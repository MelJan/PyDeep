''' This module provides functions for displaying and visualize data. 
    It extends the matplotlib.pyplot. 

    :Implemented:
        - tile a matrix
        - Show a matrix, histogram
        - Show RBM parameters

    :Version:
        1.0

    :Date:
        29.08.2016

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
