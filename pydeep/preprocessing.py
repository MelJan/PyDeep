''' This module contains several classes for data preprocessing.

    :Implemented:
        - Binarize
        - Rescale
        - Remove means
    
    :Version:
        1.0

    :Date:
        08.08.2016

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2016

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
import numpy as numx

def binarize_data(data):
    ''' Converts data to binary values. 
        For data out of [a,b] a data point p will become zero 
        if p < 0.5*(b-a) one otherwise.
        
        :Parameters:
            data: data to be binarized.
                 -type: numpy array [num data point, data dimension]
        
        :Returns:
            Binarized data .
           -type: numpy array [num data point, data dimension]
                   
    ''' 
    return numx.array(numx.where(data < 0.5, 0,1))
    
def rescale_data(data, 
                 new_min = 0.0, 
                 new_max = 1.0):
    ''' Normalize the values of a matrix. 
        e.g. [min,max] ->  [new_min,new_max]
          
    :Parameters:
        data:    Data to be normalized.
                -type: numpy array [num data point, data dimension]
                 
        new_min: New min value.
                -type: float
                 
        new_max: New max value.
                -type: float
                
    :Returns:
        Rescaled data
       -type: numpy array [num data point, data dimension]
        
    '''
    dataC = numx.array(data,numx.float64)
    minimum = numx.min(numx.min(dataC,axis=1),axis=0)
    dataC -= minimum
    maximum = numx.max(numx.max(dataC,axis=1),axis=0)
    dataC *= (new_max-new_min)/maximum
    dataC += new_min
    return dataC

def remove_rows_means(data, return_means= False):
    ''' Remove the individual mean of each row.
          
    :Parameters:
    
        data:         Data to be normalized
                     -type: numpy array [num data point, data dimension]
        return_means: If True returns also the means
        
                     -type: bool
                     
    :Returns:
        Normalized data
       -type: numpy array [num data point, data dimension]
        Means of the data (optional)
       -type: numpy array [num data point]
        
    '''
    means = numx.mean(data, axis = 1).reshape(data.shape[0],1)
    output = data - means
    if return_means == True:
        return output, means
    else:
        return output  
    
def remove_cols_means(data, return_means= False):
    ''' Remove the individual mean of each column.
          
    :Parameters:
    
        data:         Data to be normalized
                     -type: numpy array [num data point, data dimension]
        return_means: If True returns also the means
        
                     -type: bool
                     
    :Returns:
        Normalized data
       -type: numpy array [num data point, data dimension]
        Means of the data (optional)
       -type: numpy array [num data point]
        
    '''
    means = numx.mean(data, axis = 0).reshape(1,data.shape[1])
    output = data - means
    if return_means == True:
        return output, means
    else:
        return output