# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:41:15 2015

@author: susanhen
"""
from numpy import ones, convolve, zeros

def moving_average(data, window_size, axis=0):
    window= ones(int(window_size))/float(window_size)
    if len(data.shape)==1:        
        return convolve(data, window, 'same')
    else:
        ret = zeros(data.shape)
        if axis==0:
            change=1
            N_wanted = data.shape[change]
            for i in range(0, N_wanted):
                window
                ret[:,i] = convolve(data[:,i], window, 'same')
        else:
            change=0
            N_wanted = data.shape[change]
            for i in range(0, N_wanted):
                window
                ret[i,:] = convolve(data[i,:], window, 'same')
            
        return ret