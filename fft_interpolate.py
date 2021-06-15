# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 19:48:29 2015

@author: susanhen
"""

import numpy as np

def fft_interpol1d(x, y, new_N):
    '''interpolates data y(x) to the new N; the integrated spectral energy is conserved
    Parameters:
    -----------
    input: 
        x x-axis vector
        y data vector
        new_N number of points after interpolation in both x, y
    output:
        interpolated x
        interpolated y
    '''
    
    N = len(y)
    N_half = int(0.5*N)
    M_half = int(0.5*new_N)
    fact = int(new_N/N)
    dx = (x[1] - x[0])
    x_interpol = np.linspace(x[0], x[-1]+dx/fact*(fact-1), new_N)
    fft_y = np.fft.fftshift(np.fft.fft(y))
    fft_y_interpol = np.zeros(new_N, dtype=complex)
    fft_y_interpol[M_half-N_half:M_half+N_half] = fft_y/N*new_N
    y_interpol = np.fft.ifft(np.fft.fftshift(fft_y_interpol)).real
    
    return x_interpol, y_interpol
    
def fft_interpol2d(x, y, z, new_Nx, new_Ny):
    '''
    Interpolates 2d data by 2d fft; the integrated spectral energy is conserved
    Parameters:
    ------------
    input:
        
        x: vector discretizing x-dimension
        y: vector discretizeing y-dimension
        z: matrix with data on x,y
        new_Nx: desired number of points in x direction
        new_Ny: desired number of points in y direction
    return:
        interpolated x
        interpolated y
        interpolated z
    '''
    if np.mod(len(x),2) == 1:
        x = x[:-1]
        z = z[:-1,:]
    if np.mod(len(y),2) == 1:
        y = y[:-1]
        z = z[:,:-1]
    Nx, Ny = z.shape
    Nx_half, Ny_half = Nx//2, Ny//2
    new_Nx_half, new_Ny_half = new_Nx//2, new_Ny//2
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    factx = int(new_Nx/Nx)
    facty = int(new_Ny/Ny)
    x_interpol = np.linspace(x[0], x[-1]+dx/factx*(factx-1), new_Nx)
    y_interpol = np.linspace(y[0], y[-1]+dy/facty*(facty-1), new_Ny)
    
    fft_z = np.fft.fftshift(np.fft.fft2(z))
    fft_z_interpol = np.zeros((new_Nx, new_Ny), dtype=complex)
    fft_z_interpol[new_Nx_half-Nx_half:new_Nx_half+Nx_half, new_Ny_half-Ny_half:new_Ny_half+Ny_half] = fft_z*new_Nx*new_Ny/Nx/Ny
    z_interpol = np.fft.ifft2(np.fft.fftshift(fft_z_interpol)).real
    return x_interpol, y_interpol, z_interpol


if __name__=='__main__':
    import pylab as plt
    T = 8.0
    a = 1.
    N = 20
    M = 100
    t = np.linspace(0,5*T,N, endpoint=False)
    eta = a*np.sin(2*np.pi/T*t) + 0.2*np.sin(2*np.pi/6*t+0.5)
    t_interpol, eta_interpol = fft_interpol1d(t, eta, M)
    
    plt.figure()
    plt.plot(t_interpol,eta_interpol.real)
    plt.plot(t, eta)

    eta = np.outer(eta, np.ones(30))
    y = np.arange(0,30)
    new_Nx = 200
    new_Ny = 30
    x_interpol, y_interpol, z_interpol = fft_interpol2d(t,y,eta, new_Nx, new_Ny)    
    plt.figure()
    plt.plot(x_interpol, z_interpol[:,0])
    plt.plot(t, eta[:,0])
    plt.show()
    
