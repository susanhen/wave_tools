#!/usr/bin/env python

import numpy as np
import pylab as plt
#from radar_tools import deconvolution_core
from wave_tools import fft_interface
from wave_tools import fft_interpolate
from scipy import interpolate 
from wave_tools import jonswap
from wave_tools import spreading
from scipy import stats



def JonswapWave1D(t ,Tp, Hs, gamma=3.3, h=1000):
    '''
    
    TODO: h not necessary here; might add version for J(k) where h is needed
    Parameters:
            input
                    t       array
                            time for output
                    Tp      float  
                            peak period
                    Hs      float
                            significant wave height
                    gamma   optional float
                            shape factor
                    h       float
                            water depth
            output
                    eta     array
                            surface elevation according to Jonswap spec                       
    '''
    w, dw = fft_interface.grid2k(t)
    wp = 2*np.pi/Tp
    N_half = int(0.5*len(w))
    ji = jonswap.jonswap(w[N_half:], wp, Hs, gamma)
    eta_coeffs = np.zeros(len(w), dtype=complex)
    phi = stats.uniform(scale=2*np.pi).rvs(N_half)-np.pi
    eta_coeffs[N_half:] = np.sqrt(0.5*ji*dw)*len(w) * np.exp(1j*phi)
    eta_coeffs[1:N_half] = np.flipud(np.conjugate(eta_coeffs[N_half+1:]))
    eta = np.fft.ifft(np.fft.ifftshift(eta_coeffs)).real
    eta *= Hs/(4*np.sqrt(np.var(eta)))
    return eta

def JonswapWave2D(x, Tp, Hs, smax, gamma=3.3, h=1000, theta_mean=0.5*np.pi, N_theta=360):
    '''
    Construct 2D JonswapWave S(w,theta) with directional spreading
    
    Parameters:
            input
                    t       array
                            time for output
                    Tp      float  
                            peak period
                    Hs      float
                            significant wave height
                    gamma   optional float
                            shape factor
                    h       float
                            water depth
            output
                    eta     array
                            surface elevation according to Jonswap spec                       
    '''
    g=9.81
    N = len(x)//2
    k_axis, dk = fft_interface.grid2k(x)
    k = k_axis[N:]
    wp = 2*np.pi/Tp
    w = np.sqrt(k*9.81*np.tanh(k*h))
    ji = jonswap.jonswap(w, wp, Hs, h, gamma) 
    #TODO: make sure that kx, ky are returned and that these parameters can be set from here
    D_cart = spreading.mitsuyatsu_spreading(ji, theta_mean, smax, wp, k)
    
    phi = np.exp(1j*(stats.uniform(scale=2*np.pi).rvs((2*N,2*N))-np.pi))
    upper_image=phi[N:,:]*D_cart[N:,:]


    lower_image = phi[:N,:]*D_cart[:N,:]
    #lower_image = np.zeros((N,2*N), dtype=complex)
    lower_image[1:,1:N] += np.flip(upper_image[1:,N+1:]).conjugate()
    upper_image[1:,N+1:] = np.flip(lower_image[1:,1:N]).conjugate()

    upper_image[0,1:N] += np.flip(upper_image[0,N+1:]).conjugate()
    upper_image[0,N+1:] = np.flip(upper_image[0,1:N]).conjugate()

    lower_image[1:,N+1:] += np.flip(upper_image[1:,1:N]).conjugate()
    upper_image[1:,1:N] = np.flip(lower_image[1:,N+1:]).conjugate()

    lower_image[1:N,N] += np.flip(upper_image[1:,N]).conjugate()
    upper_image[1:,N] = np.flip(lower_image[1:N,N]).conjugate()

    total_image=np.block([[lower_image],[upper_image]])
    
    #TODO check that we have the correct values for interpolated kx and ky (in settings?)
    #eta2d = np.fft.ifft2(np.fft.ifftshift(total_image))
    x, y, eta2d = fft_interface.spectral2physical(total_image, [k_axis, k_axis])
    
    #define local incidence angle
    H = 45.
    def local_incidence_angle(eta, x, y, H):
        # calculate horizontal derivatives in ETA object
        deta_dx = ...
        deta_dz = ...
        r = np.sqrt(x**2 + y**2)
        n_norm = np.sqrt(deta_dx**2 + deta_dy**2 + 1)
        b_norm = np.sqrt(r**2 + (H-eta)**2)
        cos_theta_l = (x*deta_dx + y*deta_dy + (H-eta))/(r*n_norm*b_norm)
        theta_l = np.acos(theta_l)
    return x, y, eta2d.real

    def apply_geo_shad(H):
        print('not implemented!')
        #... check old functions... change to 2D after transforming to polar coordinates and transform back



if __name__=='__main__':
    t = np.linspace(0,100, 200)
    Tp = 10
    Hs = 3.0
    smax = 1
    N = 256
    eta = JonswapWave1D(t, Tp, Hs)
    print('Hs in 1d: ', np.sqrt(np.var(eta)))
    #plt.figure()
    #plt.plot(t, eta)
    

    
    
    x = np.linspace(0,2000, N)
    x, y, eta2d = JonswapWave2D(x, Tp, Hs, smax)
    print('Hs = ', np.sqrt(np.var(eta2d)))
    plt.figure()
    plt.imshow(eta2d)
    plt.show()
