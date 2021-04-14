import numpy as np
from wave_tools import fft_interface
from scipy.special import gammaln
import polarTransform
    

def mitsuyatsu_spreading(spec1d, theta_mean, smax, wp, k, h=1000, N_theta=360):
    '''
    Defines a spectral spreading matrix according to mitsuyatsu
    
    Parameters
    ----------
    spec1d    : float array
        one-sided 1d spectrum 
    theta_mean: float
        mean direction
    smax      : float
        directional spread (degrees)
    wp        : float
        peak wave frequency
    k         : float array
        one-sided uniform discretization of wave numbers
        spectrum should be built from a w calculated from this k
    h          : float
        water depth; default=1000 deep water
    N_theta    : int
        Number of azimuth directions in polar coordinates
    
    Return
    ------
    spread    : float array (2d)
        spreading matrix on kartesian grid of
    
    Details
    ------
    Wind Sea: smax = [10, 30]
    Swell:    smax = [50, 70]
    '''
    mu1 = 5.
    mu2 = -2.5
    g = 9.81
    dk = k[1]-k[0]
    theta = np.linspace(theta_mean - np.pi, theta_mean + np.pi, N_theta, endpoint=True)
    theta_mean = theta[np.argmin(np.abs(theta-theta_mean))] # ensure that this value is on the
    dtheta = theta[1]-theta[0]
    
    N = len(k)
    
    w = np.sqrt(k*9.81*np.tanh(k*h))
    w_polar, theta_polar = np.meshgrid(w, theta)
    k_polar, theta_polar = np.meshgrid(k, theta)
    spec_polar, theta_polar = np.meshgrid(spec1d, theta)
    
    A = g*np.tanh(k_polar*h)
    B = np.where(np.cosh(k_polar*h)>10**6, k_polar*0, g*h*k_polar*(1./np.cosh(k_polar*h))**2)
    C = 2*np.sqrt(9.81*k_polar*np.tanh(k_polar*h))
    print(np.max(k_polar), np.min(k_polar), np.max(C), np.min(C))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dw_dk = np.where(C>0.001, (A + B)/C, 1)
                               
    mu = np.where(w_polar<=wp, mu1, mu2)
    s = smax*(w_polar/wp)**mu
    
    D_polar = ((np.cos((theta_polar-theta_mean)/2) )**2)**s
    # Scale spreading function to maintain energy
    D_polar /= np.outer(np.ones(N_theta), np.sum(D_polar, axis=0))
    S_polar = spec_polar * D_polar * dw_dk  
    
    S_cart, settings = polarTransform.convertToCartesianImage(S_polar, imageSize=(2*N,2*N), initialAngle=theta[0], finalAngle=theta[-1])
    
    return S_cart

def mitsuyatsu_spreading_pavel(k, kp, theta, theta_mean, smax):
    kk, th = np.meshgrid(k, theta, indexing='ij')
    s = np.where(kk<=kp, smax * (np.sqrt(kk/kp))**5, smax * (np.sqrt(kk/kp))**(-2.5))
    D = (2**(2*s-1))/np.pi * (np.cos((th-theta_mean)/2)**2)**s
    D /= np.outer(np.sum(np.gradient(th, axis=1)*D, axis=1), np.ones(len(theta)))    
    return D

def asymmetric_spreading(k, kp, theta, theta_mean, smax, mu):
    kk, th = np.meshgrid(k, theta, indexing='ij')
    s = np.where(kk<=kp, smax * (np.sqrt(kk/kp))**5, smax * (np.sqrt(kk/kp))**(-2.5))
    xeta = np.where(th<theta_mean, np.exp(mu), np.exp(-mu))
    th_min = theta_mean - (2*np.pi)/(1+np.exp(2*mu))
    th_max = theta_mean + (2*np.pi)/(1+np.exp(-2*mu))
    ind_min = np.argmin(np.abs(theta-th_min))
    ind_max = np.argmin(np.abs(theta-th_max))
    G0a_inv = np.trapz( ((np.cos((th[:,ind_min:ind_max+1]-theta_mean)/2*xeta[:,ind_min:ind_max+1]))**2)**s[:,ind_min:ind_max+1], theta[ind_min:ind_max+1], axis=1)
    G0a = np.outer(1./G0a_inv, np.ones(len(theta)))
    Ga = ((np.cos((th-theta_mean)/2*xeta))**2)**s * G0a
    return Ga

if __name__=='__main__':
    import pylab as plt
    from wave_tools import jonswap as j
    from help_tools import plotting_interface
    theta_mean = np.pi/2 + 30*np.pi/180
    smax = 70
    wp = 0.2
    N = 256
    gamma = 3.3
    wp = 0.8
    Hs = 2.0
    h = 100. 
    g = 9.81
    k = np.linspace(0, 0.5, N)
    w = np.sqrt(k*g*np.tanh(k*h))


    #Mitsuyasu distribution based on codeine 2s A, normalization (integral should be one)
    '''
    ji = j.jonswap(w, wp, Hs, h, gamma) 
    plt.figure()
    plt.plot(ji)
    plt.show()
    D = mitsuyatsu_spreading(ji, theta_mean, smax, wp, k)
    plt.figure()
    plt.imshow(D)
    plt.show()
    '''
    # Mitsuyatsu distribution
    k = np.arange(0.01, 0.35, 0.005)
    theta = np.linspace(-np.pi, np.pi, 100)
    Alpha = 0.023
    kp = 2*np.pi*Alpha/Hs
    D = mitsuyatsu_spreading_pavel(k, kp, theta, theta_mean, smax)
    plotting_interface.plot_3d_as_2d(k*10, theta, D)

    # asymmetric distribution
    mu = -0.28
    D = asymmetric_spreading(k, kp, theta, theta_mean, smax, mu)
    plotting_interface.plot_3d_as_2d(k*10, theta, D)
    plotting_interface.show()