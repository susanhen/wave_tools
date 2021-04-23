#!/usr/bin/env python3

import numpy as np
import pylab as plt
#from radar_tools import deconvolution_core
from wave_tools import fft_interface
from wave_tools import fft_interpolate
from scipy import interpolate 
from wave_tools import jonswap
from wave_tools import spreading
from scipy import stats
from scipy.optimize import fsolve
from help_tools import plotting_interface
from scipy.special import gamma as gamma_func
from wave_tools import surface_core



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
    return x,y, eta2d

def JonswapWave2D_Pavel(x, y, Hs, Alpha, gamma, theta_mean, smax):
    Nx = len(x)
    Ny = len(y)
    dk = 0.005
    k = np.arange(0.01, 0.35, dk)
    dtheta=0.05
    theta=np.arange(-np.pi, np.pi, dtheta)
    Nk = len(k)
    Ntheta = len(theta)
    kp=2*np.pi*Alpha/Hs
    S = jonswap.jonswap_k_pavel(k, kp, Hs, gamma)
    D = spreading.mitsuyatsu_spreading_pavel(k, kp, theta, theta_mean, smax)
    a_mean = np.sqrt(2*np.outer(S, np.ones(Ntheta)) * D * dk * dtheta)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    kk, th = np.meshgrid(k, theta, indexing='ij')
    phase = np.outer((np.cos(th)*kk).flatten(), xx.flatten() ) + np.outer((np.sin(th)*kk).flatten(), yy.flatten())
    ascale1 = np.random.rand(Nk, Ntheta)*2 - 1
    ascale2 = np.random.rand(Nk, Ntheta)*2 - 1
    a1 = (ascale1*a_mean).flatten()
    a2 = (ascale2*a_mean).flatten()
    eta = (np.dot(a1, np.cos(phase)) + np.dot(a2, np.sin(phase))).reshape((Nx, Ny))
    return surface_core.Surface('jonswap', eta, [x, y])


def JonswapWave2D_asymetric(x, y, Hs, Alpha, gamma, theta_mean, smax, mu, h=1000):
    g = 9.81
    Nx = len(x)
    Ny = len(y)
    dk = 0.005
    k = np.arange(0.01, 0.35, dk)
    dtheta=0.05
    theta=np.arange(0, 2*np.pi, dtheta)
    Nk = len(k)
    Ntheta = len(theta)
    kp=2*np.pi*Alpha/Hs
    S = jonswap.jonswap_k_pavel(k, kp, Hs, gamma)
    D = spreading.asymmetric_spreading(k, kp, theta, theta_mean, smax, mu)
    a_mean = np.sqrt(2*np.outer(S, np.ones(Ntheta)) * D * dk * dtheta)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    kk, th = np.meshgrid(k, theta, indexing='ij')
    ascale1 = np.random.rand(Nk, Ntheta)*2 - 1
    ascale2 = np.random.rand(Nk, Ntheta)*2 - 1
    a1 = (ascale1*a_mean).flatten()
    a2 = (ascale2*a_mean).flatten()
    eta = np.zeros((Nx, Ny))
    phase = np.outer((np.cos(th)*kk).flatten(), xx.flatten() ) + np.outer((np.sin(th)*kk).flatten(), yy.flatten()) 
    eta = (np.dot(a1, np.cos(phase)) + np.dot(a2, np.sin(phase))).reshape((Nx, Ny))
    return surface_core.Surface('jonswap', eta, [x, y]) 

def JonswapWave3D_Pavel(t, x, y, Hs, Alpha, gamma, theta_mean, smax, h = 1000):
    g = 9.81
    Nt = len(t)
    Nx = len(x)
    Ny = len(y)
    dk = 0.005
    k = np.arange(0.01, 0.35, dk)
    dtheta=0.05
    theta=np.arange(0, 2*np.pi, dtheta)
    Nk = len(k)
    Ntheta = len(theta)
    kp=2*np.pi*Alpha/Hs
    S = jonswap.jonswap_k_pavel(k, kp, Hs, gamma)
    D = spreading.mitsuyatsu_spreading_pavel(k, kp, theta, theta_mean, smax)
    a_mean = np.sqrt(2*np.outer(S, np.ones(Ntheta)) * D * dk * dtheta)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    kk, th = np.meshgrid(k, theta, indexing='ij')
    ww = np.sqrt(kk*g*np.tanh(kk*h))
    ascale1 = np.random.rand(Nk, Ntheta)*2 - 1
    ascale2 = np.random.rand(Nk, Ntheta)*2 - 1
    a1 = (ascale1*a_mean).flatten()
    a2 = (ascale2*a_mean).flatten()
    eta = np.zeros((Nt, Nx, Ny))
    for i in range(0, Nt):
        phase = np.outer((np.cos(th)*kk).flatten(), xx.flatten() ) + np.outer((np.sin(th)*kk).flatten(), yy.flatten()) - np.outer(t[i]*ww, np.ones(Nx*Ny))
        eta[i,:,:] = (np.dot(a1, np.cos(phase)) + np.dot(a2, np.sin(phase))).reshape((Nx, Ny))
    return surface_core.Surface('jonswap', eta, [t, x, y]) 

def JonswapWave3D_shearCurrent(t, x, y, Hs, Alpha, gamma, theta_mean, smax, h, z, U, psi):
    g = 9.81
    Nt = len(t)
    Nx = len(x)
    Ny = len(y)
    dk = 0.005
    k = np.arange(0.01, 0.35, dk)
    dtheta=0.05
    theta=np.arange(0, 2*np.pi, dtheta)
    Nk = len(k)
    Ntheta = len(theta)
    kp=2*np.pi*Alpha/Hs
    S = jonswap.jonswap_k_pavel(k, kp, Hs, gamma)
    D = spreading.mitsuyatsu_spreading_pavel(k, kp, theta, theta_mean, smax)
    a_mean = np.sqrt(2*np.outer(S, np.ones(Ntheta)) * D * dk * dtheta)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    kk, th = np.meshgrid(k, theta, indexing='ij')
    Uk = 2*kk*np.sum(U*np.exp(np.outer(2*kk,z)), axis=1).reshape(kk.shape)
    # TODO: write a test for this program... just visualize the dispersion relation
    ww = kk*Uk*np.cos(th-psi) + np.sqrt(kk*g*np.tanh(kk*h))
    kx = kk*np.cos(th)
    ky = kk*np.sin(th)
    ascale1 = np.random.rand(Nk, Ntheta)*2 - 1
    ascale2 = np.random.rand(Nk, Ntheta)*2 - 1
    a1 = (ascale1*a_mean).flatten()
    a2 = (ascale2*a_mean).flatten()
    eta = np.zeros((Nt, Nx, Ny))
    for i in range(0, Nt):
        phase = np.outer((np.cos(th)*kk).flatten(), xx.flatten() ) + np.outer((np.sin(th)*kk).flatten(), yy.flatten()) - np.outer(t[i]*ww, np.ones(Nx*Ny))
        eta[i,:,:] = (np.dot(a1, np.cos(phase)) + np.dot(a2, np.sin(phase))).reshape((Nx, Ny))
    return surface_core.Surface('jonswap', eta, [t, x, y]) 

def JonswapWave3D_asymetric(t, x, y, Hs, Alpha, gamma, theta_mean, smax, mu, h=1000):
    g = 9.81
    Nt = len(t)
    Nx = len(x)
    Ny = len(y)
    dk = 0.005
    k = np.arange(0.01, 0.35, dk)
    dtheta=0.05
    theta=np.arange(0, 2*np.pis, dtheta)
    Nk = len(k)
    Ntheta = len(theta)
    kp=2*np.pi*Alpha/Hs
    S = jonswap.jonswap_k_pavel(k, kp, Hs, gamma)
    D = spreading.asymmetric_spreading(k, kp, theta, theta_mean, smax, mu)
    a_mean = np.sqrt(2*np.outer(S, np.ones(Ntheta)) * D * dk * dtheta)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    kk, th = np.meshgrid(k, theta, indexing='ij')
    ww = np.sqrt(kk*g*np.tanh(kk*h))
    ascale1 = np.random.rand(Nk, Ntheta)*2 - 1
    ascale2 = np.random.rand(Nk, Ntheta)*2 - 1
    a1 = (ascale1*a_mean).flatten()
    a2 = (ascale2*a_mean).flatten()
    eta = np.zeros((Nt, Nx, Ny))
    for i in range(0, Nt):
        phase = np.outer((np.cos(th)*kk).flatten(), xx.flatten() ) + np.outer((np.sin(th)*kk).flatten(), yy.flatten()) - np.outer(t[i]*ww, np.ones(Nx*Ny))
        eta[i,:,:] = (np.dot(a1, np.cos(phase)) + np.dot(a2, np.sin(phase))).reshape((Nx, Ny))
    return surface_core.Surface('jonswap', eta, [t, x, y]) 


class DirectionalSpectrum:
    def __init__(self, Tp, theta_p, gam, c, F):
        self.Tp = Tp
        self.fp = 1./Tp
        self.theta_p = theta_p
        self.gam = gam
        g = 9.81        
        U = lambda UU: 3.5*(g/UU)*(g/UU**2*F)**(-0.33)-fp
        self.U10 = fsolve(U, 10,  xtol=1e-04)[0]
        self.xxn = g/self.U10**2*F
        self.S = lambda f:(0.076*self.xxn**(-0.22)*g**2/(2*np.pi)**4*(f)**(-5)*np.exp(-5/4*(self.fp/f)**4)
         *gam**np.exp(-((f-self.fp)**2)/(2*(fp*(0.07*(1/2 + 1/2*np.sign(self.fp - f))
        +0.09*(1/2 -1/2*np.sign(self.fp - f))))**2)))
                
        self.s = lambda f:((c*(2*np.pi*self.fp*self.U10/g)**(-2.5)*(f/self.fp)**5)*(1/2 + 1/2*np.sign(self.fp - f)) +
    (c*(2*np.pi*self.fp*self.U10/g)**(-2.5)*(f/self.fp)**(-2.5))*(1/2 -1/2*np.sign(self.fp - f)))
        self.D = lambda f, theta: (2**(2*self.s(f)-1)/np.pi*(gamma_func(self.s(f)+1))**2./gamma_func(2*self.s(f)+1)*(np.abs(np.cos((theta-theta_p)/2)))**(2*self.s(f)))
        self.Sdir=lambda f, theta: (self.S(f)*self.D(f,theta))
        
    def plot_s(self, f_min, f_max, N=200):
        f = np.linspace(f_min, f_max, N)
        plt.figure()
        plt.plot(f, self.s(f))
        plt.xlabel(r'$f~[Hz]$')
        plt.ylabel(r'$\mathrm{s}(f)$')
        
    def plot_S(self, f_min, f_max, N=200):
        f = np.linspace(f_min, f_max, N)
        plt.figure()
        plt.plot(f, self.S(f))
        plt.xlabel(r'$f~[Hz]$')
        plt.ylabel(r'$\mathrm{S}(f)$')
        
    def plot_D(self, f_min, f_max, theta_min, theta_max, N_f=200, N_theta=100):
        f_1d = np.linspace(f_min, f_max, N_f)
        theta_1d = np.linspace(theta_min, theta_max, N_theta)
        f, theta = np.meshgrid(f_1d, theta_1d, indexing='ij')
        plt.figure()
        plt.contour(f, theta, self.D(f, theta), 10)
        plt.xlabel(r'$f~[Hz]$')
        plt.ylabel(r'$\theta$')
        
    def seed_f(self, f_min, f_max, N_f, plot_it=True):
        #'''
        f = []
        while len(f)<N_f:
            fi = f_min + (f_max - f_min) * np.random.uniform()
            eta = self.S(self.fp) * np.random.uniform() + 1
            if np.sqrt(eta) < np.sqrt(self.S(fi)) + 1:
                f.append(fi)
        #'''
        
        #f = f_min + (f_max - f_min) * np.random.uniform(size=N_f)
        '''
        s1 = 5
        s2 = 0.1
        f = np.random.gamma(s1, s2, size=N_f)/(s1*s2*self.Tp)
        
        '''
        f = np.sort(f)
        if plot_it:
            plt.figure()
            plt.plot(f, self.S(f), 'x')
            plt.xlabel(r'$f~[Hz]$')
            plt.ylabel(r'$\mathrm{S}(f)$')
            plt.show()
        return f
    
    
    def seed_f_theta(self, f_min, f_max, theta_min, theta_max, N_f, N_theta, plot_it=False):
        f = self.seed_f(f_min, f_max, N_f)
        print('f seeded')
        Theta = np.zeros((N_f, N_theta))
        ff = np.zeros((N_f, N_theta))
        for i in range(0, N_f):
            theta = []
            #'''
            while len(theta)<N_theta:
                thetai = theta_min + (theta_max - theta_min) * np.random.uniform()
                eta = self.D(self.fp, self.theta_p) * np.random.uniform() 
                if eta < self.D(f[i], thetai):
                    theta.append(thetai)     
            #'''
            #theta = theta_min + (theta_max - theta_min) * np.random.uniform(size=N_theta)
            Theta[i,:] = np.sort(theta)
            ff[i,:] = f[i]*np.ones(N_theta)
            
        if plot_it:                           
            xxx = np.arange(0, N_f)
            yyy = np.arange(0, N_theta)
            plotting_interface.plot_3d_surface(xxx, yyy, Theta)
        return ff, Theta
    
    
    def define_realization(self, f_min, f_max, theta_min, theta_max, N_f, N_theta, plot_it=True):
        f_r, Theta_r = self.seed_f_theta(f_min, f_max, theta_min, theta_max, N_f, N_theta)
        #print('theta seeded')
        #Q = quad(self.S, f_min, f_max)[0]
        #print('Q = ', Q)
        
        #print('test = ', integrate.nquad(self.D, [[f_min, f_max],[0, 2*np.pi]]))
        #Hs = 4*np.sqrt(Q)
        a = np.zeros((N_f, N_theta))
        '''
        for i in range(1, N_f-1):
            for j in range(1, N_theta-1):
                #a[i,j] = np.sqrt(2*integrate.nquad(self.Sdir, [[(f_r[i,j]+f_r[i-1,j])/2,(f_r[i,j]+f_r[i+1,j])/2],[(Theta_r[i,j]+Theta_r[i,j-1])/2,(Theta_r[i,j]+Theta_r[i,j+1])/2]]))[0]
                a[i,j] = np.sqrt(2*self.Sdir(f_r[i,j], Theta_r[i,j]) * ((f_r[i,j])**2-(f_r[i-1,j])**2)*(Theta_r[i,j]-Theta_r[i,j-1])/3)
        
        a[:,0] = a[:,1]
        a[0,:] = a[1,:]
        a[-1,:] = a[-2,:]
        a[:,-1] = a[:,-2]
        '''
        df = np.gradient(f_r, axis=0)
        dTheta = np.gradient(Theta_r, axis=1)
        dA = df*dTheta
        a = np.sqrt(2*self.Sdir(f_r, Theta_r) *dA)
        #aa = a.flatten()
        if plot_it:
            xxx = np.arange(0, N_f)
            yyy = np.arange(0, N_theta)
            plotting_interface.plot_3d_surface(xxx, yyy, a)
        return f_r, Theta_r, a    

    
class SpectralRealization:
    
    def __init__(self, DirSpec, f_min, f_max, theta_min, theta_max, N_f, N_theta):
        self.DirSpec = DirSpec
        self.f_min = f_min
        self.f_max = f_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.N_f = N_f
        self.N_theta = N_theta 
        self.f_r, self.Theta_r, self.a = DirSpec.define_realization(f_min, f_max, theta_min, theta_max, N_f, N_theta)

        
    def calc_wavenumber(self, Nx, Ny, h, plot_it=False):
        '''
        function to calculate the wave number for f_r
        Parameters:
        -----------
            bathy       : optional Bathymetry object
                          default: None; if provided defines the bathymetry for the calculation
            h           : float
                          default: 1000 meters; is used when bathymetry not present; defines a constant waterdepth
        '''
          
        k_loc = fsolve((lambda k: ((9.81*k*np.tanh(k*h)) - (2*np.pi*self.f_r[:,0])**2)), 0.01*np.ones(self.N_f))
        
            
        if plot_it:
            plt.figure()
            plt.plot(k_loc, self.f_r[:,0])
        return k_loc
                
    def invert(self, t, x, y, h, surf_name='inverted_surface'):

        Nx = len(x)
        Ny = len(y)
        k = self.calc_wavenumber(Nx, Ny, h)
        print('wavenumber calculated')
        w = 2*np.pi*self.f_r
        
        ti = 0#t[0]
        X, Y = np.meshgrid(x, y, indexing='ij')
        theta = self.Theta_r
        
        eta = np.zeros((Nx, Ny))
        #phase = np.random.uniform(size=self.N_f)*np.pi*2
        print('before loop')
        for i in range(0, self.N_f):
            for j in range(0, self.N_theta):                
                phase = np.random.uniform()*np.pi*2
                eta += self.a[i,j]* np.cos(phase - k[i]*(np.cos(theta[i,j])*X + np.sin(theta[i,j])*Y))
        plotting_interface.plot_3d_as_2d(x, y, eta)
        #'''
        plt.figure()
        plt.plot(eta[:,150])
        plt.figure()
        plt.plot(eta[400,:])
        #'''  
        surf = surface_core.Surface(surf_name, eta, [x,y])
        return surf



if __name__=='__main__':
    t = np.linspace(0,100, 200)
    Tp = 10
    Hs = 2.0
    Alpha = 0.023
    smax = 70
    theta_mean = np.pi/2+30*np.pi/180
    N = 256
    gamma = 3.3
    '''
    eta = JonswapWave1D(t, Tp, Hs)
    print('Hs in 1d: ', np.sqrt(np.var(eta)))
    #plt.figure()
    #plt.plot(t, eta)
    '''


    '''
    x = np.linspace(0,2000, N)
    x, y, eta2d = JonswapWave2D(x, Tp, Hs, smax)
    print('Hs = ', np.sqrt(np.var(eta2d)))
    plt.figure()
    plt.imshow(eta2d)
    '''
    '''
    dx = 2
    dy = dx
    y = np.arange(400, 1100 + dy, dy)
    x = np.arange(-500, 500 + dx, dx)
    g = 9.81
    Tp = 10
    fp = 1./Tp
    theta_p = np.pi/2-5*np.pi/180
    gam = 3.3
    N_f = 100
    f_min = 0.001
    f_max = 0.4
    N_theta = 40
    theta_min = -np.pi
    theta_max = np.pi
    c = 50
    F = 300000
    h = 1000
    
    DirSpec = DirectionalSpectrum(Tp, theta_p, gam, c, F)
    #DirSpec.plot_S(f_min, f_max)
    #DirSpec.plot_s(f_min, f_max)
    #DirSpec.plot_D(f_min, f_max, theta_min, theta_max)
    realization = SpectralRealization(DirSpec, f_min, f_max, theta_min, theta_max, N_f, N_theta)
    print('Directional Spectrum defined')

    inverted_surface = realization.invert(0, x, y, h)
    inverted_surface.save('../../Data/SimulatedWaves/inv_surf.hdf5')
    
    plt.show()
    '''

    # JONSWAP2D Pavel
    '''    
    dx = 7.5
    dy = 7.5
    x = np.arange(-250, 250, dx)
    y = np.arange(500, 1000, dy)
    surf2d = JonswapWave2D_Pavel(x, y, Hs, Alpha, gamma, theta_mean, smax)
    surf2d.plot_3d_as_2d()
    '''

    #JONSWAP 3D similar to Pavel for 2D
    dx = 7.5
    dy = 7.5
    dt = 1.
    h = 1000
    t = np.arange(0, 5, dt)
    x = np.arange(-250, 250, dx)
    y = np.arange(500, 1000, dy)
    #surf3d = JonswapWave3D_Pavel(t, x, y, Hs, Alpha, gamma, theta_mean, smax, h)
    z = np.linspace(-100,0,100)
    Ux = 0.5*np.exp(5*z)
    Uy = 0
    surf3d = JonswapWave3D_shearCurrent(t, x, y, Hs, Alpha, gamma, theta_mean, smax, h, z, Ux, Uy)
    surf3d.plot_3d_as_2d(0)

    # asymmetric Jonswap
    '''
    mu = -0.28
    surf3d_asym = JonswapWave3D_asymetric(t, x, y, Hs, Alpha, gamma, theta_mean, smax, mu, h)
    surf3d_asym.plot_3d_as_2d(0)
    '''

    plt.show()

