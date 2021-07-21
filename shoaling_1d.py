#shoaling in 1 spatial dimension
import numpy as np
from numpy.lib import scimath as SM
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.interpolate import interp1d
from scipy.integrate import dblquad, quad, simps
from scipy import integrate
from matplotlib import cm
from help_tools import plotting_interface
import h5py

class Bathymetry:

    def __init__(self, x, bathy_filename='RR23605_bathy.hdmf', test=False):
        dx = x[1] - x[0]
        x_u = x
        self.x = x_u
        self.Nx = len(self.x)
        if test==False:
            # read profile from file
            hf = h5py.File(bathy_filename, 'r')
            h = np.array(hf['bathy'])
            r = np.array(hf['r'])
            hf.close()
            # interpolate profile to given grid where available
            if x[0] >= r[0]:
                x_max_ind = np.argwhere(x>r[-1])[0][0]
                x_int = x[:x_max_ind]
                bathy_func = interp1d(r, h, kind='cubic')
                bathy1 = bathy_func(x_int)        
            else:
                print('Error not yet implemented!')

            bathy2 = -0.005*((np.arange(0, len(x)-x_max_ind))*dx) + bathy1[-1] 
            bathy = np.block([bathy1, bathy2])
            h_func = interp1d(x_u, bathy, kind='cubic')
            self.h = h_func(x)
            self.H = -self.h
        else:
            bathy1 = -10 * (x<=700)
            bathy2 = (-0.05*x + 25)*(np.logical_and(x>700, x<=1700))
            bathy3 = -60*(x>1700)
            b = bathy1 + bathy2 + bathy3
            self.h = b
            self.H = -b       

    def plot(self):
        plt.figure()
        plt.plot(self.x, self.h, 'x')

    def calc_wavenumber(self, f_r):
        N_f = np.size(f_r)
        k_out = np.zeros((N_f, self.Nx))
        eps = 10**(-6)
        N_max = 100
        for i in range(N_f):
            w = 2*np.pi*f_r[i]
            ki = w**2/(9.81)
            wt = np.sqrt(9.81*ki*np.tanh(ki*(-self.h)))
            count = 0
            while np.max(np.abs(w-wt))>eps and count<N_max:
                latter = 9.81*np.tanh(ki*(-self.h))
                ki = w**2/(latter)
                wt = np.sqrt(latter)
                count = count + 1
            
            k_out[i,:] = ki
        return k_out

class DirectionalSpectrum:

    def __init__(self, Tp, gam, F):
        self.Tp = Tp
        self.fp = 1./Tp
        self.gam = gam
        g = 9.81        
        U = lambda UU: 3.5*(g/UU)*(g/UU**2*F)**(-0.33)-self.fp
        self.U10 = fsolve(U, 10,  xtol=1e-04)[0]
        self.xxn = g/self.U10**2*F
        self.S = lambda f:(0.076*self.xxn**(-0.22)*g**2/(2*np.pi)**4*(f)**(-5)*np.exp(-5/4*(self.fp/f)**4)
         *gam**np.exp(-((f-self.fp)**2)/(2*(self.fp*(0.07*(1/2 + 1/2*np.sign(self.fp - f))
        +0.09*(1/2 -1/2*np.sign(self.fp - f))))**2)))

    def seed_f(self, f_min, f_max, N_f, plot_it=False):
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

    def define_realization(self, f_min, f_max, N_f, plot_it=False):
        f_r = self.seed_f(f_min, f_max, N_f)
        a = np.zeros(N_f)
        df = np.gradient(f_r)
        a = np.sqrt(2*self.S(f_r)*df)
        return f_r, a

class SpectralRealization:

    def __init__(self, DirSpec, f_min, f_max, N_f, dx, test=False, phase=None):
        self.N_f = N_f
        self.dx = dx
        if test == False:
            self.DirSpec = DirSpec
            self.f_min = f_min
            self.f_max = f_max
            self.f_r, self.a = DirSpec.define_realization(f_min, f_max, N_f)
            self.phase = np.random.uniform(0,2*np.pi,size=self.N_f)
        else:
            self.a = np.array([1])
            self.f_r = np.array([0.1])
            self.phase = phase


    def calc_wavenumber(self, Nx, bathy=None, h=1000):

        if bathy==None:
            k_loc_f = fsolve((lambda k: ((9.81*k*np.tanh(k*h)) - (2*np.pi*self.f_r[:,0])**2)), 0.01*np.ones(self.N_f))
            k_loc = np.outer(k_loc_f, np.ones(self.Nx)).reshape((self.N_f, Nx))
        else:
            k_loc = bathy.calc_wavenumber(self.f_r)

        return k_loc

    def invert(self, bathy, ti, x, plot_it=False):
        Nx = len(x)
        k = self.calc_wavenumber(Nx, bathy)
        print('wavenumber calculated')
        w = 2*np.pi*self.f_r
        H = bathy.H
        
        zeta = np.zeros((np.size(ti),Nx))
        print ('before loop')

        for i in range(0, self.N_f):
            k2H_by_sinh_2kH = np.where(k[i,:]*H < 50,  2*k[i,:]*H / np.sinh(2*k[i,:]*H), 0)
            ksh = np.cumsum(k[i,:]*self.dx)
            Cgx = w[i]/(2*k[i]*(1+k2H_by_sinh_2kH))
            Cg0x = w[-1]/(2*k[-1]*(1+k2H_by_sinh_2kH[-1]))
            for j in range(0, np.size(ti)):
                zeta[j,:] = zeta[j,:] + self.a[i]*np.abs(SM.sqrt(Cg0x/Cgx))*np.cos(self.phase[i]+w[i]*ti[j]+ksh)
            '''    
            if plot_it:
                plt.figure()
                plt.plot(x, zeta)
                plt.xlabel('$x~[\mathrm{m}]$')
                plt.ylabel('$\eta~[\mathrm{m}]$')
            '''
        return zeta


    def vel(self, eta, bathy, ti, x, plot_it=False):
        Nx = len(x)
        k = self.calc_wavenumber(Nx, bathy)
        
        w = 2*np.pi*self.f_r
        H = bathy.H
        
        vel = np.zeros((np.size(ti),Nx))
        
        for i in range(0, self.N_f):
            k2H_by_sinh_2kH = np.where(k[i,:]*H < 50,  2*k[i,:]*H / np.sinh(2*k[i,:]*H), 0)
            ksh = np.cumsum(k[i,:]*self.dx)
            Cgx = w[i]/(2*k[i]*(1+k2H_by_sinh_2kH))
            Cg0x = w[-1]/(2*k[-1]*(1+k2H_by_sinh_2kH[-1]))
            for j in range(0,np.size(ti)):
                vel[j,:] += self.a[i]*np.abs(SM.sqrt(Cg0x/Cgx))*(-w[i])/np.sinh(k[i,:]*H)*np.cosh(k[i,:]*(eta[j,:]+H))*np.cos(self.phase[i]+w[i]*ti[j]+ksh)
        if plot_it:
            plt.figure()
            plt.plot(x, vel)
            plt.xlabel('$x~[\mathrm{m}]$')
            plt.ylabel('$\eta~[\mathrm{m}]$')
        return vel
    
