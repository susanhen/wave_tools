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
from wave_tools import surface_core, peak_tracking

class Bathymetry:

    def __init__(self, x, bathy_filename=None):
        dx = x[1] - x[0]
        x_u = x
        self.x = x_u
        self.Nx = len(self.x)
        if not bathy_filename is None:
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
        plt.plot(self.x, self.h, 'k', linewidth=0.8)
        plt.xlabel(r'$x~[\mathrm{m}]$')
        plt.ylabel(r'$z~[\mathrm{m}]$')

    def calc_wavenumber(self, f):
        N_f = np.size(f)
        k_out = np.zeros((N_f, self.Nx))
        eps = 10**(-6)
        N_max = 100
        for i in range(N_f):
            w = 2*np.pi*f[i]
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

class Spectrum:

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

    def distribute_f(self, f_min, f_max, N_f, plot_it=False):
        #'''
        f = np.zeros(N_f)
        N_found = 0
        while N_found<N_f:
            fi = f_min + (f_max - f_min) * np.random.uniform()
            eta = self.S(self.fp) * np.random.uniform() + 1
            if np.sqrt(eta) < np.sqrt(self.S(fi)) + 1:
                f[N_found] = fi
                N_found = N_found + 1


        f = np.sort(f)
        if plot_it:
            plt.figure()
            plt.plot(f, self.S(f), 'x')
            plt.xlabel(r'$f~[Hz]$')
            plt.ylabel(r'$\mathrm{S}(f)$')
            plt.show()
        return f

    def define_realization(self, f_min, f_max, N_f, plot_it=False):
        f = self.distribute_f(f_min, f_max, N_f)
        a = np.zeros(N_f)
        df = np.gradient(f)
        a = np.sqrt(2*self.S(f)*df)
        return f, a

    def plot(self):
        f = self.distribute_f(0, 0.3, 200)
        plt.figure()
        plt.plot(f, self.S(f), 'k', linewidth=0.8)
        plt.xlabel(r'$f~[Hz]$')
        plt.ylabel(r'$\mathrm{S}(f)$')


class SpectralRealization:

    def __init__(self, DirSpec, f_min, f_max, N_f, dx):
        self.N_f = N_f
        self.dx = dx
        self.DirSpec = DirSpec
        self.f_min = f_min
        self.f_max = f_max
        self.f, self.a = DirSpec.define_realization(f_min, f_max, N_f)
        self.w = 2*np.pi*self.f
        self.phase = np.random.uniform(0,2*np.pi,size=self.N_f)


    def calc_wavenumber(self, Nx, bathy=None, h=1000):

        if bathy==None:
            k_loc_f = fsolve((lambda k: ((9.81*k*np.tanh(k*h)) - (self.w[:,0])**2)), 0.01*np.ones(self.N_f))
            k_loc = np.outer(k_loc_f, np.ones(self.Nx)).reshape((self.N_f, Nx))
        else:
            k_loc = bathy.calc_wavenumber(self.f)

        return k_loc

    def invert(self, bathy, ti, x):
        Nx = len(x)
        k = self.calc_wavenumber(Nx, bathy)
        H = bathy.H
        Nt = len(ti)
        eta = np.zeros((Nt,Nx))

        for i in range(0, self.N_f):
            K2H = 2*k[i,:]*H 
            k2H_by_sinh_2kH = np.where(K2H>0,  K2H / np.sinh(K2H), 0)
            ksh = np.cumsum(k[i,:]*self.dx)
            Cgx = self.w[i]/(2*k[i]*(1+k2H_by_sinh_2kH))
            Cg0x = self.w[-1]/(2*k[-1]*(1+k2H_by_sinh_2kH[-1]))

            for j in range(0, Nt):
                eta[j,:] = eta[j,:] + self.a[i]*np.abs(SM.sqrt(Cg0x/Cgx))*np.cos(self.phase[i]+self.w[i]*ti[j]+ksh)


            '''
            TODO: make faster!
            eta += np.outer(self.a[i]*np.abs(SM.sqrt(Cg0x/Cgx)),np.cos(self.phase[i]*np.ones(Nt)+w[i]*ti+np.outer(ksh*,np.ones(Nt))))
            '''

            
        return eta


    def vel(self, eta, bathy, ti, x):
        Nx = len(x)
        k = self.calc_wavenumber(Nx, bathy)
        w = 2*np.pi*self.f
        H = bathy.H
        vel = np.zeros((np.size(ti),Nx))
        
        for i in range(0, self.N_f):
            k2H_by_sinh_2kH = np.where(k[i,:]*H < 50,  2*k[i,:]*H / np.sinh(2*k[i,:]*H), 0)
            ksh = np.cumsum(k[i,:]*self.dx)
            Cgx = w[i]/(2*k[i]*(1+k2H_by_sinh_2kH))
            Cg0x = w[-1]/(2*k[-1]*(1+k2H_by_sinh_2kH[-1]))
            for j in range(0,np.size(ti)):
                vel[j,:] += self.a[i]*np.abs(SM.sqrt(Cg0x/Cgx))*(-w[i])/np.sinh(k[i,:]*H)*np.cosh(k[i,:]*(eta[j,:]+H))*np.cos(self.phase[i]+w[i]*ti[j]+ksh)
            # TODO: make it faster!
            # TODO combine vel and eta
        return vel
    
if __name__=='__main__':
    from_file=True
    fn = 'example_data/surfprofile'
    #from_file=False
    #fn = 'example_data/test'

    if not from_file:
        dx = 0.5
        x = np.arange(200, 2200+dx, dx)
        g = 9.81
        Tp = 10
        fp = 1./Tp
        gam  = 3.3
        N_f = 100
        f_min = 0.001
        f_max = 0.4
        F = 300000

        # Define Spectrum
        spec = Spectrum(Tp, gam, F)
        realization = SpectralRealization(spec, f_min, f_max, N_f, dx)
        print('Directional Spectrum defined')

        # Define bathymetry
        bathy_filename = 'RR23605_bathy.hdmf'
        b = Bathymetry(x, bathy_filename)
        #b.plot()
        #plotting_interface.show()
        print('Bathymetry defined')

        # Construct wave field from spectrum
        Nt = 1200
        Nx = len(x)
        eta = np.zeros((Nt, Nx))
        vel = np.zeros((Nt, Nx))
        t = np.linspace(0, 120, Nt)

        eta = realization.invert(b, t, x)
        vel = realization.vel(eta, b,  t, x)
        bsurf = surface_core.spacetempSurface('surfprofile', eta, [x, t])
        bsurf.save(fn, 'eta', False)
        bsurf.save_velocity(fn, vel)

    else:
        bsurf = surface_core.surface_from_file(fn, spaceTime=True)
        t = bsurf.t
        x = bsurf.x
        eta = bsurf.eta
        bsurf.load_velocity(fn)
        vel = bsurf.vel


    ax = bsurf.plot_3d_as_2d()
    pt = peak_tracking.get_PeakTracker(x, t, eta, vel)
    pt.plot_all_tracks(ax=ax)
    ax2 = bsurf.plot_3d_as_2d()
    pt.plot_breaking_tracks(ax=ax2)

    ids_breaking_peaks = pt.get_ids_breaking_peaks()
    
    #gt = peak_tracking.get_GroupTracker(x, t, eta, vel)
    #gt.plot_all_tracks(ax=ax)

    # follow one track
    
    peak_dict = pt.get_peak_dict()

    plotting_interface.show()