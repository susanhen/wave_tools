#shoaling
import numpy as np
from numpy.lib import scimath as SM
from scipy.optimize import fsolve
import pylab as plt
from scipy.special import gamma
from scipy.interpolate import interp1d
from scipy.integrate import dblquad, quad, simps
from scipy import integrate
from matplotlib import cm
from help_tools import plotting_interface


class Pol2Cart:
    

    def __init__(self, r, theta, x, y):
        '''
        Based on uniform axis defined in x and y the 
        '''
        self.x_mesh, self.y_mesh = np.meshgrid(x, y, indexing='ij')
        self.Nx, self.Ny = self.x_mesh.shape
        self.r_mesh, self.theta_mesh = np.meshgrid(r, theta, indexing='ij')
        x_pol = (self.r_mesh*np.cos(self.theta_mesh)).flatten()
        y_pol = (self.r_mesh*np.sin(self.theta_mesh)).flatten()
        if np.max(x_pol)>np.max(x):
            print('Out of range along positive x axis')
        if np.min(x_pol)<np.min(x):
            print('Out of range along negative x axis')
        self.x_cart_indices = np.zeros(len(x_pol), dtype=int)
        self.y_cart_indices = np.zeros(len(y_pol), dtype=int)
        if np.max(y_pol)>np.max(y):
            print('Out of range along positive y axis')
        if np.min(y_pol)<np.min(y):
            print('Out of range along negative y axis')
        for i in range(0, len(x_pol)):  
            self.x_cart_indices[i] = np.argmin(np.abs(x_pol[i] - x))
            self.y_cart_indices[i] = np.argmin(np.abs(y_pol[i] - y))
        
    def transform(self, A):
        if A.shape != self.r_mesh.shape:
            print('input format does not have the right shape')
        else:
            A_out = np.zeros((self.Nx, self.Ny), dtype=A.dtype)
            A_flat = A.flatten()
            for i in range(0, A_flat.size):
                mapped_i = self.x_cart_indices[i]
                mapped_j = self.y_cart_indices[i]            
                A_out[mapped_i, mapped_j] = A_flat[i]
        return A_out
        

class Bathymetry:
    def __init__(self, x, y):
        dy = y[1] - y[0]
        y1 = np.arange(400, 500 + dy, dy)
        y2 = np.arange(550, 700 + dy, dy)
        y3 = np.arange(750, 1100 + dy, dy)
        y_u = np.block([y1, y2, y3])
        self.x = x
        self.y = y
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        '''
        bathy1 = -25 * (y_u<=500)
        bathy2 = (-0.33*y_u+140)*(np.logical_and(y_u>=550, y_u<= 700))
        bathy3 = -100*(y_u>=750)
        '''
        bathy1 = -2 * (y_u<=500)
        bathy2 = (-0.1*y_u+50)*(np.logical_and(y_u>=550, y_u<= 700))
        bathy3 = -25*(y_u>=750)
        bathy = bathy1 + bathy2 + bathy3
        h_func = interp1d(y_u, bathy, kind='cubic')
        self.h = h_func(y)
        self.H = -np.outer(np.ones(len(x)), self.h) 
        
    def plot1d(self):
        plt.figure()
        plt.plot(self.y, self.h, 'x')
        
    def plot2d(self):
        X, Y= np.meshgrid(self.x, self.y, indexing='ij')
        fig = plt.figure(figsize=(14,8))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, self.H, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
                
    def calc_wavenumber(self, f_r):        
        N_f = len(f_r)  
        k_out = np.zeros((N_f, len(self.x), len(self.y)))
        for i in range(N_f):
        #kt = np.zeros(len(self.h))
        #   for j in range(0, len(self.h)):
            #kt = np.abs(fsolve(lambda k: ((2*np.pi*f_r[i,0])**2 - 9.81*k*np.tanh(k*(-self.h)) ), 0.01*np.ones(self.Nx), xtol=0.01))
            
            w = 2*np.pi*f_r[i,0]
            ki = w**2/(9.81)
            wt = np.sqrt(9.81*ki*np.tanh(ki*(-self.h)))
            eps = 10**(-6)
            N_max = 100
            count=0
            while np.max(np.abs(w - wt))>eps and count<N_max:
                latter = 9.81*np.tanh(ki*(-self.h))
                ki = w**2/(latter)
                wt = np.sqrt(latter)                              
                count = count + 1
                
                
            #kt = (2*np.pi*f_r[i,0])**2 - 9.81*k*np.tanh(k*(-self.h)) )
            
            k_out[i,:,:] = np.outer(np.ones(self.Nx), ki)
        return k_out
            

class DirectionalSpectrum:
    def __init__(self, Tp, theta_p, gam, c, F):
        self.Tp = Tp
        self.fp = 1./Tp
        self.theta_p = theta_p
        self.gam = gam
        g = 9.81        
        U = lambda UU: 3.5*(g/UU)*(g/UU**2*F)**(-0.33)-self.fp
        self.U10 = fsolve(U, 10,  xtol=1e-04)[0]
        self.xxn = g/self.U10**2*F
        self.S = lambda f:(0.076*self.xxn**(-0.22)*g**2/(2*np.pi)**4*(f)**(-5)*np.exp(-5/4*(self.fp/f)**4)
         *gam**np.exp(-((f-self.fp)**2)/(2*(self.fp*(0.07*(1/2 + 1/2*np.sign(self.fp - f))
        +0.09*(1/2 -1/2*np.sign(self.fp - f))))**2)))
                
        self.s = lambda f:((c*(2*np.pi*self.fp*self.U10/g)**(-2.5)*(f/self.fp)**5)*(1/2 + 1/2*np.sign(self.fp - f)) +
    (c*(2*np.pi*self.fp*self.U10/g)**(-2.5)*(f/self.fp)**(-2.5))*(1/2 -1/2*np.sign(self.fp - f)))
        self.D = lambda f, theta: (2**(2*self.s(f)-1)/np.pi*(gamma(self.s(f)+1))**2./gamma(2*self.s(f)+1)*(np.abs(np.cos((theta-theta_p)/2)))**(2*self.s(f)))
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
            fig = plt.figure(figsize=(14,8))
            ax = fig.gca(projection='3d')
            xxx = np.arange(0, N_f)
            yyy = np.arange(0, N_theta)
            bla, blu = np.meshgrid(xxx, yyy, indexing='ij')
            surf = ax.plot_surface(bla, blu, Theta, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        return ff, Theta
    
    
    def define_realization(self, f_min, f_max, theta_min, theta_max, N_f, N_theta, plot_it=False):
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
            fig = plt.figure(figsize=(14,8))
            ax = fig.gca(projection='3d')
            xxx = np.arange(0, N_f)
            yyy = np.arange(0, N_theta)
            bla, blu = np.meshgrid(xxx, yyy, indexing='ij')
            surf = ax.plot_surface(bla, blu, a, cmap=cm.coolwarm, linewidth=0, antialiased=True)
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

        
    def calc_wavenumber(self, Nx, Ny, bathy=None, h=1000, plot_it=False):
        '''
        function to calculate the wave number for f_r
        Parameters:
        -----------
            bathy       : optional Bathymetry object
                          default: None; if provided defines the bathymetry for the calculation
            h           : float
                          default: 1000 meters; is used when bathymetry not present; defines a constant waterdepth
        '''
        if bathy==None:            
            k_loc_f = fsolve((lambda k: ((9.81*k*np.tanh(k*h)) - (2*np.pi*self.f_r[:,0])**2)), 0.01*np.ones(self.N_f))
            k_loc = np.outer(k_loc_f, np.ones(Nx*Ny)).reshape((self.N_f, Nx, Ny))
        else:
            k_loc = bathy.calc_wavenumber(self.f_r)
            
        if plot_it:
            plt.figure()
            plt.plot(k_loc[:,0,0], self.f_r[:,0])
            if bathy!=None:
                for i in range(1,bathy.Ny):
                    plt.plot(k_loc[i,:,0], self.f_r[:,0])
        return k_loc
                
    def invert(self, bathy, ti, x, y, h=1000, plot_it=True):
        # es sollte hier 2 Methoden geben: das argument des cos zu kartesischen konvertieren
        # man könnte auch einfach auf dem kart. System arbeiten (uniform) und dann nur die richtigen as dran multiplizieren. Dann bräuchte man vorher sicher auch die integrale nicht?
        # natürlich könnte man auch einfach zu Polarkoordinaten transferieren
        #man braucht auch noch ein omega-grid... wie kann man dann da die echten omega dranmachen?
        #zeta = np.dot(A_cart, np.outer(np.cos(w*t[i]-kx_mesh*)))
        # TODO different option for bathymetry or const depth
        Nx = len(x)
        Ny = len(y)
        k = self.calc_wavenumber(Nx, Ny, bathy)
        print('wavenumber calculated')
        w = 2*np.pi*self.f_r    

        X, Y = np.meshgrid(x, y, indexing='ij')
        theta = self.Theta_r
        dy = np.gradient(y)
        
        if not bathy is  None:
            H = bathy.H
        else:
            H = h
        
        zeta = np.zeros((Nx, Ny))
        #phase = np.random.uniform(size=self.N_f)*np.pi*2
        print('before loop')
        for i in range(0, self.N_f):
            k2H_by_sinh_2kH = np.where(k[i,:,:]*H < 50,  2*k[i,:,:]*H / np.sinh(2*k[i,:,:]*H), 0)
            for j in range(0, self.N_theta):
                thetaxy = np.abs(np.arcsin(np.sin(theta[i,j])*k[i,-1,-1]/k[i,:,:]))
                ky = k[i,:,:] * np.cos(thetaxy)
                ksh = np.cumsum(ky[0,:])*dy
                Cgx =  w[i,j]/(2*k[i,:,:] * (1+k2H_by_sinh_2kH))*(np.cos(thetaxy))
                Cg0x = w[i,j]/(2*k[i,-1,-1] * (1+k2H_by_sinh_2kH[-1,-1]))*(np.cos(theta[i,j]))
                phase = np.random.uniform()*np.pi*2
                zeta += self.a[i,j]* np.abs(SM.sqrt(Cg0x/Cgx))*np.cos(phase+w[i, j]*ti-k[i,-1,-1]*np.sin(theta[i,j])*X
                +np.outer(np.ones(Nx), ksh))
        if plot_it:
            plotting_interface.plot_3d_as_2d(x, y, zeta)
            plt.colorbar()
            plt.xlabel('$x~[\mathrm{m}]$')
            plt.xlabel('$y~[\mathrm{m}]$')
            plt.figure()
            plt.plot(x, zeta[:,150])
            plt.xlabel('$x~[\mathrm{m}]$')
            plt.ylabel('$\eta~[\mathrm{m}]$')
            plt.figure()
            plt.plot(y, zeta[400,:])
            plt.xlabel('$y~[\mathrm{m}]$')
            plt.ylabel('$\eta~[\mathrm{m}]$')
        return zeta
                

    

if __name__=='__main__':

    dx = 2
    dy = dx
    x = np.arange(-500, 500 + dx, dx)
    y = np.arange(400, 1100 + dy, dy)
    g = 9.81
    Tp = 10
    fp = 1./Tp
    theta_p = -5*np.pi/180
    gam = 3.3
    N_f = 100
    f_min = 0.001
    f_max = 0.4
    N_theta = 40
    theta_min = -np.pi
    theta_max = np.pi
    c = 50
    F = 300000
    
    DirSpec = DirectionalSpectrum(Tp, theta_p, gam, c, F)
    #DirSpec.plot_S(f_min, f_max)
    #DirSpec.plot_s(f_min, f_max)
    #DirSpec.plot_D(f_min, f_max, theta_min, theta_max)
    realization = SpectralRealization(DirSpec, f_min, f_max, theta_min, theta_max, N_f, N_theta)
    print('Directional Spectrum defined')
    b = Bathymetry(x, y)
    print('Bathymetry defined')
    #b.plot1d()
    #b.plot2d()
    #plt.show()
    zeta = realization.invert(b, 0, x, y)    
    
    plt.show()
    