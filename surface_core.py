import numpy as np
from wave_tools import find_peaks, find_freak_waves, fft_interface, SpectralAnalysis, fft_interpolate
import matplotlib.pyplot as plt
import polarTransform
from help_tools import plotting_interface, polar_coordinates
import h5py
from fractions import Fraction
import scipy

class _Surface1D(object):
    '''
    1D surface
  
    '''
    def __init__(self, eta, grid):
        self.eta = eta
        if type(grid)==list:
            self.x = grid[0]
        else:
            self.x=grid
        self.N = len(self.x)
        self.dx = self.x[1]-self.x[0]      

    def fft_interpolate(self, inter_factor_x): 
        return fft_interpolate.fft_interpol1d(self.x, self.eta, inter_factor_x*self.N)

    def get_sub_surface(self, extent, dx_new):
        if dx_new is None:
            dx_new = self.dx
        frac_x = Fraction(dx_new/self.dx).limit_denominator(4)
        inter_factor_x = frac_x.denominator
        if inter_factor_x>1:
            x_new, eta_new = self.fft_interpolate(inter_factor_x)
        else:
            x_new = self.x
            eta_new = self.eta     
        x_out = np.arange(extent[0], extent[1]+dx_new/2, dx_new)
        interpol_eta = scipy.interpolate.interp1d(x_new, eta_new)
        eta_out = interpol_eta(x_out)           
        return [x_out], eta_out
        
    def get_r_grid(self):
        return np.abs(self.x)           

    def get_deta_dx(self):
        return np.gradient(self.eta, self.x)

    def get_local_incidence_angle(self, H, approx=False):
        deta_dx = self.get_deta_dx()
        r = np.abs(self.x)        
        if approx:
            n_norm = 1
            b_norm = r
            cos_theta_l = (self.x*deta_dx  )/(n_norm*b_norm)
        else:
            n_norm = np.sqrt(deta_dx**2 + 1)
            b_norm = np.sqrt(r**2 + (H-self.eta)**2)
            cos_theta_l = (self.x*deta_dx + (H-self.eta))/(n_norm*b_norm)
        theta_l = np.arccos(cos_theta_l)
        return theta_l   
            
    def get_illumination_function(self, H):
        # Assuming that dimension is along the radar beam
        r = np.abs(self.x)
        radar_point_angle = np.arctan2(r, (H - self.eta))        
        illumination = np.ones(r.shape)
        for i in range(0,self.N-1): 
            illumination[i+1:,:] *= radar_point_angle[i,:] < radar_point_angle[i+1:,:] 
        return illumination              


class _Surface2D(object):
    '''
    surface over 2 dimentions, may be over
    two spatial dimensions or over spatio-temporal domain
    Not to be used directly but by using class Surface
    '''
    def __init__(self, eta, grid):
        self.eta = eta
        self.x = grid[0]
        self.y = grid[1]
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]

    def save(self, fn, name, window_applied):
        '''
        saves a Surface to hdf5 file format
        '''
        hf = h5py.File(fn, 'w')
        hf.create_dataset('eta', data=self.eta)
        hf.create_dataset('x', data=self.x)
        hf.create_dataset('y', data=self.y)
        hf.attrs['window_applied'] = window_applied 
        hf.attrs['name'] = name
        hf.attrs['ND'] = 2
        hf.close()
        
    def get_surf(self, x_sub, y_sub):
        '''
        return surface information (x,y,z)
        if x_sub or y_sub are given return subspace
        Parameters:
            
            x_sub       array/tule containing two values
                        this values define values within x-grid used to define subspace
            y_sub       array/tule containing two values
                        this values define values within y-grid used to define subspace       
        '''
        if x_sub==None:
            x_ind0 = 0
            x_indN = self.Nx
        else:
            x_ind0 = np.argmin(abs(self.x/x_sub[0]))
            x_indN = np.argmin(abs(self.x/x_sub[1]))
            if np.logical_or(x_ind0>=self.Nx, x_ind0<0):
                print('Error: First values of x_sub is outide x.')
                return None
            if np.logical_or(x_indN>=self.Nx, x_indN<0):
                print('Error: Last values of x_sub is outide x.')
                return None
        if y_sub==None:
            y_ind0 = 0
            y_indN = self.Ny
        else:
            y_ind0 = np.argmin(abs(self.y/y_sub[0]))
            y_indN = np.argmin(abs(self.y/y_sub[1]))
            if np.logical_or(y_ind0>=self.Ny, y_ind0<0):
                print('Error: First values of y_sub is outide y.')
                return None
            if np.logical_or(y_indN>=self.Ny, y_indN<0):
                print('Error: Last values of y_sub is outide y.')
                return None
        return self.x[x_ind0:x_indN], self.y[y_ind0:y_indN], (self.eta.copy())[x_ind0:x_indN, y_ind0:y_indN]
        
        
    def eta_at_xi(self, xi, y_sub=None):
        '''
        return subset of eta for given xi value.
        nearest point in grid is chosen for evaluation
        returns two arrays, y and eta(xi)
        '''
        if y_sub==None:
            y_sub = [0, self.Ny]
        y_sub_ind = np.where(np.logical_and(self.y>y_sub[0], abs(self.y)<y_sub[1]))#[0]
        if len(y_sub_ind)<=0:
            print('y_sub does not define subspace on the y-axis')
            return None
        else:
            y_sub_ind = y_sub_ind[0]
        x_ind = np.argmin(abs(self.x/xi-1))
        if np.logical_or(x_ind<self.Nx, x_ind>=0):
            return self.y[y_sub_ind], self.eta[x_ind,y_sub_ind]
        else:
            print('Error: Chosen value of xi was outside of x')
            return None
        
    def eta_at_yi(self, yi, x_sub=None):
        '''
        return subset of eta for given yi value.
        nearest point in grid is chosen for evaluation
        returns two arrays, x and eta(yi)
        '''
        if x_sub==None:
            x_sub = [0, self.Nx]
        x_sub_ind = np.where(np.logical_and(self.x>x_sub[0], abs(self.x)<x_sub[1]))#[0]
        if len(x_sub_ind)<=0:
            print('x_sub does not define subspace on the x-axis')
            return None
        else:
            x_sub_ind = x_sub_ind[0]     
        y_ind = np.argmin(abs(self.x/yi-1))
        if np.logical_or(y_ind<self.Ny, y_ind>=0):
            return self.x[x_sub_ind], self.eta[x_sub_ind,y_ind]
        else:
            print('Error: Chosen value of yi was outside of y')
            return None
        
    def get_r_grid(self):        
        x_mesh, y_mesh = np.meshgrid(self.x, self.y, indexing='ij')
        return np.sqrt(x_mesh**2 + y_mesh**2)

    def get_deta_dx(self):
        deta_dx, deta_dy = np.gradient(self.eta, self.x, self.y)
        return deta_dx

    def get_deta_dy(self):
        deta_dx, deta_dy = np.gradient(self.eta, self.x, self.y)
        return deta_dy
    def plot_3d_surface(self):
        plotting_interface.plot_3d_surface(self.x, self.y, self.eta)

    def plot_3d_as_2d(self):
        plotting_interface.plot_3d_as_2d(self.x, self.y, self.eta)
            
    def fft_interpolate(self, inter_factor_x, inter_factor_y):
        x_inter, y_inter, eta_inter = fft_interpolate.fft_interpol2d(self.x, self.y, self.eta, inter_factor_x*self.Nx, inter_factor_y*self.Ny)
        return  [x_inter, y_inter], eta_inter

    def get_sub_surface(self, extent, dx_new, dy_new):
        if dx_new is None:
            dx_new = self.dx
        if dy_new is None:
            dy_new = self.dy
        frac_x = Fraction(dx_new/self.dx).limit_denominator(4)
        inter_factor_x = frac_x.denominator
        frac_y = Fraction(dy_new/self.dy).limit_denominator(4)
        inter_factor_y = frac_y.denominator
        if inter_factor_x>1 or inter_factor_y>1:
            grid_new, eta_new = self.fft_interpolate(inter_factor_x, inter_factor_y)
            x_new, y_new = grid_new
        else:
            x_new = self.x
            y_new = self.y
            eta_new = self.eta     
        x_out = np.arange(extent[0], extent[1]+dx_new/2, dx_new)
        y_out = np.arange(extent[2], extent[3]+dy_new/2, dy_new)
        interpol_eta = scipy.interpolate.RectBivariateSpline(x_new, y_new, eta_new)
        eta_out = interpol_eta(x_out, y_out)           
        return [x_out, y_out], eta_out 

    def get_local_incidence_angle(self, H, approx=False):
        deta_dx = self.get_deta_dx()
        deta_dy = self.get_deta_dy()
        r = self.get_r_grid()
        x_mesh, y_mesh = np.meshgrid(self.x, self.y, indexing='ij')
        if approx:
            n_norm = 1
            b_norm = r
            cos_theta_l = (x_mesh*deta_dx + y_mesh*deta_dy )/(n_norm*b_norm)
        else:
            n_norm = np.sqrt(deta_dx**2 + deta_dy**2 + 1)
            b_norm = np.sqrt(r**2 + (H-self.eta)**2)
            cos_theta_l = (x_mesh*deta_dx + y_mesh*deta_dy + (H-self.eta))/(n_norm*b_norm)
        theta_l = np.arccos(cos_theta_l)
        return theta_l
        
    def get_geometric_shadowing(self, H):
        # TODO: differentiate between kx-ky and k,w
        # create polar image
        # improve check for x grid and y grid!
        # Extension needed so that antenna position is in the image
        '''
        if self.x[0]<=0:
            x_all = self.x
        else:
            x_all = np.arange(0, self.x[-1]+self.dx, self.dx)
        if self.y[0]<=0:
            y_all = self.y
        else:
            y_all = np.arange(0, self.y[-1]+self.dy, self.dy)

        x_mesh, y_mesh = np.meshgrid(x_all, y_all, indexing='ij')
        r_mesh = np.sqrt(x_mesh**2 + y_mesh**2)
        theta_mesh = np.arctan2(y_mesh, x_mesh)
        
        Nx_all, Ny_all = x_mesh.shape
        eta_all = np.zeros(x_mesh.shape)
        eta_all[Nx_all-self.Nx:,Ny_all-self.Ny:] = self.eta
        x_center_ind = np.argmin(np.abs(x_all))
        y_center_ind = np.argmin(np.abs(y_all))
        
        radiusSize = self.Nx + self.Ny
        initAngle = np.min(theta_mesh)
        finAngle = np.max(theta_mesh)       

        eta_pol, settings = polarTransform.convertToPolarImage(eta_all.swapaxes(0,1),  initialRadius=0, center=[x_center_ind,y_center_ind], initialAngle=initAngle,
                                                            finalAngle=finAngle, radiusSize=radiusSize)
        eta_pol = eta_pol.swapaxes(0,1)   
        
        
    
        Nr, Ntheta = eta_pol.shape
        theta0 = settings.initialAngle
        thetaN = settings.finalAngle
        r0 = settings.initialRadius
        rN = settings.finalRadius
        r = np.linspace(r0, rN, Nr, endpoint=True)
        theta = np.linspace(theta0, thetaN, Ntheta, endpoint=True)
        #plotting_interface.plot_3d_as_2d(r, theta*180/np.pi, eta_pol)
        '''


        # second approach
        r, theta, eta_pol = polar_coordinates.cart2finePol(self.x, self.y, self.eta)
     
        # calculate shadowing for each angle
        r_pol, theta_pol = np.meshgrid(r, theta, indexing='ij')#NOTE: ??? changed? opposite indexing to match with polarTransform
        radar_point_angle = r_pol/(H - eta_pol)#np.arctan2(r_pol, (H - eta_pol))

        illumination = np.ones(eta_pol.shape, dtype=int)
        for i in range(1,10):
            illumination[i:, :] *= ((radar_point_angle[:-i, :] - radar_point_angle[i:, :])*180/np.pi < 0).astype('int')
        '''
        plt.figure()
        plt.plot(illumination[:,200], '-r')
        plt.plot(eta_pol[:,200], '-k')
        plt.figure()
        plt.plot(illumination[:,220])
        plt.plot(eta_pol[:,220])
        plt.figure()
        plt.plot(illumination[:,230])
        plt.plot(eta_pol[:,230])
        plt.figure()
        plt.plot(illumination[:,240])
        plt.plot(eta_pol[:,240])
        plt.show()
        '''
        '''
        #revert mapping 1
        #eta_cart, settings = polarTransform.convertToCartesianImage(eta_pol.swapaxes(0,1), center=[Nr//2,0], initialAngle=theta[0], finalAngle=theta[-1], imageSize=(Nx_all, Ny_all))
        illu_cart, settings = polarTransform.convertToCartesianImage(illumination.swapaxes(0,1), center=[Nr//2,0], initialAngle=theta[0], finalAngle=theta[-1], imageSize=(Nx_all, Ny_all))
        illu_cart = illu_cart.swapaxes(0,1)
        illu_cart = illu_cart[-self.Nx:,-self.Ny:]
        plotting_interface.plot_3d_as_2d(x_all, y_all, illu_cart)
        '''
        # revert mapping2
        illu_cart = polar_coordinates.averagePol2cart(r, theta, illumination, self.x, self.y)
        illu_cart = illu_cart.round().astype(int)

        #plotting_interface.plot_3d_as_2d(self.x, self.y, illu_cart)
        #plotting_interface.show()
        
        return illu_cart

    def get_illumination_function(self, H):
        return self.get_geometric_shadowing(H)
            
    def get_illumination_function_w_k(self, H, axis=0):
        if axis==0:
            r = np.outer(self.x, np.ones(self.Ny))  
            radar_point_angle = np.arctan2(r, (H - self.eta))
        else:
            r = np.outer(self.y, np.ones(self.Nx))    
            radar_point_angle = np.arctan2(r, (H - self.eta.transpose()))
        illumination = np.ones(r.shape)
        for i in range(0,illumination.shape[axis]-1): 
            illumination[i+1:,:] *= radar_point_angle[i,:] < radar_point_angle[i+1:,:] 
        if axis==1:
            illumination = illumination.transpose()
        return illumination       


class _Surface3D(object):
    '''
    Surface over 3 dimensions, 2 spacial dimensions and
    1 temporal dimension. 
    Not to be used directly but by using class Surface
    '''
    def __init__(self, eta, grid):
        self.eta = eta
        self.t = grid[0]
        self.x = grid[1]
        self.y = grid[2]
        self.Nt = len(self.t)
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.dt = self.t[1]-self.t[0]
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]

    def save(self, fn, name, window_applied):
        '''
        saves a Surface to hdf5 file format
        '''
        hf = h5py.File(fn, 'w')
        hf.create_dataset('eta', data=self.eta)
        hf.create_dataset('t', data=self.t)
        hf.create_dataset('x', data=self.x)
        hf.create_dataset('y', data=self.y)
        hf.attrs['window_applied'] = window_applied 
        hf.attrs['name'] = name
        hf.attrs['ND'] = 3
        hf.close()

    def get_surf2d_at_ti(self, ti, name):
        time_index = np.argmin(np.abs(self.t-ti))
        return self.get_surf2d_at_index(time_index, name)

    def plot_surf2d_at_index(self, time_index, name, flat=True):
        surf = self.get_surf2d_at_index(time_index, name)
        if flat:
            surf.plot_3d_as_2d()
        else:
            surf.plot_3d_surface()

    def plot_surf2d_at_ti(self, ti, name, flat=True):
        time_index = np.min(np.abs(self.t-ti))
        self.plot_surf2d_at_index(time_index, name, flat)
        
    def get_local_incidence_angle(self, H, approx=False):
        theta_l = np.zeros(self.eta.shape)
        for i in range(0, len(self.t)):
            surf2d = self.get_surf2d_at_index(i, '')
            theta_l[i,:,:] = surf2d.get_local_incidence_angle(H, approx)
        return theta_l

    def get_illumination_function(self, H):
        illumination = np.zeros(self.eta.shape)
        for i in range(0, self.Nt):
            surf2d = self.get_surf2d_at_index(i, '')
            illumination[i,:,:] = surf2d.get_illumination_function(H)
        return illumination

    def get_sub_surface(self, extent, dt_new, dx_new, dy_new):
        if dt_new is not None:
            print('\nError: The interpolation of the temporal domain has not yet been implemented\n')
            return None
        interpol_dx = 0.5
        interpol_dy = 0.5
        if dx_new is not None or dy_new is not None:
            if dx_new is None:
                dx_new = self.dx
                interpol_dx = self.dx
            if dy_new is None:
                dy_new = self.dy
                interpol_dy = self.dy
            inter_factor_x = int(self.dx/interpol_dx)
            inter_factor_y = int(self.dy/interpol_dy)
            x_new, y_new, eta_new = fft_interpolate.fft_interpol2d(self.x, self.y, self.eta, inter_factor_x*self.Nx, inter_factor_y*self.Ny)
        x_ind1 = np.argmin(np.abs(x_new - extent[0]))
        x_ind2 = np.argmin(np.abs(x_new - extent[1]))+1
        y_ind1 = np.argmin(np.abs(y_new - extent[0]))
        y_ind2 = np.argmin(np.abs(y_new - extent[1]))+1
        x_spacing = int(interpol_dx/dx_new)
        y_spacing = int(interpol_dy/dy_new)
        return [self.t, x_new[x_ind1:x_ind2:x_spacing], y_new[y_ind1:y_ind2:y_spacing]], eta_new[:, x_ind1:x_ind2:x_spacing, y_ind1:y_ind2:y_spacing]     


    def get_surf2d_at_index(self, time_index):
        '''
        Returns a 2d surface for the given time index (only 3d surfaces)
        '''
 
        return Surface('noName', self.eta[time_index,:,:], [self.x, self.y])

    def get_sub_surface(self, extent, dt_new, dx_new, dy_new):
        if dt_new is not None:
            print('\nError: The interpolation of the temporal domain has not yet been implemented\n')
            return None
        eta_out = []
        this_surf2d = self.get_surf2d_at_index(0)
        sub_surf = this_surf2d.get_sub_surface('noName', extent, dx_new, dy_new)
        x_out = sub_surf.etaND.x
        y_out = sub_surf.etaND.y
        eta_out.append(sub_surf.eta)
        for i in range(1, self.Nt):
            this_surf2d = self.get_surf2d_at_index(i)
            sub_surf = this_surf2d.get_sub_surface('noName', extent, dx_new, dy_new)
            eta_out.append(sub_surf.eta)
        return [self.t, x_out, y_out], np.array(eta_out)   


class Surface(object):
    '''
    Class to represent wave surfaces, will invoke specific
    classes of appropriate dimension
    '''
    def __init__(self, name, eta, grid, window_applied=False):
        '''
        
        '''
        self.name = name  
        self.window_applied = window_applied 
        self.grid=grid  
        if len(eta.shape)==1:
            self.ND = 1
            self.etaND = _Surface1D(eta, grid)
            self.x = self.etaND.x
        elif len(eta.shape)==2:
            self.ND = 2        
            self.etaND = _Surface2D(eta, grid)
            self.x = self.etaND.x
            self.y = self.etaND.y
        elif len(eta.shape)==3:
            self.ND = 3        
            self.etaND = _Surface3D(eta, grid)
            self.t = self.etaND.t
            self.x = self.etaND.x
            self.y = self.etaND.y
        else:
            print('\n\nError: Input data spectrum is not of the correct type\n\n')
        self.eta = self.etaND.eta

    def copy(self, name):
        return Surface(name, self.etaND.eta, self.grid, self.window_applied)    

    def get_sub_surface(self, name, extent, dx_new=None, dy_new=None, dt_new=None):
        if self.window_applied:
            print('\nA window has been applied to the surface, creating a subsurface does not make sense \n')
            return None
        if self.ND==1:
            grid, eta = self.etaND.get_sub_surface(extent, dx_new) 
        if self.ND==2:
            grid, eta = self.etaND.get_sub_surface(extent, dx_new, dy_new)            
        if self.ND==3:
            grid, eta = self.etaND.get_sub_surface(extent, dt_new, dx_new, dy_new)
        return Surface(name, eta, grid, window_applied=False)
            
    def get_name(self):
        return self.name

    def replace_grid(self, new_grid):
        self.grid = new_grid
        if self.ND==1:
            if type(new_grid)==list:
                self.x = new_grid[0]
            else:
                self.x=new_grid
            self.etaND.x = self.x
        if self.ND==2:
            self.x = new_grid[0]
            self.y = new_grid[1]
            self.etaND.x = self.x
            self.etaND.y = self.y
        if self.ND==3:
            self.t = new_grid[0]
            self.x = new_grid[1]
            self.y = new_grid[2]
            self.etaND.t = self.t
            self.etaND.x = self.x
            self.etaND.y = self.y

    def copy2newgrid(self, name, new_grid):
        if type(new_grid)!=list:
            if self.ND!=1:
                print('Error: grid provided does not match for the surface!')
                raise Exception
        elif len(new_grid) != self.ND:
            print('Error: grid provided does not match for the surface!')
            raise Exception
        return Surface(name, self.etaND.eta.copy(), new_grid)        
        
    def get_surf(self, x_sub=None, y_sub=None, z_sub=None):
        if self.ND==1:
            return self.etaND.get_surf(x_sub)
        elif self.ND==2:
            return self.etaND.get_surf(x_sub, y_sub)
        elif self.ND==3:
            return self.etaND.get_surf(x_sub, y_sub, z_sub)

    def get_surf2d_at_index(self, time_index):
        '''
        Returns a 2d surface for the given time index (only 3d surfaces)
        '''
        if self.ND==3:  
            return Surface(self.name+'_at_{0:.2f}'.format(self.t[time_index]), self.eta[time_index,:,:], [self.x, self.y])
        else:
            return NotImplemented

    def get_surf2d_at_ti(self, ti, name):
        if self.ND==3:
            return self.etaND.get_surf2d_at_ti(ti, name)
        else:
            return NotImplemented            
      
    def get_deta_dx(self):
        if self.ND==1:
            return self.etaND.get_deta_dx()
        elif self.ND==2:
            return self.etaND.get_deta_dx()
        elif self.ND==3:
            return self.etaND.get_deta_dx()        
          
    def get_deta_dy(self):
        if self.ND==1:
            return self.etaND.get_deta_dy()
        elif self.ND==2:
            return self.etaND.get_deta_dy()
        elif self.ND==3:
            return self.etaND.get_deta_dy()            
              
    def get_r_grid(self):
        if self.ND==1:
            return self.etaND.get_r_grid()
        elif self.ND==2:
            return self.etaND.get_r_grid()
        elif self.ND==3:
            return self.etaND.get_r_grid() 
        
    def plot_3d_surface(self, time_index=0):
        if self.ND==1:
            print('Waring: 3d plotting not enabled in 1d case')
        elif self.ND==2:
            self.etaND.plot_3d_surface()
        elif self.ND==3:
            self.etaND.plot_surf2d_at_index(time_index, self.name, flat=False)

    def plot_3d_as_2d(self, time_index=0):
        if self.ND==1:
            print('Waring: 3d plotting not enabled in 1d case')
        elif self.ND==2:
            self.etaND.plot_3d_as_2d()
        elif self.ND==3:
            self.etaND.plot_surf2d_at_index(time_index, self.name)

    def apply_window(self, window):
        if self.window_applied==True:
            print('\nWarning: a window has already been applied, the window is not applied!')
        else:
            self.etaND.eta *= window
            self.window_applied = True

    def remove_window(self, window):
        if self.window_applied==False:
            print('Warning: no window has been applied, no window can be removed!')
        else:
            self.etaND.eta /= window
            self.window_applied = False
        
    def fft_interpolate(self, inter_factor_x, inter_factor_y=None, inter_factor_t=None):
        '''
        Interpolate eta by truncated Fourier expansion
        
        Parameters:
        -----------
        input
                inter_factor_x      int
                                    interpolated Nx = Nx*inter_factor_x
                inter_factor_y      int
                                    interpolated Ny = Ny*inter_factor_y
                inter_factor_t      int
                                    interpolated Nt = Nt*inter_factor_t
        output 
                surface_inter       object
                                    instance of Surface class with interpolated eta and grid
        '''
        if self.ND==1:
            grid_new, eta_new = self.etaND.fft_interpolate(inter_factor_x)
        elif self.ND==2:
            grid_new, eta_new = self.etaND.fft_interpolate(inter_factor_x, inter_factor_y)
        elif self.ND==3:
            grid_new, eta_new = self.etaND.fft_interpolate(inter_factor_t, inter_factor_x, inter_factor_y)        
        return Surface('noName_inter', eta_new, grid_new, window_applied=False)
    
    def find_crests(self, axis, method='zero_crossing'): 
        '''
        Returns the indices where the data has peaks according to the given methods:
        The maximum is calculated in one dimension indicated by the axis
        So far limited to 2d, first dimension is the interesting one!
        Parameters:
        -----------
        input:
        ------- 
        axis        int: number of axis where max should be calculated
        method      method for finding peaks: "zero_crossing": the peak 
                    between two zero crossings; zeros crossings does not 
                    find single peaks with negative values to each side
                    method "all_peaks" finds all individiual peaks
        return:     1d array for 1d data input, two arrays for 2d data input
        --------
        peak_indices            
        ''' 
        return find_peaks.find_peaks(self.etaND.eta, axis, method)
    
    def find_freak_waves(self, axis):
        '''
        
        '''
        return find_freak_waves.find_freak_waves(self.etaND.eta, axis)
        
    def replace_eta(self, new_eta):
        self.etaND.eta =new_eta

    def get_local_incidence_angle(self, H, approx=False):
        '''
        Gets the local incidence angle based on the radar elevation H above the mean sea level
        Parameters:
        -----------
        input
                H       float
                        height of radar
                approx  int
                        True: the local incidence angle is approximated by assuming...
                        False(default): the local incidence angle is calculated exactly
        '''
        return self.etaND.get_local_incidence_angle(H, approx)

    def get_local_incidence_surface(self, name, H, k_cut_off=None, approx=False):
        '''
        Gets the surface with the local incidence angle based on the radar elevation H above 
        the mean sea level
        Parameters:
        -----------
        input
                H       float
                        height of radar
                approx  int
                        True: the local incidence angle is approximated by assuming...
                        False(default): the local incidence angle is calculated exactly
        '''
        theta_l = self.etaND.get_local_incidence_angle(H, approx)
        return Surface(name, theta_l, self.grid, self.window_applied)
        
    def get_geometric_shadowing(self, name, H):
        '''
        Shadowing mask along given axis
        Parameters:
        -----------
        input
                name    string
                        define name for shadwoing mask object
                H       float
                        height of radar
                axis    int
                        define range axis      
        '''
        return Surface(name, self.etaND.get_geometric_shadowing(name, H), self.grid, self.window_applied)    

    def get_illumination_function(self, H):
        '''
        Return illumination function (shadows=0), where shadowing is 
        calculated along given axis
        Parameters:
        -----------
        input
                H       float
                        height of radar
        '''
        return self.etaND.get_illumination_function(H)


    def get_illuminated_surface(self, name, H):
        '''
        Return illuminated surface (shadows=0), where shadowing is 
        calculated along given axis
        Parameters:
        -----------
        input
                name    string
                        define name for shadwoing mask object
                H       float
                        height of radar   
        '''
        illumination_function = self.etaND.get_illumination_function(H)     
        return Surface(name, illumination_function*self.eta.copy(), self.grid)  

    def get_local_incidence_angle_with_shadowing_surface(self, name, H, approx=False):
        theta_l = self.etaND.get_local_incidence_angle(H, approx)
        illumination_function = self.etaND.get_illumination_function(H) 
        return Surface(name, illumination_function*theta_l, self.grid)  
        
    def eta_at_xi(self, xi, y_sub=None, z_sub=None):
        if self.ND==2:
            return self.etaND.eta_at_xi(xi, y_sub)
        elif self.ND==3:
            return self.etaND.eta_at_xi(xi, y_sub, z_sub)
        
    def eta_at_yi(self, yi, x_sub=None, z_sub=None):
        if self.ND==2:
            return self.etaND.eta_at_yi(yi, x_sub)
        elif self.ND==3:
            return self.etaND.eta_at_yi(yi, x_sub, z_sub) 
            
    def define_SpectralAnalysis(self, grid_cut_off=None):
        '''
        Create object in spectral domain that corresponds to the given surface and the given grid      
        '''
        if self.ND==1:
            grid = [self.x]
            x, coeffs = fft_interface.physical2spectral(self.etaND.eta.copy(), grid)  
            k_grid = [x]            
        elif self.ND==2:
            grid = [self.x, self.y]
            x, y, coeffs = fft_interface.physical2spectral(self.etaND.eta.copy(), grid)            
            k_grid = [x,y]            
        elif self.ND==3:
            grid = [self.t, self.x, self.y]   
            w, kx, ky, coeffs = fft_interface.physical2spectral(self.etaND.eta.copy(), grid)                   
            k_grid = [w, kx, ky]
        return SpectralAnalysis.SpectralAnalysis(coeffs, abs(coeffs)**2, k_grid, self.window_applied, grid_cut_off)

    def save(self, fn):
        self.etaND.save(fn, self.name, self.window_applied)

    def add_illumination_to_file(self, fn, H):
        '''calculate illumination function and add it to file'''
        hf = h5py.File(fn, 'a')
        illumination = self.get_illumination_function(H)
        hf.create_dataset('illumination_{0:d}'.format(H), data=illumination)
        hf.close()

    def add_local_incidence_angle_to_file(self, fn, H, approx=False):
        '''calculate local incidence angle and add it to file'''
        hf = h5py.File(fn, 'a')
        loc_inc = self.get_local_incidence_angle(H, approx)
        hf.create_dataset('loc_inc_{0:d}'.format(H), data=loc_inc) 
        hf.close()  

    def add_current_to_file(self, fn, z, U, psi):
        hf = h5py.File(fn, 'a')
        hf.create_dataset('z', data=z)
        hf.create_dataset('U', data=U)
        hf.create_dataset('psi', data=psi)
        hf.close()

    def add_wave_parameters_to_file(self, fn, Hs, lambda_p, gamma, theta_mean, smax, h):
        hf = h5py.File(fn, 'a')
        hf.attrs['Hs'] = Hs
        hf.attrs['lambda_p'] = lambda_p
        hf.attrs['gamma'] = gamma
        hf.attrs['theta_mean'] = theta_mean
        hf.attrs['smax'] = smax
        hf.attrs['h'] = h
        hf.close()
        


def surface_from_file(fn):
    '''
    Read surface from file and create instance surface from file
    '''  
    hf = h5py.File(fn, 'r')
    name = hf.attrs['name']
    window_applied = hf.attrs['window_applied']
    eta = np.array(hf.get('eta'))
    ND = hf.attrs['ND']
    x = np.array(hf.get('x'))
    if ND == 1:
        grid = [x]
    elif ND==2:
        y = np.array(hf.get('y') )
        grid = [x, y]
    elif ND==3:
        t = np.array(hf.get('t'))
        y = np.array(hf.get('y') )
        grid = [t, x, y]
    return Surface(name, eta, grid, window_applied)

def illumination_from_file(fn, H):
    '''
    Read illumination from file and create instance surface from file
    '''  
    hf = h5py.File(fn, 'r')
    name = hf.attrs['name']
    window_applied = hf.attrs['window_applied']
    eta = np.array(hf.get('illumination_{0:d}'.format(H)))
    ND = hf.attrs['ND']
    x = np.array(hf.get('x'))
    if ND == 1:
        grid = [x]
    elif ND==2:
        y = np.array(hf.get('y') )
        grid = [x, y]
    elif ND==3:
        t = np.array(hf.get('t'))
        y = np.array(hf.get('y') )
        grid = [t, x, y]
    return Surface('illumination_{0:d}'.format(H)+name, eta, grid, window_applied)


def local_incidence_angle_from_file(fn, H):
    '''
    Read local incidence from file and create instance surface from file
    '''  
    hf = h5py.File(fn, 'r')
    name = hf.attrs['name']
    window_applied = hf.attrs['window_applied']
    eta = np.array(hf.get('loc_inc_{0:d}'.format(H)))
    ND = hf.attrs['ND']
    x = np.array(hf.get('x'))
    if ND == 1:
        grid = [x]
    elif ND==2:
        y = np.array(hf.get('y') )
        grid = [x, y]
    elif ND==3:
        t = np.array(hf.get('t'))
        y = np.array(hf.get('y') )
        grid = [t, x, y]
    hf.close()
    return Surface('loc_inc_{0:d}'.format(H)+name, eta, grid, window_applied)

def current_from_file(fn):
    hf = h5py.File(fn, 'r')
    z = np.array(hf.get('z'))
    U = np.array(hf.get('U'))
    psi = np.array(hf.get('psi')) + 0 #+0 to convert possible float
    hf.close()
    return z, U, psi

def wave_parameters_from_file(fn):
    hf = h5py.File(fn, 'r')
    Hs = hf.attrs['Hs']
    lambda_p = hf.attrs['lambda_p']
    gamma = hf.attrs['gamma']
    theta_mean = hf.attrs['theta_mean']
    smax = hf.attrs['smax']
    h = hf.attrs['h']
    hf.close()
    return Hs, lambda_p, gamma, theta_mean, smax, h

        
def plot_surfaces2d(list_of_surfaces, xi, yi, y_sub=None, x_sub=None):
    '''
    function for plotting surfaces
    plot surface along lines at center of the grid
    Parameters:
    ------------
    input
            list_of_surfaces        list
                                    surface objects                                   
            xi                      float
                                    point along x-axis                                
            yi                      float
                                    point along y-axis
            y_sub
            
            x_sub
    
    '''        
    N_surf = len(list_of_surfaces)
    fig1, fig2 = plt.figure(), plt.figure()
    ax1, ax2 = fig1.add_subplot(111), fig2.add_subplot(111)
    for i in range(0,N_surf):
        y, surf_y = list_of_surfaces[i].eta_at_xi(xi, y_sub)
        x, surf_x = list_of_surfaces[i].eta_at_yi(yi, x_sub)       
        ax1.plot(y, surf_y, label=list_of_surfaces[i].get_name())
        ax2.plot(x, surf_x, label=list_of_surfaces[i].get_name())
        ax1.legend()
        ax2.legend()
    plt.show()
        
        
           
            
    
        
        
