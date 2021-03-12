import numpy as np
from wave_tools import find_peaks, find_freak_waves, fft_interface, SpectralAnalysis, fft_interpolate
import pylab as plt
import polarTransform
from help_tools import plotting_interface
import h5py

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
        
    def get_r_grid(self):
        return np.abs(self.x)    
    


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
        hf.close()

    def load(self, fn):
        hf = h5py.File(fn, 'r')
        eta = hf.get('eta')
        x = hf.get('x')
        y = hf.get('y')
        name = hf.get('name')
        window_applied = hf.get('window_applied')
        grid = [x,y]
        return Surface(name, eta, grid, window_applied)
        
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
        return subset of eta for given xi value.
        nearest point in grid is chosen for evaluation
        returns two arrays, x and eta(xi)
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
        
    def get_deta_dx(self, cut_off_kx=None):
        kx, ky, eta_fft = fft_interface.physical2spectral(self.eta, [self.x, self.y])
        #TODO: this depends on the indexing can you make a check when initializing the surface?
        kx_cut = np.where(np.abs(kx)<cut_off_kx, kx, 0)
        kx_mesh = np.outer(kx_cut, np.ones(len(ky)))
        #kx_mesh = np.outer(np.ones(len(ky)), kx)
        deta_dx_fft = 1.0j*kx_mesh*eta_fft
        tmp_x, tmp_y, deta_dx = fft_interface.spectral2physical(deta_dx_fft, [kx, ky])
        return deta_dx    
          
    def get_deta_dy(self, cut_off_ky=None):
        kx, ky, eta_fft = fft_interface.physical2spectral(self.eta, [self.x, self.y])
        #TODO: this depends on the indexing can you make a check when initializing the surface?
        ky_cut = np.where(np.abs(ky)<cut_off_ky, ky, 0)
        if cut_off_ky==None:
            ky_mesh = np.outer(np.ones(len(kx)), ky)
        else:
            ky_mesh = np.outer(np.ones(len(kx)), ky_cut)
        #ky_mesh = np.outer(ky, np.ones(len(kx)))
        deta_dy_fft = 1j*ky_mesh*eta_fft
        tmp_x, tmp_y, deta_dy = fft_interface.spectral2physical(deta_dy_fft, [kx, ky])
        return deta_dy  
    
    def plot_3d_surface(self):
        plotting_interface.plot_3d_surface(self.x, self.y, self.eta)
            
    def fft_interpolate(self, inter_factor_x, inter_factor_y):
        '''
        Interpolate eta by truncated Fourier expansion
        
        Parameters:
        -----------
        input
                inter_factor_x      int
                                    interpolated Nx = Nx*inter_factor_x
                inter_factor_y      int
                                    interpolated Ny = Ny*inter_factor_y
        output 
                surface_inter       object
                                    instance of Surface class with interpolated eta and grid
        ''' 
        x_inter, y_inter, eta_inter = fft_interpolate.fft_interpol2d(self.x, self.y, self.eta, inter_factor_x*self.Nx, inter_factor_y*self.Ny)
        return Surface('noName_inter', eta_inter, [x_inter, y_inter]) 
        
    def get_shadowing_mask(self, name, H):
        # TODO: differentiate between kx-ky and k,w
        # create polar image
        x_mesh, y_mesh = np.meshgrid(self.x, self.y, indexing='ij')
        r_mesh = np.sqrt(x_mesh**2 + y_mesh**2)
        theta_mesh = np.arctan2(y_mesh, x_mesh)
        print(r_mesh, theta_mesh)
        eta_pol, settings = polarTransform.convertToPolarImage(self.eta.swapaxes(0,1),  initialRadius=0, center=[32,0], initialAngle=0,
                                                            finalAngle=np.pi, radiusSize=128)
        print(settings)
        #(self.eta, imageSize=(2*N,2*N), initialAngle=theta[0], finalAngle=theta[-1])
        # calculate shadowing for each angle
        #'''
        import pylab as plt
        plt.figure()
        plt.imshow(eta_pol)
        #plt.show()
        #'''
        
        '''
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
        '''
        
        
        N_theta, Nr = eta_pol.shape
        r = np.linspace(0, np.max(r_mesh), Nr)
        theta = np.linspace(0, np.pi, N_theta)
        r_pol, theta_pol = np.meshgrid(r, theta, indexing='xy')#NOTE: opposite indexing to match with polarTransform
        radar_point_angle = np.arctan2(r_pol, (H - eta_pol)).T
        illumination = np.ones(eta_pol.shape).T
        for i in range(0, Nr):
            illumination[i+1:, :] *= radar_point_angle[i, :] < radar_point_angle[i+1:, :]
        illumination = illumination.T
        eta_cart, settings = polarTransform.convertToCartesianImage(eta_pol, center=[Nr//2,0], imageSize=(64, 64), initialAngle=theta[0], finalAngle=theta[-1])
        illu_cart, settings = polarTransform.convertToCartesianImage(illumination, center=[Nr//2,0], imageSize=(64, 64), initialAngle=theta[0], finalAngle=theta[-1])
        plt.figure()
        plt.imshow(illumination)
        plt.figure()
        plt.imshow(illu_cart)
        plt.show()
        return Surface(name, illumination, [self.x, self.y])
            
    def get_illuminated_surface(self, name, H, axis=0):
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

        return Surface(name, illumination*self.eta.copy(), [self.x, self.y])

            

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
            self.x = self.etaND.x
            self.y = self.etaND.y
            self.z = self.etaND.z
        else:
            print('\n\nError: Input data spectrum is not of the correct type\n\n')
        self.eta = self.etaND.eta

    def copy(self, name):
        return Surface(name, self.etaND.eta, self.grid, self.window_applied)        
            
    def get_name(self):
        return self.name
        
    def get_surf(self, x_sub=None, y_sub=None, z_sub=None):
        if self.ND==1:
            return self.etaND.get_surf(x_sub)
        elif self.ND==2:
            return self.etaND.get_surf(x_sub, y_sub)
        elif self.ND==3:
            return self.etaND.get_surf(x_sub, y_sub, z_sub)
      
    def get_deta_dx(self, kx_cut_off):
        if self.ND==1:
            return self.etaND.get_deta_dx(kx_cut_off)
        elif self.ND==2:
            return self.etaND.get_deta_dx(kx_cut_off)
        elif self.ND==3:
            return self.etaND.get_deta_dx(kx_cut_off)        
          
    def get_deta_dy(self, ky_cut_off):
        if self.ND==1:
            return self.etaND.get_deta_dy(ky_cut_off)
        elif self.ND==2:
            return self.etaND.get_deta_dy(ky_cut_off)
        elif self.ND==3:
            return self.etaND.get_deta_dy(ky_cut_off)    
        
              
    def get_r_grid(self):
        if self.ND==1:
            return self.etaND.get_r_grid()
        elif self.ND==2:
            return self.etaND.get_r_grid()
        elif self.ND==3:
            return self.etaND.get_r_grid() 
        
    def plot_3d_surface(self):
        if self.ND==1:
            print('Waring: 3d plotting not enabled in 1d case')
        elif self.ND==2:
            self.etaND.plot_3d_surface()
        elif self.ND==3:
            self.etaND.plot_3d_surface()

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
        
    def fft_interpolate(self, inter_factor_x, inter_factor_y=None, inter_factor_z=None):
        '''
        Interpolate eta by truncated Fourier expansion
        
        Parameters:
        -----------
        input
                inter_factor_x      int
                                    interpolated Nx = Nx*inter_factor_x
                inter_factor_y      int
                                    interpolated Ny = Ny*inter_factor_y
                inter_factor_z      int
                                    interpolated Ny = Ny*inter_factor_z
        output 
                surface_inter       object
                                    instance of Surface class with interpolated eta and grid
        '''
        if self.ND==1:
            return self.etaND.fft_interpolate(inter_factor_x)
        elif self.ND==2:
            return self.etaND.fft_interpolate(inter_factor_x, inter_factor_y)
        elif self.ND==3:
            return self.etaND.fft_interpolate(inter_factor_x, inter_factor_y, inter_factor_z)        
    
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
        
    def get_local_incidence_surface(self, H, k_cut_off=None, approx=False):
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
        #TODO make all functions available in all dimensions and return 0 if suitable

        if k_cut_off == None:
            deta_dx = self.get_deta_dx(0)
            deta_dy = self.get_deta_dy(0)
        else:
            deta_dx = self.get_deta_dx(k_cut_off[0])
            deta_dy = self.get_deta_dy(k_cut_off[1])
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
        grid = [self.x, self.y]
        return Surface('loc_incidence({0:s})'.format(self.name), theta_l, grid, self.window_applied)
        
    def get_shadowing_mask(self, name, H):
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
        return self.etaND.get_shadowing_mask(name, H)    
        
    def get_illuminated_surface(self, name, H, axis=0):
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
                axis    int
                        define range axis      
        '''
        return self.etaND.get_illuminated_surface(name, H, axis)            
        
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
            grid = [self.x, self.y, self.z]   
            x, y, z, coeffs = fft_interface.physical2spectral(self.etaND.eta.copy(), grid)                   
            k_grid = [x,y,z]
        return SpectralAnalysis.SpectralAnalysis(coeffs, abs(coeffs)**2, k_grid, self.window_applied, grid_cut_off)

    def save(self, fn):
        self.etaND.save(fn, self.name, self.window_applied)
        
        
        
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
        
        
           
            
    
        
        
