import numpy as np
from wave_tools import moving_average, surface_core, fft_interface
#from PythonCode import SpectralPlotting
from radar_tools import dispersion_filter, filter_core

def _symmetrize2d(surf):
    Nx, Ny = surf.shape
    Nx //= 2
    Ny //= 2
    # ensure symmetric axes
    surf[1:Nx,Ny] = np.flip(surf[Nx+1:,Ny]).conjugate()
    surf[Nx,Ny+1:] = np.flip(surf[Nx,1:Ny]).conjugate()
    
    # ensure symmetry off axes
    surf[1:Nx, 1:Ny] = np.flip(surf[Nx+1:,Ny+1:]).conjugate()
    surf[Nx+1:, 1:Ny] = np.flip(surf[1:Nx,Ny+1:]).conjugate()
    return surf


class _SpectralAnalysis1d(object):
    def __init__(self, coeffs, spectrum, x_grid, grid_cut_off=None):
        '''
        1D specturm, coeffs may be None, typically used for integrated
        subspectra for 2D and 3D spectra
        '''
        self.coeffs = coeffs
        self.spectrum = spectrum
        if type(x_grid) == list:
            self.x_grid = x_grid[0]
        else:   
            self.x_grid = x_grid
        self.SAx = self
        self.dx = abs(self.x_grid[1]-self.x_grid[0])
        
    def get_S(self, mov_av=1):
        '''
        Return the 1d spectrum with the corresponding grid
        '''
        return self.x_grid, moving_average.moving_average(self.spectrum.copy(), mov_av)          
        
    def get_C(self):
        '''
        Return the 1d complex coeffiecients with the corresponding grid
        If the functions returns None as self.coeffs, only the spectral values are available
        '''
        return self.x_grid, self.coeffs.copy()           
        
    def get_S_one_sided(self, mov_av=1):
        '''
        Return upper half of the 1d spectrum  with the corresponding grid 
        '''
        N_half = int(0.5*len(self.x_grid))
        return self.x_grid[N_half:], moving_average.moving_average(2*self.spectrum[N_half:], mov_av)
       
    def get_characteristic_peak(self, mov_av=80, moment=1, plot_spec=False):
        '''
        from Tc definition 
        return characteristic values for wave period. 
        
        Parameters:
        ----------
        moment      moment to calculate; default = 1 which defines the peak period
        mov_av      moving average to smoothen the spectrum default = 80
        plot_spec   plotting the spectrum with 80% of the central values
        '''
        
        N_half = int(0.5*len(self.x_grid))
        S_smooth = moving_average.moving_average(self.spectrum[N_half:], mov_av) 
        lower_ind = np.argwhere(S_smooth>=0.8*max(S_smooth))[0][0]
        upper_ind = np.argwhere( S_smooth >=0.8*max( S_smooth ))[-1][0] 
        if plot_spec:    
            plt.figure()
            plt.plot(self.x_grid[N_half:], S_smooth/max(S_smooth))
            plt.plot(self.x_grid[N_half:][lower_ind:upper_ind], S_smooth[lower_ind:upper_ind]/max( S_smooth ))
        m_neg1  = np.sum(( S_smooth [lower_ind :upper_ind ]*self.dx)*(self.x_grid[N_half:][lower_ind :upper_ind ])**(-1))
        m0  = np.sum(( S_smooth [lower_ind :upper_ind ]*self.dx)*(self.x_grid[N_half:][lower_ind :upper_ind ])**(0))
        m1  = np.sum(( S_smooth [lower_ind :upper_ind ]*self.dx)*(self.x_grid[N_half:][lower_ind :upper_ind ])**(1))
        m2  = np.sum(( S_smooth [lower_ind :upper_ind ]*self.dx)*(self.x_grid[N_half:][lower_ind :upper_ind ])**(2))

        if moment==0:
            Tc = 2*np.pi*(m_neg1/m0)
        elif moment==1:
            Tc = 2*np.pi*(m0/m1)
        elif moment==2:
            Tc = 2*np.pi*sqrt(m0/m2)
        else:
            print('Error: given moment was not defined')
        return Tc
        
    def plot(self, fn, extent, save=False):
        '''
        Parameters:
        -----------
        fn          filename for saving plot, also used as title
        extent      tupel of limits (x_lower, x_upper), y_lower, y_upper) #FIXME CHECK if None option is implemented        
        '''
        SpectralAnalysis.plot_1D_spec(self.spectrum, self.x_grid, fn, x_label, y_label='', save=save, fn=fn)   # FIXME always plot scaled spectra!!! # FIXME apply extent
        
class _SpectralAnalysis2d(object):
    def __init__(self, coeffs, spectrum, xy_grid, grid_cut_off=None):
        self.coeffs = coeffs
        self.spectrum = spectrum
        self.x_grid = xy_grid[0]
        self.y_grid = xy_grid[1]
        if grid_cut_off == None:
            self.x_cut_off = self.x_grid[-1]
            self.y_cut_off = self.y_grid[-1]
        else:
            self.x_cut_off = grid_cut_off[0]
            self.y_cut_off = grid_cut_off[1]
        self.Nx = len(self.x_grid)
        self.Ny = len(self.y_grid)
        self.dx = abs(self.x_grid[1]-self.x_grid[0])
        self.dy = abs(self.y_grid[1]-self.y_grid[0])        
        self.SAx = _SpectralAnalysis1d(None, self.dy*np.sum(spectrum,axis=1), self.x_grid) 
        self.SAy = _SpectralAnalysis1d(None, self.dx*np.sum(spectrum,axis=0), self.y_grid)#
        
    def get_S(self, mov_av=1):
        '''
        Return the 2d spectrum with the corresponding grid
        '''
        return self.x_grid, self.y_grid, self.spectrum.copy()           
        
    def get_C(self):
        '''
        Return the 2d complex coefficients with the corresponding grid
        '''
        return self.x_grid, self.y_grid, self.coeffs.copy()  
        
    def plot(self, fn, waterdepth, extent, U, dB=True, vmin=-60, save=False):
        '''
        Parameters:
        -----------
        fn          filename for saving plot, also used as title
        waterdepth  used when defining dispersion relation. If unkonw maybe set to high value #FIXME do this internally and option to set NONE or is there?
        extent      tupel of limits (x_lower, x_upper, y_lower, y_upper) #FIXME CHECK if None option is implemented        
        '''
        SpectralPlotting.plot_k_w_mod2D(self.x_grid, self.y_grid, self.spectrum, fn, waterdepth, r'$\mathrm{dB}$', extent=extent, U=U, dB=dB, vmin=vmin, fn=fn, save=save)  
        
    def get_disp_filtered_spec(self, U, h, filter_width_up, filter_width_down, filter_width_axis, first_axis_k):
        if first_axis_k==True:
            disp_filt = dispersion_filter.k_w_filter(self.Nx, self.Ny, U, h, self.x_grid, self.y_grid, filter_width_up, filter_width_down, filter_width_axis, N_fine=2000)                    
            return SpectralAnalysis(disp_filt*self.coeffs.copy(), disp_filt*self.spectrum.copy(), [self.x_grid, self.y_grid])            
        else:
            print('The configuration where omega is the first axis has not yet been implemented. Do it!')
            return None
        
    def apply_HP_filter(self, limit):
        # TODO: ensure that it is distinguished between 2D and 3D ... this is for 2d with k (previously (k,w))
        '''
        bf = filter_core.BasicFilter((self.Nx,self.Ny))
        dx =  7.5
        dkx = self.dx
        dky = self.dy
        kx_hp_filter = bf.high_pass_filter(np.abs(self.x_grid), limits[0], 0)
        ky_hp_filter = bf.high_pass_filter(np.abs(self.y_grid), limits[1], 1)
        #TODO check if correct and move into filter_core possibly not needed either!
        kx_hp_filter = _symmetrize2d(kx_hp_filter)
        ky_hp_filter = _symmetrize2d(ky_hp_filter)
        '''
        kx, ky = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')
        k = np.sqrt(kx**2 + ky**2)
        k_hp_filter = (k>limit).astype('int')
        self.coeffs *= k_hp_filter
        self.spectrum *= k_hp_filter
        
    def get_2d_MTF(self, grid_offset):
        # TODO improve handling of cut_off to be set globally
        #TODO: this depends on the indexing can you make a check when initializing the surface?
        kx = self.x_grid
        ky = self.y_grid
        kx_cut = np.where(np.abs(kx)<self.x_cut_off, kx, 0)
        ky_cut = np.where(np.abs(ky)<self.y_cut_off, ky, 0)
        kx_mesh = np.outer(kx_cut, np.ones(len(ky)))
        ky_mesh = np.outer( np.ones(len(kx)), ky_cut)
        #x, y, tmp_eta = fft_interface.spectral2physical(self.coeffs, [kx, ky])
        x, dx = fft_interface.k2grid(kx)
        y, dy = fft_interface.k2grid(ky)
        x += grid_offset[0]
        y += grid_offset[1]
        x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
        r_mesh = np.sqrt(x_mesh**2 + y_mesh**2)
        #kx_mesh = np.outer(np.ones(len(ky)), kx)
        tmp1, tmp2, F_cos_theta = fft_interface.physical2spectral(x_mesh/r_mesh, [x, y])
        tmp1, tmp2, F_sin_theta = fft_interface.physical2spectral(y_mesh/r_mesh, [x, y])
        '''
        import pylab as plt
        plt.imshow(np.abs(F_cos_theta))
        plt.figure()
        plt.imshow(np.abs(F_sin_theta))
        plt.show()
        '''
        
        F_cos_theta *= np.sqrt(self.dx*self.dy)
        F_sin_theta *= np.sqrt(self.dx*self.dy)
        Nx = len(kx)
        Ny = len(ky)
        MTF = F_cos_theta[Nx//2, Ny//2]*1.0j*kx_mesh + F_sin_theta[Nx//2, Ny//2]*1.0j*ky_mesh
        return MTF   
            
    def invert(self, name, grid_offset):
        #FIXME: right now symmetry is ensured but this should work anyway!? Or is my filter not symmetric? CHECK not yet correct!!!
        data = self.coeffs.copy()
        #Nx, Ny = data.shape
        #mid_w = int(0.5*Ny)
        #data[1:,1:mid_w] = np.conjugate(np.flipud(np.fliplr(data[1:,mid_w+1:])))
        #data[0,:] = 0
        #data[:,0] = 0          
        x, y, eta_invers = fft_interface.spectral2physical(data, [self.x_grid, self.y_grid])
        x += grid_offset[0]
        y += grid_offset[1]
        return surface_core.Surface(name, eta_invers, [x,y])
        
class _SpectralAnalysis3d(object):
    def __init__(self, coeffs, spectrum, xy_grid, grid_cut_off=None):
        self.coeffs = coeffs
        self.spectrum = spectrum
        self.x_grid = xyz_grid[0]
        self.y_grid = xyz_grid[1]
        self.z_grid = xyz_grid[2]
        self.SAx = _SpectralAnalysis1d(None, np.sum(np.sum(spectrum,axis=1), axis=2), self.x_grid) 
        self.SAy = _SpectralAnalysis1d(None, np.sum(np.sum(spectrum,axis=0), axis=2), self.y_grid)
        self.SAz = _SpectralAnalysis1d(None, np.sum(np.sum(spectrum,axis=0), axis=1), self.z_grid)
        
    def get_S(self, mov_av=1):
        '''
        Return the 2d spectrum with the corresponding grid
        '''
        return self.x_grid, self.y_grid, self.z_grid, self.spectrum.copy()           

    def get_C(self):
        '''
        Return the 2d complex coefficients with the corresponding grid
        '''
        return self.x_grid, self.y_grid, self.z_grid, self.coeffs.copy()           
        
    def plot(self,fn, waterdepth, extent, U, dB, vmin, save):
        '''
        ...
        '''
        #FIXME implement: options of 3d something and slices along different axis, single slices or a bunch, use subspectra in 2D!
        print('Not implemented yet')
        
        

class SpectralAnalysis(object):
    '''
    Class for Analysis 1d, 2d and 3d spectra over a uniform grid.    
    '''
    def __init__(self, coeffs, spectrum, axes_grid, grid_cut_off=None):
        '''
        Parameters:
        ----------
        input:
                coeffs          array
                                complex coefficients, if available, if not None
                spectrum        array
                                spectrum in 1d, 2d, 3d
                axis_grid       list of arrays defining grid for each given axis
        '''
        if len(spectrum.shape)==1:
            self.ND = 1
            self.spectrumND = _SpectralAnalysis1d(coeffs, spectrum, axes_grid, grid_cut_off)
        elif len(spectrum.shape)==2:
            self.ND = 2        
            self.spectrumND = _SpectralAnalysis2d(coeffs, spectrum, axes_grid, grid_cut_off)
        elif len(spectrum.shape)==3:
            self.ND = 3        
            self.spectrumND = _SpectralAnalysis3d(coeffs, spectrum, axes_grid, grid_cut_off)
        else:
            print('\n\nError: Input data spectrum is not of the correct type\n\n')
        
    def get_S(self, mov_av=1):
        '''
        Return spectrum
        '''
        return self.spectrumND.get_S(mov_av)

    def get_C(self):
        '''
        Return complex coefficients, if not given they are returned as None
        '''
        return self.spectrumND.get_C()        

        
    def get_S_integrated(self, axis, one_sided=True):
        '''
        Return the 1d spectrum with the corresponding grid
        over the given axis. Does not work for initial spectrum of 1d       
        
        Parameters:
        -----------
                    input:
                            axis        int
                                        axis over which the 1d spectrum should be given
        '''
        if axis==0:
            sub_spec = self.spectrumND.SAx
        elif (axis==1 and self.ND>1):
            sub_spec = self.spectrumND.SAy
        elif (axis==2 and self.ND>2):
            sub_spec = self.spectrumND.SAz
        else:
            print('wrong axis input for given spectrum')
        if one_sided:
            return sub_spec.get_S_one_sided() 
        else:
            return sub_spec.get_S()  
            
    def get_S_one_sided(self, mov_av=1):
        '''
        Return one sided spectrum for the case of 1D data
        '''          
        if self.ND==1:
            return self.spectrumND.get_S_one_sided(mov_av)
        else:
            print('\n\n Error: Your spectrum has more than one dimensions. Method works only in 1D. Use get S_integrated instead!')
            return 0                  
               
 
    def get_peak(self):
        '''
        Return peak value
        '''
        return max(self.spectrumND.spectrum)

    def get_peak_index(self):
        '''
        Return index of peak value
        if spectrum is 1d, 1 value, if spectrum is 2d, two etc.
        '''
        return np.argmax(self.spectrumND.spectrum)

    def get_peak_axis_values(self):
        '''
        Returns grid values of the peak point
        '''
        peak_indices = self.get_peak_index()
        grip_points = zeros(self.ND)
        for i in range(0, self.ND):
            grid_points[i] = (self.get_S()[i])[peak_indices[i]]
        return grid_points
        
    def get_characteristic_peak(self, axis=0, mov_av=80, moment=1, plot_spec=False):
        '''
        Return characteristic peak
        For 2D and 3D another axis may be chosen
        '''
        if axis==0:
            return self.spectrumND.SAx.get_characteristic_peak(mov_av, moment, plot_spec)
        elif (axis==1 and self.ND>1):
            return self.spectrumND.SAy.get_characteristic_peak(mov_av, moment, plot_spec)   
        elif (axis==2 and self.ND>2):
            return self.spectrumND.SAz.get_characteristic_peak(mov_av, moment, plot_spec) 
        else:
            print('wrong axis input for given specturm!')
            return 0  
            
    def plot(self, fn, waterdepth=None, extent=None, U=None, dB=True, vmin=-60, save=False):
        '''
        
        '''
        #FIXME description!
        if self.ND==1:
            self.spectrumND.plot(fn, extent, save)
        elif self.ND==2:
            self.spectrumND.plot(fn, waterdepth, extent, U, dB, vmin, save)
        elif self.ND==3:
            self.spectrumND.plot(fn, waterdepth, extent, U, dB, vmin, save)  
            
    def get_disp_filtered_spec(self, U, h, filter_width_up=1, filter_width_down=1, filter_width_axis=0, first_axis_k=True):
        '''
        Return an object similar to this except with the dispersion filter applied
        The dispersion filter is defined from the stream velocity U, and waterdepth h
        Paramters:
            U           current
            h           waterdepth
            width_up    number of pixles to enlarge up along the 0-axis; integer values; affects if >0
            width_down  number of pixles to enlarge down along the 0-axis; integer values; affects if >0 
            width_axis  axis along which the filter should be broadend as for now only one axis!    
        
        '''
        return self.spectrumND.get_disp_filtered_spec(U, h, filter_width_up, filter_width_down, filter_width_axis, first_axis_k)
    
    def apply_HP_filter(self, limits):
        self.spectrumND.apply_HP_filter(limits)
        
        
    def get_2d_MTF(self, grid_offset):
        if self.ND==2:
            return self.spectrumND.get_2d_MTF(grid_offset)    
        
    def invert(self, name, grid_offset=None):
        '''
        Return a surface object with the inverted surface and the corresponding grid
        Parameters:
        ----------
            input
                    name        string
                                name for eta surface
                    offset      list of ND floats
                                default None (no offset)
        '''
        if grid_offset==None:
            return self.spectrumND.invert(name, np.zeros(self.ND))        
        else:
            return self.spectrumND.invert(name, grid_offset)        
        
        
        
        
        
            
          
