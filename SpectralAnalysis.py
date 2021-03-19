import numpy as np
from wave_tools import moving_average, surface_core, fft_interface
from help_tools import plotting_interface
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
    def __init__(self, coeffs, spectrum, kx, type, grid_cut_off=None):
        '''
        1D specturm, coeffs may be None, typically used for integrated
        subspectra for 2D and 3D spectra
        '''
        self.coeffs = coeffs
        self.spectrum = spectrum
        if type == 'wavenumber':
            self.type = 1
        elif type == 'frequency':
            self.type = 0
        else:
            print('Error: given type is not valid use "wavenumber" or "frequency"!')
            
        if type(kx) == list:
            self.kx = kx[0]
        else:   
            self.kx = kx
        self.SAx = self
        self.dkx = abs(self.kx[1]-self.kx[0])
        self.Nx = len(self.kx)
        
    def get_S(self, mov_av=1):
        '''
        Return the 1d spectrum with the corresponding grid
        '''
        return self.kx, moving_average.moving_average(self.spectrum.copy(), mov_av)          
        
    def get_C(self):
        '''
        Return the 1d complex coeffiecients with the corresponding grid
        If the functions returns None as self.coeffs, only the spectral values are available
        '''
        return self.kx, self.coeffs.copy()           
        
    def get_S_one_sided(self, mov_av=1):
        '''
        Return upper half of the 1d spectrum  with the corresponding grid 
        '''
        N_half = int(0.5*len(self.kx))
        return self.kx[N_half:], moving_average.moving_average(2*self.spectrum[N_half:], mov_av)
       
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
        
        N_half = int(0.5*len(self.kx))
        S_smooth = moving_average.moving_average(self.spectrum[N_half:], mov_av) 
        lower_ind = np.argwhere(S_smooth>=0.8*max(S_smooth))[0][0]
        upper_ind = np.argwhere( S_smooth >=0.8*max( S_smooth ))[-1][0] 
        if plot_spec:  
            plotting_interface.plot_wavenumber_specs([self.kx[N_half:], self.kx[N_half:][lower_ind:upper_ind]], [S_smooth, S_smooth[lower_ind:upper_ind]])
        m_neg1  = np.sum(( S_smooth [lower_ind :upper_ind ]*self.dkx)*(self.kx[N_half:][lower_ind :upper_ind ])**(-1))
        m0  = np.sum(( S_smooth [lower_ind :upper_ind ]*self.dkx)*(self.kx[N_half:][lower_ind :upper_ind ])**(0))
        m1  = np.sum(( S_smooth [lower_ind :upper_ind ]*self.dkx)*(self.kx[N_half:][lower_ind :upper_ind ])**(1))
        m2  = np.sum(( S_smooth [lower_ind :upper_ind ]*self.dkx)*(self.kx[N_half:][lower_ind :upper_ind ])**(2))

        if moment==0:
            Tc = 2*np.pi*(m_neg1/m0)
        elif moment==1:
            Tc = 2*np.pi*(m0/m1)
        elif moment==2:
            Tc = 2*np.pi*np.sqrt(m0/m2)
        else:
            print('Error: given moment was not defined')
        return Tc
        
    def plot(self, extent=None):
        '''
        Parameters:
        -----------
        fn          filename for saving plot, also used as title
        extent      tupel of limits (x_lower, x_upper), y_lower, y_upper) #FIXME CHECK if None option is implemented        
        '''
        #FIXME: fix extent
        plotting_interface.plot_wavenumber_spec(self.kx, self.spectrum, scaled=True)

    def remove_zeroth(self):
        self.coeffs[self.Nx//2] = 0
        self.spectrum[self.Nx//2] = 0
        
class _SpectralAnalysis2d(object):
    def __init__(self, coeffs, spectrum, k, grid_cut_off=None):
        self.coeffs = coeffs
        self.spectrum = spectrum
        self.kx = k[0]
        self.ky = k[1]
        if grid_cut_off == None:
            self.x_cut_off = self.kx[-1]
            self.y_cut_off = self.ky[-1]
        else:
            self.x_cut_off = grid_cut_off[0]
            self.y_cut_off = grid_cut_off[1]
        self.Nx = len(self.kx)
        self.Ny = len(self.ky)
        self.dkx = abs(self.kx[1]-self.kx[0])
        self.dky = abs(self.ky[1]-self.ky[0])
        self.SAx = _SpectralAnalysis1d(None, self.dky*np.sum(spectrum,axis=1), self.kx, 'wavenumber') 
        self.SAy = _SpectralAnalysis1d(None, self.dkx*np.sum(spectrum,axis=0), self.ky, 'wavenumber')
        
    def get_S(self, mov_av=1):
        '''
        Return the 2d spectrum with the corresponding grid
        '''
        return self.kx, self.ky, self.spectrum.copy()           
        
    def get_C(self):
        '''
        Return the 2d complex coefficients with the corresponding grid
        '''
        return self.kx, self.ky, self.coeffs.copy()  
        
    def plot(self):#, waterdepth, extent, U, dB=True, vmin=-60):
        '''
        Parameters:
        -----------
        fn          filename for saving plot, also used as title
        waterdepth  used when defining dispersion relation. If unkonw maybe set to high value #FIXME do this internally and option to set NONE or is there?
        extent      tupel of limits (x_lower, x_upper, y_lower, y_upper) #FIXME CHECK if None option is implemented        
        '''
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, self.spectrum)  #.plot_k_w_mod2D(self.kx, self.ky, self.spectrum, fn, waterdepth, r'$\mathrm{dB}$', extent=extent, U=U, dB=dB, vmin=vmin, fn=fn, save=save)  
        
    def get_disp_filtered_spec(self, U, h, filter_width_up, filter_width_down, filter_width_axis, first_axis_k):
        if first_axis_k==True:
            disp_filt = dispersion_filter.k_w_filter(self.Nx, self.Ny, U, h, self.kx, self.ky, filter_width_up, filter_width_down, filter_width_axis, N_fine=2000)                    
            return SpectralAnalysis(disp_filt*self.coeffs.copy(), disp_filt*self.spectrum.copy(), [self.kx, self.ky])            
        else:
            print('The configuration where omega is the first axis has not yet been implemented. Do it!')
            return None
        
    def apply_HP_filter(self, limit):
        # TODO: ensure that it is distinguished between 2D and 3D ... this is for 2d with k (previously (k,w))
        '''
        bf = filter_core.BasicFilter((self.Nx,self.Ny))
        dx =  7.5
        dkx = self.dkx
        dky = self.dky
        kx_hp_filter = bf.high_pass_filter(np.abs(self.kx), limits[0], 0)
        ky_hp_filter = bf.high_pass_filter(np.abs(self.ky), limits[1], 1)
        #TODO check if correct and move into filter_core possibly not needed either!
        kx_hp_filter = _symmetrize2d(kx_hp_filter)
        ky_hp_filter = _symmetrize2d(ky_hp_filter)
        '''
        kx, ky = np.meshgrid(self.kx, self.ky, indexing='ij')
        k = np.sqrt(kx**2 + ky**2)
        k_hp_filter = (k>limit).astype('int')
        self.coeffs *= k_hp_filter
        self.spectrum *= k_hp_filter
        
    def get_2d_MTF(self, grid_offset):
        # TODO improve handling of cut_off to be set globally
        #TODO: this depends on the indexing can you make a check when initializing the surface?
        kx_cut = np.where(np.abs(self.kx)<self.x_cut_off, self.kx, 0)
        ky_cut = np.where(np.abs(self.ky)<self.y_cut_off, self.ky, 0)
        kx_mesh = np.outer(kx_cut, np.ones(self.Ny))
        ky_mesh = np.outer( np.ones(self.Nx), ky_cut)
        #x, y, tmp_eta = fft_interface.spectral2physical(self.coeffs, [kx, ky])
        x, dx = fft_interface.k2grid(self.kx)
        y, dy = fft_interface.k2grid(self.ky)
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
        F_cos_theta *= np.sqrt(self.dkx*self.dky)
        F_sin_theta *= np.sqrt(self.dkx*self.dky)
        MTF_inv = -1.0j*(F_cos_theta[self.Nx//2, self.Ny//2]*kx_mesh + F_sin_theta[self.Nx//2, self.Ny//2]*ky_mesh)
        MTF = np.where(np.abs(MTF_inv)>10**(-6), 1./MTF_inv, 1)
        return MTF  

    def apply_MTF(self, grid_offset, percentage_of_max=0.01):
        MTF = self.get_2d_MTF(grid_offset)
        '''
        from help_tools import plotting_interface
        import pylab as plt
        plotting_interface.plot_3d_as_2d(self.kx, self.ky, MTF.real)
        plotting_interface.plot_3d_as_2d(self.kx, self.ky, MTF.imag)
        plt.show()
        '''
        threshold = percentage_of_max * np.max(np.abs(self.coeffs))
        self.coeffs = np.where(np.abs(self.coeffs)>threshold, self.coeffs* MTF, 0)
        self.spectrum = np.abs(self.coeffs)**2
            
    def invert(self, name, grid_offset, window_applied):    
        x, y, eta_invers = fft_interface.spectral2physical(self.coeffs, [self.kx, self.ky])
        x += grid_offset[0]
        y += grid_offset[1]
        return surface_core.Surface(name, eta_invers, [x,y], window_applied)
    
    def remove_zeroth(self):
        self.coeffs[self.Nx//2, self.Ny//2] = 0
        self.spectrum[self.Nx//2, self.Ny//2] = 0
    
        
class _SpectralAnalysis3d(object):
    def __init__(self, coeffs, spectrum, grid, grid_cut_off=None):
        self.coeffs = coeffs
        self.spectrum = spectrum
        self.w = grid[0]
        self.kx = grid[1]
        self.ky = grid[2]
        self.dw = self.w[1] - self.w[0]
        self.dkx = self.kx[1] - self.kx[0]
        self.dky = self.ky[1] - self.ky[0]
        self.Nt, self.Nx, self.Ny = self.coeffs.shape
        if grid_cut_off == None:
            self.w_cut_off = self.w[-1]
            self.kx_cut_off = self.kx[-1]
            self.ky_cut_off = self.ky[-1]
        else:
            self.w_cut_off = grid_cut_off[0]
            self.kx_cut_off = grid_cut_off[1]
            self.ky_cut_off = grid_cut_off[2]
        self.SAw = _SpectralAnalysis1d(None, np.sum(np.sum(spectrum,axis=2), axis=1), self.w, 'frequency') 
        self.SAx = _SpectralAnalysis1d(None, np.sum(np.sum(spectrum,axis=2), axis=0), self.kx, 'wavenumber')
        self.SAy = _SpectralAnalysis1d(None, np.sum(np.sum(spectrum,axis=1), axis=0), self.ky, 'wavenumber')
        
    def get_S(self, mov_av=1):
        '''
        Return the 2d spectrum with the corresponding grid
        '''
        return self.w, self.kx, self.ky, self.spectrum.copy()           

    def get_C(self):
        '''
        Return the 2d complex coefficients with the corresponding grid
        '''
        return self.w, self.kx, self.ky, self.coeffs.copy()   

    def plot(self,fn=None, waterdepth=None, extent=None, U=None, dB=None, vmin=None, save=False):
        '''
        plots the integrated kx-ky-spectrum and the integrated k-omega-spectrum
        '''
        #FIXME implement: options of 3d something and slices along different axis, single slices or a bunch, use subspectra in 2D!
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, np.sum(self.spectrum[self.Nt//2:], axis=0)*self.dw)
        if save:
            plotting_interface.savefig(fn)
        #FIXME: k-omega-spectrum... 
        #plotting_interface.plot_k_w_spec(self.)

    def remove_zeroth(self):
        self.coeffs[self.Nt//2, self.Nx//2, self.Ny//2] = 0
        self.spectrum[self.Nt//2, self.Nx//2, self.Ny//2] = 0  

    def get_w_slice(self, name, window_applied, grid_cut_off):
        '''
        spectral objects for all positve (negative???) frequencies
        '''
        spec2d_list = []
        for i in range(self.Nt//2, self.Nt):
            spec2d_list.append(SpectralAnalysis(name+r'$_\omega_{0:d}$'.format(i), [self.kx, self.ky], window_applied, grid_cut_off))
        return spec2d_list

    def invert(self, name, grid_offset, window_applied):    
        '''
        The grid offset is only provided for x and y (a 2d list/array is needed)
        '''
        t, x, y, eta_invers = fft_interface.spectral2physical(self.coeffs, [self.w, self.kx, self.ky])        
        x += grid_offset[0]
        y += grid_offset[1]
        return surface_core.Surface(name, eta_invers, [t, x, y], window_applied)
        
        

class SpectralAnalysis(object):
    '''
    Class for Analysis 1d, 2d and 3d spectra over a uniform grid.    
    '''
    def __init__(self, coeffs, spectrum, axes_grid, type='wavenumber', window_applied=False, grid_cut_off=None):
        '''
        Parameters:
        ----------
        input:
                coeffs          array
                                complex coefficients, if available, if not None
                spectrum        array
                                spectrum in 1d, 2d, 3d
                axis_grid       list of arrays defining grid for each given axis

                type            string
                                only meaningful for 1d spectrum... default: 'wavenumber', 
                                can be set to 'frequency'
        '''
        self.axes_grid = axes_grid
        self.grid_cut_off = grid_cut_off
        self.window_applied = window_applied
        if len(spectrum.shape)==1:
            self.ND = 1
            self.spectrumND = _SpectralAnalysis1d(coeffs, spectrum, axes_grid, type, grid_cut_off)
            self.kx = self.spectrumND.kx
        elif len(spectrum.shape)==2:
            self.ND = 2        
            self.spectrumND = _SpectralAnalysis2d(coeffs, spectrum, axes_grid, grid_cut_off)
            # TODO: distinguish different 2d spectra?
            self.kx = self.spectrumND.kx
            self.ky = self.spectrumND.ky
        elif len(spectrum.shape)==3:
            self.ND = 3        
            self.spectrumND = _SpectralAnalysis3d(coeffs, spectrum, axes_grid, grid_cut_off)
            self.w = self.spectrumND.w
            self.kx = self.spectrumND.kx
            self.ky = self.spectrumND.ky
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


    def spectrum(self):
        return self.spectrumND.spectrum

    def coeffs(self):
        return self.spectrumND.coeffs    
        
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
        
    def copy(self):
        if self.grid_cut_off==None:
            return SpectralAnalysis(self.spectrumND.coeffs.copy(), self.spectrumND.spectrum.copy(), self.axes_grid.copy(), self.window_applied, grid_cut_off=None)
        else:
            return SpectralAnalysis(self.spectrumND.coeffs.copy(), self.spectrumND.spectrum.copy(), self.axes_grid.copy(), self.window_applied, grid_cut_off=self.grid_cut_off.copy())
 
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
        grid_points = np.zeros(self.ND)
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
            
    def plot(self):#, fn, waterdepth=None, extent=None, U=None, dB=True, vmin=-60, save=False):
        '''
        TODO: recheck all arguments sensible to include
        '''
        #FIXME description!
        if self.ND==1:
            self.spectrumND.plot()#fn, extent, save)
        elif self.ND==2:
            self.spectrumND.plot()#fn, waterdepth, extent, U, dB, vmin, save)
        elif self.ND==3:
            self.spectrumND.plot()#fn, waterdepth, extent, U, dB, vmin, save)  
            
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

    def remove_zeroth(self):
        self.spectrumND.remove_zeroth()
        
    def get_2d_MTF(self, grid_offset):
        if self.ND==2:
            return self.spectrumND.get_2d_MTF(grid_offset)  
    def apply_MTF(self, grid_offset):
        self.spectrumND.apply_MTF(grid_offset)
        
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
        if grid_offset is None:
            return self.spectrumND.invert(name, np.zeros(self.ND), self.window_applied)        
        else:
            return self.spectrumND.invert(name, grid_offset, self.window_applied)        
        
        
        
        
        
            
          
