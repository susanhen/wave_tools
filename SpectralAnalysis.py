import numpy as np
from wave_tools import surface_core, fft_interface, dispersionRelation
from help_tools import plotting_interface, polar_coordinates, convolutional_filters, moving_average
from radar_tools import dispersion_filter, filter_core
import functools
import operator
import scipy
from scipy.ndimage import gaussian_filter

from scipy.signal import savgol_filter

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
    def __init__(self, coeffs, spectrum, kx, spec_type, grid_cut_off=None):
        '''
        1D specturm, coeffs may be None, typically used for integrated
        subspectra for 2D and 3D spectra
        '''
        self.coeffs = coeffs
        self.spectrum = spectrum
        if spec_type == 'wavenumber':
            self.type = 1
        elif spec_type == 'frequency':
            self.type = 0
        else:
            print('Error: given type is not valid use "wavenumber" or "frequency"!')       
        if type(kx) == list:
            self.kx = np.array(functools.reduce(operator.iconcat, kx, []))
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

    def get_peak_dir(self):
        print('\n\nError: peak dir not implemented for 1D\n')        
        return None
       
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
        
    def plot(self, extent=None, ax = None):
        '''
        Parameters:
        -----------
        extent      tupel of limits (x_lower, x_upper), y_lower, y_upper) #FIXME CHECK if None option is implemented        
        '''
        #FIXME: fix extent
        plotting_interface.plot_wavenumber_spec(self.kx, self.spectrum, scaled=True, ax=ax)

    def remove_zeroth(self):
        self.coeffs[self.Nx//2] = 0
        self.spectrum[self.Nx//2] = 0 

    def get_sub_spec2d(self, k_limit):
        '''
        Returns new spec3d object based on new  kx, ky and spectrum within the given limits
        '''
        kx_max_ind = np.argwhere(self.kx >= k_limit)[0][0] + 1
        kx_min_ind = self.Nx - kx_max_ind
        # create new vectors, the uppermost value is equal to the second entry
        kx_new = self.kx[kx_min_ind: kx_max_ind]
        grid = [kx_new]
        coeffs_new = self.coeffs[kx_min_ind:kx_max_ind]
        spectrum_new = self.spectrum[kx_min_ind:kx_max_ind]
        return coeffs_new, spectrum_new, grid 

    def get_MTF(self):
        kx_cut = np.where(np.abs(self.kx)<self.x_cut_off, self.kx, 0)
        MTF_inv = -1.0j*kx_cut
        MTF = np.where(np.abs(MTF_inv)>10**(-6), 1./MTF_inv, 1)
        return MTF
        
class _SpectralAnalysis2d(object):
    def __init__(self, coeffs, spectrum, k, grid_cut_off=None):
        self.coeffs = coeffs
        self.spectrum = spectrum
        self.kx = k[0]
        self.ky = k[1]
        self.grid_cut_off = grid_cut_off
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

    def get_peak_dir(self):
        k, theta, spec2d_pol = polar_coordinates.cart2pol(self.kx, self.ky, self.spectrum, Nr=200, Ntheta=360)
        spec1d = np.mean(spec2d_pol, axis=0)
        return theta[np.argmax(spec1d)]
        
    def plot(self, extent, ax=None):# dB=True, vmin=-60):
        '''
        Parameters:
        -----------
            input
                    extent      tupel of limits (x_lower, x_upper, y_lower, y_upper) #FIXME CHECK if None option is implemented        
        '''
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, self.spectrum, extent=extent, ax=ax)  #.plot_k_w_mod2D(self.kx, self.ky, self.spectrum, fn, waterdepth, r'$\mathrm{dB}$', extent=extent, U=U, dB=dB, vmin=vmin, fn=fn, save=save)        

    def get_sub_spectrum(self, kx_min, kx_max, ky_min, ky_max):
        kx_min_ind = np.argwhere(self.kx > kx_min)[0][0]
        kx_max_ind = np.argwhere(self.kx > kx_max)[0][0]
        ky_min_ind = np.argwhere(self.ky > ky_min)[0][0]
        ky_max_ind = np.argwhere(self.ky > ky_max)[0][0]
        kx_new = self.kx[kx_min_ind: kx_max_ind]
        ky_new = self.ky[ky_min_ind: ky_max_ind]
        return kx_new, ky_new, self.spectrum[kx_min_ind:kx_max_ind, ky_min_ind:ky_max_ind]
 
    def get_sub_spec2d(self, k_limit):
        '''
        Returns new spec3d object based on new  kx, ky and spectrum within the given limits
        '''
        kx_max_ind = np.argwhere(self.kx >= k_limit)[0][0] + 1
        kx_min_ind = self.Nx - kx_max_ind
        ky_max_ind = np.argwhere(self.ky >= k_limit)[0][0] + 1
        ky_min_ind = self.Ny - ky_max_ind
        # create new vectors, the uppermost value is equal to the second entry
        kx_new = self.kx[kx_min_ind: kx_max_ind]
        ky_new = self.ky[ky_min_ind: ky_max_ind]
        grid = [kx_new, ky_new]
        coeffs_new = self.coeffs[kx_min_ind:kx_max_ind, ky_min_ind:ky_max_ind]
        spectrum_new = self.spectrum[kx_min_ind:kx_max_ind, ky_min_ind:ky_max_ind]
        return coeffs_new, spectrum_new, grid      


    def plot_orig_disp_rel(self, w, z, Ux, Uy, h, ax=None, extent=None):
        plotting_interface.plot_disp_rel_at(w, h, z, Ux, Uy, 'w', ax, extent)  

    def plot_0current_disp_rel(self, w, h, ax=None, extent=None):
        plotting_interface.plot_disp_rel_at(w, h, 0, 0, 0, 'r', ax, extent)

    def plot_disp_rel_kx_ky(self, w, h=1000):
        #check if there is enough wave energy on the y-axis for this procedure:
        Ux0, Uy0 = 0, 0
        spec_max = np.max(self.spectrum)
        relevant_indices = np.argwhere((self.spectrum).flatten()>0.1*spec_max).T[0,:]
        kx_mesh, ky_mesh = np.meshgrid(self.kx, self.ky, indexing='ij')
        kx_rel = kx_mesh.flatten()[relevant_indices]
        ky_rel = ky_mesh.flatten()[relevant_indices]
        k_mesh = np.sqrt(kx_mesh**2 + ky_mesh**2)
        spec_filt0 = np.where(np.abs(np.sqrt(k_mesh*9.81*np.tanh(k_mesh*h))-w)<0.2, self.spectrum, 0)
        spec_filt = np.where(spec_filt0>0.3*np.max(spec_filt0), spec_filt0, 0)
        k, theta, spec_pol = polar_coordinates.cart2pol(self.kx, self.ky, spec_filt)
        kk, th = np.meshgrid(k, theta, indexing='ij')
        k_peak = np.sum(spec_pol * kk, axis=0)/np.sum(spec_pol, axis=0)
        k_peak2 = (np.sum(spec_pol * kk**2, axis=0)/np.sum(spec_pol, axis=0))**(1./2)
        k_peak3 = (np.sum(spec_pol * kk**3, axis=0)/np.sum(spec_pol, axis=0))**(1./3)


        window_length = 10
        selected_indices = np.argwhere(k_peak<1)[:,0]
        k_peak_selected = k_peak[selected_indices]
        k_peak_selected = np.block([k_peak_selected[0] * np.ones(window_length-1), k_peak_selected, k_peak_selected[-1] * np.ones(window_length-1)])
        k_peak_smooth = moving_average.moving_average(k_peak_selected, window_length)[window_length-1:-window_length+1]


        k_peak_selected2 = k_peak2[selected_indices]
        k_peak_selected2 = np.block([k_peak_selected2[0] * np.ones(window_length-1), k_peak_selected2, k_peak_selected2[-1] * np.ones(window_length-1)])
        k_peak_smooth2 = moving_average.moving_average(k_peak_selected2, window_length)[window_length-1:-window_length+1]


        k_peak_selected3= k_peak3[selected_indices]
        k_peak_selected3 = np.block([k_peak_selected3[0] * np.ones(window_length-1), k_peak_selected3, k_peak_selected3[-1] * np.ones(window_length-1)])
        k_peak_smooth3 = moving_average.moving_average(k_peak_selected3, window_length)[window_length-1:-window_length+1]
        
        weights = (self.spectrum.flatten()[relevant_indices])**(1./8)
        max_weight = np.max(weights)
        weights = np.where(weights>0.1*max_weight, weights, 0)

        def disp_rel(U):
            Ux, Uy = U
            #N = (len(x) - 2)//3
            #weights = x[:N]
            kx = kx_rel#x[N:2*N]
            ky = ky_rel#x[2*N:3*N]
            k = np.sqrt(kx**2 + ky**2)
            return np.dot(weights, (w - np.sqrt(9.81*k*np.tanh(k*h)) - Ux*kx - Uy*ky)**2)

        opt = scipy.optimize.minimize(disp_rel, [Ux0, Uy0])
        Ux, Uy = opt.x
  
        print('The current was defined to by Ux, Uy = ', Ux, Uy)
        import pylab as plt
        plt.plot(k_peak[selected_indices]*np.cos(theta[selected_indices]), k_peak[selected_indices]*np.sin(theta[selected_indices]))
        plt.plot(k_peak_smooth*np.cos(theta[selected_indices]), k_peak_smooth*np.sin(theta[selected_indices]))
        plt.plot(k_peak_smooth2*np.cos(theta[selected_indices]), k_peak_smooth2*np.sin(theta[selected_indices]))
        plt.plot(k_peak_smooth3*np.cos(theta[selected_indices]), k_peak_smooth3*np.sin(theta[selected_indices]))
        #plt.plot(k_peak2*np.cos(theta), k_peak2*np.sin(theta))
        #plt.plot(k_peak3*np.cos(theta), k_peak3*np.sin(theta))
        print('ha')

    def get_disp_filtered_spec(self, U, h, filter_width_up, filter_width_down, filter_width_axis, first_axis_k):
        if first_axis_k==True:
            disp_filt = dispersion_filter.k_w_filter(self.Nx, self.Ny, U, h, self.kx, self.ky, filter_width_up, filter_width_down, filter_width_axis, N_fine=2000)                    
            return SpectralAnalysis(disp_filt*self.coeffs.copy(), disp_filt*self.spectrum.copy(), [self.kx, self.ky])            
        else:
            print('The configuration where omega is the first axis has not yet been implemented. Do it!')
            return None
        
    def apply_HP_filter(self, limit):
        kx, ky = np.meshgrid(self.kx, self.ky, indexing='ij')
        k = np.sqrt(kx**2 + ky**2)
        k_hp_filter = (k>limit).astype('int')
        self.coeffs *= k_hp_filter
        self.spectrum *= k_hp_filter

    def apply_LP_filter(self, limit):
        kx, ky = np.meshgrid(self.kx, self.ky, indexing='ij')
        k = np.sqrt(kx**2 + ky**2)
        k_lp_filter = (k<limit).astype('int')
        self.coeffs *= k_lp_filter
        self.spectrum *= k_lp_filter        

    def get_1d_MTF(self, ky_only):
        kx_cut = np.where(np.abs(self.kx)<self.x_cut_off, self.kx, 0)
        ky_cut = np.where(np.abs(self.ky)<self.y_cut_off, self.ky, 0)
        kx_mesh = np.outer(kx_cut, np.ones(self.Ny))
        ky_mesh = np.outer( np.ones(self.Nx), ky_cut)
        k_mesh = np.sqrt(kx_mesh**2 + ky_mesh**2)
        if ky_only:
            MTF_inv = -1.0j*ky_mesh
        else:
            MTF_inv = np.where(ky_mesh>0, -1j*k_mesh, 1j*k_mesh)
        with np.errstate(divide='ignore', invalid='ignore'):
            MTF = np.where(np.abs(MTF_inv)>10**(-6), 1./MTF_inv, 1)
        return MTF

        
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
        with np.errstate(divide='ignore', invalid='ignore'):
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

    def apply_1d_MTF(self, percentage_of_max=0.01, use_1D_MTF=False):

        MTF = self.get_1d_MTF(ky_only=False)

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

    def transform_spectrum_to_polar(self, Nr, Ntheta):        
        return polar_coordinates.cart2pol(self.kx, self.ky, self.spectrum, Nr=Nr, Ntheta=Ntheta)

    def integrate_theta(self, Nr, Ntheta):
        k, theta, spectrum_pol = self.transform_spectrum_to_polar(Nr, Ntheta)
        dtheta = theta[1] - theta[0]
        integrated_spec = dtheta * np.sum(spectrum_pol, axis=1)
        return k, integrated_spec

    def get_theta_slice(self, at_theta, Nr, Ntheta):
        k, theta, spectrum_pol = self.transform_spectrum_to_polar(Nr, Ntheta)
        at_theta_ind = np.argmin(np.abs(theta- at_theta))
        return k, spectrum_pol[:,at_theta_ind]

        
class _SpectralAnalysis3d(object):
    def __init__(self, coeffs, spectrum, grid, grid_cut_off=None):
        self.coeffs = coeffs
        self.spectrum = spectrum
        self.w = grid[0]
        self.kx = grid[1]
        self.ky = grid[2]
        self.dw = np.abs(self.w[1] - self.w[0])
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

    def get_peak_dir(self):
        spec2d = np.sum(self.spectrum[:self.Nt//2,:,:], axis=0)
        k, theta, spec2d_pol = polar_coordinates.cart2pol(self.kx, self.ky, spec2d, Nr=200, Ntheta=400)
        spec1d = np.mean(spec2d_pol, axis=0)
        return theta[np.argmax(spec1d)]

    def plot(self, extent, ax=None, dB=None, vmin=None, save=False):
        '''
        plots the integrated kx-ky-spectrum and the integrated k-omega-spectrum
        '''
        #FIXME implement: options of 3d something and slices along different axis, single slices or a bunch, use subspectra in 2D!
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, np.sum(self.spectrum[self.Nt//2:], axis=0)*self.dw, extent=extent, ax=ax)

    def remove_zeroth(self):
        self.coeffs[self.Nt//2, self.Nx//2, self.Ny//2] = 0
        self.spectrum[self.Nt//2, self.Nx//2, self.Ny//2] = 0  

    def get_w_slice(self, window_applied, grid_cut_off, w_limit, k_limit, N_average):
        '''
        spectral objects for all negative frequencies
        '''
        grid, coeffs_new, spec_new = self.get_sub_spec3d(w_limit, k_limit)
        w, kx, ky = grid
        Nw = len(w)
        w_upper = w[Nw//2 + N_average//2 : -N_average//2 : N_average]
        spec2d_list = []
        for i in np.arange(Nw//2 - N_average//2, N_average//2, -N_average):
            coeffs_mean = np.zeros((len(kx), len(ky)), dtype=complex)
            spec_mean = np.zeros((len(kx), len(ky)))
            for j in range(-(N_average//2), N_average//2+1):       
                coeffs_mean += coeffs_new[i+j,:,:].copy()
                spec_mean += spec_new[i+j,:,:].copy()
            spec2d_i = SpectralAnalysis(coeffs_mean/N_average, spec_mean/N_average, [kx, ky], window_applied=window_applied, grid_cut_off=grid_cut_off)
            spec2d_list.append(spec2d_i)
        return [w_upper, kx, ky], spec2d_list        

    def integrate_theta(self, radiusSize):
        w_k_spec = np.zeros((self.Nt, radiusSize))
        k = None
        for i in range(0, self.Nt):
            spec2d = SpectralAnalysis(self.coeffs[i,:,:], self.spectrum[i,:,:], [self.kx, self.ky])
            k, integrated_spec = spec2d.integrate_theta()
            w_k_spec[i,:] = integrated_spec#*k
        k_w_spec = w_k_spec.T
        interpol = scipy.interpolate.interp1d(k, k_w_spec, axis=0, kind='cubic')
        k_fine = np.linspace(0, k[-1], 300)[1:]

        def func(x, a, b, c):
            return a * np.exp(-b * x) + c        
        
        k_max_inds = np.argmax(interpol(k_fine), axis=0)
        measured_k1 = k_fine[k_max_inds]
        max1 = np.max(interpol(k_fine))
        choose1 = np.argwhere(np.max(interpol(k_fine), axis=0)>0.01*max1)
        k_max_inds2 = np.argmax(k_w_spec, axis=0)
        measured_k2 = k[k_max_inds2]
        max2 = np.max(k_w_spec)
        choose2 = np.argwhere(np.max(interpol(k_fine), axis=0)>0.01*max2)

        #popt, pcov = scipy.optimize.curve_fit(func, measured_k1[choose1], self.w[choose1])

        import pylab as plt        
        plotting_interface.plot_3d_as_2d(k_fine, self.w, interpol(k_fine))
        plt.plot(measured_k1[choose1], self.w[choose1])
        
        plt.plot(k_fine, np.sqrt(9.81*k_fine))
        plt.plot(k_fine, -np.sqrt(9.81*k_fine))
        #plt.plot(measured_k1[choose1], func(measured_k1[choose1], *popt))
        plotting_interface.show()
        plt.figure()
        plotting_interface.plot_3d_as_2d(k, self.w, k_w_spec)
        plt.plot(k, np.sqrt(9.81*k)+self.dw)
        plt.plot(measured_k1[choose1], self.w[choose1], '--')
        plt.plot(measured_k2[choose2], self.w[choose2], ':')
        plotting_interface.show()

    
    def get_sub_spectrum(self, w_min, w_max, kx_min, kx_max, ky_min, ky_max):
        '''
        Returns new w, kx, ky and spectrum within the given limits
        '''
        w_min_ind = np.argwhere(self.w > w_min)[0][0]
        w_max_ind = np.argwhere(self.w > w_max)[0][0]
        kx_min_ind = np.argwhere(self.kx > kx_min)[0][0]
        kx_max_ind = np.argwhere(self.kx > kx_max)[0][0]
        ky_min_ind = np.argwhere(self.ky > ky_min)[0][0]
        ky_max_ind = np.argwhere(self.ky > ky_max)[0][0]
        w_new = self.w[w_min_ind:w_max_ind]
        kx_new = self.kx[kx_min_ind: kx_max_ind]
        ky_new = self.ky[ky_min_ind: ky_max_ind]
        return w_new, kx_new, ky_new, self.spectrum[w_min_ind:w_max_ind, kx_min_ind:kx_max_ind, ky_min_ind:ky_max_ind]

 
    def get_sub_spec3d(self, w_limit, k_limit):
        '''
        Returns new spec3d object based on new w, kx, ky and spectrum within the given limits
        '''
        if w_limit is None:
            w_limit = self.w[-1]
        w_max_ind = np.argwhere(self.w >= w_limit)[0][0] + 1
        w_min_ind = self.Nt - w_max_ind
        kx_max_ind = np.argwhere(self.kx >= k_limit)[0][0] + 1
        kx_min_ind = self.Nx - kx_max_ind
        ky_max_ind = np.argwhere(self.ky >= k_limit)[0][0] + 1
        ky_min_ind = self.Ny - ky_max_ind
        # create new vectors, the uppermost value is equal to the second entry
        w_new = self.w[w_min_ind:w_max_ind]
        kx_new = self.kx[kx_min_ind: kx_max_ind]
        ky_new = self.ky[ky_min_ind: ky_max_ind]
        grid = [w_new, kx_new, ky_new]
        coeffs_new = self.coeffs[w_min_ind:w_max_ind, kx_min_ind:kx_max_ind, ky_min_ind:ky_max_ind]
        spectrum_new = self.spectrum[w_min_ind:w_max_ind, kx_min_ind:kx_max_ind, ky_min_ind:ky_max_ind]
        return grid, coeffs_new, spectrum_new

    def get_anti_aliased_spec3d(self, k_limit):
        
        k_limit = 0.4
        #grid, spectrum_new = self.get_sub_spec3d(self.w[-1], k_limit)
        grid = [self.w, self.kx, self.ky]
        spectrum_new = self.spectrum
        w, kx, ky = grid
        Nt = 2*self.Nt-2
        Nx = len(kx)
        Ny = len(ky)        
        theta_mean = self.get_peak_dir()
        dtheta_max = 70*np.pi/180
        coeffs_extended = np.zeros((Nt, Nx, Ny), dtype=complex)
        spectrum_extended = np.zeros((Nt, Nx, Ny))
        w_extended = self.dw * np.arange(-Nt//2, Nt//2) 
        interval = self.Nt//2
        w_mesh, kx_mesh, ky_mesh = np.meshgrid(w, kx, ky, indexing='ij')
        kk = np.sqrt(kx_mesh**2 + ky_mesh**2)
        th = np.arctan2(ky_mesh, kx_mesh)
        mask_lower = np.where(np.abs(th - (theta_mean - np.pi))<= dtheta_max, 1, 0)
        mask_upper = np.where(np.abs(th - theta_mean)<= dtheta_max, 1, 0)
        
        # bin ich eigentlich in der richtigen Richtung? Habe ich das richtig simuliert? hmmm muss angepasst werden, aber wo?
        coeffs_extended[Nt//2:Nt//2+interval,:,:] = np.where(mask_lower[interval:,:,:], self.coeffs[interval:,:,:], 0).copy()
        spectrum_extended[Nt//2:Nt//2+interval,:,:] = np.where(mask_lower[interval:,:,:], self.spectrum[interval:,:,:], 0).copy()
        upper_coeffs = np.where(mask_lower[2:interval,:-1,:-1], self.coeffs[2:interval,:-1,:-1], 0)
        coeffs_extended[Nt//2+interval:-1,1:,1:] = upper_coeffs.copy()
        upper_spec = np.where(mask_lower[:interval-1,:-1,:-1], self.spectrum[:interval-1,:-1,:-1], 0)
        spectrum_extended[Nt//2+interval:,1:,1:] = upper_spec.copy()

        # mirror:
        coeffs_extended[1:Nt//2,1:,1:] = np.conjugate(np.flip(coeffs_extended[Nt//2+1:,1:,1:])).copy()
        spectrum_extended[1:Nt//2,1:,1:] = np.flip(spectrum_extended[Nt//2+1:,1:,1:]).copy()
        grid = [w_extended, kx, ky]
        return coeffs_extended, spectrum_extended, grid
        

    def get_k_w_slice_at_peak(self, Nr, Ntheta, U_eff=None, Psi=None):
        peak_indices = np.unravel_index(np.argmax(self.spectrum), (self.Nt, self.Nx, self.Ny))
        at_theta = np.arctan2(peak_indices[2], peak_indices[1])
        # test ouput... remove large k values
        spec2d = SpectralAnalysis(self.coeffs[0,:,:], self.spectrum[0,:,:], [self.kx, self.ky])
        k, test = spec2d.spectrumND.get_theta_slice(0, Nr, Ntheta)
        w_k_spec = np.zeros((self.Nt, len(test)))
        k = None
        peak_value = 0
        for i in range(0, self.Nt):
            spec2d = SpectralAnalysis(self.coeffs[i,:,:], self.spectrum[i,:,:], [self.kx, self.ky])
            k, w_k_spec[i,:] = spec2d.spectrumND.get_theta_slice(at_theta, Nr, Ntheta)
        k_w_spec = w_k_spec.T
        if U_eff is None:
            Ueff = 0
            Psi = 0
        plotting_interface.plot_3d_as_2d(k, self.w, k_w_spec)
        plotting_interface.plot(k, np.sqrt(9.81*k)+k*Ueff*np.cos(Psi))
        plotting_interface.plot(k, -np.sqrt(9.81*k)+k*Ueff*np.cos(Psi))
        #plt.plot(measured_k1[choose1], self.w[choose1], '--')
        #plt.plot(measured_k2[choose2], self.w[choose2], ':')
        plotting_interface.show()
        return k, self.w, k_w_spec


    def transform_spectrum_to_polar(self, Nr, Ntheta):        
        return polar_coordinates.cart2cylindrical(self.kx, self.ky, self.spectrum, Nr=Nr, Ntheta=Ntheta)


    def estimate_dispersion_cone(self, h, Umax, kmin=0.04, kmax=0.5, plot_it=False):
        '''
        Estimate dispersion cone. It returns a representation of the dispersion cone where the energy is high enough

        Algorithm: 
        - Pass to polar coordinates 
        - apply dispersion filter to avoid using aliased data etc. 
        - select part of arc where there is energy (>0.1max)
        - define relevant k based on upper and lower bounds of k on the grid (close to kmin, kmax provided)
        - select part of spectrum representing relevant theta and k values 
        - loop through theta:
            - apply gaussian blur
            - find peak of w for for all k of interest along the given angle
        
        

        Parameters:
        -----------
                    input       
                            h       float
                                    waterdepth
                            Umax    float
                                    maximum expected current 
                            kmin    float
                                    lower boundary for wave numbers relevant for defining cone, default 0.04
                            kmax    float
                                    upper boundary for wave numbers relevant for defining cone, default 0.3
                            plot_it bool
                                    if data and cone slice should be shown

                    output:
                            k_rel       array
                                        vector of wave numbers for which the dispersion cone is defined
                            theta_rel  array
                                        vector of azimuth angles for which the dispersion cone is defined
                            w compo. of disp_cone   2d array
                                        angular frequency for each combinatin of wave numbers and angle 
        '''
        w_upper = self.w[self.Nt//2:]
        half_spec = np.flip(self.spectrum[1:self.Nt//2+1,:,:], axis=0)
        filtered_half_spec = gaussian_filter(half_spec, (0.5,0.3,0.3))
        '''
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (half_spec[229,:,:]))
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (filtered_half_spec[229,:,:]))
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (half_spec[200,:,:]))
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (filtered_half_spec[200,:,:]))
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (half_spec[250,:,:]))
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (filtered_half_spec[250,:,:]))
        plotting_interface.show()
        '''

        #k, theta, spec_pol = polar_coordinates.cart2cylindrical(w_upper, self.kx, self.ky, half_spec, Ntheta=100)     
        k, theta, spec_pol = polar_coordinates.cart2cylindrical(w_upper, self.kx, self.ky, filtered_half_spec, Ntheta=100)            
        spec_pol = np.abs(spec_pol)
        
        # dispersion filter
        mask = dispersion_filter.w_k_theta_filter(w_upper, k, theta, Umax, h, w_min=0.6)
        masked_spec_pol = mask*spec_pol
        #masked_spec_pol = spec_pol
        
        # select arc with energy          
        theta_spec = np.sum(np.sum(masked_spec_pol, axis=0), axis=0)        
        rel_indices = np.argwhere(theta_spec>0.1*np.max(theta_spec)).transpose()[0]        
        theta_rel = theta[rel_indices]
        # k_relevant
        k_min_ind = np.argmin(np.abs(k - kmin))
        k_max_ind = np.argmin(np.abs(k - kmax))
        k_rel = k[k_min_ind:k_max_ind+1]
        # empty disp cone
        w_matrix = np.zeros((len(k_rel), len(theta_rel)))
        rel_spec = masked_spec_pol[:,k_min_ind:k_max_ind+1,rel_indices]
        min_dw = 4* np.sqrt(self.dkx*9.81)
        ww, kk = np.meshgrid(w_upper, k_rel, indexing='ij')

        
        for i in range(0, len(theta_rel)):
            input_spec = rel_spec[:,:,i]#convolutional_filters.apply_Gaussian_blur(rel_spec[:,:,i])
            max_input_spec = np.outer(np.ones(len(w_upper)), np.max(input_spec, axis=0) )
            # filtering relative to the power along w-axis otherwise spectral noise in the center moves the cone to the center
            input_spec = np.where(input_spec>0.1*max_input_spec, input_spec, 0)
            pow = 0.3
            pow2 = 1
            w_peaks = (np.sum(input_spec**pow * ww**pow2, axis=0)/np.sum(input_spec**pow, axis=0))**(1./pow2)
            w_matrix[:,i] = savgol_filter(w_peaks, 11, 3)#w_peaks
            
            if plot_it:
                scaled_2d = spec_pol[:,k_min_ind:k_max_ind,i]/np.max(spec_pol[:,k_min_ind:k_max_ind,i])
                #plotting_interface.plot_k_w_spec(k_rel, w_upper, np.log10(scaled_2d).T, extent=[0.03, 0.35, 0.5, 1.9])
                plotting_interface.plot_k_w_spec(k_rel, w_upper, (input_spec).T, extent=[0.03, 0.35, 0.5, 1.9])
                plotting_interface.plot(k_rel, w_peaks)
                plotting_interface.plot(k_rel, savgol_filter(w_peaks, 11, 3))
                #plotting_interface.plot(k_rel, np.sqrt(k_rel*9.81*np.tanh(k_rel*h)))
                plotting_interface.show()
        '''
        ax = plotting_interface.plot_k_w_spec(k_rel, theta_rel, (rel_spec[229,:,:]))
        CS = ax.contour(k_rel, theta_rel, w_matrix.T, origin='lower', levels=[1.18, 1.199, 1.21], colors='green')
        w_orig_ind = np.argmin(np.abs(w_upper-1.2))
        out = np.array(CS.allsegs[0][0])
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (half_spec[w_orig_ind,:,:]))
        x_points = out[:,0]
        y_points = out[:,1]
        plotting_interface.plot(x_points*np.cos(y_points), x_points*np.sin(y_points))
        '''
        return k_rel, theta_rel, w_matrix

    def estimate_dispersion_slices(self, h, Umax, kmin=0.04, kmax=0.3, wmin=0.8, wmax=1.8, plot_it=False):
        '''
        Esimate dispersion cone and return it as polar slices in w

        Parameters:
        -----------
                    input       
                            h       float
                                    waterdepth
                            Umax    float
                                    maximum expected current 
                            kmin    float
                                    lower boundary for wave numbers relevant for defining cone, default 0.04
                            kmax    float
                                    upper boundary for wave numbers relevant for defining cone, default 0.3
                            wmin    float
                                    lower boundary for angular frequency relevant for defining cone, default 0.8
                            wmax    float
                                    upper boundary for angular frequency relevant for defining cone, default 1.8
                            plot_it bool
                                    if data and cone slice should be shown

                    output:
                            w_rel   array
                                    vector of wave numbers for which the dispersion cone is defined
                            theta_rel  array
                                    vector of azimuth angles for which the dispersino cone is defined                            
                            k_matrix   2d array
                                        w(k, theta)
                    
        '''
        k_rel, theta_rel, w_matrix = self.estimate_dispersion_cone(h, Umax, kmin=kmin, kmax=kmax, plot_it=plot_it)
        # define max w and min w to know contour lines
        if wmin<np.min(w_matrix):
            wmin = np.min(w_matrix)
        if wmax>np.max(w_matrix):
            wmax = np.max(w_matrix)        
        min_ind = np.argwhere(wmin < self.w)[0][0]
        max_ind = np.argwhere(wmax > self.w)[-1][0]
        w_rel = self.w[min_ind:max_ind]
        k_matrix = np.zeros((len(w_rel), len(theta_rel)))
        for i in range(0, len(theta_rel)):
            # invert disp rel 
            F = scipy.interpolate.interp1d(w_matrix[:,i], k_rel, kind='cubic', bounds_error=False)
            k_matrix[:,i] = F(w_rel)
        return w_rel, theta_rel, k_matrix

    def estimate_Ueff_psi(self, h, Umax, kmin=0.04, kmax=0.3, wmin=0.8, wmax=1.8, ax=None, U0_vec=[0,0], k_limit=None, plot_it=False):
        if k_limit is None:
            k_limit = self.kx[-1]

        w_rel, theta_rel, k_matrix = self.estimate_dispersion_slices(h, Umax, kmin, kmax, wmin, wmax, plot_it=plot_it)

        # collectors for estimates
        U_effs = np.zeros(len(w_rel))
        psis = np.zeros(len(w_rel))
        i=0

        for at_w in w_rel:
            if at_w>np.max(self.w[-1]):
                print('\n\nchosen w is too large, will be set to max(w)!!!\n\n')
                at_w = np.max(self.w)
            U_effs[i], psis[i] = dispersionRelation.estimate_U_eff_psi_at_w(at_w, k_matrix[i,:], theta_rel, h, U0_vec=U0_vec)

            if ax is not None:
                w_ind = np.argmin(np.abs(at_w-self.w))
                k_for_Ueff= dispersionRelation.calc_wavenumber(w_rel, h, U_effs, psis, 360)
                plotting_interface.plot_kx_ky_spec(self.kx, self.ky, self.spectrum[w_ind,:,:], ax=ax, extent=[-k_limit, k_limit, -k_limit, k_limit])
                plotting_interface.plot_disp_rel_for_Ueff_at(at_w, h, k_for_Ueff, U_effs[i], psis[i], 'w', ax=ax, extent=[-k_limit, k_limit, -k_limit, k_limit])

            i = i + 1
        return w_rel, U_effs, psis
       

    def estimate_Ueff_psi_at_w_old(self, w_list, h, Umax, N_average, ax, N_theta=360, k_limit=None, max_filt_factor=0.1):
        if k_limit is None:
            k_limit = self.kx[-1]
        # provide list of 2d spectra
        grid, sped2d_list_extended = self.get_w_slice(False, None, None, k_limit, N_average)
        w_upper_extended, kx, ky = grid

        # collectors for estimates
        U_effs = np.zeros(len(w_list))
        psis = np.zeros(len(w_list))
        i=0
       
        # choose slice for given at_w
        for at_w in w_list:
            if at_w>np.max(w_upper_extended):
                print('\n\nchosen w is too large, will be set to max(w)!!!\n\n')
                at_w = np.max(self.w)
            chosen_index = np.argmin(np.abs(w_upper_extended-at_w))    
            at_w = w_upper_extended[chosen_index]
            spec2d = sped2d_list_extended[chosen_index]
            kx_ky_spec = spec2d.spectrum()
            
            dst = convolutional_filters.apply_Gaussian_blur(kx_ky_spec)

            kx_mesh, ky_mesh = np.meshgrid(kx, ky, indexing='ij')
            k_mesh = np.sqrt(kx_mesh**2 + ky_mesh**2)
            dw_max = k_mesh*Umax
            spec_filt0 = np.where(np.abs(np.sqrt(k_mesh*9.81*np.tanh(k_mesh*h))-at_w)<dw_max, dst, 0)  
            spec_filt = np.where(spec_filt0>max_filt_factor*np.max(spec_filt0), spec_filt0, 0)

            U_effs[i], psis[i] = dispersionRelation.estimate_U_eff_psi_directly(at_w, self.kx, self.ky, spec_filt, h, N_theta)

            if ax is not None:
                spec2d.plot(ax=ax)
                k_for_Ueff= dispersionRelation.calc_wavenumber(at_w, h, U_effs, psis, 360)
                plotting_interface.plot_disp_rel_for_Ueff_at(at_w, h, k_for_Ueff, U_effs[i], psis[i], 'w', ax=ax, extent=[-k_limit, k_limit, -k_limit, k_limit])

            i = i + 1
        return U_effs, psis
                

    def get_1d_MTF(self, ky_only):
        kx_cut = np.where(np.abs(self.kx)<self.x_cut_off, self.kx, 0)
        ky_cut = np.where(np.abs(self.ky)<self.y_cut_off, self.ky, 0)
        kx_mesh = np.outer(kx_cut, np.ones(self.Ny))
        ky_mesh = np.ones((self.Nt, self.Nx), ky_cut)
        k_mesh = np.sqrt(kx_mesh**2 + ky_mesh**2)
        if ky_only:
            MTF_inv = -1.0j*ky_mesh
        else:
            MTF_inv = -1.0j*np.where(ky_mesh>0, k_mesh, -k_mesh)
        with np.errstate(divide='ignore', invalid='ignore'):
            MTF2d = np.where(np.abs(MTF_inv)>10**(-6), 1./MTF_inv, 1)
        MTF = np.outer(np.ones(self.Nt), MTF2d).reshape((self.Nt, self.Nx, self.Ny))
        return MTF
    
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
        with np.errstate(divide='ignore', invalid='ignore'):
            MTF2d = np.where(np.abs(MTF_inv)>10**(-6), 1./MTF_inv, 1)
        MTF = np.outer(np.ones(self.Nt), MTF2d).reshape((self.Nt, self.Nx, self.Ny))
        return MTF  

    def apply_MTF(self, grid_offset, percentage_of_max=0.01):
        MTF = self.get_2d_MTF(grid_offset)
        threshold = percentage_of_max * np.max(np.abs(self.coeffs))
        self.coeffs = np.where(np.abs(self.coeffs)>threshold, self.coeffs* MTF, 0)
        self.spectrum = np.abs(self.coeffs)**2

    def apply_1d_MTF(self, percentage_of_max=0.01, ky_only=False):
        MTF = self.get_1d_MTF(ky_only)
        threshold = percentage_of_max * np.max(np.abs(self.coeffs))
        self.coeffs = np.where(np.abs(self.coeffs)>threshold, self.coeffs* MTF, 0)
        self.spectrum = np.abs(self.coeffs)**2 

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
    def __init__(self, coeffs, spectrum, axes_grid, spec_type='wavenumber', window_applied=False, grid_cut_off=None):
        '''
        Parameters:
        ----------
        input:
                coeffs          array
                                complex coefficients, if available, if not None
                spectrum        array
                                spectrum in 1d, 2d, 3d
                axis_grid       list of arrays defining grid for each given axis

                spec_type       string
                                only meaningful for 1d spectrum... default: 'wavenumber', 
                                can be set to 'frequency'
        '''
        self.axes_grid = axes_grid
        self.grid_cut_off = grid_cut_off
        self.window_applied = window_applied
        if len(spectrum.shape)==1:
            self.ND = 1
            self.spectrumND = _SpectralAnalysis1d(coeffs, spectrum, axes_grid, spec_type, grid_cut_off)
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
            print('\n\nError: Input data spectrum is not of the correct spec_type\n\n')
        
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
 

    def get_sub_spectrum(self, kx_min, kx_max, ky_min=None, ky_max=None, w_min=None, w_max=None):
        '''
        return a subset of the axes and the spectrum
        '''
        if self.ND==1:
            return self.spectrumND.get_sub_spectrum(kx_min, kx_max)
        elif self.ND==2:
            return self.spectrumND.get_sub_spectrum(kx_min, kx_max, ky_min, ky_max)
        elif self.ND==3:
            return self.spectrumND.get_sub_spectrum(w_min, w_max, kx_min, kx_max, ky_min, ky_max)

    def get_sub_spec(self, k_limit, w_limit=None):
        if self.ND==1:
            coeffs, spectrum, grid = self.spectrumND.get_sub_spec1d(k_limit)
        if self.ND==2:
            coeffs, spectrum, grid = self.spectrumND.get_sub_spec2d(k_limit)
        if self.ND==3:
            if w_limit in None:
                w_limit = self.w[-1]
            coeffs, spectrum, grid = self.spectrumND.get_sub_spec3d(w_limit, k_limit)
        return SpectralAnalysis(coeffs, spectrum, grid, self.grid_cut_off)

    def get_peak_dir(self):
        '''
        The peak direcion for averaged energy in direction
        '''
        return self.spectrumND.get_peak_dir()

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
            
    def plot(self, extent=None, ax=None):#, fn, waterdepth=None, extent=None, U=None, dB=True, vmin=-60, save=False):
        '''
        TODO: recheck all arguments for possible inclusion
        '''
        #FIXME description!
        if self.ND==1:
            self.spectrumND.plot(extent, ax)
        elif self.ND==2:
            self.spectrumND.plot(extent, ax)
        elif self.ND==3:
            self.spectrumND.plot(extent, ax)

    def plot_orig_disp_rel(self, w, z, Ux, Uy, h, extent=None):
        self.spectrumND.plot_orig_disp_rel(w, z, Ux, Uy, h, extent)

    def plot_0current_disp_rel(self, w, h, extent=None):
        self.spectrumND.plot_0current_disp_rel(w, h, extent)

    def plot_disp_rel_kx_ky(self, w, h):
        if self.ND==2:
            self.spectrumND.plot_disp_rel_kx_ky(w, h=h)
            
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

    def apply_LP_filter(self, limits):
        self.spectrumND.apply_LP_filter(limits)  

    def remove_zeroth(self):
        self.spectrumND.remove_zeroth()
        
    def get_2d_MTF(self, grid_offset):
        if self.ND==2 or self.ND==3:
            return self.spectrumND.get_2d_MTF(grid_offset) 

    def get_1d_MTF(self, ky_only=False):
        if self.ND==2 or self.ND==3:
            return self.spectrumND.get_1d_MTF(ky_only)

    def apply_MTF(self, grid_offset):
        self.spectrumND.apply_MTF(grid_offset)

    def apply_1d_MTF(self, ky_only=False):
        self.spectrumND.apply_1d_MTF(ky_only)

    def get_w_slice(self, w_limit=None, k_limit=None, N_average=1):
        if self.ND == 3:
            if w_limit is None:
                w_limit = self.w[-1]
            if k_limit is None:
                k_limit = np.min(self.kx[-1], self.ky[-1])
            return self.spectrumND.get_w_slice(self.window_applied, self.grid_cut_off, w_limit, k_limit, N_average)
        else:
            print('Error: get_w_slice is not implemented for dimensions lower than 3D')

    def get_theta_slice(self, theta_ind):
        radiusSize = int(np.sqrt(self.spectrumND.Nx**2 + self.spectrumND.Ny**2))
        if self.ND ==2:
            return self.spectrumND.get_theta_slice(theta_ind, radiusSize)    
        if self.ND ==3:
            print('Error: Not yet implemented!')
        if self.ND ==1:
            print('Error: The method is supported for 1D')

    def integrate_theta(self):
        radiusSize = int(np.sqrt(self.spectrumND.Nx**2 + self.spectrumND.Ny**2))
        if self.ND ==2:
            return self.spectrumND.integrate_theta(radiusSize)
        if self.ND ==3:
            return self.spectrumND.integrate_theta(radiusSize)
        else:
            print('Error: integrate_theta is not implemented for dimensions lower than 2D')

    def transform_spectrum_to_polar(self):
        Nr = int(np.sqrt(self.spectrumND.Nx**2 + self.spectrumND.Ny**2))
        Ntheta = 360
        if self.ND ==2:
            return self.spectrumND.transform_spectrum_to_polar(Nr, Ntheta)
        elif self.ND ==3:
            return self.spectrumND.transform_spectrum_to_polar(Nr, Ntheta)
        else:
            print('Error: get_transform_to_polar not implemented for dimensions lower than 2D')

    def get_anti_aliased_spec3d(self, k_limit):
        if self.ND==3:
            coeffs_extended, spectrum_extended, grid = self.spectrumND.get_anti_aliased_spec3d(k_limit)
            return SpectralAnalysis(coeffs_extended, spectrum_extended, grid, self.grid_cut_off)
        else:
            print('Error: Not implemented')

    def estimate_Ueff_psi_at_w_old(self, w_list, h, Umax, N_average=1, ax=None, k_limit=None, max_filt_factor=0.1):
        '''
        Estimate the current for the given spectrum.
        The estimation is done separately for all provided w in w_list
        Parameters:
        -----------
            input
                    w_list      array
                                all w for choosing corresponding w-slice
                    h           float
                                waterdepth
                    Umax        float
                                expected limit for max current
                    N_average   int
                                number of neighboring w-slices for the estimation, default=1 (central slice)
                    ax          matplotlib axes     
                                for plotting
                    k_limit     float, optional
                                for limiting the kx,ky-extent
                                
            return
                    Ueff        array
                                effective currents for each w
                    psi         array
                                current direction in rad for each Ueff
        '''
        if self.ND==3:
            return self.spectrumND.estimate_Ueff_psi_at_w_old(w_list, h, Umax, N_average, ax, k_limit=k_limit, max_filt_factor=max_filt_factor)
        else:
            print('Error: Not available, the current estimation requires a 3d surface')
        
    
    def estimate_Ueff_psi(self, h, Umax, kmin=0.04, kmax=0.3, ax=None, k_limit=None, plot_it=False):
        '''
        Estimate the current for the given spectrum.
        The estimation is performed separately for each angular frequency
        Parameters:
        -----------
            input
                    h           float
                                waterdepth
                    Umax        float
                                expected limit for max current
                    N_average   int
                                number of neighboring w-slices for the estimation, default=1 (central slice)
                    ax          matplotlib axes     
                                for plotting
                    k_limit     float, optional
                                for limiting the kx,ky-extent
                    plot_it     bool
                                switch for plotting validation of defining line segments
                                
            return
                    w_rel       array
                                levels of angular frequency for which Ueff and psi have been estimated
                    theta_rel   array
                                angles to define the relation between w_rel and k_matrix
                    k_matrix    array
                                k_values for each w_rel, theta
                    Ueff        array
                                effective currents for each w
                    psi         array
                                current direction in rad for each Ueff
        '''
        if self.ND==3:
            return self.spectrumND.estimate_Ueff_psi(h, Umax, kmin=kmin, kmax=kmax, ax=ax, k_limit=k_limit, plot_it=plot_it)
        else:
            print('Error: Not available, the current estimation requires a 3d surface')    

    def get_k_w_slice_at_peak(self, Ntheta=360, U_eff=None, Psi=None):
        Nr = np.max([self.spectrumND.Nx, self.spectrumND.Ny])        
        if self.ND == 3:
            return self.spectrumND.get_k_w_slice_at_peak(Nr, Ntheta)
        else:
            print('Error: get_k_w_slice is not implemented for dimensions lower than 2D')


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
        
        
        
        
        
            
          
