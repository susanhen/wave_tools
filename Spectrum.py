import numpy as np
from help_tools import plotting_interface, polar_coordinates
from wave_tools import surface_core, fft_interface, dispersionRelation
from radar_tools import dispersion_filter
from scipy.ndimage import gaussian_filter
import scipy

from scipy.signal import savgol_filter

class _Spectrum3d:
    def __init__(self, spectrum, grid):
        self.spectrum = spectrum
        self.w = grid[0]
        self.kx = grid[1]
        self.ky = grid[2]
        self.dw = np.abs(self.w[1] - self.w[0])
        self.dkx = self.kx[1] - self.kx[0]
        self.dky = self.ky[1] - self.ky[0]
        self.Nt, self.Nx, self.Ny = self.spectrum.shape
        #self.SAw = _SpectralAnalysis1d(None, np.sum(np.sum(spectrum,axis=2), axis=1), self.w, 'frequency') 
        #self.SAx = _SpectralAnalysis1d(None, np.sum(np.sum(spectrum,axis=2), axis=0), self.kx, 'wavenumber')
        #self.SAy = _SpectralAnalysis1d(None, np.sum(np.sum(spectrum,axis=1), axis=0), self.ky, 'wavenumber')
        
    def get_S(self):
        '''
        Return the 2d spectrum with the corresponding grid
        '''
        return self.w, self.kx, self.ky, self.spectrum.copy()            

    def get_peak_dir(self):
        spec2d = np.sum(self.spectrum[:self.Nt//2,:,:], axis=0)
        k, theta, spec2d_pol = polar_coordinates.cart2pol(self.kx, self.ky, spec2d, Nr=200, Ntheta=400)
        spec1d = np.mean(spec2d_pol, axis=0)
        return theta[np.argmax(spec1d)]

    def plot_kx_ky_integrated_w(self, extent, ax=None, dB=None, vmin=None, save=False):
        '''
        plots the integrated kx-ky-spectrum and the integrated k-omega-spectrum
        '''
        return plotting_interface.plot_kx_ky_spec(self.kx, self.ky, np.sum(self.spectrum[:self.Nt//2,:,:], axis=0)*self.dw, extent=extent, ax=ax)

    def plot_point_cloud(self):
        o3d.visualization.draw_geometries(self.spectrum)

    def remove_zeroth(self):
        self.spectrum[self.Nt//2, self.Nx//2, self.Ny//2] = 0  


    def get_anti_aliased_spec3d(self):
        '''
        
        '''       
        grid = [self.w, self.kx, self.ky]
        w, kx, ky = grid
        Nt = 2*self.Nt-2
        Nx = len(kx)
        Ny = len(ky)        
        theta_mean = self.get_peak_dir()
        dtheta_max = 70*np.pi/180
        spectrum_extended = np.zeros((Nt, Nx, Ny))
        w_extended = self.dw * np.arange(-Nt//2, Nt//2) 
        interval = self.Nt//2
        w_mesh, kx_mesh, ky_mesh = np.meshgrid(w, kx, ky, indexing='ij')
        th = np.arctan2(ky_mesh, kx_mesh)
        mask_lower = np.where(np.abs(th - (theta_mean - np.pi))<= dtheta_max, 1, 0)
        
        # bin ich eigentlich in der richtigen Richtung? Habe ich das richtig simuliert? hmmm muss angepasst werden, aber wo?
        spectrum_extended[Nt//2:Nt//2+interval,:,:] = np.where(mask_lower[interval:,:,:], self.spectrum[interval:,:,:], 0).copy()
        upper_spec = np.where(mask_lower[:interval-1,:-1,:-1], self.spectrum[:interval-1,:-1,:-1], 0)
        spectrum_extended[Nt//2+interval:,1:,1:] = upper_spec.copy()

        # mirror:
        spectrum_extended[1:Nt//2,1:,1:] = np.flip(spectrum_extended[Nt//2+1:,1:,1:]).copy()
        grid = [w_extended, kx, ky]
        return spectrum_extended, grid  
        
    def estimate_dispersion_cone(self, h, Umax, kmin=0.04, kmax=0.3, plot_it=False):
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
        
        '''
        w_upper = self.w[self.Nt//2:]
        half_spec = np.flip(self.spectrum[1:self.Nt//2+1,:,:], axis=0)
        filtered_half_spec = gaussian_filter(half_spec, (0.5,0.5,0.5), truncate=4)



        w_orig_ind = np.argmin(np.abs(w_upper-1.2))
        w_orig = w_upper[w_orig_ind]
        w_orig_ind1 = np.argmin(np.abs(w_upper-1.0))
        w_orig1 = w_upper[w_orig_ind1]
        w_orig_ind2 = np.argmin(np.abs(w_upper-1.4))
        w_orig2 = w_upper[w_orig_ind2]
        print('w chosen: ', w_orig)
        '''
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (half_spec[w_orig_ind,:,:]))
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (filtered_half_spec[w_orig_ind,:,:]))
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (half_spec[w_orig_ind+10,:,:]))
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (filtered_half_spec[w_orig_ind+10,:,:]))
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (half_spec[w_orig_ind-10,:,:]))
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (filtered_half_spec[w_orig_ind-10,:,:]))
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
        rel_indices = np.argwhere(theta_spec>0.2*np.max(theta_spec)).transpose()[0]        
        theta_rel = theta[rel_indices]
        # k_relevant
        k_min_ind = np.argmin(np.abs(k - kmin))
        k_max_ind = np.argmin(np.abs(k - kmax))
        k_rel = k[k_min_ind:k_max_ind+1]
        # empty disp cone
        w_matrix = np.zeros((len(k_rel), len(theta_rel)))
        k_matrix = np.zeros((len(w_upper), len(theta_rel)))
        rel_spec = masked_spec_pol[:,k_min_ind:k_max_ind+1,rel_indices]
        min_dw = 4* np.sqrt(self.dkx*9.81)
        dw_max = np.maximum(k_rel*Umax, min_dw)
        ww, kk = np.meshgrid(w_upper, k_rel, indexing='ij')
        w0_rel = np.sqrt(kk*9.81*np.tanh(kk*h))

        
        for i in range(0, len(theta_rel)):
            input_spec = rel_spec[:,:,i]#convolutional_filters.apply_Gaussian_blur(rel_spec[:,:,i])
            #input_spec = gaussian_filter(rel_spec[:,:,i], 0.5)
            max_input_spec = np.outer(np.ones(len(w_upper)), np.max(input_spec, axis=0) )
            input_spec = np.where(input_spec>0.1*max_input_spec, input_spec, 0)
            pow = 0.3
            pow2 = 1
            w_peaks = (np.sum(input_spec**pow * ww**pow2, axis=0)/np.sum(input_spec**pow, axis=0))**(1./pow2)
            #w_matrix[:,i] = w_peaks
            w_matrix[:,i] = savgol_filter(w_peaks, 11, 3)

            pow = 0.3
            pow2 = 1
            k_peaks = (np.sum(input_spec**pow * kk**pow2, axis=1)/np.sum(input_spec**pow, axis=1))**(1./pow2)
            k_matrix[:,i] = k_peaks
            
            if plot_it:
                scaled_2d = spec_pol[:,k_min_ind:k_max_ind,i]/np.max(spec_pol[:,k_min_ind:k_max_ind,i])
                #plotting_interface.plot_k_w_spec(k_rel, w_upper, np.log10(scaled_2d).T, extent=[0.03, 0.35, 0.5, 1.9])
                plotting_interface.plot_k_w_spec(k_rel, w_upper, np.log10(input_spec).T, extent=[0.03, 0.35, 0.5, 1.9])
                plotting_interface.plot(k_rel, w_peaks)
                plotting_interface.plot(k_peaks, w_upper)
                w_peaks_hat = savgol_filter(w_peaks, 11, 3) # window size 51, polynomial order 3
                plotting_interface.plot(k_rel, w_peaks_hat)
                #plotting_interface.plot(k_rel, np.sqrt(k_rel*9.81*np.tanh(k_rel*h)))
                plotting_interface.show()
        ax = plotting_interface.plot_k_w_spec(k_rel, theta_rel, (rel_spec[w_orig_ind,:,:]))
        
        CS = ax.contour(k_rel, theta_rel, w_matrix.T, origin='lower', levels=[w_orig1,w_orig,  w_orig2], colors='green')
        #print(np.argmin(np.abs(w_upper-1.2)))
        out = np.array(CS.allsegs[1][0])
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (half_spec[w_orig_ind,:,:]))
        x_points = out[:,0]
        y_points = out[:,1]
        plotting_interface.plot(x_points*np.cos(y_points), x_points*np.sin(y_points))



        out = np.array(CS.allsegs[0][0])
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (half_spec[w_orig_ind1,:,:]))
        x_points = out[:,0]
        y_points = out[:,1]
        plotting_interface.plot(x_points*np.cos(y_points), x_points*np.sin(y_points))


        out = np.array(CS.allsegs[2][0])
        plotting_interface.plot_kx_ky_spec(self.kx, self.ky, (half_spec[w_orig_ind2,:,:]))
        x_points = out[:,0]
        y_points = out[:,1]
        plotting_interface.plot(x_points*np.cos(y_points), x_points*np.sin(y_points))


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
                plotting_interface.plot_kx_ky_spec(self.kx, self.ky, self.spectrum[w_ind,:,:], ax=ax, extent=[-k_limit, k_limit, -k_limit, k_limit])
                plotting_interface.plot_disp_rel_for_Ueff_at(at_w, h, U_effs[i], psis[i], 'w', ax=ax, extent=[-k_limit, k_limit, -k_limit, k_limit])

            i = i + 1
        return w_rel, U_effs, psis
       





class Spectrum:
    def __init__(self, spectrum, axes_grid):
        '''
        Parameters:
        ----------
        input:
                spectrum        array
                                spectrum in 1d, 2d, 3d
                axis_grid       list of arrays defining grid for each given axis

                spec_type       string
                                only meaningful for 1d spectrum... default: 'wavenumber', 
                                can be set to 'frequency'
        '''
        self.axes_grid = axes_grid
        if len(spectrum.shape)==1:
            self.ND = 1
            self.spectrumND = _Spectrum1d(spectrum, axes_grid)
            self.kx = self.spectrumND.kx
        elif len(spectrum.shape)==2:
            self.ND = 2        
            self.spectrumND = _Spectrum2d(spectrum, axes_grid)
            # TODO: distinguish different 2d spectra?
            self.kx = self.spectrumND.kx
            self.ky = self.spectrumND.ky
        elif len(spectrum.shape)==3:
            self.ND = 3        
            self.spectrumND = _Spectrum3d(spectrum, axes_grid)
            self.w = self.spectrumND.w
            self.kx = self.spectrumND.kx
            self.ky = self.spectrumND.ky
        else:
            print('\n\nError: Input data spectrum is not of the correct spec_type\n\n')


    def get_S(self):
        return self.spectrumND.get_S()

    def get_peak_dir(self):
        return self.spectrumND.get_peak_dir()

    def plot_kx_ky_integrated_w(self, extent, ax=None, dB=None, vmin=None, save=False):
        return self.spectrumND.plot_kx_ky_integrated_w(extent, ax, dB, vmin, save)

    def plot_point_cloud(self):
        self.spectrumND.plot_point_cloud()

    def remove_zeroth(self):
        self.spectrumND.remove_zeroth()

    def get_anti_aliased_spec3d(self):
        if self.ND==3:
            spectrum_extended, grid = self.spectrumND.get_anti_aliased_spec3d()
            return Spectrum(spectrum_extended, grid)
        else:
            print('Error: Not implemented')


    def estimate_dispersion_cone(self, h, Umax, kmin=0.04, kmax=0.3, plot_it=False):
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
    
        return self.spectrumND.estimate_dispersion_cone(h, Umax, kmin, kmax, plot_it)

            
    def estimate_Ueff_psi(self, h, Umax, ax=None, k_limit=None, plot_it=False):
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
                    Ueff        array
                                effective currents for each w
                    psi         array
                                current direction in rad for each Ueff
        '''
        if self.ND==3:
            return self.spectrumND.estimate_Ueff_psi(h, Umax, ax=ax, k_limit=k_limit, plot_it=plot_it)
        else:
            print('Error: Not available, the current estimation requires a 3d surface')    