import numpy as np
from wave_tools import find_peaks, fft_interface, grouping, breaking_layers
from help_tools import plotting_interface
from scipy.signal import hilbert as hilbert
from skimage.measure import block_reduce



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def last_max_ind(eta):
    '''
    return the index of the last local maximum
    '''
    return np.argwhere(np.gradient(eta)>=0)[-1][0]
    

class Peak:
    def __init__(self, t_start, x_start, eta_start, vel_start, dt, dx, thresh = 0.85, ignore_c0=True):
        '''
        Create a peak instance to follow crestes in  a simulation

        Parameters:
        -----------
            input
                    t_start         float
                                    starting time where the peak is found
                    x_start         float
                                    starting position where peak is found
                    eta_start       float
                                    surface elevation at starting position
                    vel_start       float
                                    absolute horizontal velocity at starting point
                    dt              float
                                    resolution in time
                    dx              float
                                    resolution in space
                    thresh          float
                                    threshold for wave breaking (Bx)
                    ignore_c0       bool
                                    Switch for treating c=0, True: Bx=0, False: Bx=inf; default True
            
        '''
        self.is_active = True
        self.t_start = t_start
        self.x_start = x_start
        self.dt = dt
        self.dx = dx
        self.x = [x_start]
        self.eta = [eta_start]
        self.vel = [vel_start]
        self.c = None
        self.Bx = None
        self.threshold = thresh
        self.breaking = False
        self.ignore_c0 = ignore_c0
        self.breaking_start_ind = None

    def track(self, x, eta, vel):
        if self.is_active:
            self.x.append(x)
            self.eta.append(eta)
            self.vel.append(vel)
        else:
            print('Error: this peak is no longer active and cannot be tracked!')

    def stop_tracking(self):
        self.is_active = False
        self.x = np.array(self.x)
        self.eta = np.array(self.eta)
        self.vel = np.array(self.vel)
        self.x_len = 0
        self.eta_max = np.max(self.eta)

        if len(self.x)>1:
            self.x_len = self.x[-1] - self.x[0]
            self.c = np.gradient(self.x, self.dt)
            if self.ignore_c0:
                self.Bx = np.where(self.c==0, 0, np.abs(self.vel/self.c))
            else:
                # Bx is only defined from the second point, set to 0 in the first point where c is not known (0)
                self.Bx = np.block([0, np.abs(self.vel[1:]/self.c[1:])])

            all_breaking = self.Bx>=self.threshold
            self.breaking = np.sum(all_breaking) > 0
            self.cb = np.average(self.c)
            if self.breaking:
                self.breaking_start_ind = np.argwhere(all_breaking==True)[0][0]
        else:
            self.Bx = 0
        return self.x_len, self.eta_max
        

    def get_c(self):
        if self.is_active:
            print('Error: the crest velocity is not calculated yet')
        return self.c

    def get_Bx(self):
        if self.is_active:
            print('Error: the breaking criterion is not calculated yet')
        return self.Bx
        
    def get_track(self):
        '''
        For getting the physical coordinates of the peak track
        Parameters:
            output:
                t_tracked       float array
                                time steps of peak track
                x_tracked       float array
                                x-positions of peak track
        '''
        t_vec = self.t_start + np.arange(0, len(self.x))*self.dt
        return np.array(t_vec), np.array(self.x)

    def get_track_indices(self, x0=0, t0=0):
        '''
        For getting the indices of the peak track
        Parameters:
            input:  
                x0              float
                                offset of x-postion
                t0              float
                                offset of t-position
            output:
                t_tracked_inds  int array
                                time step indices s of peak track
                x_tracked       int array
                                x-position indices of peak track
        '''
        t_start_ind = int((self.t_start-t0)/self.dt)
        t_t_inds = t_start_ind + np.arange(0, len(self.x))
        if len(self.x)==1:
            return np.array([t_t_inds]), np.array([(self.x-x0)/self.dx]).astype('int')
        else:
            return np.array(t_t_inds), np.array((self.x-x0)/self.dx).astype('int')

    def is_breaking(self):
        return self.breaking

    def get_breaking_start_ind(self):
        '''
        gives breaking start along coordinates
        '''
        return self.breaking_start_ind

    def get_breaking_start_x(self):
        return self.x[self.breaking_start_ind]

    def get_breaking_start_t(self):
        return self.t_start + self.dt *self.breaking_start_ind

    def get_breaking_start_eta(self):
        return self.eta[self.breaking_start_ind]

    def get_breaking_start_Bx(self):
        return self.Bx[self.breaking_start_ind]

    def get_breaking_start_vel(self):
        return self.vel[self.breaking_start_ind]

    def get_breaking_start_c(self):
        return self.c[self.breaking_start_ind]

    def get_breaking_start_ind_x(self, x0=0):
        return int((self.x[self.breaking_start_ind]-x0)/self.dx)

    def get_breaking_start_ind_t(self, t0=0):
        return int((self.t_start-t0)/self.dt) + self.breaking_start_ind

    def get_breaking_indices(self, t0=0, x0=0):
        xi = []
        ti = []
        if self.breaking:
            mask = self.Bx>self.threshold
            N_breaking = sum(mask)
            x_breaking = np.ma.masked_array(self.x, mask=1-mask).compressed()
            t_breaking_ind = self.get_breaking_start_ind_t(t0=t0) + np.arange(0, N_breaking)
            x_breaking_ind = ((x_breaking-x0)/self.dx).astype(int)
        return t_breaking_ind, x_breaking_ind



    def plot_track(self, x, t, data, x_extent=70, dt_plot=1., cm_name='Blues', ax=None):
        '''
        Plots the evolution of the provided data along the track and marks the peak
        -----------
                    input       array
                                x-axis
                    input       array
                                t-axis
                    data        2d array
                                data to be plotted along the track over time and space
                    x_extent    float
                                extent that should be plot around the peak 
                    dt_plot     float
                                time stepping for plotting in seconds, default: 1.0
                    cm_name     string
                                name of the cmap utilized, default: 'Blues'
                    ax          axis
                                axis to be used from previously generated plots, 
                                if None a new axis is generated, default: None
        '''
        if ax == None:
            fig, ax = plotting_interface.subplots(figsize=(15,5))
        t_ind, x_ind = self.get_track_indices(x0=x[0], t0=t[0])
        dt = t[1] - t[0]
        dx = x[1] - x[0]
        interval_size = int(x_extent/dx)
        N_skip = np.max([1, int(dt_plot/dt)])
        N_max_peak_positions = x_ind.size
        if N_max_peak_positions<N_skip:
            N_skip = 1
        colors = plotting_interface.get_cmap(cm_name)(np.linspace(0.1,1,N_max_peak_positions))
        for i in np.arange(0, N_max_peak_positions, N_skip):
            start_ind = np.max([0, x_ind[i] - int(0.5*interval_size)])
            end_ind = np.min([x_ind[i] + int(0.5*interval_size), len(x)-2])
            ax.plot(x[start_ind:end_ind+1], data[t_ind[i], start_ind:end_ind+1], color=colors[i])
            ax.plot(x[x_ind[i]], data[t_ind[i], x_ind[i]], 'x', color=colors[i])
        return ax

    def plot_track_and_mark_breaking(self, x, t, data, x_extent=70, dt_plot=1., cm_name='Blues', ax=None):
        '''
        Plots the evolution along the track and marks where breaking occurs. 

        Parameters:
        -----------
                    input       array
                                x-axis
                    input       array
                                t-axis
                    data        2d array
                                data to be plotted along the track over time and space  
                    x_extent    float
                                extent that should be plot around the peak 
                    dt_plot     float
                                time stepping for plotting in seconds, default: 1.0
                    cm_name     string
                                name of the cmap utilized, default: 'Blues'
                    ax          axis
                                axis to be used from previously generated plots, 
                                if None a new axis is generated, default: None                
        '''
        if ax == None:
            fig, ax = plotting_interface.subplots(figsize=(15,5))
        t_ind, x_ind = self.get_track_indices(x0=x[0], t0=t[0])
        dt = t[1] - t[0]
        dx = x[1] - x[0]
        interval_size = int(x_extent/dx)
        N_skip = np.max([1, int(dt_plot/dt)])
        N_max_peak_positions = x_ind.size
        if N_max_peak_positions<N_skip:
            N_skip = 1
        colors = plotting_interface.get_cmap(cm_name)(np.linspace(0.1,1,N_max_peak_positions))
        for i in np.arange(0, N_max_peak_positions, N_skip):
            start_ind = np.max([0, x_ind[i] - int(0.5*interval_size)])
            end_ind = np.min([x_ind[i] + int(0.5*interval_size), len(x)-2])
            ax.plot(x[start_ind:end_ind+1], data[t_ind[i], start_ind:end_ind+1], color=colors[i])
            # If there is breaking happening in this time step in the observed interval
            if self.Bx[i]>self.threshold:
                ax.plot(x[x_ind[i]], data[t_ind[i], x_ind[i]], 'rx')#, color=colors[i])
        return ax
      
class PeakTracker:
    def __init__(self, x, t, eta0, vel0, cmax, high_peak_thresh=3.0, long_peak_thresh=300):
        self.x = x
        self.t = t
        self.Nx = len(x)
        self.Nt = len(t)
        self.dt = t[1] - t[0]
        self.dx = x[1] - x[0]
        self.N_max_steps_x = int(cmax/self.dt) + 1
        self.max_index_tracked = self.Nx - self.N_max_steps_x
        self.method = 'zero_crossing'
        peak_location_indices = list(find_peaks.find_peaks(eta0, method=self.method, peak_threshold=0.1))
        self.peak_location_collector = [peak_location_indices]
        self.N_peaks = len(peak_location_indices)
        self.peaks = {} # dictionary: key: peak ID, value: peak object
        self.active_peaks = {} # dictonary: key peak ID, value: peak location index
        self.ids_high_peaks = []
        self.ids_long_peaks = []
        self.ids_breaking_peaks = []
        self.ids_non_breaking_peaks = []
        self.high_peak_thresh = high_peak_thresh
        self.long_peak_thresh = long_peak_thresh
        for i in range(0, self.N_peaks):
            peak_index = peak_location_indices[i]
            self.peaks[i] = Peak(t[0], self.x[peak_index], eta0[peak_index], vel0[peak_index], self.dt, self.dx)
            self.active_peaks[i] = peak_index

    def breaking_tracker(self):
        self.Nb = 0
        self.bindex = np.array([0,0])
        self.pc = 0
        for i in range(0, self.N_peaks):
            if self.peaks[i].breaking == True:
                self.Nb += 1
                tindex = find_nearest(self.t, self.peaks[i].get_breaking_start_t())
                xindex = find_nearest(self.x, self.peaks[i].get_breaking_start_x())
                self.bindex = np.vstack([self.bindex, np.array([tindex, xindex])])
                self.pc = np.append(self.pc, self.peaks[i].cb)
        self.bindex = np.delete(self.bindex, 0, 0)

    def track_peaks(self, ti, eta, vel, max_dist=30, plot_each_iteration=False): 
        '''
        find peaks for given data track peaks found
        Old paths are continued or stopped, new paths are added

        max_dist: maximum number of grid points peak travelled since last time step
        '''
        peak_location_indices = list(find_peaks.find_peaks(eta, method=self.method, peak_threshold=0.14))
        self.peak_location_collector.append(peak_location_indices)
        indices_to_be_removed = []

        # check for all active peaks if they can be associated with a peak at the next timestep
        for peak_ID in self.active_peaks.keys():
            old_peak_index = self.active_peaks[peak_ID]
            peak = self.peaks[peak_ID]
            new_peak_location_index = None
            if len(peak_location_indices)>0:
                if old_peak_index >= self.N_max_steps_x:
                    index_difference = (old_peak_index - peak_location_indices)
                    mask = (index_difference>0)
                    index_difference = np.ma.masked_array(index_difference, mask=~mask).compressed()
                    if len(index_difference)>0:
                        chosen_index = old_peak_index - np.min(index_difference)
                        if (old_peak_index - chosen_index) <= max_dist:
                            new_peak_location_index = chosen_index

                            if plot_each_iteration:                    
                                import pylab as plt
                                plt.figure()
                                plt.plot(self.x, eta[:])
                                plt.plot(self.x, vel[:])
                                for iii in peak_location_indices:
                                    plt.plot(self.x[iii], eta[iii], 'ro')
                                plt.plot(self.x[old_peak_index], eta[old_peak_index], 'ko')
                                plt.plot(self.x[chosen_index], eta[chosen_index], 'kx')
                                plt.show()
                    else:
                        chosen_index = None

                    
                if new_peak_location_index is None:     
                    self.stop_tracking(peak_ID)           
                    indices_to_be_removed.append(peak_ID)                    
                else:
                    peak.track(self.x[new_peak_location_index], eta[new_peak_location_index], vel[new_peak_location_index])
                    self.active_peaks[peak_ID] = new_peak_location_index
                    peak_location_indices.pop(peak_location_indices.index(new_peak_location_index))
        
        for index in indices_to_be_removed:
            self.active_peaks.pop(index)

        for i in range(0, len(peak_location_indices)):
            peak_index = peak_location_indices[i]
            self.peaks[self.N_peaks + i] = Peak(ti, self.x[peak_index], eta[peak_index], vel[peak_index], self.dt, self.dx)
            self.active_peaks[self.N_peaks + i] = peak_index
        self.N_peaks = self.N_peaks + len(peak_location_indices)

    def stop_tracking(self, peak_ID, min_breaking_height=0.0):
        peak = self.peaks[peak_ID]
        x_len, eta_max = peak.stop_tracking()
        x_len = np.abs(x_len)
        if x_len >= self.long_peak_thresh:
            self.ids_long_peaks.append(peak_ID)
        if eta_max >= self.high_peak_thresh:
            self.ids_high_peaks.append(peak_ID)
        if peak.is_breaking():
            if peak.eta[peak.breaking_start_ind]>min_breaking_height:
                self.ids_breaking_peaks.append(peak_ID)
        else:
            self.ids_non_breaking_peaks.append(peak_ID)

    def stop_tracking_all(self):
        for peak_ID in self.active_peaks.keys():
            self.stop_tracking(peak_ID)

    def get_all_peaks(self):
        '''
        Return a list of peaks for each time step where peaks were tracked
        '''
        return self.peak_location_collector

    def get_active_peak_location_indices(self):
        return self.active_peak_location_indices
    
    def get_peak_dict(self):
        return self.peaks

    def get_ids_long_peaks(self):
        return np.array(self.ids_long_peaks).flatten()

    def get_ids_high_peaks(self):
        return np.array(self.ids_high_peaks).flatten()

    def get_ids_breaking_peaks(self):
        return np.array(self.ids_breaking_peaks).flatten()

    def get_ids_non_breaking_peaks(self):
        return np.array(self.ids_non_breaking_peaks).flatten()

    def get_specific_tracks(self, id_list_of_interest):
        x_list = []
        t_list = []
        for peak_ID in id_list_of_interest:
            peak = self.peaks[peak_ID]
            this_t, this_x = peak.get_track()
            x_list.append(this_x)
            t_list.append(this_t)
        return t_list, x_list


    def get_all_tracks(self):
        return self.get_specific_tracks(self.peaks.keys())

    def get_high_tracks(self):
        return self.get_specific_tracks(self.ids_high_peaks)
    
    def get_long_tracks(self):
        return self.get_specific_tracks(self.ids_long_peaks)

    def get_breaking_tracks(self):
        return self.get_specific_tracks(self.ids_breaking_peaks)

    def get_specific_track_indices(self, id_list_of_interest):
        xi_list = []
        ti_list = []
        for peak_ID in id_list_of_interest:
            peak = self.peaks[peak_ID]
            this_ti, this_xi = peak.get_track_indices()
            xi_list.append(this_xi)
            ti_list.append(this_ti)
        return ti_list, xi_list

    def get_breaking_track_indices(self):
        return self.get_specific_track_indices(self.ids_breaking_peaks)

    def get_high_track_indices(self):
        return self.get_specific_track_indices(self.ids_high_peaks)

    def get_long_track_indices(self):
        return self.get_specific_track_indices(self.ids_long_peaks)
    
    def plot_specific_tracks(self, id_list_of_interest, ax):
        t_list, x_list = self.get_specific_tracks(id_list_of_interest)
        for i in range(0, len(x_list)):
            plotting_interface.plot(t_list[i], x_list[i], ax=ax)

    def plot_all_tracks(self, ax=None):
        self.plot_specific_tracks(self.peaks.keys(), ax)

    def plot_high_tracks(self, ax=None):
        self.plot_specific_tracks(self.ids_high_peaks, ax)

    def plot_long_tracks(self, ax=None):
        self.plot_specific_tracks(self.ids_long_peaks, ax)

    def plot_breaking_tracks(self, ax=None):
        self.plot_specific_tracks(self.ids_breaking_peaks, ax)
        
    def plot_evolution_of_specific_tracks(self, data, id_list_of_interest, N=None, x_extent=70, dt_plot=1.0, ax_list=None, cm_name='Blues', envelope=False, env_x_max_dist_front=20, env_x_max_dist_back=20):
        '''
        Plots the evolution of specific tracks
        
        Parameters:
        ----------
                    input
                            data                    2d array
                                                    data to be plotted
                            id_list_of_interest     list
                                                    edge ids to be plotted (one figure for each)
                            N                       int/None
                                                    if not None: limits the edges plotted from the given list to the provided number
                            x_extent                float
                                                    extent of x-axis of surrounding to be plotted around edge, default:70
                            dt_plot                 float
                                                    step size for plotting in time, default: 1
                            ax_list                 list
                                                    list of axis, one for each of the ids that should be plotted
                            cm_name                 string
                                                    colormap name, default: 'Blues'
                            envelope                bool
                                                    if true plot envelope as well
                            env_x_max_dist_front    float
                                                    maximum distance on x-axis to search for minimum (lower envelope front)
                            env_x_max_dist_back     float
                                                    maximum distance on x-axis to search for minimum (lower envelope back)
                    output
                            out_ax_list             list
                                                    list of axis of plots
        '''
        if N is None or N>len(id_list_of_interest):
            N=len(id_list_of_interest)
        out_ax_list = []
        for i in range(0, N):
            this_peakID = id_list_of_interest[i]
            this_peak = self.peaks[this_peakID]
            if ax_list is None:
                ax = None
            else:
                ax = ax_list[i]
            ax = this_peak.plot_track(self.x, self.t, data, x_extent=x_extent, dt_plot=dt_plot, cm_name=cm_name, ax=ax)
            if envelope:
                x_env, y_env = self.get_upper_envelope(this_peakID, data, env_x_max_dist_front)
                ax.plot(x_env, y_env, 'r')
                x_env, y_env = self.get_lower_envelope_front(this_peakID, data, env_x_max_dist_front)
                ax.plot(x_env, y_env, 'darkorange')
                x_env, y_env = self.get_lower_envelope_back(this_peakID, data, env_x_max_dist_back)
                ax.plot(x_env, y_env, 'purple')
            out_ax_list.append(ax)
        return out_ax_list


    def plot_evolution_of_breaking_tracks(self, data, id_list_of_interest=None, N=None, x_extent=70, ax_list=None, cm_name='Blues', dt_plot=1, envelope=False, env_x_max_dist_front=20, env_x_max_dist_back=20):
        if id_list_of_interest is None:
            id_list_of_interest = self.ids_breaking_peaks
        else:
            id_list_of_interest = np.array(self.ids_breaking_peaks)[id_list_of_interest]
        return self.plot_evolution_of_specific_tracks(data, id_list_of_interest, N=N, x_extent=x_extent, ax_list=ax_list, cm_name=cm_name, dt_plot=dt_plot, envelope=envelope, env_x_max_dist_front=env_x_max_dist_front, env_x_max_dist_back=env_x_max_dist_back)

    def plot_specific_tracks_and_mark_breaking(self, data, id_list_of_interest, N=None, x_extent=50, dt_plot=1., cm_name='Blues', ax_list=None, envelope=False, env_x_max_dist_front=20, env_x_max_dist_back=20):
        '''
        Plots the evolution of specific tracks
        
        Parameters:
        ----------
                    input
                            data                    2d array
                                                    data to be plotted
                            id_list_of_interest     list
                                                    edge ids to be plotted (one figure for each)
                            N                       int/None
                                                    if not None: limits the edges plotted from the given list to the provided number
                            x_extent         float
                                                    extent of surrounding to be plotted around edge, default: 70
                            dt_plot                 float
                                                    step size for plotting in time, default: 1
                            cm_name                 string
                                                    colormap name, default: 'Blues'
                            ax_list                 list
                                                    list of axis, one for each of the ids that should be plotted
                            envelope                bool
                                                    if true plot envelope as well
                            env_x_max_dist          float
                                                    maximum distance on x-axis to search for minimum (lower envelope)
                    output
                            out_ax_list             list
                                                    list of axis of plots
        '''
        if N is None or N>len(id_list_of_interest):
            N=len(id_list_of_interest)
        out_ax_list = []
        for i in range(0, N):
            this_peakID = id_list_of_interest[i]
            this_peak = self.peaks[this_peakID]
            if ax_list is None:
                ax = None
            else:
                ax = ax_list[i]
            ax = this_peak.plot_track_and_mark_breaking(self.x, self.t, data, x_extent=x_extent, dt_plot=dt_plot, cm_name=cm_name, ax=ax)
            if envelope:
                x_env, y_env = self.get_upper_envelope(this_peakID, data, 10)
                ax.plot(x_env, y_env, 'r')
                x_env, y_env = self.get_lower_envelope_front(this_peakID, data, env_x_max_dist_front)
                ax.plot(x_env, y_env, 'darkorange')
                x_env, y_env = self.get_lower_envelope_back(this_peakID, data, env_x_max_dist_back)
                ax.plot(x_env, y_env, 'purple')
            out_ax_list.append(ax)
        return out_ax_list

    def plot_breaking_tracks_and_mark_breaking(self, data, id_list_of_interest=None, N=None, x_extent=70, dt_plot=1.):
        if id_list_of_interest ==None:
            ids = self.ids_breaking_peaks
        else:
            ids = np.array(self.ids_breaking_peaks)[id_list_of_interest]
        return self.plot_specific_tracks_and_mark_breaking(data, ids, N, x_extent=x_extent, dt_plot=dt_plot)   


    def get_breaking_mask_fixed_L(self, L):
        '''
        return a mask that marks areas of wave breaking by one

        Parameters:
        -----------
                    input
                            L           float
                                        extent in x-direction of breaking wave
                    output
                            mask        int array
                                        mask: 0: not breaking 1:breaking
                                        
        '''
        mask = np.zeros((self.Nt, self.Nx), dtype=int)
        L_indices = int(L/self.dx)
        for peak_ID in self.ids_breaking_peaks:
            this_peak = self.peaks[peak_ID]
            t_inds, x_inds = this_peak.get_breaking_indices(t0=self.t[0], x0=self.x[0])
            for i in range(0, len(t_inds)):
                x_ind_stop = x_inds[i]
                x_ind_start = np.max([0, x_ind_stop - L_indices])
                mask[t_inds[i], x_ind_start:x_ind_stop] = 1
        return mask

    def get_breaking_mask(self, eta):
        '''
        return a mask that marks areas of wave breaking by one, using tilt to determine wave size.

        Parameters:
        -----------
                    input
                            eta         float array
                                        2d surface elevation field
                    output
                            mask        int array
                                        mask: 0: not breaking 1:breaking
                                        
        '''
        mask = np.zeros((self.Nt, self.Nx), dtype=int)
        for peak_ID in self.ids_breaking_peaks:
            this_peak = self.peaks[peak_ID]
            t_inds, x_inds = this_peak.get_breaking_indices(t0=self.t[0], x0=self.x[0])
            for i in range(0, len(t_inds)):
                control = True
                x_ind_stop = x_inds[i]
                l = 0
                while control == True:
                    x_ind_start = x_inds[i] - l
                    tilt = np.arctan2(eta[t_inds[i], x_ind_start] - eta[t_inds[i], x_ind_start-1], this_peak.dx)
                    if tilt <= 0.1:
                        control = False
                    l = l+1
                mask[t_inds[i], x_ind_start:x_ind_stop] = 1
        return mask



    def get_breaking_tilt_and_mask(self, eta, H, polarization, plot_it):
        '''
        return a the tilt-based basis for backscatter in the breaking region and the mask that 
        marks areas of wave breaking by one, using tilt to determine wave size.

        Parameters:
        -----------
                    input
                            eta         float array
                                        2d surface elevation field
                            H           float
                                        elevation of the radar antenna above the mean level
                            polarization    string
                                        'HH' or 'VV'
                            plot_it     bool
                                        if True plotting of breaking wave with breaking layers
                    output
                            mask        int array
                                        mask: 0: not breaking 1:breaking
                                        
        '''
        mask = np.zeros((self.Nt, self.Nx), dtype=int)
        tilt_basis = np.zeros((self.Nt, self.Nx), dtype=int)
        for peak_ID in self.ids_breaking_peaks:
            this_peak = self.peaks[peak_ID]
            t_inds, x_inds = this_peak.get_breaking_indices(t0=self.t[0], x0=self.x[0])
            for i in range(0, len(t_inds)):
                control = True
                x_ind_stop = x_inds[i]
                l = 0
                while control == True:
                    x_ind_start = x_inds[i] - l
                    tilt = np.arctan2(eta[t_inds[i], x_ind_start] - eta[t_inds[i], x_ind_start-1], this_peak.dx)
                    if tilt <= 0.1:
                        control = False
                    l = l+1
                mask[t_inds[i], x_ind_start:x_ind_stop] = 1
                if (x_ind_stop - x_ind_start) > 1:
                    Ninterpolate = 10
                    N_here = (x_ind_stop - x_ind_start)
                    N_here_fine = N_here * Ninterpolate
                    x_here_fine = np.linspace(self.x[x_ind_start], self.x[x_ind_stop], N_here_fine)
                    eta_here = eta[t_inds[i], x_ind_start:x_ind_stop]
                    amp = np.max(eta_here) - np.min(eta_here)
                    if plot_it:
                        import pylab as plt
                        plt.figure()
                        plt.plot(x_here_fine[::Ninterpolate], eta_here, color='darkblue', label=r'$\eta$')
                    
                    y0 = np.min(eta_here)
                    tilt_basis_here = breaking_layers.accumulated_tilt_basis(x_here_fine, amp, H, y0, polarization='VV', plot_it=plot_it)
                    tilt_basis_here = block_reduce(tilt_basis_here, (Ninterpolate,), np.max)
                    tilt_basis[t_inds[i], x_ind_start:x_ind_stop] = tilt_basis_here
                    if plot_it:
                        plt.plot(x_here_fine[::Ninterpolate], tilt_basis_here, 'g--', label=r'tilt_basis')
                        plt.legend()
                        plt.savefig('layers.pdf', bbox_inches='tight')
                        plt.show()
        return tilt_basis, mask


    def get_breaking_crest_speeds_fixed_L(self, L):
        '''
        This function defines the speed of the particles in areas of breaking.
        The speed is defined as the crest speed

        Parameters:
        -----------
                    input       
                            L       float
                                    extent in x-direction of breaking wave
                    output
                            speeds  float array
                                    crest speed of the waves provided where wave breaking occurs, otherwise 0
        '''
        speeds = np.zeros((self.Nt, self.Nx))
        L_indices = int(L/self.dx)
        for peak_ID in self.ids_breaking_peaks:
            this_peak = self.peaks[peak_ID]
            c = this_peak.get_c()
            t_inds, x_inds = this_peak.get_breaking_indices(t0=self.t[0], x0=self.x[0])
            for i in range(0, len(t_inds)):
                x_ind_stop = x_inds[i]
                x_ind_start = np.max([0, x_ind_stop - L_indices])
                speeds[t_inds[i], x_ind_start:x_ind_stop] = c[i]
        return speeds

    def get_breaking_crest_speeds(self, eta, vel, N_extend=10):
        '''
        This function defines the speed of the particles.
        At breaking the speed is defined as the crest speed if there are orbital velocities greater than crest speed 
        their value is alternated with the crest speed.
        Outside breaking, orbital velocitites are used.

        Parameters:
        -----------
                    input       
                            eta     float
                                    extent in x-direction of breaking wave
                            vel         float array
                                        2d surface velocity
                            N_extend    int
                                        number of points by which velocities after the peak are evaluated
                    output
                            speeds  float array
                                    crest speed of the waves provided where wave breaking occurs, otherwise 0
        '''
        speeds = vel.copy()
        for peak_ID in self.ids_breaking_peaks:
            this_peak = self.peaks[peak_ID]
            c = this_peak.get_c()
            t_inds, x_inds = this_peak.get_breaking_indices(t0=self.t[0], x0=self.x[0])
            for i in range(0, len(t_inds)):
                control = True
                x_ind_stop = x_inds[i] 
                l = 0
                while control == True:
                    x_ind_start = x_inds[i] - l
                    tilt = np.arctan2(eta[t_inds[i], x_ind_start] - eta[t_inds[i], x_ind_start-1], this_peak.dx)
                    if tilt <= 0.1:
                        control = False
                    l = l+1
                x_ind_stop = x_ind_stop +  np.argwhere(vel[t_inds[i], x_ind_stop:]<np.abs(c[i]))[0][0]
                if x_ind_stop-x_ind_start > 2:
                    vel_max = np.max(vel[t_inds[i], x_ind_start:x_ind_stop])
                    speeds[t_inds[i], x_ind_start:x_ind_stop][::2] = -c[i]
                    speeds[t_inds[i], x_ind_start+1:x_ind_stop][::2] = vel_max
                else:
                    speeds[t_inds[i], x_ind_start:x_ind_stop] = c[i]
        return speeds

    def get_upper_envelope(self, peakID, eta, x_max_dist=10):
        '''
        return upper envelope of track (equivalent to peaks)
        '''
        x_dist = int(x_max_dist/self.dx)
        this_peak = self.peaks[peakID]
        t_inds, x_inds = this_peak.get_track_indices(x0=self.x[0], t0=self.t[0])
        N_track = len(t_inds)
        envelope = np.zeros(N_track)
        for i in range(0, N_track):
            first_x_ind = np.max([0, x_inds[i]-x_dist])
            last_x_ind = np.min([self.Nx-1, x_inds[i]+x_dist])
            envelope[i] = np.max(eta[t_inds[i], first_x_ind:last_x_ind])
            '''
            import pylab as plt
            plt.figure()
            plt.plot(self.x, eta[t_inds[i], :])
            plt.plot(self.x[x_inds[i]], eta[t_inds[i], x_inds[i]], 'o')
            plt.plot(self.x[first_x_ind:last_x_ind], eta[t_inds[i], first_x_ind:last_x_ind])
            plt.plot(self.x[x_inds[i]], envelope[i], 'x')
            plt.show()
            '''
        return this_peak.x, envelope

    def get_lower_envelope_front(self, peakID, eta, x_max_dist=20):
        '''
        return lower envelope of the track (lowest value closer to the shore)
        '''
        x_dist = int(x_max_dist/self.dx)
        this_peak = self.peaks[peakID]
        t_inds, x_inds = this_peak.get_track_indices(x0=self.x[0], t0=self.t[0])
        N_track = len(t_inds)
        envelope = np.zeros(N_track)
        x_pos = np.zeros(N_track)
        for i in range(0, N_track):
            this_x_ind = np.max([0, x_inds[i]-x_dist])
            '''
            next_peak_ind = find_peaks.find_peaks(eta[t_inds[i], this_x_ind:x_inds[i]], method='all_peaks')
            if len(next_peak_ind) == 0:
                x_ind_offset = np.argmin(eta[t_inds[i], this_x_ind:x_inds[i]])
                front_peak_ind = this_x_ind + x_ind_offset
            else:
                next_peak_ind = this_x_ind + next_peak_ind[-1]
                front_peak_ind = next_peak_ind + np.argmin(eta[t_inds[i], next_peak_ind:x_inds[i]])
            
            envelope[i] = eta[t_inds[i], front_peak_ind]
            x_pos[i] = self.x[front_peak_ind]
            '''
            envelope[i] = np.min(eta[t_inds[i], this_x_ind:x_inds[i]])
            x_pos[i] = self.x[this_x_ind+np.argmin(eta[t_inds[i], this_x_ind:x_inds[i]])]
            '''
            import pylab as plt
            plt.figure()
            plt.plot(self.x, eta[t_inds[i], :])
            plt.plot(self.x[x_inds[i] - x_dist:x_inds[i]], eta[t_inds[i], x_inds[i] - x_dist:x_inds[i]])
            plt.plot(self.x[x_inds[i]], eta[t_inds[i], x_inds[i]], 'o')
            plt.plot(x_pos[i], envelope[i], 'x')
            plt.show()
            '''
        return x_pos, envelope

    def get_lower_envelope_back(self, peakID, eta, x_max_dist=70):
        '''
        return lower envelope of the track (lowest value father from the shore)
        '''
        x_dist = int(x_max_dist/self.dx)
        this_peak = self.peaks[peakID]
        t_inds, x_inds = this_peak.get_track_indices(x0=self.x[0], t0=self.t[0])
        N_track = len(t_inds)
        envelope = np.zeros(N_track)
        x_pos = np.zeros(N_track)
        for i in range(0, N_track):
            this_x_ind = np.min([self.Nx-1, x_inds[i]+x_dist])
            '''
            x_ind_offset = find_peaks.find_peaks(20-eta[t_inds[i], x_inds[i]:this_x_ind], method='all_peaks')
            if len(x_ind_offset)==0:
                x_ind_offset = this_x_ind
            else:
                x_ind_offset = x_ind_offset[0]
            back_peak_ind = x_inds[i] + x_ind_offset
            envelope[i] = eta[t_inds[i], back_peak_ind]
            x_pos[i] = self.x[back_peak_ind]
            '''
            envelope[i] = np.min(eta[t_inds[i], x_inds[i]:this_x_ind])
            x_pos[i] = self.x[x_inds[i]+np.argmin(eta[t_inds[i],  x_inds[i]:this_x_ind])]
            '''
            import pylab as plt
            plt.figure()
            plt.plot(self.x, eta[t_inds[i], :])
            plt.plot(self.x[x_inds[i]:this_x_ind], eta[t_inds[i], x_inds[i]:this_x_ind])
            plt.plot(self.x[x_inds[i]], eta[t_inds[i], x_inds[i]], 'o')
            plt.plot(x_pos[i], envelope[i], 'x)
            plt.show()
            '''
        return x_pos, envelope

    def plot_envelopes(self, list_of_interest, eta, show_all=False, show_mean=True, x_max_center=0, x_max_dist_front=20, x_max_dist_back=20, mov_av=15, ylabel=r'$\eta~[\mathrm{m}]$'):
        import pylab as plt
        from help_tools.moving_average import moving_average
        if show_all:
            plt.figure()
        y_env_col_u = np.zeros(self.Nx)
        y_env_col_lf = np.zeros(self.Nx)
        y_env_col_lb = np.zeros(self.Nx)
        counter_u = np.zeros(self.Nx)
        counter_lf = np.zeros(self.Nx)
        counter_lb = np.zeros(self.Nx)
        for peakID in list_of_interest:
            x_env, y_env_u = self.get_upper_envelope(peakID, eta, x_max_center)
            x_inds = ((x_env-self.x[0])/self.dx).astype(int)
            y_env_col_u[x_inds] += y_env_u
            counter_u[x_inds] += 1
            x_env, y_env_lf = self.get_lower_envelope_front(peakID, eta, x_max_dist_front)
            x_inds = ((x_env-self.x[0])/self.dx).astype(int)
            y_env_col_lf[x_inds] += y_env_lf
            counter_lf[x_inds] += 1
            x_env, y_env_lb = self.get_lower_envelope_back(peakID, eta, 2.5*x_max_dist_back)
            x_inds = ((x_env-self.x[0])/self.dx).astype(int)
            y_env_col_lb[x_inds] += y_env_lb
            counter_lb[x_inds] += 1
            if show_all:
                plt.plot(x_env, y_env_u, 'r')
                plt.plot(x_env, y_env_lf, 'darkorange')
                plt.plot(x_env, y_env_lb, 'purple')
        if show_all:
            plt.xlabel(r'$x~[\mathrm{m}]$')
            plt.ylabel(ylabel)
            

        counter_u = np.where(counter_u==0, 1, counter_u)
        counter_lf = np.where(counter_lf==0, 1, counter_lf)
        counter_lb = np.where(counter_lb==0, 1, counter_lb)

        if show_mean:
            fig, ax = plt.subplots()
            ax.plot(self.x, moving_average(y_env_col_u/counter_u, mov_av), 'r', label=r'$\mathrm{upper~envelope}$')
            ax.plot(self.x, moving_average(y_env_col_lf/counter_lf, mov_av), 'darkorange', label=r'$\mathrm{lower~envelope~front}$')
            ax.plot(self.x, moving_average(y_env_col_lb/counter_lb, mov_av), 'purple', label=r'$\mathrm{lower~envelope~back}$')
            ax.set_xlabel(r'$x~[\mathrm{m}]$')
            ax.set_ylabel(ylabel)
            ax.legend()
        return ax


def get_PeakTracker(x, t, eta, vel, cmax=15, max_dist=30, high_peak_thresh=3, long_peak_thresh=300, plot_tracking=False, breaking_mask=None, smoothen_input=False):
    '''
    Creates and instance of Peak Tracker and tracks all peaks and returns the instance

    Parameters:
    -----------
        input:
                x       1d array 
                        x axis
                t       1d array 
                        t axis
                eta     2d array
                        surface elevation, [t, x]
                vel     2d array
                        horizontal velocity [t, x], if not available the breaking mask can be multiplied by a high number (such that vel/C >0.85) and it will work
                cmax    maximum crest speed
            
                max_dist            float
                                    maximum distance between two peaks (should be calculated from cmax...)
                high_peak_thresh    float
                                    threshold for classifying peaks as high
                long_peak_thresh    float
                                    threshold for classifying peaks as long
                plot_tracking       bool
                                    for debugging: put to True and se how tracking happens
                smoothen_input      bool
                                    True: apply smoothing before running algorithm; Default: False
    '''
    pt = PeakTracker(x, t, eta[0,:], vel[0,:], cmax=cmax, high_peak_thresh=high_peak_thresh, long_peak_thresh=long_peak_thresh)
    for i in range(1, len(t)):
        pt.track_peaks(t[i], eta[i,:], vel[i,:], max_dist=max_dist, plot_each_iteration=plot_tracking)
    pt.stop_tracking_all()
    return pt

            

