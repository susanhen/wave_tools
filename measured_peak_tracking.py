import numpy as np
from wave_tools import find_peaks, fft_interface, grouping
from help_tools import plotting_interface


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    

class Peak:
    def __init__(self, t_start, x_start, data_start, breeaking_mask_start, dt, dx, thresh = 0.85, ignore_c0=True):
        '''
        Create a peak instance to follow crestes in  a simulation

        Parameters:
        -----------
            input
                    t_start         float
                                    starting time where the peak is found
                    x_start         float
                                    starting position where peak is found
                    data_start      float
                                    surface elevation at starting position
                    breeaking_mask_start  bool

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
        self.dt = dt
        self.dx = dx
        self.x = [x_start]
        self.data = [data_start]
        self.mask = [breeaking_mask_start]
        self.c = None
        self.Bx = None
        self.threshold = thresh
        self.breaking = False
        self.ignore_c0 = ignore_c0
        self.breaking_start_ind = None

    def track(self, x, data, mask):
        if self.is_active:
            self.x.append(x)
            self.data.append(data)
            self.mask.append(mask)
        else:
            print('Error: this peak is no longer active and cannot be tracked!')

    def stop_tracking(self):
        self.is_active = False
        self.x = np.array(self.x)
        self.data = np.array(self.data)
        self.mask = np.array(self.mask)
        self.x_len = 0
        self.data_max = np.max(self.data)

        if len(self.x)>1:
            self.x_len = self.x[-1] - self.x[0]
            self.c = np.gradient(self.x, self.dt)

            self.breaking = np.sum(self.mask) > 0
            self.cb = np.average(self.c)
            if self.breaking:
                self.breaking_start_ind = np.argwhere(self.mask>0)[0][0]
        else:
            self.Bx = 0
        return self.x_len, self.data_max
        

    def get_c(self):
        if self.is_active:
            print('Error: the crest maskocity is not calculated yet')
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

    def get_breaking_start_data(self):
        return self.data[self.breaking_start_ind]

    def get_breaking_start_Bx(self):
        print('ERROR: Not yet implemented for measured data')
        return None#self.Bx[self.breaking_start_ind]

    def get_breaking_start_c(self):
        return self.c[self.breaking_start_ind]

    def get_breaking_start_ind_x(self, x0=0):
        return int((self.x[self.breaking_start_ind]-x0)/self.dx)

    def get_breaking_start_ind_t(self, t0=0):
        return int((self.t_start-t0)/self.dt) + self.breaking_start_ind
      



class PeakTracker:
    def __init__(self, x, t, data0, mask0, cmax=20.0, high_peak_thresh=3.0, long_peak_thresh=300):
        self.x = x
        self.t = t
        self.Nx = len(x)
        self.Nt = len(t)
        self.dt = t[1] - t[0]
        self.dx = x[1] - x[0]
        self.N_max_steps_x = int(cmax/self.dt) + 1
        self.max_index_tracked = self.Nx - self.N_max_steps_x
        self.method = 'zero_crossing'
        peak_location_indices = list(find_peaks.find_peaks(data0, method=self.method))
        self.peak_location_collector = [peak_location_indices]
        self.N_peaks = len(peak_location_indices)
        self.peaks = {} # dictionary: key: peak ID, value: peak object
        self.active_peaks = {} # dictonary: key peak ID, value: peak location index
        self.ids_high_peaks = []
        self.ids_long_peaks = []
        self.ids_breaking_peaks = []
        self.high_peak_thresh = high_peak_thresh
        self.long_peak_thresh = long_peak_thresh
        for i in range(0, self.N_peaks):
            peak_index = peak_location_indices[i]
            self.peaks[i] = Peak(0, self.x[peak_index], data0[peak_index], mask0[peak_index], self.dt, self.dx)
            self.active_peaks[i] = peak_index

    def breaking_tracker(self):
        self.Nb = 0
        self.bindex = np.array([0,0])
        self.pc = 0
        for i in range(0, self.N_peaks):
            if self.peaks[i].breaking == True:
                self.Nb += 1
                tindex = self.peaks[i].get_breaking_start_ind_t()
                xindex = self.peaks[i].get_breaking_start_ind_x()
                self.bindex = np.vstack([self.bindex, np.array([tindex, xindex])])
                self.pc = np.append(self.pc, self.peaks[i].cb)
        self.bindex = np.delete(self.bindex, 0, 0)

    def track_peaks(self, ti, data, mask, max_dist=4):
        '''
        find peaks for given data track peaks found
        Old paths are continued or stopped, new paths are added

        max_dist: maximum number of grid points peak tramaskled since last time step
        '''
        peak_location_indices = list(find_peaks.find_peaks(data, method=self.method))
        self.peak_location_collector.append(peak_location_indices)
        indices_to_be_removed = []

        # check for all active peaks if they can be associated with a peak at the next timestep
        for peak_ID in self.active_peaks.keys():
            old_peak_index = self.active_peaks[peak_ID]
            peak = self.peaks[peak_ID]
            new_peak_location_index = None
            found = False
            shift = 0
            if old_peak_index >= self.N_max_steps_x:

                chosen_index = np.argmin(np.abs(peak_location_indices - old_peak_index))
                if peak_location_indices[chosen_index] <= old_peak_index and -(peak_location_indices[chosen_index] - old_peak_index) <= max_dist:
                    new_peak_location_index = peak_location_indices[chosen_index]
            if new_peak_location_index is None:     
                self.stop_tracking(peak_ID)           
                indices_to_be_removed.append(peak_ID)                    
            else:
                peak.track(self.x[new_peak_location_index], data[new_peak_location_index], mask[new_peak_location_index])
                self.active_peaks[peak_ID] = new_peak_location_index
                peak_location_indices.pop(peak_location_indices.index(new_peak_location_index))
        
        for index in indices_to_be_removed:
            self.active_peaks.pop(index)

        for i in range(0, len(peak_location_indices)):
            peak_index = peak_location_indices[i]
            self.peaks[self.N_peaks + i] = Peak(ti, self.x[peak_index], data[peak_index], mask[peak_index], self.dt, self.dx)
            self.active_peaks[self.N_peaks + i] = peak_index
        self.N_peaks = self.N_peaks + len(peak_location_indices)

    def stop_tracking(self, peak_ID, min_breaking_height=0.0):
        peak = self.peaks[peak_ID]
        x_len, data_max = peak.stop_tracking()
        if x_len >= self.long_peak_thresh:
            self.ids_long_peaks.append(peak_ID)
        if data_max >= self.high_peak_thresh:
            self.ids_high_peaks.append(peak_ID)
        if peak.is_breaking():
            if peak.data[peak.breaking_start_ind]>min_breaking_height:
                self.ids_breaking_peaks.append(peak_ID)

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
        return self.ids_long_peaks

    def get_ids_high_peaks(self):
        return self.ids_high_peaks

    def get_ids_breaking_peaks(self):
        return self.ids_breaking_peaks

    def get_specific_tracks(self, id_list_of_interest):
        x_list = []
        t_list = []
        for peak_ID in id_list_of_interest:
            peak = self.peaks[peak_ID]
            this_t, this_x = peak.get_track()
            x_list.append(this_x)
            t_list.append(this_t)
        return x_list, t_list

    def get_all_tracks(self):
        return self.get_specific_tracks(self.peaks.keys())

    def get_high_tracks(self):
        return self.get_specific_tracks(self.ids_high_peaks)
    
    def get_long_tracks(self):
        return self.get_specific_tracks(self.ids_long_peaks)

    def get_breaking_tracks(self):
        return self.get_specific_tracks(self.ids_breaking_peaks)

    def plot_specific_tracks(self, id_list_of_interest, ax):
        x_list, t_list = self.get_specific_tracks(id_list_of_interest)
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



def get_PeakTracker(x, t, data, mask, cmax=15):
    '''
    Creates and instance of Peak Tracker and tracks all peaks and returns the instance

    Parameters:
    -----------
        input:
                x       1d array 
                        x axis
                t       1d array 
                        t axis
                data    2d array
                        data for searching peak in 2d (t,x)
                mask     2d array
                        breaking mask
                cmax    maximum crest speed
    '''
    pt = PeakTracker(x, t, data[0,:], mask[0,:], cmax=cmax)
    for i in range(1, len(t)):
        pt.track_peaks(t[i], data[i,:], mask[i,:])
    pt.stop_tracking_all()
    return pt