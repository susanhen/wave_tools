import numpy as np
from scipy.signal import hilbert as hilbert
from wave_tools import fft_interface, find_peaks
from scipy.interpolate import interp1d
from help_tools import plotting_interface



def smooth_shifted_envelope(x, eta, wave_number_threshold): 
    envelope = np.abs(hilbert(eta))
    k, fft_envelope = fft_interface.physical2spectral(envelope, x)
    fft_envelope_filtered = np.where(np.abs(k)>wave_number_threshold, 0, fft_envelope)
    x, smooth_envelope = fft_interface.spectral2physical(fft_envelope_filtered, k)   
    return smooth_envelope.real - np.mean(smooth_envelope.real)

def interpolated_envelope(x, eta):
    '''
    1d first
    '''
    peak_indices = find_peaks.find_peaks(eta, method='all_peaks')
    peak_indices = np.block([0,peak_indices,-1])
    envelope_interp_function = interp1d(x[peak_indices], eta[ peak_indices], kind='quadratic')
    return envelope_interp_function(x)

class Group:
    def __init__(self, t_index, start_index, stop_index, contour, Nx, Nt):
        self.start_indices = [start_index]
        self.stop_indices = [stop_index]
        self.t_indices = [t_index]
        self.contours = [contour]
        self.cg = None
        self.Nt = Nt # length of dataset
        self.Nx = Nx
        self.group_marker = np.zeros((Nt, Nx))
        self.group_marker[t_index, start_index:stop_index] = 1

    def get_start_index(self):
        return self.start_index

    def get_stop_index(self):
        return self.stop_index

    def get_peak_index(self):
        return self.peak_index

    def get_group_marker(self):
        return self.group_marker

    def track(self, t_index, start_index, stop_index, contour):
        self.start_indices.append(start_index)
        self.t_indices.append(t_index)
        self.stop_indices.append(stop_index)
        self.contours.append(contour)
        self.group_marker[t_index, start_index:stop_index] = 1

    def stop_tracking(self):
        self.start_indices = np.array(self.start_indices)
        self.stop_indices = np.array(self.stop_indices)
        self.t_indices = np.array(self.t_indices)



class GroupTracker:
    '''
    Track groups along their peak
    '''
    def __init__(self, x, t, eta0, vel0, cgmax=10.0, wave_number_threshold=0.07, Nx_min=100):
        self.x = x
        self.t = t
        self.Nx = len(x)
        self.dt = t[1] - t[0]
        self.N_max_steps_x = int(cgmax/self.dt) + 1
        self.wave_number_threshold = wave_number_threshold
        self.max_index_tracked = self.Nx - self.N_max_steps_x
        self.peak_method = 'zero_crossing'
        self.Nx_min = Nx_min
        self.group_list = grouping.define_groups(x, eta0, wave_number_threshold, Nx_min)
        self.N_groups = len(self.group_list)
        self.peaks = {} # dictionary: key: group ID, value: peak object
        self.active_peaks = {} # dictonary: key group ID, value: peak location index

    def track_groups(self, ti, eta, vel):
        '''
        find peaks for given data track peaks found
        Old paths are continued or stopped, new paths are added

        max_dist: maximum number of grid points peak travelled since last time step
        '''

        new_group_location_indices, new_group_list = grouping.define_groups(self.x, eta, self.wave_number_threshold, self.Nx_min)
        self.group_location_collector.append(new_group_location_indices)
        indices_to_be_removed = []

        # check for all active peaks if they can be associated with a peak at the next timestep
        for group_ID in self.active_peaks.keys():
            old_peak_index = self.active_peaks[group_ID]
            peak = self.peaks[group_ID]
            new_peak_location_index = None
            found = False
            new_group_list_index = 0
            if old_peak_index >= self.N_max_steps_x:
                '''
                Find the group where the old_peak_index is contained
                '''
                while not found and new_group_list_index<len(new_group_location_indices):
                    this_group = new_group_list[new_group_list_index]
                    if this_group.ind_within_group(old_peak_index):
                        new_peak_location_index = new_group_location_indices[new_group_list_index]
                        found = True
                    else:
                        new_group_list_index += 1
                    
            if new_peak_location_index is None:     
                self.stop_tracking(group_ID)           
                indices_to_be_removed.append(group_ID)                    
            else:
                # if a peak from the previous time step can be associated to a new peak its position is tracked and no longer part of new peaks to be registered
                peak.track(self.x[new_peak_location_index], eta[new_peak_location_index], vel[new_peak_location_index])
                self.active_peaks[group_ID] = new_peak_location_index
                new_group_location_indices.pop(new_group_location_indices.index(new_peak_location_index))
        
        for index in indices_to_be_removed:
            self.active_peaks.pop(index)

        # Register new peaks that could not be associated to previous peaks
        for i in range(0, len(new_group_location_indices)):
            peak_index = new_group_location_indices[i]
            self.peaks[self.N_groups + i] = Peak(ti, self.x[peak_index], eta[peak_index], vel[peak_index], self.dt)
            self.active_peaks[self.N_groups + i] = peak_index
        self.N_groups = self.N_groups + len(new_group_location_indices)

    def stop_tracking(self, peak_ID):
        peak = self.peaks[peak_ID]
        x_len, eta_max = peak.stop_tracking()

    def stop_tracking_all(self):
        for peak_ID in self.active_peaks.keys():
            self.stop_tracking(peak_ID)

    def get_all_tracks(self):
        x_list = []
        t_list = []
        for peak_ID in self.peaks.keys():
            peak = self.peaks[peak_ID]
            this_t, this_x = peak.get_track()
            x_list.append(this_x)
            t_list.append(this_t)
        return x_list, t_list

    def plot_all_tracks(self, ax=None):
        x_list, t_list = self.get_all_tracks()
        for i in range(0, len(x_list)):
            plotting_interface.plot(t_list[i], x_list[i], ax=ax)

def get_GroupTracker(x, t, eta, vel, cgmax=15):
    '''
    Creates and instance of Group Tracker and tracks all groups and returns the instance

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
                        horizontal velocity [t, x]
                cmax    maximum crest speed
    '''
    gt = GroupTracker(x, t, eta[0,:], vel[0,:], cgmax=cgmax)
    for i in range(1, len(t)):
        gt.track_groups(t[i], eta[i,:], vel[i,:])
    gt.stop_tracking_all()
    return gt            
            


def define_groups(x, eta, Nx_min, Nt):
    '''
    define wave groups

    Parameters:
    ----------
                input:
                        x       array
                                x range (could also be time)
                        eta     array
                                surface elevation
                        Nx_min                  float
                                                minimum distance along x for a group
                        Nt      int
                                
                output:
                        group_list              list
                                                list of group instances

    '''

    #TODO: first find a good envelope and then do as for peaks just that what is regsitered is different or attach a peak to the group...
    Nx = len(x)
    group_list = []
    dx = x[1] - x[0]
    N_min = int(Nx_min/dx)
    #envelope = smooth_shifted_envelope(x, eta, wave_number_threshold)
    envelope = interpolated_envelope(x, eta)
    upcrossing = np.block([0, np.diff(np.sign(envelope))])
    start_indices = np.argwhere(upcrossing>0).transpose()[0]
    start_index = start_indices[0]
    stop_indices = start_index  + np.argwhere(upcrossing[start_index:]<0).transpose()[0]
    if len(stop_indices)<len(start_indices):
        start_indices = start_indices[:-1]
    for i in range(0, len(start_indices)):
        if (stop_indices[i] - start_indices[i])>=N_min:
            group_list.append(Group(0, start_indices[i], stop_indices[i], envelope[start_indices[i]:stop_indices[i]], Nt, Nx))
    return group_list




if __name__=='__main__':
    import matplotlib.pyplot as plt
    from scipy.signal import chirp as chirp
    t = np.linspace(0, 1, 1000)
    eta = chirp(t, 20.0, t[-1], 100.0)
    eta *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t))


    analytic_signal = hilbert(eta)
    env2 = interpolated_envelope(t, np.array([eta, eta, eta]))
    plt.plot(t, eta)
    plt.plot(t, np.abs(analytic_signal))
    plt.plot(t, env2)
    plt.show()