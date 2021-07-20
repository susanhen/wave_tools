import numpy as np
from wave_tools import find_peaks
from help_tools import plotting_interface


def last_max_ind(eta):
    '''
    return the index of the last local maximum
    '''
    return np.argwhere(np.gradient(eta)>=0)[-1][0]
    

class Peak:
    def __init__(self, t_start, x_start, eta_start, vel_start, dt, thresh = 0.85):
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
            
        '''
        self.is_active = True
        self.t_start = t_start
        self.dt = dt
        self.x = [x_start]
        self.eta = [eta_start]
        self.vel = [vel_start]
        self.c = None
        self.Bx = None
        self.threshold = thresh
        self.breaking = False

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
        
        if len(self.x)>1:
            self.c = np.gradient(self.x, self.dt)
            self.Bx = np.abs(self.vel)/np.abs(self.c)
            '''
            if sum(np.where(self.Bx>0.4, 1,0))>1:
                print('Bx = ', self.Bx)
            '''
            for i in range(0, np.size(self.Bx)):
                if self.Bx[i]>=self.threshold and self.c[i] != 0:
                    self.breaking = True
                    self.position = np.array([self.t_start + self.dt*i, self.x[i]])
                    self.cb = self.c[i]
                    break
        else:
            self.Bx = 0
        

    def get_c(self):
        if self.is_active:
            print('Error: the crest velocity is not calculated yet')
        return self.c

    def get_Bx(self):
        if self.is_active:
            print('Error: the breaking criterion is not calculated yet')
        return self.Bx
        
    def get_track(self):
        t_vec = self.t_start + np.arange(0, len(self.x))*self.dt
        return np.array(self.x), np.array(t_vec)

class PeakTracker:
    def __init__(self, x, t, eta0, vel0, cmax=10.0):
        self.x = x
        self.t = t
        self.Nx = len(x)
        self.dt = t[1] - t[0]
        self.dx_max = int(cmax/self.dt) + 1
        self.max_index_tracked = self.Nx - self.dx_max
        peak_indices = list(find_peaks.find_peaks(eta0))
        self.peak_collector = [peak_indices]
        self.N_peaks = len(peak_indices)
        self.peaks = {}
        self.active_peaks = {}
        for i in range(0, self.N_peaks):
            peak_index = peak_indices[i]
            self.peaks[i] = Peak(0, self.x[peak_index], eta0[peak_index], vel0[peak_index], self.dt)
            self.active_peaks[i] = peak_index

    def breaking_tracker(self):
        self.Nb = 0
        self.bindex = np.array([0,0])
        self.pc = 0
        for i in range(0, self.N_peaks):
            if self.peaks[i].breaking == True:
                self.Nb += 1
                tindex = find_nearest(self.t, self.peaks[i].position[0])
                xindex = find_nearest(self.x, self.peaks[i].position[1])
                self.bindex = np.vstack([self.bindex, np.array([tindex, xindex])])
                self.pc = np.append(self.pc, self.peaks[i].cb)
        self.bindex = np.delete(self.bindex, 0, 0)

    def track_peaks(self, ti, eta, vel):
        '''
        find peaks for given data track peaks found
        Old paths are continued or stopped, new paths are added
        '''
        peak_indices = list(find_peaks.find_peaks(eta))
        self.peak_collector.append(peak_indices)
        indices_to_be_removed = []
        for peak_number in self.active_peaks.keys():
            old_peak_index = self.active_peaks[peak_number]
            peak = self.peaks[peak_number]
            new_peak_index = None
            if old_peak_index >= self.dx_max:
                #close_peak_index = old_peak_index - self.dx_max + np.argmax(eta[old_peak_index-self.dx_max:old_peak_index+1])
                close_peak_index = old_peak_index - self.dx_max + last_max_ind(eta[old_peak_index-self.dx_max:old_peak_index+1])
                ''' if the maximum value in a small interval closer to shore of previous max is detected
                as peak, register, and remove this peak form the list
                otherwise: stop tracking of this peak
                '''
                if close_peak_index in peak_indices:
                    new_peak_index = close_peak_index
                elif close_peak_index + 1 in peak_indices:
                    new_peak_index = close_peak_index + 1
                elif close_peak_index - 1 in peak_indices:
                    new_peak_index = close_peak_index - 1
                    
            if new_peak_index is None:
                peak.stop_tracking()
                indices_to_be_removed.append(peak_number)                    
            else:
                peak.track(self.x[new_peak_index], eta[new_peak_index], vel[new_peak_index])
                self.active_peaks[peak_number] = new_peak_index
                peak_indices.pop(peak_indices.index(new_peak_index))
        
        for index in indices_to_be_removed:
            self.active_peaks.pop(index)

        for i in range(0, len(peak_indices)):
            peak_index = peak_indices[i]
            self.peaks[self.N_peaks + i] = Peak(ti, self.x[peak_index], eta[peak_index], vel[peak_index], self.dt)
            self.active_peaks[self.N_peaks + i] = peak_index
        self.N_peaks = self.N_peaks + len(peak_indices)

    def stop_tracking(self):
        for peak_number in self.active_peaks.keys():
            peak = self.peaks[peak_number]
            peak.stop_tracking()


    def get_all_peaks(self):
        '''
        Return a list of peaks for each time step where peaks were tracked
        '''
        return self.peak_collector

    def get_active_peak_indices(self):
        return self.active_peak_indices
    
    def get_peak_dict(self):
        return self.peaks

    def get_all_tracks(self):
        x_list = []
        t_list = []
        for peak_number in self.peaks.keys():
            peak = self.peaks[peak_number]
            this_x, this_t = peak.get_track()
            x_list.append(this_x)
            t_list.append(this_t)
        return x_list, t_list

    def plot_all_tracks(self):
        x_list, t_list = self.get_all_tracks()
        for i in range(0, len(x_list)):
            plotting_interface.plot(t_list[i], x_list[i])

def get_PeakTracker(x, t, eta, vel, cmax=15):
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
                        horizontal velocity [t, x]
                cmax    maximum crest speed
    '''
    pt = PeakTracker(x, t, eta[0,:], vel[0,:], cmax=cmax)
    for i in range(1, len(t)):
        pt.track_peaks(t[i], eta[i,:], vel[i,:])
    pt.stop_tracking()
    return pt

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

            