import numpy as np
from scipy.signal import hilbert as hilbert
from wave_tools import fft_interface, find_peaks
from scipy.interpolate import interp1d



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