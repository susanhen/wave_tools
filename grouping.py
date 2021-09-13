import numpy as np
from scipy.signal import hilbert as hilbert
from wave_tools import fft_interface, find_peaks



def smooth_shifted_envelope(x, eta, wave_number_threshold): 
    envelope = np.abs(hilbert(eta))
    k, fft_envelope = fft_interface.physical2spectral(envelope, x)
    fft_envelope_filtered = np.where(np.abs(k)>wave_number_threshold, 0, fft_envelope)
    x, smooth_envelope = fft_interface.spectral2physical(fft_envelope_filtered, k)   
    return smooth_envelope.real - np.mean(smooth_envelope.real)

class Group:
    def __init__(self, start_index, stop_index, peak_index, N):
        self.start_index = start_index
        self.stop_index = stop_index
        self.peak_index = peak_index
        self.N = N # length of dataset
        self.group_marker = np.zeros(N)
        self.group_marker[start_index:stop_index] = 1

    def get_start_index(self):
        return self.start_index

    def get_stop_index(self):
        return self.stop_index

    def get_peak_index(self):
        return self.peak_index

    def get_group_marker(self):
        return self.group_marker

    def overlap_with_group(self, group):
        return np.sum(self.group_marker*group.get_group_marker())>0

    def ind_within_group(self, ind):
        return ind>=self.start_index and ind<self.stop_index


            


def define_groups(x, eta, wave_number_threshold, Nx_min):
    '''
    define wave groups

    Parameters:
    ----------
                input:
                        x       array
                                x range (could also be time)
                        eta     array
                                surface elevation
                        wave_number_threshold   float
                                                threshold for filtering, angular frequency if x is time
                        Nx_min                  float
                                                minimum distance along x for a group
                output:
                        group_list              list
                                                list of group instances

    '''
    N = len(x)
    group_list = []
    peak_list = []
    dx = x[1] - x[0]
    N_min = int(Nx_min/dx)
    envelope = smooth_shifted_envelope(x, eta, wave_number_threshold)
    upcrossing = np.block([0, np.diff(np.sign(envelope))])
    start_indices = np.argwhere(upcrossing>0).transpose()[0]
    start_index = start_indices[0]
    stop_indices = start_index  + np.argwhere(upcrossing[start_index:]<0).transpose()[0]
    if len(stop_indices)<len(start_indices):
        start_indices = start_indices[:-1]
    for i in range(0, len(start_indices)):
        if (stop_indices[i] - start_indices[i])>=N_min:
            peak_indices = start_indices[i]+list(find_peaks.find_peaks(eta, method='all_peaks'))
            #peak_index = peak_indices[np.argmin(np.abs(peak_indices-)]
            group_list.append(Group(start_indices[i], stop_indices[i], peak_index, N))
            peak_list.append(peak_index)
    return peak_list, group_list




if __name__=='__main__':
    import matplotlib.pyplot as plt
    from scipy.signal import chirp as chirp
    t = np.linspace(0, 1, 1000)
    eta = chirp(t, 20.0, t[-1], 100.0)
    eta *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t))

    analytic_signal = hilbert(eta)
    plt.plot(t, eta)
    plt.plot(t, np.abs(analytic_signal))
    plt.show()