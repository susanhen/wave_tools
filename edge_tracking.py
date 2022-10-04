import numpy as np
from wave_tools import find_peaks, fft_interface, grouping
from help_tools import plotting_interface, convolutional_filters
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from skimage.feature import canny
from skimage.filters import gaussian

def get_common_indices(index_list1, index_list2):
    list1_set = set(index_list1)
    return list(list1_set.intersection(index_list2))

class Edge:
    def __init__(self, t_start, x_start, data_start, breaking, dt, dx):
        '''
        Create a edge instance to follow crestes in  a simulation

        Parameters:
        -----------
            input
                    t_start         float
                                    starting time where the edge is found
                    x_start         float
                                    starting position where edge is found
                    data_start      float
                                    surface elevation at starting position
                    breaking     bool    
                                    indicates if there is wavebreaking along the track
                    dt              float
                                    resolution in time
                    dx              float
                                    resolution in space
            
        '''
        self.is_active = True
        self.t_start = t_start
        self.dt = dt
        self.dx = dx
        self.b_mask_around = None # if set, True if breaking is occuring around the edge, False if not
        self.x = [x_start]
        self.data = [data_start]
        self.c = None
        self.Bx = None
        self.breaking = breaking

    def track(self, x, data, breaking):
        if self.is_active:
            self.x.append(x)
            self.data.append(data)
            self.breaking = self.breaking or breaking
        else:
            print('Error: this edge is no longer active and cannot be tracked!')

    def stop_tracking(self):
        self.is_active = False
        self.x = np.array(self.x)
        self.data = np.array(self.data)
        self.x_len = 0
        self.data_max = np.max(self.data)
        if len(self.x)>1:
            self.x_len = np.max(self.x) - np.min(self.x)
        # TODO calc Bx if available
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
        For getting the physical coordinates of the edge track
        Parameters:
            output:
                t_tracked       float array
                                time steps of edge track
                x_tracked       float array
                                x-positions of edge track
        '''
        t_vec = self.t_start + np.arange(0, len(self.x))*self.dt
        return np.array(t_vec), np.array(self.x)

    def get_track_indices(self, x0=0, t0=0):
        '''
        For getting the indices of the edge track
        Parameters:
            input:  
                x0              float
                                offset of x-postion
                t0              float
                                offset of t-position
            output:
                t_tracked_inds  int array
                                time step indices s of edge track
                x_tracked       int array
                                x-position indices of edge track
        '''
        t_start_ind = int((self.t_start-t0)/self.dt)
        t_t_inds = t_start_ind + np.arange(0, len(self.x))
        return np.array(t_t_inds), np.array((self.x-x0)/self.dx).astype('int')

    def get_b_mask_around(self, mask, x0=0, t0=0, interval_size=3):
        '''
        fill b_mask_around with bools, true if breaking occurs close to the edge, false if not
        Parameters:
            input:  
                mask            int array
                                0: non-breaking, 1: breaking
                x0              float
                                offset of x-postion
                t0              float
                                offset of t-position
                intervalsize    int
                                number of points to be tested around edge
        '''
        if self.b_mask_around is None:
            tis, xis = self.get_track_indices(x0=x0, t0=t0)
            self.b_mask_around = np.zeros(len(tis))
            Nx = mask.shape[1]   
            for i in range(0, len(tis)):
                start_ind = np.max([0, xis[i]-interval_size])
                end_ind = np.min([xis[i] +interval_size, Nx-2])
                self.b_mask_around[i] = np.sum(mask[tis[i], start_ind:end_ind]) > 0
        return self.b_mask_around


    def get_indices_before_and_at_breaking(self, mask, x0=0, t0=0, interval_size=3):
        '''
        return t and x indidces just before breaking and at breaking, as breaking can occur
        multiple times along the track the indices are returned as lists
        Parameters:
            input:  
                mask            int array
                                0: non-breaking, 1: breaking
                x0              float
                                offset of x-postion
                t0              float
                                offset of t-position
            output:
                t_tracked_inds_before   int array 
                                        time step indices of edge track before breaking starts
                x_tracked_inds_before   int array
                                        x-position index of edge track
                t_tracked_inds_after    int array 
                                        time step indices of edge track where breaking starts
                x_tracked_inds_after    int array
                                        x-position index of edge track where breaking starts
        '''
        if not self.breaking or len(self.x)<3:
            return None, None, None, None
        tis, xis = self.get_track_indices(x0=x0, t0=t0)
        loc_mask = self.get_b_mask_around(mask, x0, t0, interval_size)
        t_tracked_inds_before = []
        x_tracked_inds_before = []
        t_tracked_inds_at = []
        x_tracked_inds_at = []
        mask_grad = np.gradient(loc_mask)
        relevant_indices = np.argwhere(mask_grad==0.5).T[0]
        if len(relevant_indices)<2:
            return None, None, None, None
        else:
            for i in range(0, len(relevant_indices)//2):
                if tis[relevant_indices[2*i+1]] - tis[relevant_indices[2*i]] ==1:
                    t_tracked_inds_before.append(tis[relevant_indices[2*i]])
                    t_tracked_inds_at.append(tis[relevant_indices[2*i+1]])
                    x_tracked_inds_before.append(xis[relevant_indices[2*i]])
                    x_tracked_inds_at.append(xis[relevant_indices[2*i+1]])
            return t_tracked_inds_before, x_tracked_inds_before, t_tracked_inds_at, x_tracked_inds_at


    def is_breaking(self):
        '''
        checks if wave within is breaking close to the edge a long the track
        '''
        return self.breaking

    def plot_track(self, x, t, data, label, x_extent=70, dt_plot=1., cm_name='Blues', ax=None, interpolate=False):
        '''
        Plots the evolution along the track and marks the edge
        '''
        if ax == None:
            fig, ax = plt.subplots(figsize=(15,5))
        t_ind, x_ind = self.get_track_indices(x0=x[0], t0=t[0])
        dt = t[1] - t[0]
        dx = x[1] - x[0]
        interval_size = int(x_extent/dx)
        N_skip = np.max([1, int(dt_plot/dt)])
        N_max_peak_positions = x_ind.size
        if N_max_peak_positions<N_skip:
            N_skip = 1
        colors = plt.get_cmap(cm_name)(np.linspace(0.1,1,N_max_peak_positions))
        for i in np.arange(0, N_max_peak_positions, N_skip):
            start_ind = np.max([0, x_ind[i] - int(0.2*interval_size)])
            end_ind = np.min([x_ind[i] + int(0.8*interval_size), len(x)-2])
            if interpolate:
                N_inter = 100
                r_inter = np.linspace(x[start_ind], x[end_ind], N_inter)
                f_data = interp1d(x[start_ind: end_ind+1], data[t_ind[i], start_ind: end_ind+1], kind='cubic')
                data_inter = f_data(r_inter)
            else:
                r_inter = x[start_ind: end_ind+1]
                data_inter = data[t_ind[i], start_ind: end_ind+1]
            ax.plot(r_inter, data_inter, color=colors[i])
            ax.plot(x[x_ind[i]], data[t_ind[i], x_ind[i]], 'x', color=colors[i])
        ax.set_ylabel(label)
        ax.set_xlabel(r'$r~[m]$')
        return ax

    def get_trough_peak_trough(self, x, t, data, mask=None, x_extent=200, control_plot=False):
        '''
        Get the evolution of trough1, peak trough2  along the track 
        '''

        t_ind, x_ind = self.get_track_indices(x0=x[0], t0=t[0])
        peaks = np.zeros(len(t_ind))
        x_pos_peaks = np.zeros(len(t_ind))
        troughs1 = np.zeros(len(t_ind))
        x_pos_troughs1 = np.zeros(len(t_ind))
        troughs2 = np.zeros(len(t_ind))
        x_pos_troughs2 = np.zeros(len(t_ind))
        if not mask is None:
            track_mask = np.zeros(len(t_ind))
        else:
            track_mask = None
        dt = t[1] - t[0]
        dx = x[1] - x[0]
        interval_size = int(x_extent/dx)
        for i in np.arange(0, len(peaks)):
            start_ind = np.max([0, x_ind[i] - int(0.5*interval_size)])
            end_ind = np.min([x_ind[i] + int(0.5*interval_size), len(x)-2])
            gradfunc = np.gradient(data[t_ind[i], start_ind:end_ind+1])
            signgrad = np.sign(gradfunc)
            gradsign = np.gradient(signgrad)
            trough_indices = start_ind + np.argwhere(gradsign==1).T[0][::2]
            trough_inds_before_edge = np.argwhere(trough_indices<=x_ind[i])
            if len(trough_inds_before_edge)>0:
                trough1_ind = trough_indices[trough_inds_before_edge[-1][0]]
                trough1_ind = trough1_ind - 1 + np.argmin(data[t_ind[i], trough1_ind-1:trough1_ind+2])
                x_pos_troughs1[i] = x[trough1_ind]
                trough1 = data[t_ind[i],trough1_ind]
                start_ind_cut = trough1_ind
            else:
                trough1 = np.nan
                start_ind_cut = start_ind

            trough_inds_behind_edge = np.argwhere(trough_indices>x_ind[i])
            if len(trough_inds_behind_edge)>0:
                trough2_ind = trough_indices[trough_inds_behind_edge[0][0]]
                trough2_ind = trough2_ind + np.argmin(data[t_ind[i], trough2_ind:trough2_ind+3])
                x_pos_troughs2[i] = x[trough2_ind]
                end_ind_cut = trough2_ind
                trough2 = data[t_ind[i],trough2_ind]
            else:
                trough2 = np.nan
                end_ind_cut = end_ind
            
            troughs1[i] = trough1
            troughs2[i] = trough2
            peaks[i] = np.max(data[t_ind[i], start_ind_cut:end_ind_cut+1])
            track_mask[i] = np.sum(mask[t_ind[i], start_ind_cut:end_ind_cut+1])>0
            peak_indices = np.argmax(data[t_ind[i], start_ind_cut:end_ind_cut+1])
            x_pos_peaks[i] = x[start_ind_cut+peak_indices]
            if control_plot:
                plt.plot(x[start_ind:end_ind+1], data[t_ind[i], start_ind:end_ind+1])
                plt.plot(x_pos_peaks[i], peaks[i], 'x')
                plt.plot(x[start_ind:end_ind+1], gradsign)
                plt.plot(x_pos_troughs1[i], troughs1[i], 'x')
                plt.plot(x_pos_troughs2[i], troughs2[i], 'x')
                plt.plot(x[x_ind[i]], data[t_ind[i], x_ind[i]], 'ko')
                plt.show()
        '''
        F_peaks = interp1d(x_pos_peaks, peaks)
        x_pos_peaks = np.linspace(x_pos_peaks[0], x_pos_peaks[-1], 100)
        peaks = F_peaks(x_pos_peaks)
        F_troughs1 = interp1d(x_pos_troughs1, troughs1)
        x_pos_troughs1 = np.linspace(x_pos_troughs1[0], x_pos_troughs1[-1], 100)
        troughs1 = F_troughs1(x_pos_troughs1)
        F_troughs2 = interp1d(x_pos_troughs2, troughs2)
        x_pos_troughs2 = np.linspace(x_pos_troughs2[0], x_pos_troughs2[-1], 100)
        troughs2 = F_troughs2(x_pos_troughs2)
        '''
        return x_pos_troughs1, troughs1, x_pos_peaks, peaks, x_pos_troughs2, troughs2, track_mask

    def plot_track_and_mark_breaking(self, x, t, data, mask, label, x_extent=70, dt_plot=1., cm_name='Blues', ax=None, interpolate=False):
        '''
        Plots the evolution along the track and marks where breaking occurs
        '''
        if ax == None:
            fig, ax = plt.subplots(figsize=(15,5))
        t_ind, x_ind = self.get_track_indices(x0=x[0], t0=t[0])
        dt = t[1] - t[0]
        dx = x[1] - x[0]
        interval_size = int(x_extent/dx)
        N_skip = np.max([1, int(dt_plot/dt)])
        N_max_peak_positions = x_ind.size
        if N_max_peak_positions<N_skip:
            N_skip = 1
        colors = plt.get_cmap(cm_name)(np.linspace(0.1,1,N_max_peak_positions))
        for i in np.arange(0, N_max_peak_positions, N_skip):
            start_ind = np.max([0, x_ind[i] - int(0.2*interval_size)])
            end_ind = np.min([x_ind[i] + int(0.8*interval_size), len(x)-2])
            if interpolate:
                N_inter = 100
                r_inter = np.linspace(x[start_ind], x[end_ind], N_inter)
                f_data = interp1d(x[start_ind: end_ind+1], data[t_ind[i], start_ind: end_ind+1], kind='cubic')
                data_inter = f_data(r_inter)
            else:
                r_inter = x[start_ind: end_ind+1]
                data_inter = data[t_ind[i], start_ind: end_ind+1]
            ax.plot(r_inter, data_inter, color=colors[i])
            # If there is breaking happening in this time step in the observed interval
            mask_x_ind = 0
            for x_breaking in x[start_ind:end_ind+1]:
                if mask[t_ind[i], start_ind + mask_x_ind]>0:
                    ax.plot(x_breaking, data[t_ind[i], start_ind+mask_x_ind], 'rx')
                mask_x_ind = mask_x_ind+1
            # mask the peak on a breaking wave
            '''
            if sum(mask[t_ind[i], start_ind:end_ind+1])>0:
                #first_breaking_ind = start_ind + np.argwhere(mask[t_ind[i], start_ind:end_ind+1]==1)#[-1]
                data_here = data[t_ind[i], start_ind:end_ind+1]
                peak_ind = np.argmax(data_here)
                x_here = x[start_ind:end_ind+1]
                #ax.plot(x[x_ind[i]], data[t_ind[i], x_ind[i]], 'rx')#, color=colors[i])
                ax.plot(x_here[peak_ind], data_here[peak_ind], 'rx')
            '''
        ax.set_ylabel(label)
        ax.set_xlabel(r'$r~[m]$')
        return ax




class EdgeTracker:
    def __init__(self, x, t, data0, max_edge_dist, mask0=None, cmax=10.0, high_edge_thresh=3.0, long_edge_thresh=200):
        self.x = x
        self.t = t
        self.Nx = len(x)
        self.Nt = len(t)
        self.dt = t[1] - t[0]
        self.dx = x[1] - x[0]
        self.N_max_steps_x = int(cmax/self.dt) + 1
        self.max_index_tracked = self.Nx - self.N_max_steps_x
        self.method = 'zero_crossing'
        edge_location_indices = list(find_peaks.find_peaks(data0, method=self.method))
        self.edge_location_collector = [edge_location_indices]
        self.N_edges = len(edge_location_indices)
        self.edges = {} # dictionary: key: edge ID, value: edge object
        self.active_edges = {} # dictonary: key edge ID, value: edge location index
        self.ids_high_edges = []
        self.ids_long_edges = []
        self.ids_breaking_edges = []
        self.ids_non_breaking_edges = []
        self.high_edge_thresh = high_edge_thresh
        self.long_edge_thresh = long_edge_thresh
        self.max_edge_dist = max_edge_dist
        if mask0 is None:
            mask0 = np.zeros(data0.shape)
        for i in range(0, self.N_edges):
            edge_index = edge_location_indices[i]
            breaking = np.sum(mask0[edge_index:edge_index+max_edge_dist])>0
            self.edges[i] = Edge(0, self.x[edge_index], data0[edge_index], breaking, self.dt, self.dx)
            self.active_edges[i] = edge_index

    def breaking_tracker(self):
        self.Nb = 0
        self.bindex = np.array([0,0])
        self.pc = 0
        for i in range(0, self.N_edges):
            if self.edges[i].breaking == True:
                self.Nb += 1
                tindex = self.edges[i].get_breaking_start_ind_t()
                xindex = self.edges[i].get_breaking_start_ind_x()
                self.bindex = np.vstack([self.bindex, np.array([tindex, xindex])])
                self.pc = np.append(self.pc, self.edges[i].cb)
        self.bindex = np.delete(self.bindex, 0, 0)

    def track_edges(self, ti, data, mask=None, max_dist=4):
        '''
        find edges for given data track edges found
        Old paths are continued or stopped, new paths are added

        max_dist: maximum number of grid points edge tramaskled since last time step
        '''
        if mask is None:
            mask = np.zeros(data.shape)
        edge_location_indices = list(find_peaks.find_peaks(data, method=self.method))
        self.edge_location_collector.append(edge_location_indices)
        indices_to_be_removed = []

        # check for all active edges if they can be associated with a edge at the next timestep
        for edge_ID in self.active_edges.keys():
            old_edge_index = self.active_edges[edge_ID]
            edge = self.edges[edge_ID]
            new_edge_location_index = None
            found = False
            shift = 0
            if len(edge_location_indices)>0:
                chosen_index = np.argmin(np.abs(edge_location_indices - old_edge_index))
                if edge_location_indices[chosen_index] <= old_edge_index and -(edge_location_indices[chosen_index] - old_edge_index) <= max_dist:
                    new_edge_location_index = edge_location_indices[chosen_index]
            if new_edge_location_index is None:     
                self.stop_tracking(edge_ID)           
                indices_to_be_removed.append(edge_ID)                    
            else:
                breaking = np.sum(mask[new_edge_location_index:new_edge_location_index + self.max_edge_dist])>0
                edge.track(self.x[new_edge_location_index], data[new_edge_location_index], breaking)
                self.active_edges[edge_ID] = new_edge_location_index
                edge_location_indices.pop(edge_location_indices.index(new_edge_location_index))
        
        for index in indices_to_be_removed:
            self.active_edges.pop(index)

        for i in range(0, len(edge_location_indices)):
            edge_index = edge_location_indices[i]
            breaking = np.sum(mask[edge_index:edge_index+self.max_edge_dist])>0
            self.edges[self.N_edges + i] = Edge(ti, self.x[edge_index], data[edge_index], breaking, self.dt, self.dx)
            self.active_edges[self.N_edges + i] = edge_index
        self.N_edges = self.N_edges + len(edge_location_indices)

    def stop_tracking(self, edge_ID):
        edge = self.edges[edge_ID]
        x_len, data_max = edge.stop_tracking()
        if x_len >= self.long_edge_thresh:
            self.ids_long_edges.append(edge_ID)
        if data_max >= self.high_edge_thresh:
            self.ids_high_edges.append(edge_ID)
        if edge.is_breaking():
            self.ids_breaking_edges.append(edge_ID)
        else:
            self.ids_non_breaking_edges.append(edge_ID)

    def plot_image_with_track(self, data, edgeID):
        ax = plotting_interface.plot_surf_time_range(self.t, self.x, data)
        self.plot_specific_tracks([edgeID], ax=ax)
        return ax

    def stop_tracking_all(self):
        for edge_ID in self.active_edges.keys():
            self.stop_tracking(edge_ID)

    def get_all_edges(self):
        '''
        Return a list of edges for each time step where edges were tracked
        '''
        return self.edge_location_collector

    def get_active_edge_location_indices(self):
        return self.active_edge_location_indices
    
    def get_edge_dict(self):
        return self.edges

    def get_ids_long_edges(self):
        return np.array(self.ids_long_edges)

    def get_ids_high_edges(self):
        return np.array(self.ids_high_edges)

    def get_ids_breaking_edges(self):
        return np.array(self.ids_breaking_edges)

    def get_ids_non_breaking_edges(self):
        return np.array(self.ids_non_breaking_edges)

    def get_specific_tracks(self, id_list_of_interest):
        x_list = []
        t_list = []
        for edge_ID in id_list_of_interest:
            edge = self.edges[edge_ID]
            this_t, this_x = edge.get_track()
            x_list.append(this_x)
            t_list.append(this_t)
        return x_list, t_list

    def get_all_tracks(self):
        return self.get_specific_tracks(self.edges.keys())

    def get_high_tracks(self):
        return self.get_specific_tracks(self.ids_high_edges)
    
    def get_long_tracks(self):
        return self.get_specific_tracks(self.ids_long_edges)

    def get_breaking_tracks(self):
        return self.get_specific_tracks(self.ids_breaking_edges)

    def plot_specific_tracks(self, id_list_of_interest, ax):
        if ax is None:
            fig, ax = plt.subplots()
        x_list, t_list = self.get_specific_tracks(id_list_of_interest)
        for i in range(0, len(x_list)):
            ax.plot(t_list[i], x_list[i])

    def plot_all_tracks(self, ax=None):
        self.plot_specific_tracks(self.edges.keys(), ax)

    def plot_high_tracks(self, ax=None):
        self.plot_specific_tracks(self.ids_high_edges, ax)

    def plot_long_tracks(self, ax=None):
        print('\n\n\n\n',self.ids_long_edges,'\n\n\n\n' )
        self.plot_specific_tracks(self.ids_long_edges, ax)

    def plot_breaking_tracks(self, ax=None):
        self.plot_specific_tracks(self.ids_breaking_edges, ax)

    def plot_evolution_of_specific_tracks(self, data, label, id_list_of_interest, N=None, x_extent=70, dt_plot=1.0, ax_list=None, cm_name='Blues', show_tracks=False, interpolate=False):
        '''
        Plots the evolution of specific tracks
        
        Parameters:
        ----------
                    input
                            data                    2d array
                                                    data to be plotted
                            label                   string
                                                    label for y-axis
                            id_list_of_interest     list
                                                    edge ids to be plotted (one figure for each)
                            N                       int/None
                                                    if not None: limits the edges plotted from the given list to the provided number
                            x_extent         float
                                                    extent of surrounding to be plotted around edge, default:70
                            dt_plot                 float
                                                    step size for plotting in time, default: 1
                            ax_list                 list
                                                    list of axis, one for each of the ids that should be plotted
                            cm_name                 string
                                                    colormap name, default: 'Blues'
                            show_tracks             bool
                                                    for each id of interest plot data with given track, default:false
                            interpolate     bool
                                            interpolate data before plotting; default:False
                    output
                            out_ax_list             list
                                                    list of axis of plots
        '''
        if N is None or N>len(id_list_of_interest):
            N=len(id_list_of_interest)
        out_ax_list = []
        for i in range(0, N):
            this_edge = self.edges[id_list_of_interest[i]]
            if ax_list is None:
                ax = None
            else:
                ax = ax_list[i]
            if show_tracks:
                ax_tracks = self.plot_image_with_track(data, id_list_of_interest[i])
                ax_tracks.set_title('ID {0:d}'.format(id_list_of_interest[i]))
                
            ax = this_edge.plot_track(self.x, self.t, data, label, x_extent=x_extent, dt_plot=dt_plot, cm_name=cm_name, ax=ax, interpolate=interpolate)
            out_ax_list.append(ax)
        return out_ax_list


    def plot_evolution_of_breaking_tracks(self, data, label, id_list_of_interest=None, N=None, x_extent=70, ax_list=None, cm_name='Blues', dt_plot=1, show_tracks=False):
        if id_list_of_interest is None:
            id_list_of_interest = self.ids_breaking_edges
        else:
            id_list_of_interest = np.array(self.ids_breaking_edges)[id_list_of_interest]
        return self.plot_evolution_of_specific_tracks(data, label, id_list_of_interest, N=N, x_extent=x_extent, ax_list=ax_list, cm_name=cm_name, dt_plot=dt_plot, show_tracks=show_tracks)

    def plot_specific_tracks_and_mark_breaking(self, data, mask, label, id_list_of_interest=None, N=None, x_extent=50, dt_plot=1., cm_name='Blues', ax_list=None, show_tracks=False, interpolate=False):
        '''
        Plots the evolution of specific tracks
        
        Parameters:
        ----------
                    input
                            data                    2d array
                                                    data to be plotted
                            mask                    2d array
                                                    data to be plotted
                            label                   string
                                                    label for y-axis
                            id_list_of_interest     list
                                                    edge ids to be plotted (one figure for each), default: None (=all breaking tracks)
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
                            show_tracks             bool
                                                    for each id of interest plot data with given track, default:false
                            interpolate             bool
                                                    interpolate data before plotting; default:False
                    output
                            out_ax_list             list
                                                    list of axis of plots
        '''
        if id_list_of_interest is None:
            id_list_of_interest = self.ids_breaking_edges
        if N is None or N>len(id_list_of_interest):
            N=len(id_list_of_interest)
        out_ax_list = []
        for i in range(0, N):
            this_edge = self.edges[id_list_of_interest[i]]
            if ax_list is None:
                ax = None
            else:
                ax = ax_list[i]
            if show_tracks:
                ax_tracks = self.plot_image_with_track(data, id_list_of_interest[i])
                ax_tracks.set_title('ID {0:d}'.format(id_list_of_interest[i]))
            ax = this_edge.plot_track_and_mark_breaking(self.x, self.t, data, mask, label, x_extent=x_extent, dt_plot=dt_plot, cm_name=cm_name, ax=ax, interpolate=interpolate)
            out_ax_list.append(ax)
        return out_ax_list

    def plot_breaking_tracks_and_mark_breaking(self, data, mask, label, id_list_of_interest=None, N=None, x_extent=70, dt_plot=1.):
        if id_list_of_interest ==None:
            ids = self.ids_breaking_edges
        else:
            ids = np.array(self.ids_breaking_edges)[id_list_of_interest]
        return self.plot_specific_tracks_and_mark_breaking(data, mask, label, ids, N, x_extent=x_extent, dt_plot=dt_plot)  

    def get_indices_before_and_at_breaking(self, mask, id_list_breaking_tracks):
        '''
        return indices of edges just before breaking

        Parameters:
        -----------
            input:
                    id_list_breaking_tracks         int list
                                                    indices of tracks that include breaking
            output:
                    t_inds                          int list
                                                    t-indices to define before breaking
                    x_inds                          int list
                                                    x-indices to define before breaking
        '''
        t_inds_before = []
        x_inds_before = []
        t_inds_at = []
        x_inds_at = []
        for i in range(0, len(id_list_breaking_tracks)):
            this_edge = self.edges[id_list_breaking_tracks[i]]
            ti_before, xi_before, ti_at, xi_at = this_edge.get_indices_before_and_at_breaking(mask, self.x[0], self.t[0])
            if not ti_before is None:
                for i in range(0, len(ti_before)):
                    t_inds_before.append(ti_before[i])
                    x_inds_before.append(xi_before[i])
                    t_inds_at.append(ti_at[i])
                    x_inds_at.append(xi_at[i])
        return t_inds_before, x_inds_before, t_inds_at, x_inds_at


    def get_indices_first_breaking(self, mask, id_list_breaking_tracks):
        '''
        return indices of edges just first breaking

        Parameters:
        -----------
            input:
                    id_list_breaking_tracks         int list
                                                    indices of tracks that include breaking
            output:
                    t_inds                          int list
                                                    t-indices to define first breaking
                    x_inds                          int list
                                                    x-indices to define first breaking
        '''
        t_inds = []
        x_inds = []
        for i in range(0, len(id_list_breaking_tracks)):
            this_edge = self.edges[id_list_breaking_tracks[i]]
            ti, xi = this_edge.get_indices_first_breaking(mask, x0=self.x[0], t0=self.t[0])
            if not ti is None:
                t_inds.append(ti)
                x_inds.append(xi)
        return t_inds, x_inds

    def plot_envelopes(self, data, mask, id_list_of_interest, N=None, extent=200, ax_list=None, show_tracks=False):
        '''
        Plot upper envelope and two lower envelopes

        Parameters:
        -----------
                    input
                            mask                float array
                                                breaking mask
                            id_list_of_interest int list     
                                                list of IDs of Tracks that should be plotted
                            N                   int, optional   
                                                number of cases to be plotted
                            extent              float
                                                extent around edge for searching for the peaks/troughs
                            ax_list             list of axes 
                                                optional
                            show_tracks         bool
                                                shows an image of the data with the current track marked

                    output  
                            out_ax_list         list of axes object
                                                contains one axes object for each plot that was generated
        '''
        if id_list_of_interest is None:
            id_list_of_interest = self.ids_breaking_edges
        if N is None or N>len(id_list_of_interest):
            N=len(id_list_of_interest)
        out_ax_list = []
        for i in range(0, N):
            this_edge = self.edges[id_list_of_interest[i]]
            if ax_list is None:
                fig, ax = plt.subplots()
            else:
                ax = ax_list[i]
            if show_tracks:
                ax_tracks = self.plot_image_with_track(data, id_list_of_interest[i])
                ax_tracks.set_title('ID {0:d}'.format(id_list_of_interest[i]))
            #ax = this_edge.plot_track_and_mark_breaking(self.x, self.t, data, mask, label, x_extent=x_extent, dt_plot=dt_plot, cm_name=cm_name, ax=ax)
            #x_pos, peaks = this_edge.get_peaks(self.x, self.t, data, x_extent=extent)
            x_pos_troughs1, troughs1, x_pos_peaks, peaks, x_pos_troughs2, troughs2, track_mask= this_edge.get_trough_peak_trough(self.x, self.t, data, mask, x_extent=extent)
            ax.plot(x_pos_peaks, peaks)
            ax.plot(x_pos_peaks, np.ma.masked_array(peaks, 1-track_mask), 'rx')
            ax.plot(x_pos_troughs1, troughs1)
            ax.plot(x_pos_troughs2, troughs2)
            #ax.plot(x_pos_peaks, track_mask)
            out_ax_list.append(ax)
        return out_ax_list


def get_EdgeTracker(x, t, data, mask, max_edge_dist, sign, cmax=15, filter_input=True, high_edge_thresh=3.0, long_edge_thresh=200):
    '''
    Creates and instance of edge Tracker and tracks all edges and returns the instance

    Parameters:
    -----------
        input:
                x       1d array 
                        x axis
                t       1d array 
                        t axis
                data    2d array
                        data for searching edge in 2d (t,x)
                mask     2d array
                        breaking mask
                max_edge_dist float
                        maximum distance from edge to wave breaking along x-axis
                sign    signed int
                        1 for Micha, -1 for Pato        
                cmax    float
                        maximum crest speed
                filter_input bool
                        switch for preprocessing inputdata
    '''
    if filter_input:
        #input = (convolutional_filters.apply_Gaussian_blur(data))
        input = gaussian(data, sigma=3.0)
        data = sign*np.gradient(convolutional_filters.apply_edge_detection(input), axis=1)
    pt = EdgeTracker(x, t, data[0,:], max_edge_dist, mask0=mask[0,:], cmax=cmax, high_edge_thresh=high_edge_thresh, long_edge_thresh=long_edge_thresh)
    for i in range(1, len(t)):
        pt.track_edges(t[i], data[i,:], mask[i,:])
    pt.stop_tracking_all()
    return pt