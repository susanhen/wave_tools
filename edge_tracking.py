import numpy as np
from wave_tools import find_peaks, fft_interface, grouping
from help_tools import plotting_interface, convolutional_filters
import matplotlib.pyplot as plt

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

    def is_breaking(self):
        '''
        checks if wave within is breaking close to the edge a long the track
        '''
        return self.breaking

    def plot_track(self, x, t, data, label, x_extent=70, dt_plot=1., cm_name='Blues', ax=None):
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
            ax.plot(x[start_ind:end_ind+1], data[t_ind[i], start_ind:end_ind+1], color=colors[i])
            ax.plot(x[x_ind[i]], data[t_ind[i], x_ind[i]], 'x', color=colors[i])
        ax.set_ylabel(label)
        return ax

    def plot_track_and_mark_breaking(self, x, t, data, mask, label, x_extent=70, dt_plot=1., cm_name='Blues', ax=None):
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
            ax.plot(x[start_ind:end_ind+1], data[t_ind[i], start_ind:end_ind+1], color=colors[i])
            # If there is breaking happening in this time step in the observed interval
            if sum(mask[t_ind[i], start_ind:end_ind+1])>0:
                #first_breaking_ind = start_ind + np.argwhere(mask[t_ind[i], start_ind:end_ind+1]==1)#[-1]
                ax.plot(x[x_ind[i]], data[t_ind[i], x_ind[i]], 'rx')#, color=colors[i])
        ax.set_ylabel(label)
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
        x_list, t_list = self.get_specific_tracks(id_list_of_interest)
        for i in range(0, len(x_list)):
            plotting_interface.plot(t_list[i], x_list[i], ax=ax)

    def plot_all_tracks(self, ax=None):
        self.plot_specific_tracks(self.edges.keys(), ax)

    def plot_high_tracks(self, ax=None):
        self.plot_specific_tracks(self.ids_high_edges, ax)

    def plot_long_tracks(self, ax=None):
        self.plot_specific_tracks(self.ids_long_edges, ax)

    def plot_breaking_tracks(self, ax=None):
        self.plot_specific_tracks(self.ids_breaking_edges, ax)

    def plot_evolution_of_specific_tracks(self, data, label, id_list_of_interest, N=None, x_extent=70, dt_plot=1.0, ax_list=None, cm_name='Blues', show_tracks=False):
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
                
            ax = this_edge.plot_track(self.x, self.t, data, label, x_extent=x_extent, dt_plot=dt_plot, cm_name=cm_name, ax=ax)
            out_ax_list.append(ax)
        return out_ax_list


    def plot_evolution_of_breaking_tracks(self, data, label, id_list_of_interest=None, N=None, x_extent=70, ax_list=None, cm_name='Blues', dt_plot=1, show_tracks=False):
        if id_list_of_interest is None:
            id_list_of_interest = self.ids_breaking_edges
        else:
            id_list_of_interest = np.array(self.ids_breaking_edges)[id_list_of_interest]
        return self.plot_evolution_of_specific_tracks(data, label, id_list_of_interest, N=N, x_extent=x_extent, ax_list=ax_list, cm_name=cm_name, dt_plot=dt_plot, show_tracks=show_tracks)

    def plot_specific_tracks_and_mark_breaking(self, data, mask, label, id_list_of_interest=None, N=None, x_extent=50, dt_plot=1., cm_name='Blues', ax_list=None, show_tracks=False):
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
            ax = this_edge.plot_track_and_mark_breaking(self.x, self.t, data, mask, label, x_extent=x_extent, dt_plot=dt_plot, cm_name=cm_name, ax=ax)
            out_ax_list.append(ax)
        return out_ax_list

    def plot_breaking_tracks_and_mark_breaking(self, data, mask, label, id_list_of_interest=None, N=None, x_extent=70, dt_plot=1.):
        if id_list_of_interest ==None:
            ids = self.ids_breaking_edges
        else:
            ids = np.array(self.ids_breaking_edges)[id_list_of_interest]
        return self.plot_specific_tracks_and_mark_breaking(data, mask, label, ids, N, x_extent=x_extent, dt_plot=dt_plot)            


def get_EdgeTracker(x, t, data, mask, max_edge_dist, cmax=15, filter_input=True):
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
                cmax    float
                        maximum crest speed
                filter_input bool
                        switch for preprocessing inputdata
    '''
    if filter_input:
        #input = (convolutional_filters.apply_Gaussian_blur(data))
        input = gaussian(data, sigma=1.0)
        data = np.gradient(convolutional_filters.apply_edge_detection(input), axis=1)
    pt = EdgeTracker(x, t, data[0,:], max_edge_dist, mask0=mask[0,:], cmax=cmax)
    for i in range(1, len(t)):
        pt.track_edges(t[i], data[i,:], mask[i,:])
    pt.stop_tracking_all()
    return pt