from types import DynamicClassAttribute
import numpy as np
from numpy.lib import scimath as SM
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.interpolate import interp1d
from scipy.integrate import dblquad, quad, simps
from scipy import integrate
from matplotlib import cm
from help_tools import plotting_interface
from wave_tools import surface_core
from wave_tools import peak_tracking
import numpy.ma as ma
import math

bsurf = surface_core.spacetempSurface.surface_from_file('surfprofile')
bsurf.load_velocity('velprofile')

def breaking_tracking_old(surf, L, T):
    pt = peak_tracking.get_PeakTracker(surf.x, surf.t, surf.eta, surf.vel)
    pt.breaking_tracker()
    msurf = np.zeros((np.size(surf.t), np.size(surf.x)))
    xind = int(math.ceil(np.round(L/surf.dx)/2.)*2)
    tind = int(np.round(T/surf.dt))
    for i in range(0, pt.Nb):
        tloc = pt.bindex[i,0]
        xloc = pt.bindex[i,1]
        dis = 0
        speed = np.abs(pt.pc[i+1])
        for j in range(0, tind):
            if tloc + j >= np.size(surf.t):
                break
            for k in range(int(-xind/2), int(xind/2)):
                if xloc - k < 0:
                    break
                msurf[tloc+j, xloc-k] = 1                   
            dis += surf.dt*(speed)
            while dis >= surf.dx:
                xloc -= 1
                dis -= surf.dx
    return msurf, pt


def breaking_tracking(surf, peakTracker, L, T):
    mask = np.zeros(surf.eta.shape)
    peak_dict = peakTracker.get_peak_dict()    
    ids_breaking_peaks = peakTracker.get_ids_breaking_peaks()
    dt = peakTracker.dt
    dx = peakTracker.dx
    Nt = peakTracker.Nt
    x_min = peakTracker.x[0]
    N_Tb = int(T/dt) # number of points in breaking region in time
    for id in ids_breaking_peaks:
        peak = peak_dict[id]
        t0_ind = peak.get_breaking_start_ind_t()
        x0 = peak.get_breaking_start_x()
        cb = peak.cb
        #N_L = int(L/(np.abs(c)*dt))  # number of points in breaking region in space
        N_L = int(L/dx)
        xb = x0 + np.arange(0, N_L) * cb*dt
        boundary_mask = np.where(xb>=0, 0, 1)
        xb = ma.masked_array(xb, mask=boundary_mask).compressed()
        xb_ind = ((xb-x_min)/dx).astype('int')
        for i in range(0, N_L):
            t_end_ind = np.min([t0_ind+i+N_Tb, Nt])
            mask[t0_ind+i:t_end_ind, xb_ind[i]] = 1.0
    return mask





'''
msurf, pt = breaking_tracking(bsurf, 10, 10)

tp, xp = 0, 0
for i in range(0, np.size(bsurf.t)):
    for j in range(0, np.size(bsurf.x)):
        if msurf[i, j] == 1:
            tp, xp = np.append(tp, bsurf.t[i]), np.append(xp, bsurf.x[j])

plotting_interface.plot_3d_as_2d(bsurf.t, bsurf.x, bsurf.eta)
plt.scatter(tp[1:], xp[1:], s=10)

plt.show()
'''
