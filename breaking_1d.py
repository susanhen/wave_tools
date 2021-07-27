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

def breaking_tracking(surf, L, T):
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

msurf, pt = breaking_tracking(bsurf, 10, 10)

tp, xp = 0, 0
for i in range(0, np.size(bsurf.t)):
    for j in range(0, np.size(bsurf.x)):
        if msurf[i, j] == 1:
            tp, xp = np.append(tp, bsurf.t[i]), np.append(xp, bsurf.x[j])

plotting_interface.plot_3d_as_2d(bsurf.t, bsurf.x, bsurf.eta)
plt.scatter(tp[1:], xp[1:], s=10)

plt.show()
