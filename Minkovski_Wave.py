import pypaya2
import numpy as np
import pylab as plt
from wave_tools import ConstructWave
from help_tools import plotting_interface

Tp = 10
Hs = 3.0
N = 256
x = np.linspace(0,2000, N)
q2 = []
smax_list = np.linspace(1, 75, 100)

for smax in smax_list:

    x, y, eta2d = ConstructWave.JonswapWave2D(x, Tp, Hs, smax)
    #plotting_interface.plot_3d_surface(x, y, eta2d)
    minkval = pypaya2.imt_for_image(eta2d, threshold=1.46)
    q2.append(minkval['q2'])

plt.figure()
plt.plot(smax_list, q2)
plt.ylabel(r'$q_2$')
plt.xlabel(r'$s_{\max}$')
plt.savefig('first_comparison.pdf', bbox_inches='tight')
plt.show()
    