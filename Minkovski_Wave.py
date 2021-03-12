import pypaya2
import numpy as np
import pylab as plt
from wave_tools import ConstructWave
from help_tools import plotting_interface
'''
Tp = 10
Hs = 3.0
N = 256
x = np.linspace(0,2000, N)
q2 = []

smax_list = np.linspace(1, 20, 100)

for smax in smax_list:

    x, y, eta2d = ConstructWave.JonswapWave2D(x, Tp, Hs, smax)
    #plotting_interface.plot_3d_surface(x, y, eta2d)
    minkval = pypaya2.imt_for_image(eta2d, threshold=0)
    q2.append(minkval['q2'])

plt.figure()
plt.plot(smax_list, q2)
plt.ylabel(r'$q_2$')
plt.xlabel(r'$s_{\max}$')
plt.savefig('first_comparison.pdf', bbox_inches='tight')
plt.show()
'''

name = r'$\eta$'
Nx = 64#1024
Ny = 64#2*1024
x = np.linspace(-250, 250, Nx)
y = np.linspace(0, 500, Ny)
kx = 0.066
ky = 0.066
k = np.sqrt(kx**2 + ky**2)
grid_cut_off = [0.15, 0.15]

xx, yy = np.meshgrid(x, y, indexing='ij')
eta1 = np.sin(kx*xx + ky*yy + np.random.uniform()*2*np.pi)
eta1 += np.sin(-kx*xx - ky*yy+ np.random.uniform()*2*np.pi)

eta2 = np.sin(kx*xx + ky*yy+ np.random.uniform()*2*np.pi)
#surf2d = surface_core.Surface(name, eta, [x,y])
#plotting_interface.plot_3d_surface(x, y, eta1)
#plt.show()
minkval1 = pypaya2.imt_for_image(eta1, threshold=0)
minkval2 = pypaya2.imt_for_image(eta2, threshold=0)
print('q2 = ', minkval1['q2'] - minkval2['q2'])
print('q4 = ', minkval1['q4'] - minkval2['q4'])
print('q6 = ', minkval1['q6'] - minkval2['q6'])
print('q8 = ', minkval1['q8'] - minkval2['q8'])
print('q10 = ', minkval1['q10'] - minkval2['q10'])
print('q12 = ', minkval1['q12'] - minkval2['q12'])


print('q3 = ', minkval1['q3'] - minkval2['q3'])
print('q5 = ', minkval1['q5'] - minkval2['q5'])
print('q7 = ', minkval1['q7'] - minkval2['q7'])
print('q9 = ', minkval1['q9'] - minkval2['q9'])
print('q11 = ', minkval1['q11'] - minkval2['q11'])


