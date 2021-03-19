import pypaya2
import numpy as np
import pylab as plt
from wave_tools import ConstructWave
from help_tools import plotting_interface
#'''
Hs = 2.0
q2 = []
q4 = []
q6 = []
q8 = []
q10 = []
q12 = []
dx = 7.5
dy = 7.5
x = np.arange(-250, 250, dx)
y = np.arange(500, 1000, dy)
Alpha = 0.023
gamma = 3.3
theta_mean = np.pi/2 - 10*np.pi/180
save_files = True


smax_list = np.linspace(1, 100, 100)

for smax in smax_list:
    print('processing for surface for smax {0:.0f} final smax will be {1:.0f}'.format(smax, smax_list[-1]))

    surf2d = ConstructWave.JonswapWave2D_Pavel(x, y, Hs, Alpha, gamma, theta_mean, smax)
    #surf2d.plot_3d_as_2d()

    #plotting_interface.plot_3d_surface(x, y, eta2d)
    minkval = pypaya2.imt_for_image(surf2d.eta, threshold=0)
    q2.append(minkval['q2'])
    q4.append(minkval['q4'])
    q6.append(minkval['q6'])
    q8.append(minkval['q8'])
    q10.append(minkval['q10'])
    q12.append(minkval['q12'])

plt.figure()
plt.plot(smax_list, q2)
plt.ylabel(r'$q_{2}$')
plt.xlabel(r'$s_{\max}$')
if save_files:
    plt.savefig('smax_q2.pdf', bbox_inches='tight')
plt.figure()
plt.plot(smax_list, q4)
plt.ylabel(r'$q_{4}$')
plt.xlabel(r'$s_{\max}$')
if save_files:
    plt.savefig('smax_q4.pdf', bbox_inches='tight')
plt.figure()
plt.plot(smax_list, q6)
plt.ylabel(r'$q_{6}$')
plt.xlabel(r'$s_{\max}$')
if save_files:
    plt.savefig('smax_q6.pdf', bbox_inches='tight')
plt.figure()
plt.plot(smax_list, q8)
plt.ylabel(r'$q_{8}$')
plt.xlabel(r'$s_{\max}$')
if save_files:
    plt.savefig('smax_q8.pdf', bbox_inches='tight')
plt.figure()
plt.plot(smax_list, q10)
plt.ylabel(r'$q_{10}$')
plt.xlabel(r'$s_{\max}$')
if save_files:
    plt.savefig('smax_q10.pdf', bbox_inches='tight')
plt.figure()
plt.plot(smax_list, q12)
plt.ylabel(r'$q_{12}$')
plt.xlabel(r'$s_{\max}$')
if save_files:
    plt.savefig('smax_q12.pdf', bbox_inches='tight')
plt.show()
#'''
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


'''