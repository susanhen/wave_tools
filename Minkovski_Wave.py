import pypaya2
import numpy as np
import pylab as plt
from wave_tools import ConstructWave
from help_tools import plotting_interface
from statistic_tools import confidence
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
mu = 0.5
theta_mean = np.pi/2 - 10*np.pi/180
save_files = True
N_parameters = 20
N_cases = 10
smax_list = np.linspace(1, 100, N_parameters)


def define_container_dict(q_list, N_cases, N_parameters):
    container_dict = {}
    for q in q_list:
        container_dict[q] = confidence.Container((N_cases, N_parameters))
    return container_dict

def add_case_at_all_qs(container_dict, minkval, q_list, k):
    for q in q_list:
        container_dict[q].add_case_at(minkval[q], k)

def plot_all(container_dict, q_list, smax_list, q_labels, save_files):
    for q in q_list:
        plt.figure()
        plt.plot(smax_list, container_dict[q].mean(), 'r-')
        conf_lower, conf_upper = container_dict[q].conf_int(0.9)
        plt.plot(smax_list, conf_lower, 'k--')
        plt.plot(smax_list, conf_upper, 'k--')
        plt.ylabel(q_labels[q])
        plt.xlabel(r'$s_{\max}$')
        if save_files:
            plt.savefig('smax_mu+05_{0:s}.pdf'.format(q), bbox_inches='tight')

q_list = ['q2', 'q3', 'q4', 'q5', 'q6', 'q7','q8', 'q9', 'q10','q11', 'q12']
q_labels = {'q2':r'$q_{2}$', 'q3':r'$q_{3}$', 'q4':r'$q_{4}$', 'q5':r'$q_{5}$', 'q6':r'$q_{6}$', 'q7':r'$q_{7}$', 'q8':r'$q_{8}$', 'q9':r'$q_{9}$', 'q10':r'$q_{10}$', 'q11':r'$q_{11}$', 'q12':r'$q_{12}$'}
container_dict = define_container_dict(q_list, N_cases, N_parameters)


for k in range(0, N_parameters):
    smax = smax_list[k]
    print('processing for surface for smax {0:.0f} final smax will be {1:.0f}'.format(smax, smax_list[-1]))
    for i in range(0, N_cases):
        #surf2d = ConstructWave.JonswapWave2D_Pavel(x, y, Hs, Alpha, gamma, theta_mean, smax)
        surf2d = ConstructWave.JonswapWave2D_asymetric(x, y, Hs, Alpha, gamma, theta_mean, smax, mu)
        minkval = pypaya2.imt_for_image(surf2d.eta, threshold=0)
        add_case_at_all_qs(container_dict, minkval, q_list, k)


plot_all(container_dict, q_list, smax_list, q_labels, save_files)
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