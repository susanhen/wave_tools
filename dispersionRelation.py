import numpy as np
#from skimage import measure
import scipy.interpolate
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from help_tools import plotting_interface


def calc_wavenumber_no_current(w, h, Niter_max=200, eps=10**(-6)): 
    '''
    calculate the wave number for the provided angular frequency or frequencies when there is no current.
    This method is faster than the one allowing current

    Parameters
    ----------
            input
                    w       float or array
                            angular frequency or vector of angular frequencies
                    h       foat
                            water depth
            output
                    k       float or array
                            wave number or vector of wave numbers
    '''  
    g=9.81     
    # treat case that w contains zeros
    w_old = w.copy()
    chosen_indices = np.argwhere(np.abs(w)>0)
    w = w_old[chosen_indices]
    # start
    ki = w**2/g
    wt = np.sqrt(g*ki*np.tanh(ki*h))
    count=0
    while np.max(np.abs(w - wt))>eps and count<Niter_max:
        
        ki = w**2/(g*np.tanh(ki*h))
        wt = np.sqrt(ki*g*np.tanh(ki*h))                         
        count = count + 1
        
    k = np.zeros(len(w_old))
    k[chosen_indices] = ki
    return k


def calc_wavenumber(w, h, Ueff, psi, Ntheta, Niter_max=200, eps=10**(-6)): 
    '''
    calculaate the wave number for the provided angular frequency or frequencies for the given effective current
    and the angle of the current psi.

    Parameters
    ----------
            input
                    w       float or array
                            angular frequency or vector of angular frequencies
                    h       foat
                            water depth
                    Ueff    float or array
                            effective current (if w is array should be array, one value for each w)
                    psi     float
                            angle of the current (if w is array psi shoul be array, one value for each w)
                    Ntheta  resolution of azimuth angle 
            output
                    k       float or array
                            wave number or vector of wave numbers
    ''' 
    # TODO: how to deal with different directions at different slices? array input not correctly supported!!!
    g=9.81     
    theta = np.linspace(0, 2*np.pi, Ntheta)
    chosen_indices = None
    if type(w) == float or type(w)==int or type(w)==np.float64:
        ww = np.outer(w, np.ones(Ntheta))
        th = theta
    else:
        # treat case that w contains zeros
        chosen_indices = np.argwhere(np.abs(w)>0)[:,0]
        ww = np.outer(w[chosen_indices], np.ones(Ntheta))
        th = np.outer(np.ones(len(chosen_indices)), theta)
    # start
    ki = ww**2/g
    phi = th - psi
    wwt = np.sqrt(g*ki*np.tanh(ki*h)) + ki*Ueff*np.cos(phi)
    
    count=0
    while np.max(np.abs(ww - wwt))>eps and count<Niter_max:
        
        ki = (ww-ki*Ueff*np.cos(phi))**2/(g*np.tanh(ki*h))
        wwt = np.sqrt(ki*g*np.tanh(ki*h))  + ki*Ueff*np.cos(phi)                       
        count = count + 1
    # map back to original size where 0 values could be included
    if type(w) == float or type(w)==int or type(w)==np.float64:
        kk = ki[0,:]
    else:
        kk = np.zeros((len(w), Ntheta))
        kk[chosen_indices,:] = ki
        th = np.outer(np.ones(len(w)), theta)
    return kk, th


    
def get_U_eff_at(k, z, U):   
    '''
    Calculate the effective current according to Stewart and Joy

    Parameters:
    -----------
    input
            k       float
                    wavenumber related to Bragg waves
            z       array
                    depth coordinates with velocity profile
            U       array
                    strength of the current for z
    '''
    #return 2*k*simpson(U*np.exp(2*k*z), z)
    return 2*k*np.sum(U*np.exp(2*k*z))*np.abs(z[1]-z[0])    

    
def get_dispersion_cone_at(at_w, h, z, U, psi, extent=None, polar=False):
    CS = plotting_interface.plot_disp_rel_at(at_w, h, z, U, psi, 'w', extent=extent)    
    kx_theor, ky_theor = CS.collections[0].get_paths()[0].vertices.T
    k_theor = np.sqrt(kx_theor**2 + ky_theor**2)
    th_theor = np.arctan2(ky_theor, kx_theor)
    # convert to only positive angles
    th_theor = np.where(th_theor<0, th_theor+(2*np.pi), th_theor)
    # sort so that angles go from lowest to highest
    inds = np.argsort(th_theor)
    th_theor = th_theor[inds]
    k_theor = k_theor[inds]
    if polar:
        return k_theor, th_theor
    else:
        return k_theor*np.cos(th_theor), k_theor*np.sin(th_theor)

def estimate_U_eff_psi_directly(at_w, kx, ky, spec, h, Ntheta, Umax=1, thresh_fact=0.1):
    kx_mesh, ky_mesh = np.meshgrid(kx, ky, indexing='ij')
    k_mesh = np.sqrt(kx_mesh**2 + ky_mesh**2)
    k_estim = at_w**2/9.81
    z = np.linspace(-10,0, 10)
    Umax_eff = get_U_eff_at(k_estim, z, Umax)
    dw_max = k_mesh*Umax_eff
    spec_filt0 = np.where(np.abs(np.sqrt(k_mesh*9.81*np.tanh(k_mesh*h))-at_w)<dw_max, spec, 0)  
    spec_filt = np.where(spec_filt0>thresh_fact*np.max(spec_filt0), spec_filt0, 0)
    kx_mesh, ky_mesh = np.meshgrid(kx, ky, indexing='ij')
    k_mesh = np.sqrt(kx_mesh**2 + ky_mesh**2)
    th_mesh = np.arctan2(ky_mesh, kx_mesh)
    th_mesh = np.where(th_mesh<0, th_mesh+2*np.pi, th_mesh)
    max_spec = np.max(np.abs(spec_filt))

    def minimize_distances(U_vec):
        Ux, Uy = U_vec
        Ueff = np.sqrt(Ux**2 + Uy**2)
        psi = np.arctan2(Uy, Ux) 
        k_now, theta = calc_wavenumber(at_w, h, Ueff, psi, Ntheta)
        f_k_now = scipy.interpolate.interp1d(theta, k_now)
        return np.sum(spec_filt * np.abs(k_mesh-f_k_now(th_mesh))**2)

    opt = minimize(minimize_distances, [0.9, 0.9])
    Ux, Uy = opt.x
    U = np.sqrt(Ux**2 + Uy**2)
    psi = np.arctan2(Uy, Ux)
    return U, psi


def estimate_U_eff_psi_at_w(w, k_cur, th_cur, h=1000, U0_vec=[0, 0]):
    w0_cur = np.sqrt(9.81*k_cur*np.tanh(k_cur*h))
    def disp_rel(U_vec):
        Ux, Uy = U_vec
        return np.sum(np.abs(w-w0_cur - k_cur*(np.cos(th_cur)*Ux + np.sin(th_cur)*Uy))**2)

    opt = minimize(disp_rel, U0_vec)
    Ux, Uy = opt.x
    U_eff = np.sqrt(Ux**2 + Uy**2)
    psi = np.arctan2(Uy, Ux)
    return U_eff, psi





if __name__=='__main__':
    import pylab as plt
    # test without current
    N = 20
    h = 20
    k = np.linspace(0.0, 0.2, N)
    w = np.sqrt(k*9.81*np.tanh(k*h))
    k_estim = calc_wavenumber_no_current(w, h)
    plt.plot(k, w)
    plt.plot(k_estim, w, '--')

    #test with current 
    h = 100
    Ueff = 0.1
    psi = 30/180*np.pi
    Ntheta = 100
    theta = np.linspace(0, 2*np.pi, Ntheta)
    kk, th = np.meshgrid(k, theta, indexing='ij')
    ww = np.sqrt(kk*9.81*np.tanh(kk*h)) + kk*Ueff*np.cos(th-psi)

    k_th, theta = calc_wavenumber(w, h, Ueff, psi, Ntheta)
    plt.figure()
    i = 0
    plt.plot(k, ww[:,i])
    plt.plot(k_th[:,i], w, '--')
    plt.figure()
    i = 30
    plt.plot(k, ww[:,i])
    plt.plot(k_th[:,i], w, '--')
    plt.figure()
    i = 70
    plt.plot(k, ww[:,i])
    plt.plot(k_th[:,i], w, '--')
    
    plt.show()