import numpy as np
from skimage import measure


def calc_wavenumber_no_current(w, h, Niter_max=200, eps=10**(-6)):   
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
    g=9.81     
    theta = np.linspace(0, 2*np.pi, Ntheta)
    chosen_indices = None
    if type(w) == float:
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
    if type(w) == float:
        kk = ki
    else:
        kk = np.zeros((len(w), Ntheta))
        kk[chosen_indices,:] = ki
        th = np.outer(np.ones(len(w)), theta)
    return kk, th

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