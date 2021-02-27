import numpy as np

def jonswap(w, wp, Hs, gamma=3.3, w_cut=None):
    '''
    Evalutesthe 1d jonswap spectrum for a given frequency vector; only positive w
    Parameters:
    -----------
    input: 
    -----------
        w       array:  of frequencies for which the jonswap should be returned; positive!!!
        wp      float:  peak frequency
        Hs      float:  significant wave height
        gamma   float:  defines width of spectrum 3.3 per default
    returns:
    --------
        jonswap array      
    '''
    ww = w[1:]
    if w_cut == None:
        w_cut = w[-1]
    s = np.where(ww>wp, 0.09, 0.07)
    b = 5./4
    a = 0.0081
    r = np.exp(-(ww-wp)**2/(2*(wp*s)**2))
    c = np.where(ww>0, np.exp(-b*(wp/ww)**4), 0)
    g = 9.81
    jonny = np.zeros(len(w))
    jonny[1:] = np.where(c>10**(-10), a*g**2/ww**5 * c * gamma**r, 0)
    jonny = np.where(w<w_cut, jonny, 0)
    return jonny
    
def jonswap_k(k, wp, Hs, h, gamma=3.3, k_cut=None):  
    '''
    Evalutesthe 1d jonswap spectrum for a given wave number vector; only positive k
    Parameters:
    -----------
    input: 
    -----------
        k       array:  of wave numbers for which the jonswap should be returned; should start from 0!!
        wp      float:  peak frequency
        Hs      float:  significant wave height
        h       float:  water depth
        gamma   float:  defines width of spectrum 3.3 per default
    returns:
    --------
        jonswap array      
    '''
    if k_cut==None:
        k_cut = k[-1]
    g = 9.81
    w = np.sqrt(g*np.abs(k) * np.tanh(np.abs(k)*H))
    jonny_w = jonswap(w, wp, Hs, gamma)
    
    A = g*np.tanh(k_polar*h)
    B = np.where(np.cosh(k_polar*h)>10**6, k_polar*0, g*h*k_polar*(1./np.cosh(k_polar*h))**2)
    C = 2*np.sqrt(9.81*k_polar*np.tanh(k_polar*h))
    dw_dk = np.where(C>0, (A + B)/C, 1)
    
    '''
    if np.max(k*H)>10:
        #print('\n\nDeep water assumed in jonswap!\n\n')
        with np.errstate(divide='ignore'):
            Jac_factor = np.where(np.abs(w)>0.1, g/(2*w), 0)
        #Jac_factor = np.where(np.bitwise_and(w>0, k>0), 2*np.sqrt(k/g), 0)      
    else:
        Jac_factor = np.where(np.bitwise_and(w>0, k>0), g/(2*w)*(k*/(np.cosh(k*h))**2 + np.tanh(k*h)), 0)   #FIXME see above!!!
    '''
    jonny = jonny_w / dw_dk
    jonny = np.where(k<k_cut, jonny, 0)
    jonny[0] = 0
    return jonny

if __name__=='__main__':
    import pylab as plt
    # generate J(w)
    w = np.linspace(0.03,1.8,1000)
    g = 9.81
    wp = 0.8
    gamma = 3.3
    Hs = 3.
    plt.figure()
    H = 100
    for gamma in [1., 1.8, 2.5, 3.3]:
        ji = jonswap(w,wp, Hs, gamma)
        print('Hs ( S(w)) = ', np.sqrt(np.sum(ji*np.gradient(w)))*4)
        plt.plot(w, ji)
        
    # generate J(k)        
    #k = np.linspace(0.0001, 0.2, 1000) 
    k = np.linspace(0, 0.2, 1000)

    plt.figure()
    for gamma in [1., 1.8, 2.5, 3.3]:
        ji = jonswap_k(k,wp, Hs, H, gamma)
        print('Hs ( S(k)) = ', np.sqrt(np.sum(ji*np.gradient(k)))*4)
        plt.plot(k, ji)     
        ji = jonswap(w,wp, Hs, gamma)
        
    # generate example with surface elevation
    from numpy import pi, cos, outer, sum, var
    from scipy import stats
    k = np.linspace(0.0000001, 1.0, 66)
    dk = k[-1]-k[-2]
    ji = jonswap_k(k,wp, Hs, H, 5., 0.15)
    x = np.linspace(0,2*pi/dk, 2*len(k))
    
    phi = stats.uniform(scale=2*pi).rvs(len(k))-pi
    eta = 2*sum( np.sqrt(0.5*ji*dk)*np.cos(np.outer(np.ones(len(x)), k)*np.outer(x, np.ones(len(k)))+np.outer(np.ones(len(x)), phi)), axis=1)
    from numpy import fft, zeros, flipud, conjugate
    eta2_coeffs = np.zeros(2*len(k), dtype=complex)
    eta2_coeffs[len(k)+1:] = (np.sqrt(0.5*ji*dk)*len(k) * np.exp(1j*phi))[1:]
    eta2_coeffs[1:len(k)] = np.flipud(conjugate(eta2_coeffs[len(k)+1:]))
    eta2 = 2*np.fft.ifft(np.fft.ifftshift(eta2_coeffs)).real
    plt.figure()
    plt.plot(x,eta)
    plt.plot(x,eta2)
    print(4*np.sqrt(np.var(eta)))
    plt.show()
		

