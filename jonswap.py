
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
        H       float:  water depth
        gamma   float:  defines width of spectrum 3.3 per default
    returns:
    --------
        jonswap array      
    '''
    if k_cut==None:
        k_cut = k[-1]
    g = 9.81
    w = np.sqrt(g*abs(k) * np.tanh(abs(k)*h))
    jonny_w = jonswap(w, wp, Hs, gamma)
    
    A = g*np.tanh(k*h)
    B = np.where(np.cosh(k*h)>10**6, k*0, g*h*k*(1./np.cosh(k*h))**2)
    C = 2*np.sqrt(9.81*k*np.tanh(k*h))
    with np.errstate(divide='ignore'):
        dw_dk = np.where(C>0, (A + B)/C, 1)
    jonny = jonny_w * dw_dk
    jonny = np.where(k<k_cut, jonny, 0)
    jonny[0] = 0
    return jonny

def jonswap_k_pavel(k, kp, Hs, gamma):
    if k[0] == 0:
        kk = k[1:]
    else:
        kk = k
    s = np.where(kk<=kp, 0.07, 0.09)**2
    S = (1/(2*(kk**3)))*np.exp((-5/4)*(kp/kk)**2) * gamma**(np.exp(-((np.sqrt(kk)-np.sqrt(kp))**2)/(2*s*kp)))
    S *=(Hs/(4*np.sqrt(np.trapz(S,kk))))**2
    if k[0]==0:
        return np.block([0, S])
    else:
        return S

if __name__=='__main__':
    import pylab as plt
    # generate J(w)
    w = np.linspace(0.03,1.8,1000)
    g = 9.81
    wp = 0.8
    gamma = 3.3
    Hs = 3.
    plt.figure()
    h = 100
    for gamma in [1., 1.8, 2.5, 3.3]:
        ji = jonswap(w,wp, Hs, gamma)
        print('Hs ( S(w)) = ', np.sqrt(sum(ji*np.gradient(w)))*4)
        plt.plot(w, ji)
        
    # generate J(k)        
    #k = linspace(0.0001, 0.2, 1000) 
    k = np.linspace(0, 0.2, 1000)

    plt.figure()
    for gamma in [1., 1.8, 2.5, 3.3]:
        ji = jonswap_k(k,wp, Hs, h, gamma)
        print('Hs ( S(k)) = ', np.sqrt(sum(ji*np.gradient(k)))*4)
        plt.plot(k, ji)     
        ji = jonswap(w,wp, Hs, gamma)
        plt.plot(w**2/9.81, 0.5*ji*np.sqrt(k/9.81))
        
    # generate example with surface elevation
    from numpy import pi, cos, outer, sum, var
    from scipy import stats
    k = np.linspace(0.0000001, 1.0, 66)
    dk = k[-1]-k[-2]
    ji = jonswap_k(k,wp, Hs, h, 5., 0.15)
    x = np.linspace(0,2*pi/dk, 2*len(k))
    
    phi = stats.uniform(scale=2*pi).rvs(len(k))-pi
    eta = 2*sum( np.sqrt(0.5*ji*dk)*cos(outer(np.ones(len(x)), k)*outer(x, np.ones(len(k)))+outer(np.ones(len(x)), phi)), axis=1)
    from numpy import fft, zeros, flipud, conjugate
    eta2_coeffs = zeros(2*len(k), dtype=complex)
    eta2_coeffs[len(k)+1:] = (np.sqrt(0.5*ji*dk)*len(k) * np.exp(1j*phi))[1:]
    eta2_coeffs[1:len(k)] = flipud(conjugate(eta2_coeffs[len(k)+1:]))
    eta2 = 2*fft.ifft(fft.ifftshift(eta2_coeffs)).real
    plt.figure()
    plt.plot(x,eta)
    plt.plot(x,eta2)
    print(4*np.sqrt(var(eta)))
    plt.show()
		

