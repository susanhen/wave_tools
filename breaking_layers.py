import numpy as np


def cos_loc_inc_angle(eta, x, H):
    deta_dx = np.gradient(eta, x)
    r = np.abs(x)        
    n_norm = np.sqrt(deta_dx**2 + 1)
    b_norm = np.sqrt(r**2 + (H-eta)**2)
    cos_theta_l = (x*deta_dx + (H-eta))/(n_norm*b_norm)
    return cos_theta_l   

def up_down(start, mid, end):
    return np.block([np.linspace(start,mid,N//2), np.linspace(mid,end,N//2)])

def assymmetric_sigmoid(lower_bound, upper_bound, N, factor=1):
    x = np.linspace(lower_bound, upper_bound, N)
    return 1./(1+np.exp(-factor*x))    

def plunging_breaker(N, amp, N_layers_max=9):
    '''
    returns scaled layers
    '''
    layers = []
    sig_factor = 0.62
    layers.append(amp*(sig_factor*assymmetric_sigmoid(-4, 3, N, 3) + up_down(0, 0.5, 0.4)))
    layers.append(amp*(sig_factor*assymmetric_sigmoid(-3.5,3.5,N, 2.5) + up_down(0.05, 0.505, 0.42)))
    layers.append(amp*(sig_factor*assymmetric_sigmoid(-3,4,N, 2) + up_down(0.1, 0.51, 0.44)))
    layers.append(amp*(sig_factor*assymmetric_sigmoid(-2.5,4.5,N, 2) + up_down(0.15, 0.52, 0.46)))
    layers.append(amp*(sig_factor*assymmetric_sigmoid(-2,5,N, 2) + up_down(0.16, 0.53, 0.48)))
    layers.append(amp*(sig_factor*assymmetric_sigmoid(-1.5,5.5,N, 2) + up_down(0.17, 0.54, 0.5)))
    layers.append(amp*(sig_factor*assymmetric_sigmoid(-1,6,N, 2) + up_down(0.17, 0.55, 0.52)))
    layers.append(amp*(sig_factor*assymmetric_sigmoid(-0.7,8,N, 3) + up_down(0.19, 0.56, 0.54)))
    layers.append(amp*(sig_factor*assymmetric_sigmoid(-0.5,10,N, 5) + up_down(0.22, 0.57, 0.56)))
    if N_layers_max>len(layers):
        print('Number of max layers extends number of defined layers!')
        return None
    return layers[:N_layers_max]

def accumulated_tilt_basis(N, amp, H, N_layers_max=9, breaker_type='plunging', polarisation='HH'):
    if polarisation == 'HH':
        power = 2
    else:
        power = 1
    if breaker_type=='plunging':
        layers = plunging_breaker(N, amp, N_layers_max)
    else:
        print('not decided')
        return None
    a_added = np.zeros(N)
    for i in range(0, len(layers)):
        a_added += 1./len(layers) * cos_loc_inc_angle(layers[i], x, H)**power
    return a_added

    
    


if __name__=='__main__':
    import matplotlib.pyplot as plt
    N = 50
    H = 10
    amp = 1
    x = 500 + np.linspace(0, 10, N)
    y = amp*np.blackman(2*N)[:N]
    a0 = cos_loc_inc_angle(y, x, H)
    a_added = np.zeros(N)
    # define layers
    layers = plunging_breaker(N, amp)
    # define cosine directly
    accum_tilt = accumulated_tilt_basis(N, amp, H)
    plt.plot(x,y)
    for i in range(0, len(layers)):
        plt.plot(x, layers[i])
        ai = cos_loc_inc_angle(layers[i], x, H)
        a_added += 1./len(layers) * ai**2
    plt.plot(x, a_added)
    plt.plot(x, accum_tilt, ':')
    plt.plot(x, a0**2, '--')
    plt.show()