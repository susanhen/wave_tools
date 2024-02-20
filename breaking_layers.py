import numpy as np
import matplotlib.pyplot as plt


def cos_loc_inc_angle(eta, x, H):
    deta_dx = np.gradient(eta, x)
    r = np.abs(x)        
    n_norm = np.sqrt(deta_dx**2 + 1)
    b_norm = np.sqrt(r**2 + (H-eta)**2)
    cos_theta_l = (x*deta_dx + (H-eta))/(n_norm*b_norm)
    return cos_theta_l   

def up_down(start, mid, end, N):
    if np.mod(N,2)==0:
        return np.block([np.linspace(start,mid,N//2), np.linspace(mid,end,N//2)])
    else:
        return np.block([np.linspace(start,mid,N//2), np.linspace(mid,end,N//2+1)])

def assymmetric_sigmoid(lower_bound, upper_bound, N, factor=1):
    x = np.linspace(lower_bound, upper_bound, N)
    return 1./(1+np.exp(-factor*x))    

def plunging_breaker(N, amp, y0, N_layers_max=9):
    '''
    returns scaled layers
    '''
    layers = []
    sig_factor = 0.62
    amp_fact = 1
    layers.append(y0 + amp*(sig_factor*assymmetric_sigmoid(-4, 3, N, amp_fact*3) + up_down(0, 0.5, 0.4, N)))
    layers.append(y0 + amp*(sig_factor*assymmetric_sigmoid(-3.5,3.5,N, amp_fact*2.5) + up_down(0.05, 0.505, 0.42, N)))
    layers.append(y0 + amp*(sig_factor*assymmetric_sigmoid(-3,4,N, amp_fact*2) + up_down(0.1, 0.51, 0.44, N)))
    layers.append(y0 + amp*(sig_factor*assymmetric_sigmoid(-2.5,4.5,N, amp_fact*2) + up_down(0.15, 0.52, 0.46, N)))
    layers.append(y0 + amp*(sig_factor*assymmetric_sigmoid(-2,5,N, amp_fact*2) + up_down(0.16, 0.53, 0.48, N)))
    layers.append(y0 + amp*(sig_factor*assymmetric_sigmoid(-1.5,5.5,N, amp_fact*2) + up_down(0.17, 0.54, 0.5, N)))
    layers.append(y0 + amp*(sig_factor*assymmetric_sigmoid(-1,6,N, amp_fact*2) + up_down(0.17, 0.55, 0.52, N)))
    layers.append(y0 + amp*(sig_factor*assymmetric_sigmoid(-0.7,8,N, amp_fact*3) + up_down(0.19, 0.56, 0.54, N)))
    layers.append(y0 + amp*(sig_factor*assymmetric_sigmoid(-0.5,10,N, amp_fact*5) + up_down(0.22, 0.57, 0.56, N)))
    if N_layers_max>len(layers):
        print('Number of max layers extends number of defined layers!')
        return None
    return layers[:N_layers_max]

def accumulated_tilt_basis(x, amp, H, y0=0, N_layers_max=9, breaker_type='plunging', polarization='HH', plot_it=False, ax=None):
    '''
    Calculate accumulated basis for tilt modulation of a breaking wave based on multiple layers
    Parameters:
    -----------
                input:
                        x           float array
                                    x values for breaking region
                        amp         float
                                    amplitude of breaking region
                        H           float
                                    elevation of radar antenna
                        y0          float 
                                    lowest surface elevation of breaking wave
                        N_layers_max    int
                                    maximum number of breaking layers
                        breaker_type    string
                                    'plunging', no other type for now
                        polarization    string
                                    'HH' will give power of 2, VV power of 1
                        plot_it     bool    
                                    plotting every breaker with breaking layers
                        ax          axis; default None
                                    for plotting
    '''
    N = len(x)
    if breaker_type=='plunging':
        layers = plunging_breaker(N, amp, y0, N_layers_max)
    else:
        print('not decided')
        return None
    a_added = np.zeros(N)
    for i in range(0, len(layers)):
        if plot_it:
            if i ==0:
                label=r'$\mathrm{layers}~1-$'+'{0:d}'.format(N_layers_max)
            else:
                label=None
            ax.plot(x, layers[i], 'k-', linewidth=0.8, label=label)
        
        if polarization == 'VV':
            cos_theta_l = cos_loc_inc_angle(layers[i], x, H)
            theta_l = np.arccos(cos_theta_l)
            addition = 2./len(layers) * (1+np.sin(theta_l)**2)*cos_theta_l 
            addition = np.where(addition>0, addition, 0)
        else:
            addition = 2./len(layers) * cos_loc_inc_angle(layers[i], x, H)**2
            addition = np.where(addition>0, addition, 0)
        a_added += addition

        
    return a_added

    
    


if __name__=='__main__':
    N = 50
    H = 10
    amp = 1
    x = 500 + np.linspace(0, 10, N)
    y = amp*np.blackman(2*N)[:N]
    y0 = np.min(y)
    a0 = cos_loc_inc_angle(y, x, H)
    a_added = np.zeros(N)
    a_added1 = np.zeros(N)
    a_added2 = np.zeros(N)
    # define layers
    layers = plunging_breaker(N, amp, y0)
    # define cosine directly
    accum_tilt = accumulated_tilt_basis(x, amp, H, y0, polarization='VV')
    plt.plot(x,y)
    for i in range(0, len(layers)):
        plt.plot(x, layers[i], 'k-')
        ai = cos_loc_inc_angle(layers[i], x, H)
        a_added += 1./len(layers) * ai**2
        a_added1 += 1./len(layers) * ai
        a_added2 += 1./len(layers) * ai**4
    plt.plot(x, cos_loc_inc_angle(y, x, H), '--', label='cos(theta_l)')
    #plt.plot(x, a_added, label='pow2')
    #plt.plot(x, a_added1, label='pow1')
    #plt.plot(x, a_added2, label='pow4')
    plt.plot(x, accum_tilt, ':')
    plt.plot(x, a0**2, '--')
    plt.legend()
    plt.show()