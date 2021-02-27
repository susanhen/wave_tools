import numpy as np
import pylab as plt

import jonswap as j
import polarTransform
from scipy import stats
import imageio

N = 256
N_theta=50
k = np.linspace(0, 1.5, N)
gamma = 3.3
wp = 0.8
Hs = 3.0
h = 100. 
g = 9.81
w = np.sqrt(k*g*np.tanh(k*h))


#Mitsuyasu distribution based on codeine 2s A, normalization (integral should be one)

ji = j.jonswap(w, wp, Hs, h, gamma) 
theta_mean = 0.5*np.pi #TODO: works only correctly for angle of 90 deg. fix this by copying data from all parts and adding it to previous parts
theta = np.linspace(theta_mean - np.pi, theta_mean + np.pi, N_theta, endpoint=True)
theta_mean = theta[np.argmin(np.abs(theta-theta_mean))] # ensure that this value is on the 


k_all, theta_all = np.meshgrid(k, theta)
w_all, theta_all = np.meshgrid(w, theta)
ji_all, theta_all = np.meshgrid(ji, theta)


mu1 = 5
mu2 = -2.5
smax = 10 # windsea 10, swell short decay distance 25, swell long decay distance 75
mu = np.where(w_all<wp, mu1, mu2)
s = smax * (w_all/wp)**mu
# ds/dk = ds/dw dw/dk

H = (np.cos((theta_all-theta_mean)/2) )**s#(2*s)

print(s.shape, mu.shape, H.shape, theta_all.shape, np.max(s), np.min(s))
print('max s, max H ', np.max(s), np.max(H))
#H /= np.sum(H) # right scaling? Dk missing or not necessary? Should be part of the formula above?
#image =  np.array([k_all, theta_all, ji_all]).flatten().reshape(100, 20, 3)
#imageio.imwrite('test.jpg', image)
if h>10:
    dw_dk = g/(2*np.sqrt(g*k_all))
else:
    C1 =  np.tanh(h*k_all) + h * k_all * (np.cosh(h*k_all))**(-2)
    C2 = 2*np.sqrt(g*k_all*np.tanh(h*k_all))
    dw_dk = np.where(np.bitwise_and(w_all>0, k_all>0), g*C1/C2, 1)
    
print(ji_all.shape, H.shape, dw_dk.shape)
D_k_theta = ji_all*H #/ dw_dk
print(np.sum(H))
cart_image, settings = polarTransform.convertToCartesianImage(D_k_theta, imageSize=(2*N,2*N), initialAngle=theta[0], finalAngle=theta[-1])
print(settings)
plt.figure()
plt.imshow(cart_image)

phi = np.exp(1j*(stats.uniform(scale=2*np.pi).rvs((2*N,2*N))-np.pi))
upper_image=phi[N:,:]*cart_image[N:,:]

#total_image = np.block([[np.zeros(2*N)], [np.flipud(complex_cart_image[N+1:,:]).conjugate()], [complex_cart_image[N:,:]]])
#'''

lower_image = phi[:N,:]*cart_image[:N,:]
#lower_image = np.zeros((N,2*N), dtype=complex)
lower_image[1:,1:N] += np.flip(upper_image[1:,N+1:]).conjugate()
upper_image[1:,N+1:] = np.flip(lower_image[1:,1:N]).conjugate()

upper_image[0,1:N] += np.flip(upper_image[0,N+1:]).conjugate()
upper_image[0,N+1:] = np.flip(upper_image[0,1:N]).conjugate()

lower_image[1:,N+1:] += np.flip(upper_image[1:,1:N]).conjugate()
upper_image[1:,1:N] = np.flip(lower_image[1:,N+1:]).conjugate()

lower_image[1:N,N] += np.flip(upper_image[1:,N]).conjugate()
upper_image[1:,N] = np.flip(lower_image[1:N,N]).conjugate()

total_image=np.block([[lower_image],[upper_image]])


#'''
plt.figure()
plt.imshow((np.abs(total_image)**2), origin='lower')
#'''
eta2d = np.fft.ifft2(np.fft.ifftshift(total_image))
print('Hs out: ', 4*np.sqrt(np.var(eta2d)))
#imageio.imwrite('test_spread100_wp06_3.jpg', eta2d.real)
plt.figure()
plt.subplot(2,1,1)
plt.imshow(eta2d.real)
plt.subplot(2,1,2)
plt.imshow(eta2d.imag)

plt.show()


