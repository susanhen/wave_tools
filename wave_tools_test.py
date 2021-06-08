#!/usr/bin/env python3

import unittest
import numpy as np
from wave_tools import surface_core
from help_tools import convolution

#TODO make sure all dimensions are tested!

class SeaSurface(unittest.TestCase):

    def setUp(self):
        self.name = r'$\eta$'
        self.Nx = 64#1024
        self.Ny = 64#2*1024
        self.x = np.linspace(-250, 250, self.Nx)
        self.y = np.linspace(0, 500, self.Ny)
        self.kx = 0.0#66
        self.ky = 0.066
        self.k = np.sqrt(self.kx**2 + self.ky**2)
        grid_cut_off = [0.15, 0.15]
        
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')
        self.eta = 5*np.sin(self.kx*self.xx + self.ky*self.yy)
        self.surf2d = surface_core.Surface(self.name, self.eta, [self.x,self.y])
        self.spec2d = self.surf2d.define_SpectralAnalysis(grid_cut_off)
        
        #from wave_tools import fft_interface
        #kx, ky, eta_fft = fft_interface.physical2spectral(self.eta, [self.x, self.y])
        #tmp_x, tmp_y, deta_dx = fft_interface.spectral2physical(eta_fft, [kx, ky])
                            
        '''
        import pylab as plt
        plt.figure()
        plt.imshow(self.eta)
        plt.figure()
        plt.imshow(deta_dx)
        plt.figure()
        plt.imshow(self.surf2d.etaND.eta)
        plt.show()
        '''
        
    def test_inversion(self):
        surf2d_inversion = self.spec2d.invert('inversion_test', grid_offset=[self.x[0], self.y[0]])
        for i in range(0, self.Nx):            
            for j in range(0, self.Ny):
                self.assertAlmostEqual(surf2d_inversion.etaND.eta[i,j], self.eta[i,j])
        for i in range(0, self.Nx):
            self.assertAlmostEqual(self.x[i], surf2d_inversion.etaND.x_grid[i])        
        for i in range(0, self.Ny):
            self.assertAlmostEqual(self.y[i], surf2d_inversion.etaND.y_grid[i])
            
        
    def test_name(self):
        self.assertEqual(self.name, self.surf2d.get_name())
    
    def test_1dconvolution(self, plot_it=False):
        N = len(self.x)
        mat1 = np.sin(self.kx*self.x)
        mat2 = self.x**2
        mat2 /= np.max(mat2)
        Res1 = np.fft.fftshift(np.fft.fft(mat1 * mat2))/N
        mat1_hat = np.fft.fftshift(np.fft.fft(mat1))/N
        mat2_hat = np.fft.fftshift(np.fft.fft(mat2))/N
        Res2 = np.convolve(mat1_hat, mat2_hat)[self.Nx//2:self.Nx//2+self.Nx]
        if plot_it:
            import pylab as plt
            plt.subplot(2,1,1)
            plt.plot(Res1.real)
            plt.plot(Res2.real)
            plt.subplot(2,1,2)
            plt.plot(Res1.imag)
            plt.plot(Res2.imag)
            plt.show()
        '''
        for i in range(0, self.Nx):
            self.assertAlmostEqual(Res1[i], Res2[i])
        '''
        
    
    def test_2dconvolution(self, plot_it=False):
        mat2 = self.xx + self.yy
        mat2 /= np.max(mat2)
        Res1 = np.fft.fftshift(np.fft.fft2(self.eta*mat2))/(self.Nx*self.Ny)
        F_mat1 = np.fft.fftshift(np.fft.fft2(self.eta))/(self.Nx*self.Ny)
        F_mat2 = np.fft.fftshift(np.fft.fft2(mat2))/(self.Nx*self.Ny)
        Res2 = convolution.convolve2d(F_mat1, F_mat2)
        if plot_it:
            import pylab as plt
            plt.imshow(np.abs(Res1))
            plt.figure()
            plt.imshow(np.abs(Res2))
            plt.show()
        '''
        for i in range(0, self.Nx):
            for j in range(0, self.Ny):
                self.assertAlmostEqual(Res1[i,j], Res2[i,j])
        '''        
        
    def test_shadowing(self):
        H = 45.0
        #self.surf2d.get_shadowing_mask('shad', H)
        
    def test_incidence_angle(self):
        HP_limits = [0.05, 0.05]
        grid_offset = [self.x[0], self.y[0]]
        grid_cut_off=[0.2, 0.2]
        
        '''
        surf_theta_l_10 = self.surf2d.get_local_incidence_surface(10, k_cut_off=grid_cut_off, approx=False)
        surf_theta_l_10_approx = self.surf2d.get_local_incidence_surface(10, k_cut_off=grid_cut_off, approx=True)
        surf_theta_l_45 = self.surf2d.get_local_incidence_surface(45, k_cut_off=grid_cut_off, approx=False)
        surf_theta_l_45_approx = self.surf2d.get_local_incidence_surface(45, k_cut_off=grid_cut_off, approx=True)
        spec_theta_l_10 = surf_theta_l_10.define_SpectralAnalysis()
        spec_theta_l_10.apply_HP_filter(HP_limits)
        surf_theta_l_10_filtered = spec_theta_l_10.invert('test1')
        
        spec_theta_l_45 = surf_theta_l_45.define_SpectralAnalysis()
        spec_theta_l_45.apply_HP_filter(HP_limits)
        surf_theta_l_45_filtered = spec_theta_l_45.invert('test2')
        
        spec_theta_l_45_approx = surf_theta_l_45_approx.define_SpectralAnalysis(grid_cut_off)
        #spec_theta_l_45_approx.apply_HP_filter(HP_limits)
        surf_theta_l_45_approx_filtered = spec_theta_l_45_approx.invert('test3')
        
        test1 = surf_theta_l_10.etaND.eta
        test1 -= np.mean(test1)
        test2 = surf_theta_l_10_approx.etaND.eta
        test2 -= np.mean(test2)
        rms_error = np.sqrt(np.mean((test1 - test2)**2))
        sigma = np.sqrt(np.var(test1))
        print('rms_error/sigma = ', rms_error/sigma)
        ''''''
        
        
        
        #'''
        '''
        import pylab as plt
        plt.figure()
        plt.imshow(surf_theta_l_10_filtered.etaND.eta*180/np.pi)
        plt.colorbar()
        plt.figure()
        plt.imshow(surf_theta_l_10_approx.etaND.eta*180/np.pi)
        plt.figure()
        plt.imshow(surf_theta_l_45_approx_filtered.etaND.eta*180/np.pi)
        plt.colorbar()
        plt.figure()
        plt.imshow(surf_theta_l_45_approx.etaND.eta*180/np.pi)
        plt.colorbar()
        plt.show()
        '''
        import pylab as plt
        '''
        surf_theta_l_10.plot_3d_surface()
        surf_theta_l_10_approx.plot_3d_surface()
        plt.show()
        
        # test
        kx_mesh, F_cos_theta, ky_mesh, F_sin_theta = spec_theta_l_45_approx.get_2d_MTF(grid_offset)
        #plt.figure()
        #plt.imshow(np.abs(1.0j*kx_mesh * self.spec2d.spectrumND.coeffs))
        should_work = convolution.convolve2d(F_cos_theta, (1.0j*kx_mesh * self.spec2d.spectrumND.coeffs))
        please_work = 1.0j*kx_mesh * convolution.convolve2d( F_cos_theta, (self.spec2d.spectrumND.coeffs))
        #magic = convolution.convolve2d( F_cos_theta, 1.0j*kx_mesh) * self.spec2d.spectrumND.coeffs
        #magic = convolution.convolve2d(1.0j*kx_mesh * F_cos_theta, self.spec2d.spectrumND.coeffs)
        magic = (F_cos_theta[self.Nx//2, self.Ny//2]*1.0j*kx_mesh + F_sin_theta[self.Nx//2, self.Ny//2]*1.0j*ky_mesh)* self.spec2d.spectrumND.coeffs
        #plt.figure()
        #plt.imshow(np.abs(should_work))
        #plt.figure()
        #plt.imshow(np.abs(please_work))
        plt.figure()
        plt.imshow(np.abs(magic))
        
        #MTF = spec_theta_l_45_approx.get_2d_MTF(self.x[0], self.y[0])
        #plt.imshow(np.abs(convolution.convolve2d(MTF, ( self.spec2d.spectrumND.coeffs))))
        #plt.figure()
        #plt.imshow(np.abs(MTF))
        plt.figure()
        spec_theta_l_45_approx.spectrumND.coeffs[self.Nx//2, self.Ny//2] = 0
        plt.imshow(np.abs(spec_theta_l_45_approx.spectrumND.coeffs))
        #plt.figure()
        #plt.plot(np.abs(spec_theta_l_45_approx.spectrumND.coeffs[:, self.Ny//2]))
        #plt.plot(np.abs(please_work)[:, self.Ny//2], '--')
        #plt.plot(np.abs(should_work)[:, self.Ny//2], '-.')
        #plt.plot(np.abs(1.0j*kx_mesh*self.spec2d.spectrumND.coeffs), '-')
        #plt.plot(np.abs(magic)[:, self.Ny//2], ':')
        
        plt.figure()
        plt.plot(np.abs(spec_theta_l_45_approx.spectrumND.coeffs[self.Nx//2, :]))
        plt.plot(np.abs(magic)[self.Nx//2, :], ':')
        '''
        plt.show()
        




if __name__ == '__main__':
    unittest.main()
