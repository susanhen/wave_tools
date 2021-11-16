import numpy as np

def grid2k(grid):
    dx = abs(grid[-1]-grid[-2])    
    N = len(grid)
    kmin = -np.pi/dx
    dk = 2*np.pi/(dx*N)
    k = kmin + dk*np.arange(0,N)
    return k, dk
    
def k2grid(k):   
    dk = abs(k[-1]-k[-2])
    N = len(k)
    dx = 2*np.pi/(dk*N)
    x = np.arange(0,N)*dx
    return x, dk

  
def _physical2spectral1d(data, grid):
    k, dk = grid2k(grid)    
    N = len(data)
    fft_data = np.fft.fftshift(np.fft.fft(data))/np.sqrt(dk)/N
    return k, fft_data
    
def _physical2spectral2d(data, grid):
    kx, dkx = grid2k(grid[0])
    ky, dky = grid2k(grid[1])
    Nx, Ny = data.shape
    fft_data = np.fft.fftshift(np.fft.fft2(data))/np.sqrt(dkx*dky)/(Nx*Ny)
    return kx, ky, fft_data

def _physical2spectral3d(data, grid):
    kx, dkx = grid2k(grid[0])
    ky, dky = grid2k(grid[1])
    kz, dkz = grid2k(grid[2])
    Nx, Ny, Nz = data.shape
    fft_data = np.fft.fftshift(np.fft.fftn(data))/np.sqrt(dkx*dky*dkz)/(Nx*Ny*Nz)
    return kx, ky, kz, fft_data
    

def physical2spectral(data, grid):
    '''
    Pass from data and grid from physical domain to spectral domain
    Parameters:
    -----------
    input       
            data        array
                        physical data
            grid        array/list
                        defining the grid along each axis
                        if 1d array
                        if >2d list_of_arrays   
    output
            1d: k, fft_data
            2d: kx, ky, fft_data
            3d: kx, ky, kz, fft_data
            
    '''
    if len(data.shape)==1:
        if type(grid)==list:
            grid = grid[0]
        return _physical2spectral1d(data, grid)
    elif len(data.shape)==2:
        return _physical2spectral2d(data, grid)
    elif len(data.shape)==3:
        return _physical2spectral3d(data, grid)
    else:
        print('Error: data input only allowed up to 3d')
        return None
        
def _spectral2physical1d(coeffs, k_grid):
    x, dk = k2grid(k_grid)    
    N = len(coeffs)
    data = np.fft.ifft(np.fft.ifftshift(coeffs))*np.sqrt(dk)*N
    return x, data
    
def _spectral2physical2d(coeffs, k_grid):
    x, dkx = k2grid(k_grid[0])
    y, dky = k2grid(k_grid[1])
    Nx, Ny = coeffs.shape
    data = np.fft.ifft2(np.fft.ifftshift(coeffs))*np.sqrt(dkx*dky)*(Nx*Ny)
    return x, y, data.real

def _spectral2physical3d(coeffs, k_grid):
    x, dkx = k2grid(k_grid[0])
    y, dky = k2grid(k_grid[1])
    z, dkz = k2grid(k_grid[2])
    Nx, Ny, Nz = coeffs.shape
    data = np.fft.ifftn(np.fft.ifftshift(coeffs))*np.sqrt(dkx*dky*dkz)*(Nx*Ny*Nz)
    return x, y, z, data.real
    

def spectral2physical(coeffs, k_grid):
    '''
    Pass from spectral domain to physical domain
    Parameters:
    -----------
    input       
            coeffs      array
                        scaled fft_coeffs of data
            k_grid      array/list
                        k_grid
                        if 1d array
                        if >2d list_of_arrays   
    output
            1d: x, data
            2d: x, y, data
            3d: x, y, z, data
            
    '''
    if len(coeffs.shape)==1:
        return _spectral2physical1d(coeffs, k_grid)
    elif len(coeffs.shape)==2:
        return _spectral2physical2d(coeffs, k_grid)
    elif len(coeffs.shape)==3:
        return _spectral2physical3d(coeffs, k_grid)
    else:
        print('Error: data input only allowed up to 3d')
        return None
                
