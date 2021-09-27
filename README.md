# wave_tools

A framework for working with ocean-wave modelling. Including:

- Construction of wave fields
- Tracking peaks
- Tracking edges
- Detecting wavebreaking (at tracked peaks or close to edges)
- Spectral analysis and filtering
- Interface for writing to file and reading to file by h5py (hdmf)

# Construct a 2D Jonswap wave in space
```python
import numpy as np
from wave_tools import ConstructWave, surface_core, peak_tracking
import matplotlib.pyplot as plt

Hs = 2.0
Alpha = 0.023
smax = 70
theta_mean = np.pi/2+30*np.pi/180
gamma = 3.3
dx = 7.5
dy = 7.5
x = np.arange(-250, 250, dx)
y = np.arange(500, 1000, dy)
surf2d = ConstructWave.JonswapWave2D(x, y, Hs, Alpha, gamma, theta_mean, smax)
surf2d.plot_3d_as_2d()
surf2d.plot_3d_surface()
plt.show()
```
<img src="figures/surf2d.jpg" width="500">
<img src="figures/surf3d.jpg" width="500">

# Convert to Fourier domain
```python
spec2d = surf2d.define_SpectralAnalysis()
# plot the symmetric 2d spectrum
spec2d.plot()
plt.savefig('spec2d.jpg', bbox_inches='tight')
# plot the symmetric 2d spectrum for the given extent
spec2d.plot(extent=[-0.2,0.2,-0.2,0.2])
plt.savefig('spec2d_extent.jpg', bbox_inches='tight')
```

<img src="figures/spec2d.jpg" width="500">
<img src="figures/spec2d_extent.jpg" width="500">

# Filter high frequencies


# Plot the 
