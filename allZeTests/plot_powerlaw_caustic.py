import numpy as np
import pylab
import lens_models


ng = 21
gamma_grid = np.linspace(1.8, 2.2, ng)

nm = 101
mag_grid = np.logspace(-2., 0., nm)

lens = lens_models.powerlaw()
for i in range(ng):
    lens.gamma = gamma_grid[i]

    area_grid = np.zeros(nm)
    for j in range(nm):
        lens.get_xy_minmag(min_mag=mag_grid[j], xtol=1e-6)
        area_grid[j] = lens.yminmag

    pylab.loglog(mag_grid, area_grid, label='$\gamma=%2.1f$'%gamma_grid[i])

pylab.show()

gamma1_mag_grid = np.zeros(nm)
gamma2_mag_grid = np.zeros(nm)

min_mag = 0.1

for i in range(nm):
    lens.gamma = 1.9
    for j in range(nm):
        lens.get_xy_minmag(min_mag=mag_grid[j], xtol=1e-6)
        gamma1_mag_grid[j] = lens.yminmag
    lens.gamma = 2.1
    for j in range(nm):
        lens.get_xy_minmag(min_mag=mag_grid[j], xtol=1e-6)
        gamma2_mag_grid[j] = lens.yminmag

pylab.plot(mag_grid, gamma2_mag_grid/gamma1_mag_grid)
pylab.show()

