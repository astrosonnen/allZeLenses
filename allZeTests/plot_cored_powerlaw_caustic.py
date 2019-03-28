import numpy as np
import pylab
import lens_models


ng = 5
gamma_grid = np.linspace(1.8, 2.2, ng)

nc = 101
rc_grid = np.logspace(-5., -3., nc)

lens = lens_models.cored_powerlaw()
for i in range(ng):
    lens.gamma = gamma_grid[i]

    caustic_grid = np.zeros(nc)
    for j in range(nc):
        lens.rc = rc_grid[j]

        lens.get_caustic()

        caustic_grid[j] = lens.caustic

    pylab.plot(rc_grid, caustic_grid, label='$\gamma=%2.1f$'%gamma_grid[i])

pylab.legend()
pylab.show()

# now plots inner image magnification as a function of gamma, for various source positions

lens.rc = 1e-4

ns = 20
source_grid = np.linspace(0.1, 2., ns)
for i in range(ng):
    lens.gamma = gamma_grid[i]
    lens.get_caustic()
    inmag_grid = np.zeros(ns)
    for j in range(ns):
        if source_grid[j] < lens.caustic:
            lens.source = source_grid[j]
            lens.get_images()
            inmag_grid[j] = lens.mu(lens.images[1])

    pylab.plot(source_grid, inmag_grid, label='$\gamma=%2.1f$'%gamma_grid[i])

pylab.axhline(-1., linestyle='--', color='k')

pylab.legend()
pylab.show()

