import numpy as np
import pylab
import lens_models


ng = 5
gamma_grid = np.linspace(1.8, 2.2, ng)

nb = 51
beta_grid = np.linspace(-0.2, 0.2, nb)

bpl = lens_models.sps_ein_break()

for i in range(ng):
    psi2 = np.zeros(nb)

    bpl.gamma = gamma_grid[i]
    for j in range(nb):
        bpl.beta = beta_grid[j]

        psi2[j] = bpl.const()/(3.-bpl.gamma)*((3.-bpl.gamma)*(2.-bpl.gamma)*2.**bpl.beta + 2.*(3.-bpl.gamma)*bpl.beta*2.**(bpl.beta-1.) + bpl.beta*(bpl.beta-1.)*2.**(bpl.beta-2.))

    pylab.plot(beta_grid, psi2, label='%2.1f'%gamma_grid[i])

pylab.axhline(0., linestyle='--', color='k')

pylab.legend()
pylab.show()

