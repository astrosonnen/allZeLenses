import numpy as np
import ndinterp
from scipy.interpolate import splrep
import lens_models
import pickle


ng = 1601
gamma_min = 1.2
gamma_max = 2.8

gamma_grid = np.linspace(gamma_min, gamma_max, ng)

nb = 801
beta_min = -0.4
beta_max = 0.4
beta_grid = np.linspace(beta_min, beta_max, nb)

gamma_spline = splrep(gamma_grid, np.arange(ng))
beta_spline = splrep(beta_grid, np.arange(nb))

axes = {0: gamma_spline, 1: beta_spline}

psi2 = np.zeros((ng, nb))

psi3 = np.zeros((ng, nb))

lens = lens_models.sps_ein_break(rein=1.)

for i in range(ng):
    lens.gamma = gamma_grid[i]
    for j in range(nb):
        lens.beta = beta_grid[j]
        psi2[i, j] = lens.psi2()
        psi3[i, j] = lens.psi3()

psi2_ndinterp = ndinterp.ndInterp(axes, psi2, order=1)
psi3_ndinterp = ndinterp.ndInterp(axes, psi3, order=1)

f = open('brokenpowerlaw_psi2_ndinterp.dat', 'w')
pickle.dump(psi2_ndinterp, f)
f.close()

f = open('brokenpowerlaw_psi3_ndinterp.dat', 'w')
pickle.dump(psi3_ndinterp, f)
f.close()


