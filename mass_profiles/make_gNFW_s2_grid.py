#definitions of the Sersic profile

import numpy as np
import pickle
from scipy.interpolate import splrep, splev
from sonnentools.cgsconstants import *
from vdmodel_2013 import sigma_model
from vdmodel_2013.profiles import deVaucouleurs
import ndinterp
from mass_profiles import gNFW


beta_min = 0.2
beta_max = 2.8
nb = 27

beta_grid = np.linspace(beta_min, beta_max, nb)

nr3d = 1001
r3d_grid = np.logspace(-3., 3., nr3d)

s2_grid = 0.*beta_grid

for i in range(nb):

    norm = 1./gNFW.M2d(1., 10., beta_grid[i])

    m3d_grid = norm * gNFW.M3d(r3d_grid, 10., beta_grid[i])
    
    s2_grid[i] = sigma_model.sigma2general((r3d_grid, m3d_grid), 0.5, lp_pars=1., seeing=None, light_profile=deVaucouleurs)

s2_grid = G * M_Sun / 10.**10 / kpc * s2_grid
s2_spline = splrep(beta_grid, s2_grid)

f = open('gNFW_rs10reff_re2_s2_spline.dat', 'w')
pickle.dump(s2_spline, f)
f.close()

