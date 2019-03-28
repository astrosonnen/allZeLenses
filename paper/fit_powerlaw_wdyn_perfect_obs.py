import numpy as np
import lens_models
import pickle
from scipy.interpolate import splev, splrep
from scipy.optimize import brentq
from mass_profiles import NFW, gNFW
import os
import h5py
from vdmodel_2013 import sigma_model
from vdmodel_2013.profiles import deVaucouleurs
from lensingtools import powerlaw
from sonnentools.cgsconstants import *


mockname = 'mockQ'

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

gamma_min = 1.5
gamma_max = 2.5
ng = 21

gamma_grid = np.linspace(gamma_min, gamma_max, ng)
s2_grid = 0.*gamma_grid

r3d_grid = np.logspace(-3., 3., 1001)

for i in range(ng):
    m3d_grid = powerlaw.M3d(r3d_grid, gamma_grid[i])/powerlaw.M2d(1., gamma_grid[i])

    s2_grid[i] = sigma_model.sigma2general((r3d_grid, m3d_grid), 0.5, lp_pars=1., seeing=None, light_profile=deVaucouleurs) * G * M_Sun / kpc / 10.**10

s2_spline = splrep(gamma_grid, s2_grid)

grid_dir = os.environ.get('ATL_GRIDDIR')

nlens = len(mock['lenses'])

psi1_fit = np.zeros(nlens)
psi2_fit = np.zeros(nlens)
psi3_fit = np.zeros(nlens)
gamma_fit = np.zeros(nlens)
dt_fit = np.zeros(nlens)

for i in range(nlens):

    lens = mock['lenses'][i]

    xa, xb = lens.images
    imerr = lens.obs_images[1]
    
    radmagrat_obs, radmagrat_err = lens.obs_radmagrat
    radmagrat = lens.radmag_ratio

    sigma_true = mock['sigma_sample'][i]
    sigma_err = 10.

    eps = 1e-4

    model_lens = lens_models.powerlaw(zd=lens.zd, zs=lens.zs, rein=1., gamma=2., images=lens.images, source=lens.source)

    def b_from_gamma(gamma):
        model_lens.gamma = gamma
        model_lens.b = 1.

        norm = (xa - xb) / (model_lens.alpha(xa) - model_lens.alpha(xb))

        b = norm**(1./(model_lens.gamma - 1.))
        return b

    def pl_sigma(gamma):

        b = b_from_gamma(gamma)
        model_lens.gamma = gamma
        model_lens.b = b

        m2d_e = np.pi * model_lens.m(lens.reff) * model_lens.S_cr / M_Sun

        return (m2d_e * splev(model_lens.gamma, s2_spline) / lens.reff_phys)**0.5

    print i, pl_sigma(gamma_min), pl_sigma(gamma_max), sigma_true

    gamma_fit = brentq(lambda g: pl_sigma(g) - sigma_true, gamma_min, gamma_max)

    model_lens.images = (xa, xb)
    model_lens.gamma = gamma_fit
    model_lens.b = b_from_gamma(gamma_fit)
    model_lens.get_rein_from_b()
    model_lens.source = xa - model_lens.alpha(xa)
    model_lens.get_timedelay()

    psi1_fit[i] = model_lens.rein
    psi2_fit[i] = 2. - gamma_fit
    psi3_fit[i] = (2. - gamma_fit) * (1. - gamma_fit) / model_lens.rein

    dt_fit[i] = model_lens.timedelay

    print '%d %3.2f %d %d %2.1f'%(i, gamma_fit, pl_sigma(gamma_fit), sigma_true, 70.*model_lens.timedelay/lens.timedelay)

fit_file = h5py.File('%s_powerlaw_perfectobs_wdyn.hdf5'%mockname, 'w')
fit_file.create_dataset('psi1', data=psi1_fit)
fit_file.create_dataset('psi2', data=psi2_fit)
fit_file.create_dataset('psi3', data=psi3_fit)
fit_file.create_dataset('timedelay', data=dt_fit)
fit_file.create_dataset('gamma', data=gamma_fit)

