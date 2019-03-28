import numpy as np
import lens_models
import pickle
import os
import h5py
from lensingtools import powerlaw
from sonnentools.cgsconstants import *
from scipy.optimize import brentq


mockname = 'mockP'

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

gamma_min = 1.3
gamma_max = 2.8

psi1_fit = np.zeros(nlens)
psi2_fit = np.zeros(nlens)
psi3_fit = np.zeros(nlens)
gamma_fit = np.zeros(nlens)
dt_fit = np.zeros(nlens)
rmur_fit = np.zeros(nlens)

for i in range(nlens):

    lens = mock['lenses'][i]

    xa, xb = lens.images
    imerr = lens.obs_images[1]
    
    radmagrat_obs, radmagrat_err = lens.obs_radmagrat
    radmagrat = lens.radmag_ratio

    eps = 1e-4

    model_lens = lens_models.powerlaw(zd=lens.zd, zs=lens.zs, rein=1., gamma=2., images=lens.images, source=lens.source)
    unitb_lens = lens_models.powerlaw(zd=lens.zd, zs=lens.zs, rein=1., gamma=2., images=lens.images, source=lens.source)
    unitb_lens.b = 1.

    def b_from_gamma(gamma):
        unitb_lens.gamma = gamma
        norm = (xa - xb) / (unitb_lens.alpha(xa) - unitb_lens.alpha(xb))
        b = norm**(1./(unitb_lens.gamma - 1.))
        return b

    def rmur_model(gamma):
        b = b_from_gamma(gamma)
        model_lens.gamma = gamma
        model_lens.b = b
        model_lens.get_rein_from_b()
        return model_lens.mu_r(xa)/model_lens.mu_r(xb)

    if (rmur_model(gamma_min) - radmagrat) * (rmur_model(gamma_max) - radmagrat) < 0.:
        gamma_ml = brentq(lambda g: rmur_model(g) - radmagrat, gamma_min, gamma_max)

    else:
        gamma_ml = 1.01

    model_lens.images = (xa, xb)
    b_here = b_from_gamma(gamma_ml)
    model_lens.b = b_here
    model_lens.gamma = gamma_ml

    model_lens.get_rein_from_b()
    model_lens.source = xa - model_lens.alpha(xa)
    model_lens.get_timedelay()
    model_lens.get_radmag_ratio()

    psi1_fit[i] = model_lens.rein
    psi2_fit[i] = 2. - gamma_ml
    psi3_fit[i] = (2. - gamma_ml) * (1. - gamma_ml) / model_lens.rein

    dt_fit[i] = model_lens.timedelay
    gamma_fit[i] = gamma_ml
    rmur_fit[i] = model_lens.radmag_ratio

    print '%d %3.2f %3.2f %3.2f %f'%(i, gamma_ml, rmur_model(gamma_ml) - radmagrat, rmur_model(gamma_ml) - model_lens.radmag_ratio, radmagrat)

fit_file = h5py.File('%s_powerlaw_perfectobs_lensing.hdf5'%mockname, 'w')
fit_file.create_dataset('psi1', data=psi1_fit)
fit_file.create_dataset('psi2', data=psi2_fit)
fit_file.create_dataset('psi3', data=psi3_fit)
fit_file.create_dataset('timedelay', data=dt_fit)
fit_file.create_dataset('gamma', data=gamma_fit)
fit_file.create_dataset('rmur', data=rmur_fit)

