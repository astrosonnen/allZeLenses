import pickle
import numpy as np
import h5py
import os
import pymc
import lens_models
from vdmodel_2013 import sigma_model
from vdmodel_2013.profiles import deVaucouleurs
from lensingtools import powerlaw
from sonnentools.cgsconstants import *
from scipy.interpolate import splrep, splev


gamma_min = 1.2
gamma_max = 2.8
ng = 33

gamma_grid = np.linspace(gamma_min, gamma_max, ng)
s2_grid = 0.*gamma_grid

r3d_grid = np.logspace(-3., 3., 1001)

for i in range(ng):
    m3d_grid = powerlaw.M3d(r3d_grid, gamma_grid[i])/powerlaw.M2d(1., gamma_grid[i])

    s2_grid[i] = sigma_model.sigma2general((r3d_grid, m3d_grid), 0.5, lp_pars=1., seeing=None, light_profile=deVaucouleurs) * G * M_Sun / kpc / 10.**10

s2_spline = splrep(gamma_grid, s2_grid)

mockname = 'mockP'

nstep = 20000
burnin = 10000
thin = 1

f = open(mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

chain_file = h5py.File('%s_powerlaw_impos_dyn_chains.hdf5'%mockname, 'w')

for i in range(nlens):
    print i

    group = chain_file.create_group('lens_%02d'%i)

    lens = mock['lenses'][i]
    print 'sampling lens %d...'%i

    xA, xB = lens.images

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    radmagrat_obs, radmagrat_err = lens.obs_radmagrat

    sigma_obs, sigma_err = lens.obs_sigma

    rein_guess = 0.5*(xA - xB)

    model_lens = lens_models.powerlaw(zd=lens.zd, zs=lens.zs, rein=rein_guess, gamma=2., images=lens.images, source=lens.source)
    model_lens.make_grids(err=imerr, nsig=5.)

    gamma_par = pymc.Uniform('gamma', lower=gamma_min, upper=gamma_max, value=2.)
    rein_par = pymc.Uniform('rein', lower=0.2*rein_guess, upper=3.*rein_guess, value=rein_guess)
    s2_par = pymc.Uniform('s2', lower=0., upper=(xa_obs)**2, value=(xA - model_lens.alpha(xA))**2)

    @pymc.deterministic()
    def images(rein=rein_par, gamma=gamma_par, s2=s2_par):
        model_lens.rein = rein
        model_lens.gamma = gamma
        model_lens.source = s2**0.5
        model_lens.get_b_from_rein()

        model_lens.get_images()

        return (model_lens.images[0], model_lens.images[1])
   
    @pymc.deterministic()
    def imA(rein=rein_par, gamma=gamma_par, s2=s2_par):
        return float(images[0])

    @pymc.deterministic()
    def imB(rein=rein_par, gamma=gamma_par, s2=s2_par):
        return float(images[1])

    @pymc.deterministic()
    def radmagrat(rein=rein_par, gamma=gamma_par, s2=s2_par):
        model_imA, model_imB = images

        model_lens.rein = rein
        model_lens.gamma = gamma
        model_lens.source = s2**0.5

        model_lens.get_b_from_rein()

        return model_lens.mu_r(model_imA)/model_lens.mu_r(model_imB)

    @pymc.deterministic()
    def timedelay(rein=rein_par, gamma=gamma_par, s2=s2_par):
        model_imA, model_imB = images

        model_lens.rein = rein
        model_lens.gamma = gamma
        model_lens.source = s2**0.5

        model_lens.get_b_from_rein()

        model_lens.images = (model_imA, model_imB)

        model_lens.get_timedelay()

        return model_lens.timedelay

    @pymc.deterministic()
    def sigma(rein=rein_par, gamma=gamma_par):
        model_lens.rein = rein
        model_lens.gamma = gamma

        model_lens.get_b_from_rein()

        m2d_e = np.pi * model_lens.m(lens.reff) * model_lens.S_cr / M_Sun
        return (m2d_e * splev(model_lens.gamma, s2_spline) / lens.reff_phys)**0.5

    ima_logp = pymc.Normal('ima_logp', mu=imA, tau=1./imerr**2, value=xa_obs, observed=True)
    imb_logp = pymc.Normal('imb_logp', mu=imB, tau=1./imerr**2, value=xb_obs, observed=True)

    #radmag_logp = pymc.Normal('radmag_logp', mu=radmagrat, tau=1./radmagrat_err**2, value=radmagrat_obs, observed=True)

    sigma_logp = pymc.Normal('sigma_logp', mu=sigma, tau=1./sigma_err**2, value=sigma_obs, observed=True)

    pars = [gamma_par, rein_par, s2_par]
    allpars = pars + [images, radmagrat, timedelay, sigma]

    M = pymc.MCMC(allpars)
    M.use_step_method(pymc.AdaptiveMetropolis, pars)
    M.sample(nstep, burnin, thin=thin)

    for par in allpars:
        group.create_dataset(str(par), data=M.trace(par)[:])

