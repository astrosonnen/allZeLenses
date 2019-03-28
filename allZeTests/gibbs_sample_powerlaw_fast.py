import numpy as np
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq
import pymc
import pickle
import lens_models
import h5py
import pylab


mockname = 'mockI'

save_every = 100

rein_tol = 0.001

ngrid_source = 101

gamma_min = 1.2
gamma_max = 2.8
ngrid_gamma = 161

gamma_grid = np.linspace(gamma_min, gamma_max, ngrid_gamma)

day = 24.*3600.

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

nsamp = 3000

mu_gamma = np.zeros(nsamp)
sig_gamma = np.zeros(nsamp)
invH0 = np.zeros(nsamp)

gamma = np.zeros((nlens, nsamp))
s2 = np.zeros((nlens, nsamp))
rein = np.zeros((nlens, nsamp))

xA_obs = np.zeros(nlens)
xB_obs = np.zeros(nlens)

xA_err = np.zeros(nlens)
xB_err = np.zeros(nlens)

dt_obs = np.zeros(nlens)
dt_err = np.zeros(nlens)

xA_model = np.zeros((nlens, nsamp))
xB_model = np.zeros((nlens, nsamp))
radmagrat_model = np.zeros((nlens, nsamp))
dt_model = np.zeros((nlens, nsamp))

radmagrat_obs = np.zeros(nlens)
radmagrat_err = np.zeros(nlens)

model_lenses = []
rein_grids = []
for i in range(nlens):
    lens = mock['lenses'][i]

    xA, xB = lens.images

    xA_obs[i] = lens.obs_images[0][0]
    xB_obs[i] = lens.obs_images[0][1]

    xA_err[i] = lens.obs_images[1]
    xB_err[i] = lens.obs_images[1]

    radmagrat_obs[i] = lens.obs_radmagrat[0]
    radmagrat_err[i] = lens.obs_radmagrat[1]

    dt_obs[i] = lens.obs_timedelay[0] / day
    dt_err[i] = lens.obs_timedelay[1] / day

    rein_guess = 0.5*(xA - xB)

    model_lens = lens_models.powerlaw(zd=lens.zd, zs=lens.zs, rein=rein_guess, gamma=2.)
    model_lens.source = xA - model_lens.alpha(xA)

    model_lens.get_b_from_rein()
    model_lens.get_images()
    model_lens.get_timedelay()

    dt_model[i, 0] = model_lens.timedelay / day

    gamma[i, 0] = model_lens.gamma
    rein[i, 0] = model_lens.rein
    s2[i, 0] = model_lens.source**2

    rein_minhere = 0.5*(xA_obs[i] - 5.*xA_err[i] - min(0., xB_obs[i] + 5.*xB_err[i]))
    rein_maxhere = 0.5*(xA_obs[i] + 5.*xA_err[i] - (xB_obs[i] - 5.*xB_err[i]))

    rein_gridhere = np.arange(rein_minhere, rein_maxhere, rein_tol)

    rein_grids.append(rein_gridhere)

    model_lenses.append(model_lens)

mu_gamma[0] = 2.
sig_gamma[0] = 0.1
invH0[0] = 1./70.

sqrtn = float(nlens)**0.5

ngrid = 100
sig_min = 0.03
sig_max = 1.
sig_grid = np.linspace(sig_min, sig_max, ngrid)

for i in range(nsamp-1):

    mean_g = gamma[:, i].mean()

    mu_gamma[i+1] = np.random.normal(mean_g, sig_gamma[i]/sqrtn, 1)

    psig_logpgrid = 0.*sig_grid
    for j in range(ngrid):
        logp = -0.5*nlens*np.log(2.*np.pi) - nlens*np.log(sig_grid[j]) -0.5/sig_grid[j]**2 * ((mu_gamma[i+1] - gamma[:, i])**2).sum()
        psig_logpgrid[j] = logp

    psig_logpgrid -= psig_logpgrid.max()
    psig_grid = np.exp(psig_logpgrid)

    psig_spline = splrep(sig_grid, psig_grid)
     
    intfunc = lambda s: splint(sig_min, s, psig_spline)

    norm = intfunc(sig_max)

    F = np.random.rand(1) * norm

    sig_gamma[i+1] = brentq(lambda s: intfunc(s) - F, sig_min, sig_max)

    # now draws new sample for inverse H0

    mu_invH0 = dt_obs/(dt_model[:, i]*70.)
    sig_invH0 = dt_err/(dt_model[:, i]*70.)

    eff_sigma = ((1./sig_invH0**2).sum())**-0.5
    eff_mu = (mu_invH0/sig_invH0**2).sum() * eff_sigma**2

    invH0[i+1] = np.random.normal(eff_mu, eff_sigma, 1)

    print i, invH0[i+1]**-1

    # now loops through individual objects

    # sampling in Einstein radius
    for j in range(nlens):

        logp_ein = 0.*rein_grids[j]

        for k in range(len(rein_grids[j])):
            model_lenses[j].rein = rein_grids[j][k]
            model_lenses[j].get_b_from_rein()
            model_lenses[j].get_images()

            if np.isfinite(model_lenses[j].images[0]):
                model_lenses[j].get_timedelay()
            
                logp_ein[k] = -0.5*(model_lenses[j].images[0] - xA_obs[j])**2/xA_err[j]**2 - 0.5*(model_lenses[j].images[1] - xB_obs[j])**2/xB_err[j]**2
                logp_ein[k] += -0.5*(model_lenses[j].mu_r(model_lenses[j].images[0])/model_lenses[j].mu_r(model_lenses[j].images[1]) - radmagrat_obs[j])**2/radmagrat_err[j]**2
                logp_ein[k] += -0.5*(invH0[i+1] * model_lenses[j].timedelay/day * 70. - dt_obs[j])**2/dt_err[j]**2
            else:
                logp_ein[k] = -np.inf

        logp_ein -= logp_ein.max()

        p_ein_grid = np.exp(logp_ein)

        p_ein_spline = splrep(rein_grids[j], p_ein_grid)
     
        intfunc = lambda t: splint(rein_grids[j][0], t, p_ein_spline)

        norm = intfunc(rein_grids[j][-1])

        F = np.random.rand(1) * norm

        rein[j, i+1] = brentq(lambda t: intfunc(t) - F, rein_grids[j][0], rein_grids[j][-1])

        model_lenses[j].rein = rein[j, i+1]
        model_lenses[j].get_b_from_rein()

    # sampling in source position
    for j in range(nlens):

        source_max = xA_obs[j] + 5.*xA_err[j] - model_lenses[j].alpha(xA_obs[j] + 5.*xA_err[j])
        source_min = max(0., xA_obs[j] - 5.*xA_err[j] - model_lenses[j].alpha(xA_obs[j] - 5.*xA_err[j]))

        s2_grid = np.linspace(source_min**2, source_max**2, ngrid_source)

        logp_s2 = 0. * s2_grid

        for k in range(ngrid_source):
            model_lenses[j].source = s2_grid[k]**0.5
            model_lenses[j].get_images()

            if np.isfinite(model_lenses[j].images[0]):
                model_lenses[j].get_timedelay()
            
                logp_s2[k] = -0.5*(model_lenses[j].images[0] - xA_obs[j])**2/xA_err[j]**2 - 0.5*(model_lenses[j].images[1] - xB_obs[j])**2/xB_err[j]**2
                logp_s2[k] += -0.5*(model_lenses[j].mu_r(model_lenses[j].images[0])/model_lenses[j].mu_r(model_lenses[j].images[1]) - radmagrat_obs[j])**2/radmagrat_err[j]**2
                logp_s2[k] += -0.5*(invH0[i+1] * model_lenses[j].timedelay/day * 70. - dt_obs[j])**2/dt_err[j]**2
            else:
                logp_s2[k] = -np.inf

        logp_s2 -= logp_s2.max()

        p_s2_grid = np.exp(logp_s2)
        
        p_s2_spline = splrep(s2_grid, p_s2_grid)
     
        intfunc = lambda t: splint(s2_grid[0], t, p_s2_spline)

        norm = intfunc(s2_grid[-1])

        F = np.random.rand(1) * norm

        s2[j, i+1] = brentq(lambda t: intfunc(t) - F, s2_grid[0], s2_grid[-1])

        model_lenses[j].source = s2[j, i+1]**0.5

    # sampling in gamma
    for j in range(nlens):

        logp_gamma = 0.*gamma_grid

        for k in range(ngrid_gamma):
            model_lenses[j].gamma = gamma_grid[k]
            model_lenses[j].get_b_from_rein()
            model_lenses[j].get_images()

            if np.isfinite(model_lenses[j].images[0]):
                model_lenses[j].get_timedelay()
            
                logp_gamma[k] = -0.5*(model_lenses[j].images[0] - xA_obs[j])**2/xA_err[j]**2 - 0.5*(model_lenses[j].images[1] - xB_obs[j])**2/xB_err[j]**2
                logp_gamma[k] += -0.5*(model_lenses[j].mu_r(model_lenses[j].images[0])/model_lenses[j].mu_r(model_lenses[j].images[1]) - radmagrat_obs[j])**2/radmagrat_err[j]**2
                logp_gamma[k] += -0.5*(invH0[i+1] * model_lenses[j].timedelay/day * 70. - dt_obs[j])**2/dt_err[j]**2
            else:
                logp_gamma[k] = -np.inf

        logp_gamma += -0.5*(gamma_grid - mu_gamma[i+1])**2/sig_gamma[i+1]**2 - np.log(sig_gamma[i+1])

        logp_gamma -= logp_gamma.max()

        p_gamma_grid = np.exp(logp_gamma)

        p_gamma_spline = splrep(gamma_grid, p_gamma_grid)
     
        intfunc = lambda t: splint(gamma_min, t, p_gamma_spline)

        norm = intfunc(gamma_max)

        F = np.random.rand(1) * norm

        gamma[j, i+1] = brentq(lambda t: intfunc(t) - F, gamma_min, gamma_max)

        model_lenses[j].gamma = gamma[j, i+1]
        model_lenses[j].get_b_from_rein()
        model_lenses[j].get_images()
        model_lenses[j].get_timedelay()

        xA_model[j, i+1] = model_lenses[j].images[0]
        xB_model[j, i+1] = model_lenses[j].images[1]
        radmagrat_model[j, i+1] = model_lenses[j].mu_r(model_lenses[j].images[0])/model_lenses[j].mu_r(model_lenses[j].images[1])
        dt_model[j, i+1] = model_lenses[j].timedelay / day

    for j in range(nlens):
        print '%d %4.3f %4.3f %4.3f'%(j, rein[j, i+1], gamma[j, i+1], s2[j, i+1]**0.5)

    if (i+1) % save_every == 0:
        chain_file = h5py.File('tmp_%s_powerlaw_gibbs_sample_fast.hdf5'%mockname, 'w')
        chain_file.create_dataset('invH0', data=invH0)
        chain_file.create_dataset('mu_gamma', data=mu_gamma)
        chain_file.create_dataset('sig_gamma', data=sig_gamma)
        chain_file.create_dataset('rein', data=rein)
        chain_file.create_dataset('gamma', data=gamma)
        chain_file.create_dataset('s2', data=s2)
        chain_file.create_dataset('xA_model', data=xA_model)
        chain_file.create_dataset('xB_model', data=xB_model)
        chain_file.create_dataset('radmagrat_model', data=radmagrat_model)
        chain_file.create_dataset('dt_model', data=dt_model)
        
        chain_file.close()

chain_file = h5py.File('%s_powerlaw_gibbs_sample_fast.hdf5'%mockname, 'w')
chain_file.create_dataset('invH0', data=invH0)
chain_file.create_dataset('mu_gamma', data=mu_gamma)
chain_file.create_dataset('sig_gamma', data=sig_gamma)
chain_file.create_dataset('rein', data=rein)
chain_file.create_dataset('gamma', data=gamma)
chain_file.create_dataset('s2', data=s2)
chain_file.create_dataset('xA_model', data=xA_model)
chain_file.create_dataset('xB_model', data=xB_model)
chain_file.create_dataset('radmagrat_model', data=radmagrat_model)
chain_file.create_dataset('dt_model', data=dt_model)

chain_file.close()

