import lens_models
from toy_models import sample_generator
import pylab
import pickle
import h5py
import os
import pymc
import numpy as np


mockname = 'mockJ'
chaindir = '/Users/sonnen/allZeChains/'

nstep = 20000
burnin = 10000
thin = 10

f = open(mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

gamma_prior = {'mu': 2., 'sigma': 0.3}
beta_prior = {'mu': 0., 'sigma': 0.2}

for i in range(nlens):
    print i
    chainname = chaindir+'%s_lens_%02d_brokenpowerlaw.hdf5'%(mockname, i)
    if not os.path.isfile(chainname):
        print 'sampling lens %d...'%i
        chain_file = h5py.File(chainname, 'w')

        lens = mock['lenses'][i]

        xa_obs, xb_obs = lens.obs_images[0]
        imerr = lens.obs_images[1]

        rein_guess = 0.5*(xa_obs - xb_obs)

        radmagrat_obs, radmagrat_err = lens.obs_radmagrat

        model_lens = lens_models.sps_ein_break(zd=lens.zd, zs=lens.zs, rein=rein_guess)

        model_lens.images = (xa_obs, xb_obs)

        model_lens.get_caustic()
        model_lens.make_grids(err=imerr, nsig=5.)

        rein_par = pymc.Uniform('rein', lower=0.5*rein_guess, upper=2.*rein_guess, value=rein_guess)
        gamma_par = pymc.Normal('gamma', mu=gamma_prior['mu'], tau=1./gamma_prior['sigma']**2, value=2.)
        beta_par = pymc.Normal('beta', mu=beta_prior['mu'], tau=1./beta_prior['sigma']**2, value=0.)

        s2_par = pymc.Uniform('s2', lower=0., upper=xa_obs**2, value=(xa_obs - model_lens.alpha(xa_obs))**2)

        pars = [rein_par, gamma_par, beta_par, s2_par]

        @pymc.deterministic()
        def images(p=pars):

            rein, gamma, beta, s2 = p

            model_lens.source = s2**0.5
            model_lens.rein = rein
            model_lens.gamma = gamma
            model_lens.beta = beta
    
            model_lens.get_images()
            if len(model_lens.images) < 2:
                return np.inf, -np.inf
            else:
                return model_lens.images
    
        @pymc.deterministic()
        def image_a(imgs=images):
            return imgs[0]
    
        @pymc.deterministic()
        def image_b(imgs=images):
            return imgs[1]
    
        @pymc.deterministic()
        def radmagrat(p=pars, imgs=images):
    
            rein, gamma, beta, s2 = p

            model_lens.source = s2**0.5
            model_lens.rein = rein
            model_lens.gamma = gamma
            model_lens.beta = beta

            model_lens.images = imgs
    
            if not np.isfinite(imgs[0]):
                return 0.
            else:
                model_lens.get_radmag_ratio()
                return model_lens.radmag_ratio
    
        @pymc.deterministic()
        def timedelay(p=pars, imgs=images):

            rein, gamma, beta, s2 = p
    
            model_lens.source = s2**0.5
            model_lens.rein = rein
            model_lens.gamma = gamma
            model_lens.beta = beta

            model_lens.images = imgs
    
            if not np.isfinite(imgs[0]):
                return 0.
            else:
                model_lens.get_timedelay()
                return model_lens.timedelay
    
        ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
        imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

        radmag_logp = pymc.Normal('radmag_logp', mu=radmagrat, tau=1./radmagrat_err**2, value=radmagrat_obs, observed=True)
    
        allpars = pars + [timedelay, image_a, image_b, radmagrat]
    
        M = pymc.MCMC(allpars)
        M.use_step_method(pymc.AdaptiveMetropolis, pars)
        M.sample(nstep, burnin, thin=thin)
    
        for par in allpars:
            chain_file.create_dataset(str(par), data=M.trace(par)[:])

        gp_group = chain_file.create_group('gamma_prior')
        gp_group.create_dataset('mu', data=gamma_prior['mu'])
        gp_group.create_dataset('sigma', data=gamma_prior['sigma'])

        bp_group = chain_file.create_group('beta_prior')
        bp_group.create_dataset('mu', data=beta_prior['mu'])
        bp_group.create_dataset('sigma', data=beta_prior['sigma'])

        chain_file.close()

