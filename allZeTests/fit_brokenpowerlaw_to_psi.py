import lens_models
from toy_models import sample_generator
import pylab
import pickle
import h5py
import os
import pymc
import numpy as np


mockname = 'mockF'
chaindir = '/Users/sonnen/allZeChains/'

nstep = 20000
burnin = 10000
thin = 10

eps = 1e-6
psi2_err = 0.01
psi3_err = 0.01

f = open(mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

gamma_prior = {'lower': 1.2, 'upper': 2.8}
beta_prior = {'lower': -1., 'upper': 1.}

for i in range(nlens):
    print i
    chainname = chaindir+'%s_lens_%02d_psifit_brokenpl_flatprior.hdf5'%(mockname, i)
    if not os.path.isfile(chainname):
        print 'sampling lens %d...'%i
        chain_file = h5py.File(chainname, 'w')

        lens = mock['lenses'][i]

        xa_obs, xb_obs = lens.obs_images[0]
        imerr = lens.obs_images[1]


        psi2_true = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
        psi3_true = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

        rein_guess = 0.5*(xa_obs - xb_obs)

        model_lens = lens_models.sps_ein_break(zd=lens.zd, zs=lens.zs, rein=rein_guess)

        model_lens.images = (xa_obs, xb_obs)

        model_lens.get_caustic()

        rein_par = pymc.Uniform('rein', lower=0.5*rein_guess, upper=2.*rein_guess, value=rein_guess)
        gamma_par = pymc.Uniform('gamma', lower=gamma_prior['lower'], upper=gamma_prior['upper'], value=2.)
        beta_par = pymc.Uniform('beta', lower=beta_prior['lower'], upper=beta_prior['upper'], value=0.)

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

        @pymc.deterministic()
        def psi2(p=pars, imgs=images):

            rein, gamma, beta, s2 = p
    
            model_lens.source = s2**0.5
            model_lens.rein = rein
            model_lens.gamma = gamma
            model_lens.beta = beta

            model_lens.images = imgs

            const_bpl = model_lens.const()

            psi2_bpl = const_bpl/(3.-gamma)*((3.-gamma)*(2.-gamma)*2.**beta + 2.*(3.-gamma)*beta*2.**(beta-1.) + beta*(beta-1.)*2.**(beta-2.))

            return psi2_bpl

        @pymc.deterministic()
        def psi3(p=pars, imgs=images):

            rein, gamma, beta, s2 = p
    
            model_lens.source = s2**0.5
            model_lens.rein = rein
            model_lens.gamma = gamma
            model_lens.beta = beta

            const_bpl = model_lens.const()

            psi3_bpl = const_bpl/(3.-gamma)*((3.-gamma)*(2.-gamma)*(1.-gamma)*2.**beta + 3.*beta*(3.-gamma)*(2.-gamma)*2.**(beta-1.) + 3.*beta*(beta-1.)*(3.-gamma)*2.**(beta-2.) + beta*(beta-1.)*(beta-2.)*2.**(beta-3.))

            return psi3_bpl
     
        ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
        imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

        psi2_logp = pymc.Normal('psi2_logp', mu=psi2, tau=1./psi2_err**2, value=psi2_true, observed=True)
        psi3_logp = pymc.Normal('psi3_logp', mu=psi3, tau=1./psi3_err**2, value=psi3_true, observed=True)
    
        allpars = pars + [timedelay, image_a, image_b, psi2, psi3]
    
        M = pymc.MCMC(allpars)
        M.use_step_method(pymc.AdaptiveMetropolis, pars)
        M.sample(nstep, burnin, thin=thin)
    
        for par in allpars:
            chain_file.create_dataset(str(par), data=M.trace(par)[:])

        gp_group = chain_file.create_group('gamma_prior')
        gp_group.create_dataset('lower', data=gamma_prior['lower'])
        gp_group.create_dataset('upper', data=gamma_prior['upper'])

        bp_group = chain_file.create_group('beta_prior')
        bp_group.create_dataset('lower', data=beta_prior['lower'])
        bp_group.create_dataset('upper', data=beta_prior['upper'])

        chain_file.close()

