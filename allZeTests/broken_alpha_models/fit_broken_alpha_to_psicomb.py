import lens_models
from toy_models import sample_generator
import pylab
import pickle
import h5py
import os
import pymc
import numpy as np


mockname = 'mockI'
chaindir = '/Users/sonnen/allZeChains/'
mockdir = '/gdrive/projects/allZeLenses/allZeTests/'

nstep = 20000
burnin = 10000
thin = 10

eps = 1e-4
psicomb_err = 0.01

f = open(mockdir+mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

gamma_prior = {'lower': 1.2, 'upper': 2.8}
beta_prior = {'lower': -1., 'upper': 1.}

for i in range(nlens):
    print i
    chainname = chaindir+'%s_lens_%02d_psifit_brokenalpha_flatprior.hdf5'%(mockname, i)
    if not os.path.isfile(chainname):
        print 'sampling lens %d...'%i
        chain_file = h5py.File(chainname, 'w')

        lens = mock['lenses'][i]

        xa_obs, xb_obs = lens.obs_images[0]
        imerr = lens.obs_images[1]

        psi2_true = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
        psi3_true = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

        print psi3_true * lens.rein

        a_true = psi3_true / (1. - psi2_true)

        rein_guess = 0.5*(xa_obs - xb_obs)

        model_lens = lens_models.broken_alpha_powerlaw(zd=lens.zd, zs=lens.zs, rein=rein_guess)

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
        def psi2(rein=rein_par, gamma=gamma_par, beta=beta_par):

            model_lens.rein = rein
            model_lens.gamma = gamma
            model_lens.beta = beta

            return model_lens.psi2()

        @pymc.deterministic()
        def psi3(rein=rein_par, gamma=gamma_par, beta=beta_par):

            model_lens.rein = rein
            model_lens.gamma = gamma
            model_lens.beta = beta

            return model_lens.psi3()
    
        ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
        imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

        psicomb_logp = pymc.Normal('psi2_logp', mu=psi3/(1.-psi2), tau=1./psicomb_err**2, value=a_true, observed=True)
    
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

