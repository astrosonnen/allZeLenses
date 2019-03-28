import lens_models
from toy_models import sample_generator
import pylab
import pickle
import h5py
import os
import pymc
import numpy as np
from scipy.interpolate import splev


mockname = 'mockL'
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

f = open('/gdrive/projects/cs82_weaklensing/PyBaWL/nfw_re2_s2_grid.dat', 'r')
nfw_re2_s2_spline = pickle.load(f)
f.close()

f = open('/gdrive/projects/cs82_weaklensing/PyBaWL/deV_re2_s2.dat', 'r')
deV_re2_s2 = pickle.load(f)
f.close()

for i in range(nlens):
    print i
    chainname = chaindir+'%s_lens_%02d_psifit_nfwdev_wsigma_flatprior.hdf5'%(mockname, i)
    if not os.path.isfile(chainname):
        print 'sampling lens %d...'%i
        chain_file = h5py.File(chainname, 'w')

        lens = mock['lenses'][i]

        xa_obs, xb_obs = lens.obs_images[0]
        imerr = lens.obs_images[1]

        sigma_obs, sigma_err = lens.obs_sigma

        psi2_true = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
        psi3_true = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

        a_true = psi3_true / (1. - psi2_true)

        rein_guess = 0.5*(xa_obs - xb_obs)

        model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, reff_phys=lens.reff_phys, delta_halo=200., cvir=lens.cvir)

        model_lens.images = (xa_obs, xb_obs)

        model_lens.normalize()
        model_lens.get_caustic()

        if lens.source > model_lens.caustic:
            model_lens.source = 0.2 * model_lens.caustic
        else:
            model_lens.source = lens.source

        mhalo_par = pymc.Uniform('mhalo', lower=11., upper=15., value=np.log10(lens.mhalo))
        mstar_par = pymc.Uniform('mstar', lower=10., upper=12.5, value=np.log10(lens.mstar))
        cvir_par = pymc.Uniform('cvir', lower=0., upper=2., value=1.)

        s2_par = pymc.Uniform('s2', lower=0., upper=xa_obs**2, value=model_lens.source**2)

        pars = [mhalo_par, mstar_par, cvir_par, s2_par]

        @pymc.deterministic()
        def images(p=pars):

            mhalo, mstar, cvir, s2 = p

            model_lens.source = s2**0.5
            model_lens.mstar = 10.**mstar
            model_lens.mhalo = 10.**mhalo
            model_lens.cvir = 10.**cvir
    
            model_lens.normalize()

            model_lens.get_caustic()
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

            mhalo, mstar, cvir, s2 = p

            model_lens.source = s2**0.5
            model_lens.mstar = 10.**mstar
            model_lens.mhalo = 10.**mhalo
            model_lens.cvir = 10.**cvir
    
            model_lens.normalize()

            model_lens.images = imgs
    
            if not np.isfinite(imgs[0]):
                return 0.
            else:
                model_lens.get_timedelay()
                return model_lens.timedelay

        @pymc.deterministic()
        def psi2(mstar=mstar_par, mhalo=mhalo_par, cvir=cvir_par):

            model_lens.mstar = 10.**mstar
            model_lens.mhalo = 10.**mhalo
            model_lens.cvir = 10.**cvir
    
            model_lens.normalize()

            model_lens.get_rein()

            return (model_lens.alpha(model_lens.rein + eps) - model_lens.alpha(model_lens.rein - eps))/(2.*eps)

        @pymc.deterministic()
        def psi3(mstar=mstar_par, mhalo=mhalo_par, cvir=cvir_par):

            model_lens.mstar = 10.**mstar
            model_lens.mhalo = 10.**mhalo
            model_lens.cvir = 10.**cvir
    
            model_lens.normalize()

            model_lens.get_rein()

            return (model_lens.alpha(model_lens.rein + eps) - 2.*model_lens.alpha(model_lens.rein) + model_lens.alpha(model_lens.rein - eps))/eps**2
    
        @pymc.deterministic()
        def sigma(mstar=mstar_par, mhalo=mhalo_par, cvir=cvir_par):

            model_lens.mstar = 10.**mstar
            model_lens.mhalo = 10.**mhalo
            model_lens.cvir = 10.**cvir
    
            model_lens.normalize()

            m200tomrs = (np.log(2.) - 0.5)/(np.log(1. + lens.cvir) - lens.cvir/(1. + lens.cvir))

            s2_halo = model_lens.mhalo * m200tomrs * splev(lens.rs/lens.reff, nfw_re2_s2_spline)/(lens.reff * lens.arcsec2kpc)
            s2_bulge = model_lens.mstar * deV_re2_s2 / (lens.reff * lens.arcsec2kpc)

            return (s2_halo + s2_bulge)**0.5

        ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
        imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

        psicomb_logp = pymc.Normal('psi2_logp', mu=psi3/(1.-psi2), tau=1./psicomb_err**2, value=a_true, observed=True)

        sigma_logp = pymc.Normal('sigma_logp', mu=sigma, tau=1./sigma_err, value=sigma_obs, observed=True)
    
        allpars = pars + [timedelay, image_a, image_b, psi2, psi3, sigma]
    
        M = pymc.MCMC(allpars)
        M.use_step_method(pymc.AdaptiveMetropolis, pars)
        M.sample(nstep, burnin, thin=thin)
    
        for par in allpars:
            chain_file.create_dataset(str(par), data=M.trace(par)[:])

        chain_file.close()

