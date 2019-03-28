import numpy as np
import lens_models
import pymc
import pickle
from scipy.interpolate import splev
from mass_profiles import NFW, gNFW
import os
import h5py


mockname = 'mockO'

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

chaindir = '/Users/sonnen/allZeChains/'

grid_dir = os.environ.get('ATL_GRIDDIR')

nlens = len(mock['lenses'])

f = open('/gdrive/projects/cs82_weaklensing/PyBaWL/deV_re2_s2.dat', 'r')
deV_re2_s2 = pickle.load(f)
f.close()

f = open(grid_dir+'/gNFW_rs10reff_re2_s2_spline.dat', 'r')
gnfw_re2_s2_spline = pickle.load(f)
f.close()
 
for i in range(nlens):

    chainname = chaindir+'%s_lens_%02d_gnfwdev.hdf5'%(mockname, i)

    print i

    if not os.path.isfile(chainname):
        chain_file = h5py.File(chainname, 'w')

        lens = mock['lenses'][i]
    
        xa_obs, xb_obs = lens.obs_images[0]
        imerr = lens.obs_images[1]
        
        radmagrat_obs, radmagrat_err = lens.obs_radmagrat
        
        sigma_obs, sigma_err = lens.obs_sigma

        eps = 1e-4
        
        mdme_true = lens.mhalo / NFW.M3d(lens.rvir*lens.arcsec2kpc, lens.rs*lens.arcsec2kpc) * NFW.M2d(lens.reff_phys, lens.rs*lens.arcsec2kpc)
        
        print np.log10(mdme_true)
        
        model_lens = lens_models.gNfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mdme=mdme_true, reff_phys=lens.reff_phys, beta=1., rs_phys=10.*lens.reff_phys, delta_halo=200.)
        
        model_lens.normalize()
        model_lens.get_caustic()
        
        if lens.source < model_lens.caustic:
            model_lens.source = lens.source
        else:
            model_lens.source = 0.8 * model_lens.caustic

        mdme_par = pymc.Uniform('mdme', lower=9., upper=13., value=np.log10(model_lens.mdme))
        mstar_par = pymc.Uniform('mstar', lower=10., upper=13., value=np.log10(lens.mstar))
        
        beta_par = pymc.Uniform('beta', lower=0.2, upper=2.8, value=1.)
        
        s2_par = pymc.Uniform('s2', lower=0., upper=xa_obs**2, value=model_lens.source**2)
        
        pars = [mdme_par, mstar_par, beta_par, s2_par]
        
        @pymc.deterministic()
        def images(p=pars):
        
            mdme, mstar, beta, s2 = p
        
            model_lens.source = s2**0.5
            model_lens.mstar = 10.**mstar
            model_lens.mdme = 10.**mdme
            model_lens.beta = beta
            
            model_lens.normalize()
        
            model_lens.get_images()

            if len(model_lens.images) < 2:
                return (np.inf, -np.inf)
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
        
            mdme, mstar, beta, s2 = p
        
            model_lens.source = s2**0.5
            model_lens.mstar = 10.**mstar
            model_lens.mdme = 10.**mdme
            model_lens.beta = beta
            
            model_lens.normalize()
        
            model_lens.images = imgs
            
            if not np.isfinite(imgs[0]):
                return 0.
            else:
                model_lens.get_timedelay()
                return model_lens.timedelay
        
        @pymc.deterministic()
        def rein(mstar=mstar_par, mdme=mdme_par, beta=beta_par):
        
            model_lens.mstar = 10.**mstar
            model_lens.mdme = 10.**mdme
            model_lens.beta = beta
            
            model_lens.normalize()
        
            model_lens.get_rein()
        
            return model_lens.rein
        
        @pymc.deterministic()
        def sigma(mstar=mstar_par, mdme=mdme_par, beta=beta_par):
        
            s2_bulge = 10.**mstar * deV_re2_s2 / model_lens.reff_phys
            s2_halo = 10.**mdme * splev(beta, gnfw_re2_s2_spline) / model_lens.reff_phys
        
            return (s2_bulge + s2_halo)**0.5
        
        @pymc.deterministic()
        def radmagratio(mstar=mstar_par, mdme=mdme_par, beta=beta_par, imgs=images):
        
            model_lens.mstar = 10.**mstar
            model_lens.mdme = 10.**mdme
            model_lens.beta = beta
            
            model_lens.normalize()
        
            model_lens.images = imgs
        
            model_lens.get_radmag_ratio()
        
            return model_lens.radmag_ratio
        
        @pymc.deterministic()
        def psi2(mstar=mstar_par, mdme=mdme_par, beta=beta_par, re=rein):
        
            model_lens.mstar = 10.**mstar
            model_lens.mdme = 10.**mdme
            model_lens.beta = beta
            
            model_lens.normalize()
        
            return (model_lens.alpha(re + eps) - model_lens.alpha(re - eps))/(2.*eps)
        
        @pymc.deterministic()
        def psi3(mstar=mstar_par, mdme=mdme_par, beta=beta_par, re=rein):
        
            model_lens.mstar = 10.**mstar
            model_lens.mdme = 10.**mdme
            model_lens.beta = beta
            
            model_lens.normalize()
        
            return (model_lens.alpha(re + eps) - 2.*model_lens.alpha(re) + model_lens.alpha(re - eps))/eps**2
            
        ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
        imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)
        
        sigma_logp = pymc.Normal('sigma_logp', mu=sigma, tau=1./sigma_err**2, value=sigma_obs, observed=True)
        
        radmagrat_logp = pymc.Normal('radmagrat_logp', mu=radmagratio, tau=1./radmagrat_err**2, value=radmagrat_obs, observed=True)
        
        allpars = pars + [timedelay, image_a, image_b, rein, psi2, psi3, sigma, radmagratio]
            
        M = pymc.MCMC(allpars)
        M.use_step_method(pymc.AdaptiveMetropolis, pars)
        M.sample(110000, 10000)

        for par in allpars:
            chain_file.create_dataset(str(par), data=M.trace(par)[:])


