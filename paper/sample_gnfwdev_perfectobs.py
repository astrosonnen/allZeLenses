import numpy as np
import lens_models
import pymc
import pickle
from scipy.interpolate import splev
from mass_profiles import NFW, gNFW
import os
import h5py


mockname = 'mockP'

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

f = open('/gdrive/projects/cs82_weaklensing/PyBaWL/nfw_re2_s2_grid.dat', 'r')
nfw_re2_s2_spline = pickle.load(f)
f.close()

 
for i in range(nlens):

    chainname = chaindir+'%s_lens_%02d_gnfwdev_perfect_obs.hdf5'%(mockname, i)

    print i

    if not os.path.isfile(chainname):
        chain_file = h5py.File(chainname, 'w')

        lens = mock['lenses'][i]
    
        xa, xb = lens.images
        xa_obs, xb_obs = lens.obs_images[0]
        imerr = lens.obs_images[1]
        
        radmagrat_obs, radmagrat_err = lens.obs_radmagrat
        
        sigma_obs, sigma_err = lens.obs_sigma

        sigma_true = mock['sigma_sample'][i]

        eps = 1e-4
        
        mdme_true = lens.mhalo / NFW.M3d(lens.rvir*lens.arcsec2kpc, lens.rs*lens.arcsec2kpc) * NFW.M2d(lens.reff_phys, lens.rs*lens.arcsec2kpc)
        
        print np.log10(mdme_true)
        
        model_lens = lens_models.gNfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mdme=mdme_true, reff_phys=lens.reff_phys, beta=1., rs_phys=10.*lens.reff_phys, delta_halo=200.)

        model_lens.images = (xa, xb)
        
        dmonly_lens = lens_models.gNfwDev(zd=lens.zd, zs=lens.zs, mstar=0., mdme=mdme_true, reff_phys=lens.reff_phys, beta=1., rs_phys=10.*lens.reff_phys, delta_halo=200.)
        unitstar_lens = lens_models.gNfwDev(zd=lens.zd, zs=lens.zs, mstar=1., mdme=0., reff_phys=lens.reff_phys, beta=1., rs_phys=10.*lens.reff_phys, delta_halo=200.)

        model_lens.normalize()
        dmonly_lens.normalize()
        unitstar_lens.normalize()

        model_lens.get_caustic()
        
        mdme_par = pymc.Uniform('mdme', lower=9., upper=13., value=np.log10(model_lens.mdme))
        beta_par = pymc.Uniform('beta', lower=0.2, upper=2.8, value=1.)
        
        pars = [mdme_par, beta_par]
        
        @pymc.deterministic()
        def mstar(p=pars):
            mdme, beta = p

            dmonly_lens.mdme = 10.**mdme
            dmonly_lens.beta = beta
            dmonly_lens.normalize()

            alpha_diff = xa - xb - dmonly_lens.alpha(xa) + dmonly_lens.alpha(xb)

            mstar = alpha_diff / (unitstar_lens.alpha(xa) - unitstar_lens.alpha(xb))

            return np.log10(mstar)

        @pymc.deterministic()
        def source(p=pars, ms=mstar):
            mdme, beta = p

            model_lens.mdme = 10.**mdme
            model_lens.beta = beta
            model_lens.mstar = 10.**ms
            model_lens.normalize()

            return xa - model_lens.alpha(xa)
             
        @pymc.deterministic()
        def timedelay(p=pars, ms=mstar, s=source):
        
            mdme, beta = p
        
            model_lens.source = s
            model_lens.mstar = 10.**ms
            model_lens.mdme = 10.**mdme
            model_lens.beta = beta
            
            model_lens.normalize()
        
            model_lens.get_timedelay()
            return model_lens.timedelay
        
        @pymc.deterministic()
        def rein(ms=mstar, mdme=mdme_par, beta=beta_par):
        
            model_lens.mstar = 10.**ms
            model_lens.mdme = 10.**mdme
            model_lens.beta = beta
            
            model_lens.normalize()
        
            model_lens.get_rein()
        
            return model_lens.rein
        
        @pymc.deterministic()
        def sigma(ms=mstar, mdme=mdme_par, beta=beta_par):
        
            s2_bulge = 10.**ms * deV_re2_s2 / model_lens.reff_phys
            s2_halo = 10.**mdme * splev(beta, gnfw_re2_s2_spline) / model_lens.reff_phys
        
            return (s2_bulge + s2_halo)**0.5
        
        @pymc.deterministic()
        def radmagratio(ms=mstar, mdme=mdme_par, beta=beta_par):
        
            model_lens.mstar = 10.**ms
            model_lens.mdme = 10.**mdme
            model_lens.beta = beta
            
            model_lens.normalize()
        
            model_lens.get_radmag_ratio()
        
            return model_lens.radmag_ratio
        
        @pymc.deterministic()
        def psi2(ms=mstar, mdme=mdme_par, beta=beta_par, re=rein):
        
            model_lens.mstar = 10.**ms
            model_lens.mdme = 10.**mdme
            model_lens.beta = beta
            
            model_lens.normalize()
        
            return (model_lens.alpha(re + eps) - model_lens.alpha(re - eps))/(2.*eps)
        
        @pymc.deterministic()
        def psi3(ms=mstar, mdme=mdme_par, beta=beta_par, re=rein):
        
            model_lens.mstar = 10.**ms
            model_lens.mdme = 10.**mdme
            model_lens.beta = beta
            
            model_lens.normalize()
        
            return (model_lens.alpha(re + eps) - 2.*model_lens.alpha(re) + model_lens.alpha(re - eps))/eps**2
            
        @pymc.deterministic()
        def logp(sig=sigma, rmu=radmagratio):
            return - 0.5*(sig - sigma_true)**2/sigma_err**2 - 0.5*(rmu - lens.radmag_ratio)**2/radmagrat_err**2

        sigma_logp = pymc.Normal('sigma_logp', mu=sigma, tau=1./sigma_err**2, value=sigma_true, observed=True)
        
        radmagrat_logp = pymc.Normal('radmagrat_logp', mu=radmagratio, tau=1./radmagrat_err**2, value=lens.radmag_ratio, observed=True)
        
        allpars = pars + [mstar, source, timedelay, rein, psi2, psi3, sigma, radmagratio, logp]
            
        M = pymc.MCMC(allpars)
        M.use_step_method(pymc.AdaptiveMetropolis, pars)
        M.sample(11000, 1000)

        for par in allpars:
            chain_file.create_dataset(str(par), data=M.trace(par)[:])


