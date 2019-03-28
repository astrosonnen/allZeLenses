import numpy as np
import lens_models
import pickle
from scipy.interpolate import splev
from scipy.optimize import minimize
from mass_profiles import NFW, gNFW
import os
import h5py


mockname = 'mockQ'

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
  
psi1_fit = np.zeros(nlens)
psi2_fit = np.zeros(nlens)
psi3_fit = np.zeros(nlens)
dt_fit = np.zeros(nlens)
mstar_fit = np.zeros(nlens)
mdme_fit = np.zeros(nlens)
beta_fit = np.zeros(nlens)
source_fit = np.zeros(nlens)

for i in range(nlens):

    lens = mock['lenses'][i]

    xa, xb = lens.images
    imerr = lens.obs_images[1]
    
    radmagrat_obs, radmagrat_err = lens.obs_radmagrat
    radmagrat = lens.radmag_ratio

    sigma_true = mock['sigma_sample'][i]
    sigma_err = 10.

    eps = 1e-4
    
    mdme_true = lens.mhalo / NFW.M3d(lens.rvir*lens.arcsec2kpc, lens.rs*lens.arcsec2kpc) * NFW.M2d(lens.reff_phys, lens.rs*lens.arcsec2kpc)
    
    print np.log10(mdme_true)
    
    model_lens = lens_models.gNfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mdme=mdme_true, reff_phys=lens.reff_phys, beta=1., rs_phys=10.*lens.reff_phys, delta_halo=200.)
    dmonly_lens = lens_models.gNfwDev(zd=lens.zd, zs=lens.zs, mstar=0., mdme=mdme_true, reff_phys=lens.reff_phys, beta=1., rs_phys=10.*lens.reff_phys, delta_halo=200.)
    unitstar_lens = lens_models.gNfwDev(zd=lens.zd, zs=lens.zs, mstar=1., mdme=0., reff_phys=lens.reff_phys, beta=1., rs_phys=10.*lens.reff_phys, delta_halo=200.)
    
    model_lens.images = (xa, xb)

    model_lens.normalize()
    dmonly_lens.normalize()
    unitstar_lens.normalize()

    beta_min = 0.2
    beta_max = 2.8

    mdme_min = np.log10(model_lens.mdme) - 0.5
    mdme_max = np.log10(model_lens.mdme) + 0.5

    start = np.array([1., np.log10(model_lens.mdme)])

    bounds = np.array(((beta_min, beta_max), (mdme_min, mdme_max)))

    scale_free_bounds = 0.*bounds
    scale_free_bounds[:, 1] = 1.

    scale_free_guess = (start - bounds[:, 0])/(bounds[:, 1] - bounds[:, 0])

    feps = 1e-5

    minimizer_kwargs = dict(method='L-BFGS-B', bounds=scale_free_bounds, tol=eps)

    def mlogp(x):

        chi2 = 0.

        p = x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

        beta, mdme = p

        dmonly_lens.mdme = 10.**mdme
        dmonly_lens.beta = beta
        dmonly_lens.normalize()
        alpha_diff = xa - xb - dmonly_lens.alpha(xa) + dmonly_lens.alpha(xb)

        mstar = alpha_diff / (unitstar_lens.alpha(xa) - unitstar_lens.alpha(xb))

        model_lens.mdme = 10.**mdme
        model_lens.beta = beta
        model_lens.mstar = mstar
        model_lens.normalize()

        model_lens.get_radmag_ratio()
        
        s2_bulge = mstar * deV_re2_s2 / model_lens.reff_phys
        s2_halo = 10.**mdme * splev(beta, gnfw_re2_s2_spline) / model_lens.reff_phys

        s2_tot = s2_bulge + s2_halo
        if s2_tot < 0.:
            sigma_model = 0.
        else:
            sigma_model = s2_tot**0.5

        chi2 = 0.5*(sigma_model - sigma_true)**2/sigma_err**2

        chi2 += 0.5*(model_lens.radmag_ratio - lens.radmag_ratio)**2/radmagrat_err**2

        return chi2

    res = minimize(mlogp, scale_free_guess, bounds=scale_free_bounds)

    beta, mdme = res.x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    dmonly_lens.mdme = 10.**mdme
    dmonly_lens.beta = beta
    dmonly_lens.normalize()
    alpha_diff = xa - xb - dmonly_lens.alpha(xa) + dmonly_lens.alpha(xb)

    mstar = alpha_diff / (unitstar_lens.alpha(xa) - unitstar_lens.alpha(xb))

    model_lens.mdme = 10.**mdme
    model_lens.beta = beta
    model_lens.mstar = mstar
    model_lens.normalize()

    model_lens.get_radmag_ratio()
    model_lens.source = xa - model_lens.alpha(xa)
 
    beta_fit[i] = beta
    mdme_fit[i] = mdme
    mstar_fit[i] = np.log10(mstar)
    source_fit[i] = model_lens.source
    model_lens.get_timedelay()

    s2_bulge = mstar * deV_re2_s2 / model_lens.reff_phys
    s2_halo = 10.**mdme * splev(beta, gnfw_re2_s2_spline) / model_lens.reff_phys

    sigma_model = (s2_bulge + s2_halo)**0.5

    model_lens.get_rein()

    psi1_fit[i] = model_lens.rein

    psi2_fit[i] = (model_lens.alpha(model_lens.rein + eps) - model_lens.alpha(model_lens.rein - eps))/(2.*eps)
        
    psi3_fit[i] = (model_lens.alpha(model_lens.rein + eps) - 2.*model_lens.alpha(model_lens.rein) + model_lens.alpha(model_lens.rein - eps))/eps**2
  
    dt_fit[i] = model_lens.timedelay

    print lens.radmag_ratio, model_lens.radmag_ratio
    print sigma_true, sigma_model

    print model_lens.timedelay/lens.timedelay

fit_file = h5py.File('%s_gnfwdev_perfect_data.hdf5'%mockname, 'w')

fit_file.create_dataset('psi1', data=psi1_fit)
fit_file.create_dataset('psi2', data=psi2_fit)
fit_file.create_dataset('psi3', data=psi3_fit)
fit_file.create_dataset('timedelay', data=dt_fit)
fit_file.create_dataset('mstar', data=mstar_fit)
fit_file.create_dataset('mdme', data=mdme_fit)
fit_file.create_dataset('beta', data=beta_fit)
fit_file.create_dataset('source', data=source_fit)

fit_file.close()


