import numpy as np
import lens_models
import pickle
from scipy.interpolate import splev, splrep, splint
from scipy.optimize import brentq
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
nlens = 20

save_every = 100

f = open('/gdrive/projects/cs82_weaklensing/PyBaWL/deV_re2_s2.dat', 'r')
deV_re2_s2 = pickle.load(f)
f.close()

f = open(grid_dir+'/gNFW_rs10reff_re2_s2_spline.dat', 'r')
gnfw_re2_s2_spline = pickle.load(f)
f.close()

f = open('/gdrive/projects/cs82_weaklensing/PyBaWL/nfw_re2_s2_grid.dat', 'r')
nfw_re2_s2_spline = pickle.load(f)
f.close()

nsamp = 1000

invH0 = np.zeros(nsamp)

beta_samp = np.zeros((nlens, nsamp))
mdme_samp = np.zeros((nlens, nsamp))
dt_model = np.zeros((nlens, nsamp))

sigma_samp = np.zeros((nlens, nsamp))
rmur_samp = np.zeros((nlens, nsamp))

xa_obs = np.zeros(nlens)
xb_obs = np.zeros(nlens)
sigma_obs = np.zeros(nlens)
rmur_obs = np.zeros(nlens)
dt_obs = np.zeros(nlens)

rmur_err = np.zeros(nlens)
sigma_err = np.zeros(nlens)
dt_err = np.zeros(nlens)

model_lenses = []
unitstar_lenses = []
dmonly_lenses = []

mdme_grids = []
nm = 101

nb = 101
beta_grid = np.linspace(0.2, 2.8, nb)

for i in range(nlens):

    lens = mock['lenses'][i]
    
    xa_obs[i] = lens.images[0]
    xb_obs[i] = lens.images[1]

    rmur_obs[i] = lens.radmag_ratio
    rmur_err[i] = lens.obs_radmagrat[1]

    dt_obs[i] = lens.timedelay
    dt_err[i] = lens.obs_timedelay[1]

    sigma_obs[i] = mock['sigma_sample'][i]
    sigma_err[i] = lens.obs_sigma[1]

    mdme_true = lens.mhalo / NFW.M3d(lens.rvir*lens.arcsec2kpc, lens.rs*lens.arcsec2kpc) * NFW.M2d(lens.reff_phys, lens.rs*lens.arcsec2kpc)
        
    model_lens = lens_models.gNfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mdme=mdme_true, reff_phys=lens.reff_phys, beta=1., rs_phys=10.*lens.reff_phys, delta_halo=200.)

    xa, xb = lens.images
    model_lens.images = (xa, xb)
        
    dmonly_lens = lens_models.gNfwDev(zd=lens.zd, zs=lens.zs, mstar=0., mdme=mdme_true, reff_phys=lens.reff_phys, beta=1., rs_phys=10.*lens.reff_phys, delta_halo=200.)
    unitstar_lens = lens_models.gNfwDev(zd=lens.zd, zs=lens.zs, mstar=1., mdme=0., reff_phys=lens.reff_phys, beta=1., rs_phys=10.*lens.reff_phys, delta_halo=200.)

    model_lens.normalize()
    dmonly_lens.normalize()
    unitstar_lens.normalize()

    model_lens.get_caustic()
        
    model_lenses.append(model_lens)
    dmonly_lenses.append(dmonly_lens)
    unitstar_lenses.append(unitstar_lens)

    mdme_samp[i, 0] = np.log10(mdme_true)
    beta_samp[i, 0] = 1.

    alpha_diff = xa - xb - dmonly_lens.alpha(xa) + dmonly_lens.alpha(xb)

    mstar = alpha_diff / (unitstar_lens.alpha(xa) - unitstar_lens.alpha(xb))

    model_lens.mstar = mstar
    model_lens.normalize()

    model_lens.source = xa - model_lens.alpha(xa)

    model_lens.get_timedelay()

    dt_model[i, 0] = model_lens.timedelay

    mdme_grid = np.linspace(9., np.log10(mdme_true) + 0.5, nm)
    mdme_grids.append(mdme_grid)
 
invH0[0] = 1./70.

for i in range(nsamp-1):

    mu_invH0 = dt_obs/(dt_model[:, i]*70.)
    sig_invH0 = dt_err/(dt_model[:, i]*70.)

    eff_sigma = ((1./sig_invH0**2).sum())**-0.5
    eff_mu = (mu_invH0/sig_invH0**2).sum() * eff_sigma**2

    invH0[i+1] = np.random.normal(eff_mu, eff_sigma, 1)

    print i, 1./invH0[i+1]

    for j in range(nlens):

        logp_mdme = 0.*mdme_grids[j]

        for k in range(nm):
            dmonly_lenses[j].mdme = 10.**mdme_grids[j][k]
            dmonly_lenses[j].normalize()
            alpha_diff = xa_obs[j] - xb_obs[j] - dmonly_lenses[j].alpha(xa_obs[j]) + dmonly_lenses[j].alpha(xb_obs[j])

            mstar = alpha_diff / (unitstar_lenses[j].alpha(xa_obs[j]) - unitstar_lenses[j].alpha(xb_obs[j]))

            model_lenses[j].mstar = mstar
            model_lenses[j].normalize()

            model_lenses[j].get_radmag_ratio()
            logp_mdme[k] = -0.5*(model_lenses[j].radmag_ratio - rmur_obs[j])**2/rmur_err[j]**2

            s2_bulge = mstar * deV_re2_s2 / model_lenses[j].reff_phys
            s2_halo = 10.**mdme_grids[j][k] * splev(beta_samp[j, i], gnfw_re2_s2_spline) / model_lenses[j].reff_phys
            sigma_here = (s2_bulge + s2_halo)**0.5
            if sigma_here == sigma_here:
                logp_mdme[k] += -0.5*(sigma_here - sigma_obs[j])**2/sigma_err[j]**2
            else:
                logp_mdme[k] = -np.inf
             
        logp_mdme -= logp_mdme.max()

        p_mdme_grid = np.exp(logp_mdme)

        p_mdme_spline = splrep(mdme_grids[j], p_mdme_grid)
     
        intfunc = lambda t: splint(mdme_grids[j][0], t, p_mdme_spline)

        norm = intfunc(mdme_grids[j][-1])

        F = np.random.rand(1) * norm

        mdme_samp[j, i+1] = brentq(lambda t: intfunc(t) - F, mdme_grids[j][0], mdme_grids[j][-1])

        model_lenses[j].mdme = 10.**mdme_samp[j, i+1]
        dmonly_lenses[j].mdme = 10.**mdme_samp[j, i+1]

        dmonly_lenses[j].normalize()

        alpha_diff = xa_obs[j] - xb_obs[j] - dmonly_lenses[j].alpha(xa_obs[j]) + dmonly_lenses[j].alpha(xb_obs[j])

        mstar = alpha_diff / (unitstar_lenses[j].alpha(xa_obs[j]) - unitstar_lenses[j].alpha(xb_obs[j]))

        model_lenses[j].mstar = mstar

        logp_beta = 0.*beta_grid

        for k in range(nb):
            dmonly_lenses[j].beta = beta_grid[k]
            dmonly_lenses[j].normalize()

            alpha_diff = xa_obs[j] - xb_obs[j] - dmonly_lenses[j].alpha(xa_obs[j]) + dmonly_lenses[j].alpha(xb_obs[j])

            mstar = alpha_diff / (unitstar_lenses[j].alpha(xa_obs[j]) - unitstar_lenses[j].alpha(xb_obs[j]))

            model_lenses[j].mstar = mstar
            model_lenses[j].beta = beta_grid[k]
            model_lenses[j].normalize()

            model_lenses[j].get_radmag_ratio()
            logp_beta[k] = -0.5*(model_lenses[j].radmag_ratio - rmur_obs[j])**2/rmur_err[j]**2

            s2_bulge = mstar * deV_re2_s2 / model_lenses[j].reff_phys
            s2_halo = model_lenses[j].mdme * splev(model_lenses[j].beta, gnfw_re2_s2_spline) / model_lenses[j].reff_phys
            sigma_here = (s2_bulge + s2_halo)**0.5
            if sigma_here == sigma_here:
                logp_beta[k] += -0.5*(sigma_here - sigma_obs[j])**2/sigma_err[j]**2
            else:
                logp_beta[k] = -np.inf

        logp_beta -= logp_beta.max()

        p_beta_grid = np.exp(logp_beta)

        p_beta_spline = splrep(beta_grid, p_beta_grid)
     
        intfunc = lambda t: splint(beta_grid[0], t, p_beta_spline)

        norm = intfunc(beta_grid[-1])

        F = np.random.rand(1) * norm

        beta_samp[j, i+1] = brentq(lambda t: intfunc(t) - F, beta_grid[0], beta_grid[-1])

        model_lenses[j].beta = beta_samp[j, i+1]
        dmonly_lenses[j].beta = beta_samp[j, i+1]

        dmonly_lenses[j].normalize()

        alpha_diff = xa_obs[j] - xb_obs[j] - dmonly_lenses[j].alpha(xa_obs[j]) + dmonly_lenses[j].alpha(xb_obs[j])

        mstar = alpha_diff / (unitstar_lenses[j].alpha(xa_obs[j]) - unitstar_lenses[j].alpha(xb_obs[j]))

        model_lenses[j].mstar = mstar

        model_lenses[j].normalize()

        model_lenses[j].source = xa_obs[j] - model_lenses[j].alpha(xa_obs[j])

        model_lenses[j].get_timedelay()

        dt_model[j, i+1] = model_lenses[j].timedelay

        model_lenses[j].get_radmag_ratio()

        s2_bulge = mstar * deV_re2_s2 / model_lenses[j].reff_phys
        s2_halo = model_lenses[j].mdme * splev(model_lenses[j].beta, gnfw_re2_s2_spline) / model_lenses[j].reff_phys
        sigma_here = (s2_bulge + s2_halo)**0.5

        sigma_samp[j, i+1] = sigma_here
        rmur_samp[j, i+1] = model_lenses[j].radmag_ratio

        print j, model_lenses[j].radmag_ratio, rmur_obs[j], sigma_here, sigma_obs[j]

    if (i+1) % save_every == 0:
        gibbs_file = h5py.File('tmp_%s_gibbs_perfect_impos_truesigma.hdf5'%mockname, 'w')
        gibbs_file.create_dataset('invH0', data=invH0)
        gibbs_file.create_dataset('mdme', data=mdme_samp)
        gibbs_file.create_dataset('beta', data=beta_samp)
        gibbs_file.create_dataset('sigma', data=sigma_samp)
        gibbs_file.create_dataset('rmur', data=rmur_samp)

        gibbs_file.close()
        
gibbs_file = h5py.File('%s_gibbs_perfect_impos_truesigma.hdf5'%mockname, 'w')
gibbs_file.create_dataset('invH0', data=invH0)
gibbs_file.create_dataset('mdme', data=mdme_samp)
gibbs_file.create_dataset('beta', data=beta_samp)
gibbs_file.create_dataset('sigma', data=sigma_samp)
gibbs_file.create_dataset('rmur', data=rmur_samp)

