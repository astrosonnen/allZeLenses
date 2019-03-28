import numpy as np
from scipy.interpolate import splrep, splev, splint
import pickle
import emcee
import sys
from scipy.stats import truncnorm
import h5py
import lens_models
from scipy.optimize import minimize, brentq
from mass_profiles import gNFW as gNFW_profile
from allZeTools import cgsconstants as cgs


mockname = 'devgnfw_A'

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lmstar_samp'])

lmstar_min = 10.
lmstar_max = 13.

mstar_piv = 11.6

gamma_min = 0.2
gamma_max = 2.7

lmdm5_min = 9.
lmdm5_max = 13.

dlmdm5 = 0.01
dbeta = 0.01
dgamma = 0.01

feps = 1e-8
eps = 1e-12

ngrid = 301
lmstar_grid = np.linspace(lmstar_min, lmstar_max, ngrid)

day = 24.*3600.
dt_err = day
impos_err = 0.01

# loops on the lenses. For each lens, for each value of mstar on the grid, finds mdm5, gammadm and beta that reproduce the observed image positions and time delay. Then calculates the Jacobian determinant of the variable transformation between logmdm, gammadm, beta and the observables.

#for i in range(nlens):

ngnfw_grid = 26
gnfw_grid = np.linspace(gamma_min, gamma_max, ngnfw_grid)

grids_file = h5py.File('%s_devgnfw_perfectobs_grids.hdf5'%mockname, 'w')

for i in range(nlens):
    lens = mock['lenses'][i]

    lmdm5_grid = np.zeros(ngrid)
    gammadm_grid = np.zeros(ngrid)
    beta_grid = np.zeros(ngrid)

    Jdet_grid = np.zeros(ngrid)

    # calculates the maximum allowed stellar mass, given the image positions

    thetaA_obs, thetaB_obs = lens.images

    reff = 10.**mock['lreff_samp'][i]
    rs = 10.*reff

    model_lens = lens_models.gNfwDev(zd=lens.zd, zs=lens.zs, h=lens.h, mstar=1., mdme=0., reff_phys = reff, rs_phys=rs)
    model_lens.normalize()

    model_bulge = lens_models.gNfwDev(zd=lens.zd, zs=lens.zs, h=lens.h, mstar=1., mdme=0., reff_phys = reff, rs_phys=rs)
    model_bulge.normalize()

    alphaA_bulge = model_bulge.alpha(thetaA_obs)
    alphaB_bulge = model_bulge.alpha(-thetaB_obs)

    potA_bulge = model_bulge.lenspot(thetaA_obs)
    potB_bulge = model_bulge.lenspot(-thetaB_obs)

    model_halo = lens_models.gNfwDev(zd=lens.zd, zs=lens.zs, h=lens.h, mstar=0., mdme=1., reff_phys = reff, rs_phys=rs)
    model_halo.normalize()

    alphaA_halo_grid = 0. * gnfw_grid
    alphaB_halo_grid = 0. * gnfw_grid
    potA_halo_grid = 0. * gnfw_grid
    potB_halo_grid = 0. * gnfw_grid
    for k in range(ngnfw_grid):
        model_halo.beta = gnfw_grid[k]
        model_halo.normalize()
        alphaA_halo_grid[k] = model_halo.alpha(thetaA_obs)
        alphaB_halo_grid[k] = model_halo.alpha(-thetaB_obs)
        potA_halo_grid[k] = model_halo.lenspot(thetaA_obs)
        potB_halo_grid[k] = model_halo.lenspot(-thetaB_obs)

    alphaA_halo_spline = splrep(gnfw_grid, alphaA_halo_grid)
    alphaB_halo_spline = splrep(gnfw_grid, alphaB_halo_grid)
    potA_halo_spline = splrep(gnfw_grid, potA_halo_grid)
    potB_halo_spline = splrep(gnfw_grid, potB_halo_grid)

    mstar_max = (thetaA_obs - thetaB_obs) / (alphaA_bulge + alphaB_bulge)

    gamma_guess = 2.
    for j in range(ngrid):
        model_lens.mstar = 10.**lmstar_grid[j]

        if lmstar_grid[j] < np.log10(mstar_max):
            start = np.array((gamma_guess))

            bounds = np.array((gamma_min, gamma_max)).reshape((1, 2))
        
            minimizer_kwargs = dict(method='L-BFGS-B', bounds=bounds, tol=eps)
        
            def mdme_given_gamma(gamma):

                mdme_here = (thetaA_obs - thetaB_obs - 10.**lmstar_grid[j]*(alphaB_bulge + alphaA_bulge)) / (splev(gamma, alphaA_halo_spline) + splev(gamma, alphaB_halo_spline))
                return mdme_here

            def mlogp(gamma):
        
                mdme_here = mdme_given_gamma(gamma)
                alphaA_here = 10.**lmstar_grid[j]*alphaA_bulge + mdme_here * splev(gamma, alphaA_halo_spline)
                alphaB_here = 10.**lmstar_grid[j]*alphaB_bulge + mdme_here * splev(gamma, alphaB_halo_spline)

                potA_here = 10.**lmstar_grid[j]*potA_bulge + mdme_here * splev(gamma, potA_halo_spline)
                potB_here = 10.**lmstar_grid[j]*potB_bulge + mdme_here * splev(gamma, potB_halo_spline)

                source_here = thetaA_obs - alphaA_here

                timedelay_here = lens.Dt/cgs.c*cgs.arcsec2rad**2*(0.5*(alphaB_here**2 - alphaA_here**2) + potA_here - potB_here)

                chi2 = 0.5*(lens.timedelay - timedelay_here)**2/dt_err**2
        
                return chi2
        
            res = minimize(mlogp, start, bounds=bounds)

            gamma_sol = res.x
            gamma_guess = gamma_sol
            mdme_sol = mdme_given_gamma(gamma_sol)
 
            if mlogp(gamma_sol) < 0.01:

                model_lens.beta = gamma_sol
                model_lens.mdme = mdme_sol

                model_lens.normalize()
                source_sol = thetaA_obs - model_lens.alpha(thetaA_obs)
                model_lens.source = source_sol

                gammadm_grid[j] = gamma_sol
                lmdm5_here = np.log10(mdme_sol / gNFW_profile.fast_M2d(reff, rs, gamma_sol) * gNFW_profile.fast_M2d(5., rs, gamma_sol))
                lmdm5_grid[j] = lmdm5_here
                beta_grid[j] = source_sol

                # now calculates the Jacobian of the variable transformation
                lmdm5_up = lmdm5_here + dlmdm5
                lmdme_up = np.log10(mdme_sol) + dlmdm5

                lmdm5_dw = lmdm5_here - dlmdm5
                lmdme_dw = np.log10(mdme_sol) - dlmdm5

                model_lens.mdme = 10.**lmdme_up
                model_lens.normalize()

                model_lens.get_images()
                thetaA_lmdm5_up, thetaB_lmdm5_up = model_lens.images
                model_lens.get_timedelay()
                dt_lmdm5_up = model_lens.timedelay

                model_lens.mdme = 10.**lmdme_dw
                model_lens.normalize()

                model_lens.get_images()
                thetaA_lmdm5_dw, thetaB_lmdm5_dw = model_lens.images
                model_lens.get_timedelay()
                dt_lmdm5_dw = model_lens.timedelay

                dthetaA_dlmdm5 = (thetaA_lmdm5_up - thetaA_lmdm5_dw)/(2.*dlmdm5)
                dthetaB_dlmdm5 = (thetaB_lmdm5_up - thetaB_lmdm5_dw)/(2.*dlmdm5)
                ddt_dlmdm5 = (dt_lmdm5_up - dt_lmdm5_dw)/(2.*dlmdm5)

                model_lens.mdme = mdme_sol
                model_lens.beta = gamma_sol + dgamma
                model_lens.normalize()
                model_lens.get_images()
                thetaA_gamma_up, thetaB_gamma_up = model_lens.images
                model_lens.get_timedelay()
                dt_gamma_up = model_lens.timedelay

                model_lens.mdme = mdme_sol
                model_lens.beta = gamma_sol - dgamma
                model_lens.normalize()
                model_lens.get_images()
                thetaA_gamma_dw, thetaB_gamma_dw = model_lens.images
                model_lens.get_timedelay()
                dt_gamma_dw = model_lens.timedelay

                dthetaA_dgamma = (thetaA_gamma_up - thetaA_gamma_dw)/(2.*dgamma)
                dthetaB_dgamma = (thetaB_gamma_up - thetaB_gamma_dw)/(2.*dgamma)
                ddt_dgamma = (dt_gamma_up - dt_gamma_dw)/(2.*dgamma)

                model_lens.beta = gamma_sol
                model_lens.normalize()
                model_lens.source = source_sol + dbeta
                model_lens.get_images()
                thetaA_beta_up, thetaB_beta_up = model_lens.images
                model_lens.get_timedelay()
                dt_beta_up = model_lens.timedelay

                model_lens.source = source_sol - dbeta
                model_lens.get_images()
                thetaA_beta_dw, thetaB_beta_dw = model_lens.images
                model_lens.get_timedelay()
                dt_beta_dw = model_lens.timedelay

                dthetaA_dbeta = (thetaA_beta_up - thetaA_beta_dw)/(2.*dbeta)
                dthetaB_dbeta = (thetaB_beta_up - thetaB_beta_dw)/(2.*dbeta)
                ddt_dbeta = (dt_beta_up - dt_beta_dw)/(2.*dbeta)

                J = np.array(((dthetaA_dlmdm5, dthetaA_dgamma, dthetaA_dbeta), (dthetaB_dlmdm5, dthetaB_dgamma, dthetaB_dbeta), (ddt_dlmdm5, ddt_dgamma, ddt_dbeta)))

                Jdet_grid[j] = np.linalg.det(J)

            else:
                mstar_max = 10.**lmstar_grid[j]
                lmstar_max_eff = lmstar_grid[j]

    print i, lmstar_max_eff

    group = grids_file.create_group('lens_%03d'%i)

    group.create_dataset('lmstar_grid', data=lmstar_grid)
    group.create_dataset('gammadm_grid', data=gammadm_grid)
    group.create_dataset('lmdm5_grid', data=lmdm5_grid)
    group.create_dataset('beta_grid', data=beta_grid)
    group.create_dataset('Jdet_grid', data=Jdet_grid)

    group.create_dataset('lmstar_max', data=lmstar_max_eff)


