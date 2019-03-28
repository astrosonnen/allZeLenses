import numpy as np
import pickle
import ndinterp
import lens_models
from scipy.interpolate import splrep, splint
from scipy.integrate import quad
from vdmodel_2013 import sigma_model
from vdmodel_2013.profiles import deVaucouleurs


# there are three parameters: reff/rein ratio, beta, gamma
# before doing the dynamics calculation, we need to calculate the 3d profile. This has two free parameters (plus radius)

ng = 17
gamma_min = 1.2
gamma_max = 2.8

gamma_grid = np.linspace(gamma_min, gamma_max, ng)

nb = 20
beta_min = -1.
beta_max = 1.

beta_grid = np.linspace(beta_min, beta_max, nb)

nr = 10001
r_grid = np.logspace(-4., 2., nr)

nr3d = 1001
r3d_grid = np.logspace(-3., 3., nr3d)

nreff = 101
reff_grid = np.linspace(0.1, 10., nreff)

gamma_spline = splrep(gamma_grid, np.arange(ng))
beta_spline = splrep(beta_grid, np.arange(nb))
reff_spline = splrep(reff_grid, np.arange(nreff))

axes = {0: gamma_spline, 1: beta_spline, 2: reff_spline}

def Sigma_deriv(R, gamma, beta):
    return 2.**(beta-1.)/np.pi * ((3.-gamma)*(1.-gamma)*R**(-gamma)*(1.+R)**(-beta) - (5.- 2.*gamma)*beta*R**(1.-gamma)*(1.+R)**(-1.-beta) + beta*(1.+beta)*R**(2.-gamma)*(1.+R)**(-2.-beta))

def deproject(r, gamma, beta):
    rho = -1./np.pi*quad(lambda R: Sigma_deriv(R, gamma, beta)/np.sqrt(R**2 - r**2), r, np.inf)[0]
    return rho

def rho(r, gamma, beta):
    r = np.atleast_1d(r)
    out = 0.*r
    for i in range(0,len(r)):
        out[i] = deproject(r[i], gamma, beta)
    return out

s2_grid = np.zeros((ng, nb, nreff))

for i in range(ng):
    print i

    for j in range(nb):
        print j
        rhos = rho(r_grid, gamma_grid[i], beta_grid[j])

        rs0 = np.array([0.] + list(r_grid))
        mp0 = np.array([0.] + list(4.*np.pi*rhos*r_grid**2))

        mprime_spline = splrep(rs0, mp0)

        m3d_grid = 0.*r3d_grid
        for k in range(nr3d):
            m3d_grid[k] = splint(0., r3d_grid[k], mprime_spline)

        for k in range(nreff):
            s2_grid[i, j, k] = sigma_model.sigma2general((r3d_grid, m3d_grid), 0.5 * reff_grid[k], lp_pars=reff_grid[k], seeing=None, light_profile=deVaucouleurs)

f = open('broken_alpha_s2_grid.dat', 'w')
pickle.dump(s2_grid, f)
f.close()

s2_ndinterp = ndinterp.ndInterp(axes, s2_grid, order=1)
f = open('broken_alpha_s2_ndinterp.dat', 'w')
pickle.dump(s2_ndinterp, f)
f.close()

