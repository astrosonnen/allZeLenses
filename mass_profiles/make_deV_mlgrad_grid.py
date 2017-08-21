#definitions of the Sersic profile

import numpy as np
from scipy.special import gamma as gfunc
import pickle
from scipy.integrate import quad
from scipy.interpolate import splrep, splint
import os
from sonnentools.cgsconstants import *
from vdmodel_2013 import sigma_model
from vdmodel_2013.profiles import deVaucouleurs
import ndinterp


grid_dir = os.environ.get('ATL_GRIDDIR')

ndeV = 4.

def b(n):
    return 2*n - 1./3. + 4/405./n + 46/25515/n**2

def L(n,Re):
    return Re**2*2*np.pi*n/b(n)**(2*n)*gfunc(2*n)

def I(R, n, Re, beta):
    return np.exp(-b(n)*(R/Re)**(1./n))/L(n,Re) * (R/Re)**beta

def deproject(r, Re, beta):
    deriv = lambda R: -b(ndeV)/ndeV*(R/Re)**(1/ndeV)/R*I(R, ndeV, Re, beta) + beta/Re * I(R, ndeV, Re, beta) / (R/Re)

    rho = -1/np.pi*quad(lambda R: deriv(R)/np.sqrt(R**2 - r**2), r, np.inf)[0]
    return rho

def rho(r, Re, beta):
    r = np.atleast_1d(r)
    out = 0.*r
    for i in range(0,len(r)):
        out[i] = deproject(r[i], Re, beta)
    return out

m3d_grids = []
norms = []

nb = 21
nr = 1001
nr2d = 1001
nrs = 101

beta_min = -0.5
beta_grid = np.linspace(beta_min, 0., nb)
r3d_grid = np.logspace(-3, 3., nr)
r2d_grid = np.logspace(-2., 2., nr2d)
rs_grid = np.logspace(0., 3., nrs)

beta_spline = splrep(beta_grid, np.arange(nb))
r2d_spline = splrep(r2d_grid, np.arange(nr2d))

axes = {0: beta_spline, 1: r2d_spline}

m2d_grid = np.zeros((nb, nr2d))

norm_grid = np.zeros(nb)

pot_grid = np.zeros((nb, nr2d))

s2_grid = np.zeros(nb)

for i in range(nb):
    print i
    rhos = rho(r3d_grid, 1., beta_grid[i])
    rs0 = np.array([0.] + list(r3d_grid))
    mp0 = np.array([0.] + list(4.*np.pi*rhos*r3d_grid**2))

    mprime_spline = splrep(rs0, mp0)

    m3d_grid = 0.*r3d_grid
    for j in range(nr):
        m3d_grid[j] = splint(0., r3d_grid[j], mprime_spline)

    norm = 1./m3d_grid[-1]
    norms.append(norm)
    norm_grid[i] = norm

    m3d_grid *= norm

    # now calculates 2d enclosed masses
    for j in range(nr2d):
        m2d_grid[i, j] = norm * 2.*np.pi * quad(lambda R: R*I(R, ndeV, 1., beta_grid[i]), 0., r2d_grid[j])[0]
        pot_grid[i, j] = norm * 2*quad(lambda r: r*I(r, ndeV, 1., beta_grid[i])*np.log(r2d_grid[j]/r),0.,r2d_grid[j])[0]

    # now makes the dynamics grid

    s2_grid[i] = sigma_model.sigma2general((r3d_grid, m3d_grid), 0.5, lp_pars=1., seeing=None, light_profile=deVaucouleurs)

s2_grid = G * M_Sun / 10.**10 / kpc * s2_grid
s2_spline = splrep(beta_grid, s2_grid)

norm_spline = splrep(beta_grid, norm_grid)

f = open('deV_mlgrad_norm_spline.dat', 'w')
pickle.dump(norm_spline, f)
f.close()

f = open('deV_mlgrad_re2_s2_spline.dat', 'w')
pickle.dump(s2_spline, f)
f.close()

m2d_interp = ndinterp.ndInterp(axes, m2d_grid, order=3)

f = open('deV_mlgrad_m2d_grid.dat', 'w')
pickle.dump(m2d_interp,f)
f.close()

pot_interp = ndinterp.ndInterp(axes, pot_grid, order=3)

f = open('deV_mlgrad_lenspot_grid.dat', 'w')
pickle.dump(pot_interp,f)
f.close()


