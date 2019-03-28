import h5py
import pickle
import lens_models
from scipy.optimize import brentq
from scipy.interpolate import splrep, splev
import numpy as np
import pylab
import pymc
from plotters import cornerplot


mockname = 'mockN'
mockdir = '/gdrive/projects/allZeLenses/allZeTests/'

f = open(mockdir+mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])
nlens = 100

grid_file = h5py.File('%s_psicombfit_noerr_grids.hdf5'%mockname, 'r')

eps = 1e-4

day = 24.*3600.

psi2_grid = grid_file['psi2_grid'].value.copy()

H0_min = -np.inf
H0_max = np.inf

true_psi2 = np.zeros(nlens)

psi2_splines = []
for i in range(nlens):

    lens = mock['lenses'][i]

    true_psi2[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)

    group = grid_file['lens_%02d'%i]
    good = np.logical_not(group['outside_grid'])

    ngrid = good.sum()

    dt_grid = group['dt_grid'][good]

    dt_obs = lens.timedelay / day

    H0_grid = dt_grid / dt_obs * 70.

    H0_maxhere = H0_grid.min()
    H0_minhere = H0_grid.max()

    psi2_spline = splrep(np.flipud(H0_grid), np.flipud(psi2_grid[good]))

    psi2_splines.append(psi2_spline)

    if H0_maxhere < H0_max:
        H0_max = H0_maxhere
    if H0_minhere > H0_min:
        H0_min = H0_minhere

print H0_min, H0_max

mu_par = pymc.Uniform('mu', lower=-1., upper=1., value=0.)
sig_par = pymc.Uniform('sig', lower=0., upper=1., value=0.2)
mstar_dep = pymc.Uniform('beta', lower=-2., upper=2., value=0.)
reff_dep = pymc.Uniform('xi', lower=-2., upper=2., value=0.)
H0_par = pymc.Uniform('H0', lower=50., upper=90., value=70.)

pars = [mu_par, sig_par, mstar_dep, reff_dep, H0_par]

mstar = mock['mstar_sample']
mstar_piv = mstar.mean()

reff = np.log10(mock['reff_sample'])
reff_piv = reff.mean()

@pymc.deterministic
def like(p=pars):

    mu, sig, beta, xi, H0 = p

    muhere = mu + beta * (mstar - mstar_piv) + xi * (reff - reff_piv)

    psi2_here = np.zeros(nlens)
    for i in range(nlens):
        psi2_here[i] = splev(H0, psi2_splines[i])

    logp = -0.5*(muhere - psi2_here)**2/sig**2 - np.log(sig)

    return logp.sum()

@pymc.stochastic
def logp(p=pars, value=0., observed=True):
    return like

M = pymc.MCMC(pars)
M.sample(11000, 1000)

output = {}
for par in pars:
    output[str(par)] = M.trace(par)[:]

cp = []
for par in output:
    cp.append({'data': output[par], 'label': par})

cornerplot(cp, color='r')
pylab.show()


