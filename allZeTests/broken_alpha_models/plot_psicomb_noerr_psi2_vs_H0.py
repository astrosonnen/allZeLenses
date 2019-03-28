import h5py
import pickle
import lens_models
from scipy.optimize import brentq
from scipy.interpolate import splrep, splev
import numpy as np
import pylab
import pymc
from plotters import cornerplot


mockname = 'mockI'
mockdir = '/gdrive/projects/allZeLenses/allZeTests/'

f = open(mockdir+mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

grid_file = h5py.File('%s_psicombfit_noerr_grids.hdf5'%mockname, 'r')

eps = 1e-4

true_psi2 = np.zeros(nlens)
inferred_psi2 = np.zeros(nlens)
wrongH0_psi2 = np.zeros(nlens)

rein = np.zeros(nlens)
reff = np.zeros(nlens)

H0 = 70.
wrong_H0 = 77.

day = 24.*3600.

psi2_grid = grid_file['psi2_grid'].value.copy()

for i in range(nlens):

    lens = mock['lenses'][i]
    rein[i] = lens.rein
    reff[i] = lens.reff

    true_psi2[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)

    group = grid_file['lens_%02d'%i]
    good = np.logical_not(group['outside_grid'])

    ngrid = good.sum()

    dt_grid = group['dt_grid'][good]

    beta_grid = group['beta_grid'][good]
    gamma_grid = group['gamma_grid'][good]
    source_grid = group['source_grid'][good]

    dt_model = dt_grid * 70. / H0

    dt_obs = lens.timedelay / day

    dt_spline = splrep(np.arange(ngrid), dt_model)
    wrong_dt_spline = splrep(np.arange(ngrid), dt_grid * 70. / wrong_H0)
    gamma_spline = splrep(np.arange(ngrid), gamma_grid)
    beta_spline = splrep(np.arange(ngrid), beta_grid)
    source_spline = splrep(np.arange(ngrid), source_grid)
    psi2_spline = splrep(np.arange(ngrid), psi2_grid[good])

    ind_fit = brentq(lambda x: splev(x, dt_spline) - dt_obs, 0., ngrid-1.)
    wrong_ind_fit = brentq(lambda x: splev(x, wrong_dt_spline) - dt_obs, 0., ngrid-1.)

    inferred_psi2[i] = splev(ind_fit, psi2_spline)
    wrongH0_psi2[i] = splev(wrong_ind_fit, psi2_spline)


def fit_psi2_dist(psi2_sample):
    mu_par = pymc.Uniform('mu', lower=-1., upper=1., value=0.)
    sig_par = pymc.Uniform('sig', lower=0., upper=1., value=0.2)
    mstar_dep = pymc.Uniform('beta', lower=-2., upper=2., value=0.)
    reff_dep = pymc.Uniform('xi', lower=-2., upper=2., value=0.)
    
    pars = [mu_par, sig_par, mstar_dep, reff_dep]
    
    mstar = mock['mstar_sample']
    mstar_piv = mstar.mean()
    
    reff = np.log10(mock['reff_sample'])
    reff_piv = reff.mean()
    
    rein_piv = rein.mean()
    
    @pymc.deterministic
    def like(p=pars):
    
        mu, sig, beta, xi = p
    
        muhere = mu + beta * (mstar - mstar_piv) + xi * (reff - reff_piv)
    
        logp = -0.5*(muhere - psi2_sample)**2/sig**2 - np.log(sig)
    
        return logp.sum()
    
    @pymc.stochastic
    def logp(p=pars, value=0., observed=True):
        return like
    
    M = pymc.MCMC(pars)
    M.sample(11000, 1000)

    output = {}
    for par in pars:
        output[str(par)] = M.trace(par)[:]

    return output

truepsi2_fit = fit_psi2_dist(true_psi2)
wrongH0_fit = fit_psi2_dist(wrongH0_psi2)

cp = []
for par in truepsi2_fit:
    cp.append({'data': [truepsi2_fit[par], wrongH0_fit[par]], 'label': par})

cornerplot(cp, color=('r', 'b'))
pylab.show()


