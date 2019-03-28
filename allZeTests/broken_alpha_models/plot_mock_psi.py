import lens_models
from toy_models import sample_generator
import pylab
import pickle
import h5py
import os
import pymc
import numpy as np
from scipy.optimize import minimize
from plotters import probcontour


mockname = 'mockL'
chaindir = '/Users/sonnen/allZeChains/'
mockdir = '/gdrive/projects/allZeLenses/allZeTests/'

eps = 1e-4

ngrid = 81
psi2_min = -0.4
psi2_max = 0.4

psi2_grid = np.linspace(psi2_min, psi2_max, ngrid)

f = open(mockdir+mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

"""
f = open('brokenpowerlaw_psi2_ndinterp.dat', 'r')
psi2_ndinterp = pickle.load(f)
f.close()

f = open('brokenpowerlaw_psi3_ndinterp.dat', 'r')
psi3_ndinterp = pickle.load(f)
f.close()
"""

gamma_min = 1.2
gamma_max = 2.8

beta_min = -0.5
beta_max = 0.5

start = np.array((2., 0.))
bounds = np.array(((gamma_min, gamma_max), (beta_min, beta_max)))
scale_free_bounds = 0.*bounds
scale_free_bounds[:, 1] = 1.

scale_free_guess = (start - bounds[:, 0])/(bounds[:, 1] - bounds[:, 0])

minimizer_kwargs = dict(method="L-BFGS-B", bounds=scale_free_bounds, tol=eps)

gammas_grid = np.linspace(1.4, 2.6, 11)
betas_grid = np.linspace(-1., 1., 101)

bpl_lens = lens_models.broken_alpha_powerlaw(rein=1.)

for j in range(11):
    bpl_lens.gamma = gammas_grid[j]

    psi2_thislens = 0.*betas_grid
    psi3_thislens = 0.*betas_grid

    for k in range(101):
        bpl_lens.beta = betas_grid[k]

        psi2_thislens[k] = bpl_lens.psi2()
        psi3_thislens[k] = bpl_lens.psi3()

    pylab.plot(psi2_thislens, psi3_thislens, color='gray')

betas_grid = np.linspace(-1., 1., 11)
gammas_grid = np.linspace(1.5, 2.5, 101)

for j in range(11):
    bpl_lens.beta = betas_grid[j]

    psi2_thislens = 0.*gammas_grid
    psi3_thislens = 0.*gammas_grid

    for k in range(101):
        bpl_lens.gamma = gammas_grid[k]

        psi2_thislens[k] = bpl_lens.psi2()
        psi3_thislens[k] = bpl_lens.psi3()

    pylab.plot(psi2_thislens, psi3_thislens, color='gray')

psi1_mock = np.zeros(nlens)
psi2_mock = np.zeros(nlens)
psi3_mock = np.zeros(nlens)

for i in range(nlens):

    lens = mock['lenses'][i]

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    psi1_true = lens.rein

    psi2_true = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    psi3_true = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

    print i, psi2_true, psi1_true * psi3_true

    psi1_mock[i] = psi1_true
    psi2_mock[i] = psi2_true
    psi3_mock[i] = psi3_true

    a_true = psi3_true / (1. - psi2_true)

    gamma_grid = 0. * psi2_grid
    beta_grid = 0. * psi2_grid

    psi3_grid = a_true * (1. - psi2_grid)

    #chain = h5py.File(chaindir+mockname+'_lens_%02d_psifit_brokenalpha_flatprior.hdf5'%i, 'r')

    #probcontour(chain['psi2'], chain['rein'].value * chain['psi3'], style='g')

pylab.scatter(psi2_mock, psi1_mock * psi3_mock, color='r')
pylab.show()

