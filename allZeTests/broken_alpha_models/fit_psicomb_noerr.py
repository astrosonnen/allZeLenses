import numpy as np
import ndinterp
from scipy.interpolate import splrep
import lens_models
import pickle
from scipy.optimize import minimize
import pylab
import h5py


mockname = 'mockN'
mockdir = '/gdrive/projects/allZeLenses/allZeTests/'

gamma_min = 1.2
gamma_max = 2.8

beta_min = -1.
beta_max = 1.

day = 24.*3600.

f = open(mockdir+mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

f = open('psi2_ndinterp.dat', 'r')
psi2_ndinterp = pickle.load(f)
f.close()

f = open('psi3_ndinterp.dat', 'r')
psi3_ndinterp = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

ngrid = 101
psi2_grid = np.linspace(-0.5, 0.5, ngrid)

start = np.array((2., 0.))
bounds = np.array(((gamma_min, gamma_max), (beta_min, beta_max)))
scale_free_bounds = 0.*bounds
scale_free_bounds[:, 1] = 1.

scale_free_guess = (start - bounds[:, 0])/(bounds[:, 1] - bounds[:, 0])

eps = 1e-4

fiteps = 1e-5

minimizer_kwargs = dict(method="L-BFGS-B", bounds=scale_free_bounds, tol=fiteps)

grid_file = h5py.File('%s_psicombfit_noerr_grids.hdf5'%mockname, 'w')
grid_file.create_dataset('psi2_grid', data=psi2_grid)

for i in range(nlens):

    print i

    group = grid_file.create_group('lens_%02d'%i)

    lens = mock['lenses'][i]

    bpl_lens = lens_models.sps_ein_break(zd=lens.zd, zs=lens.zs, rein=lens.rein)

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    psi1_true = lens.rein

    psi2_true = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    psi3_true = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

    a_true = psi3_true / (1. - psi2_true)

    source_grid = 0. * psi2_grid
    dt_grid = 0. * psi2_grid
    psi3_given_psi2 = a_true * (1. - psi2_grid)
    outside_grid = np.zeros(ngrid, dtype=bool)
    gamma_grid = 0. * psi2_grid
    beta_grid = 0. * psi2_grid

    model_lens = lens_models.broken_alpha_powerlaw(zd=lens.zd, zs=lens.zs, rein=psi1_true)

    for j in range(ngrid):

        def min_func(x):
            p = x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
            
            return (psi2_ndinterp.eval(p.reshape((1, 2))) - psi2_grid[j])**2 + (psi3_ndinterp.eval(p.reshape((1, 2)))/psi1_true - psi3_given_psi2[j])**2

        res = minimize(min_func, scale_free_guess, bounds=scale_free_bounds)

        if min_func(res.x) > fiteps:
            outside_grid[j] = True

        else:
            gamma, beta = res.x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

            model_lens.gamma = gamma
            model_lens.beta = beta

            model_lens.source = lens.images[0] - model_lens.alpha(lens.images[0])

            model_lens.get_images()
            model_lens.get_timedelay()

            dt_grid[j] = model_lens.timedelay / day

            gamma_grid[j] = gamma
            beta_grid[j] = beta
            source_grid[j] = model_lens.source

    group.create_dataset('psi3_grid', data=psi3_given_psi2)
    group.create_dataset('gamma_grid', data=gamma_grid)
    group.create_dataset('beta_grid', data=beta_grid)
    group.create_dataset('dt_grid', data=dt_grid)
    group.create_dataset('outside_grid', data=outside_grid)
    group.create_dataset('source_grid', data=source_grid)


