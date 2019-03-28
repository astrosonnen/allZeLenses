import h5py
import pickle
import lens_models
from scipy.optimize import brentq
from scipy.interpolate import splrep, splev
import numpy as np
import pylab


mockname = 'mockK'
mockdir = '/gdrive/projects/allZeLenses/allZeTests/'

f = open(mockdir+mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])
nlens = 90

grid_file = h5py.File('%s_psicombfit_noerr_grids.hdf5'%mockname, 'r')

source = np.zeros(nlens)
true_source = np.zeros(nlens)
beta_max = np.zeros(nlens)
true_beta_max = np.zeros(nlens)

wrong_source = np.zeros(nlens)
wrong_beta_max = np.zeros(nlens)

max_asymm = 0.5

H0 = 70.
wrong_H0 = 72.

day = 24.*3600.

for i in range(nlens):

    lens = mock['lenses'][i]

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

    ind_fit = brentq(lambda x: splev(x, dt_spline) - dt_obs, 0., ngrid-1.)
    wrong_ind_fit = brentq(lambda x: splev(x, wrong_dt_spline) - dt_obs, 0., ngrid-1.)

    source[i] = splev(ind_fit, source_spline)
    wrong_source[i] = splev(wrong_ind_fit, source_spline)
    true_source[i] = lens.source

    model_lens = lens_models.broken_alpha_powerlaw(zd=lens.zd, zs=lens.zs, rein=lens.rein, gamma=splev(ind_fit, gamma_spline), beta=splev(ind_fit, beta_spline))

    smax = model_lens.rein*1.6 - model_lens.alpha(model_lens.rein*1.6)
    smin = model_lens.rein*1.1 - model_lens.alpha(model_lens.rein*1.1)

    def asymm_func(s):
        model_lens.source = s
        model_lens.get_images()
        asymm = (model_lens.images[0] + model_lens.images[1])/(model_lens.images[0] - model_lens.images[1])
        return asymm - max_asymm

    beta_max[i] = brentq(asymm_func, smin, smax, xtol=1e-8)

    wrong_model_lens = lens_models.broken_alpha_powerlaw(zd=lens.zd, zs=lens.zs, rein=lens.rein, gamma=splev(wrong_ind_fit, gamma_spline), beta=splev(wrong_ind_fit, beta_spline))

    wrong_smax = wrong_model_lens.rein*1.6 - wrong_model_lens.alpha(wrong_model_lens.rein*1.6)
    wrong_smin = wrong_model_lens.rein*1.1 - wrong_model_lens.alpha(wrong_model_lens.rein*1.1)

    def wrong_asymm_func(s):
        wrong_model_lens.source = s
        wrong_model_lens.get_images()
        asymm = (wrong_model_lens.images[0] + wrong_model_lens.images[1])/(wrong_model_lens.images[0] - wrong_model_lens.images[1])
        return asymm - max_asymm

    wrong_beta_max[i] = brentq(wrong_asymm_func, smin, smax, xtol=1e-8)

    def true_asymm_func(s):
        lens.source = s
        lens.get_images()
        asymm = (lens.images[0] + lens.images[1])/(lens.images[0] - lens.images[1])
        return asymm - max_asymm

    true_beta_max[i] = brentq(true_asymm_func, smin, smax, xtol=1e-8)

    print i, source[i], true_source[i], wrong_source[i]

good = (beta_max > 0.) & (true_beta_max > 0.) & (wrong_beta_max > 0.)

bins = np.linspace(0., 1., 21)
pylab.hist((source/beta_max)[good]**2, bins=bins, histtype='step')
pylab.hist((true_source/true_beta_max)[good]**2, bins=bins, histtype='step')
pylab.hist((wrong_source/wrong_beta_max)[good]**2, bins=bins, histtype='step')
pylab.show()

pylab.hist(source[good], histtype='step')
pylab.hist(true_source[good], histtype='step')
pylab.hist(wrong_source[good], histtype='step')
pylab.show()

