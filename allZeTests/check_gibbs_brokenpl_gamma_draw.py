import pickle
import h5py
import numpy as np
import pylab
import lens_models


f = open('mockI.dat', 'r')
mock = pickle.load(f)
f.close()

lensno = 0
step = 69

gibbs = h5py.File('tmp_mockI_brokenpl_gibbs_sample_fast.hdf5', 'r')

day = 24.*3600.

xA_obs, xB_obs = mock['lenses'][lensno].obs_images[0]
xA_err = mock['lenses'][lensno].obs_images[1]
xB_err = mock['lenses'][lensno].obs_images[1]

radmagrat_obs, radmagrat_err = mock['lenses'][lensno].obs_radmagrat
dt_obs, dt_err = mock['lenses'][lensno].obs_timedelay
dt_obs = dt_obs/day
dt_err = dt_err/day

lens = lens_models.sps_ein_break(zd=mock['lenses'][lensno].zd, zs=mock['lenses'][lensno].zs)

lens.rein = gibbs['rein'][lensno, step]
lens.source = gibbs['s2'][lensno, step]**0.5
lens.beta = gibbs['beta'][lensno, step-1]

gamma = gibbs['gamma'][lensno, step]

invH0 = gibbs['invH0'][step]

mu_gamma = gibbs['mu_gamma'][step]
sig_gamma = gibbs['sig_gamma'][step]

print mu_gamma, sig_gamma
print gibbs['gamma'][lensno, step]
print gibbs['invH0'][step] * gibbs['dt_model'][lensno, step] * 70., mock['lenses'][lensno].obs_timedelay[0]/day

gamma_min = 1.2
gamma_max = 2.8
ngrid_gamma = 161
gamma_grid = np.linspace(gamma_min, gamma_max, ngrid_gamma)

logp_gamma = 0.*gamma_grid

dtmodel_grid = 0.*gamma_grid
xAmodel_grid = 0.*gamma_grid
xBmodel_grid = 0.*gamma_grid
radmagratmodel_grid = 0.*gamma_grid

for k in range(ngrid_gamma):
    lens.gamma = gamma_grid[k]
    lens.get_images()

    if np.isfinite(lens.images[0]):
        lens.get_timedelay()
        lens.get_radmag_ratio()
    
        logp_gamma[k] = -0.5*(lens.images[0] - xA_obs)**2/xA_err**2 - 0.5*(lens.images[1] - xB_obs)**2/xB_err**2
        logp_gamma[k] += -0.5*(lens.radmag_ratio - radmagrat_obs)**2/radmagrat_err**2
        logp_gamma[k] += -0.5*(invH0 * lens.timedelay/day * 70. - dt_obs)**2/dt_err**2

        dtmodel_grid[k] = invH0 * lens.timedelay/day * 70.
        xAmodel_grid[k] = lens.images[0]
        xBmodel_grid[k] = lens.images[1]
        radmagratmodel_grid[k] = lens.radmag_ratio
    else:
        logp_gamma[k] = -np.inf

logp_gamma += -0.5*(gamma_grid - mu_gamma)**2/sig_gamma**2 - np.log(sig_gamma)

logp_gamma -= logp_gamma.max()

p_gamma_grid = np.exp(logp_gamma)

pylab.plot(gamma_grid, p_gamma_grid)
pylab.show()

pylab.subplot(2, 2, 1)
pylab.plot(gamma_grid, dtmodel_grid)
pylab.axhspan(dt_obs - dt_err, dt_obs + dt_err, color='gray', alpha=0.5)

pylab.subplot(2, 2, 2)
pylab.plot(gamma_grid, radmagratmodel_grid)
pylab.axhspan(radmagrat_obs - radmagrat_err, radmagrat_obs + radmagrat_err, color='gray', alpha=0.5)

pylab.subplot(2, 2, 3)
pylab.plot(gamma_grid, xAmodel_grid)
pylab.axhspan(xA_obs - xA_err, xA_obs + xA_err, color='gray', alpha=0.5)

pylab.subplot(2, 2, 4)
pylab.plot(gamma_grid, xBmodel_grid)
pylab.axhspan(xB_obs - xB_err, xB_obs + xB_err, color='gray', alpha=0.5)

pylab.show()

p_gamma_spline = splrep(gamma_grid, p_gamma_grid)
     
intfunc = lambda t: splint(gamma_min, t, p_gamma_spline)

norm = intfunc(gamma_max)

F = np.random.rand(1) * norm

gamma[j, i+1] = brentq(lambda t: intfunc(t) - F, gamma_min, gamma_max)


