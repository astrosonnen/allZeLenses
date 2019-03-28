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
dt_obs /= day
dt_err /= day

lens = lens_models.sps_ein_break(zd=mock['lenses'][lensno].zd, zs=mock['lenses'][lensno].zs)

lens.rein = gibbs['rein'][lensno, step]
lens.source = gibbs['s2'][lensno, step]**0.5
gamma = gibbs['gamma'][lensno, step]

invH0 = gibbs['invH0'][step]

mu_beta = gibbs['mu_beta'][step]
sig_beta = gibbs['sig_beta'][step]

print mu_beta, sig_beta
print gibbs['beta'][lensno, step]
print gibbs['invH0'][step] * gibbs['dt_model'][lensno, step] * 70., mock['lenses'][lensno].obs_timedelay[0]/day

beta_min = -1.
beta_max = 1.
ngrid_beta = 201
beta_grid = np.linspace(beta_min, beta_max, ngrid_beta)

logp_beta = 0.*beta_grid

dtmodel_grid = 0.*beta_grid
xAmodel_grid = 0.*beta_grid
xBmodel_grid = 0.*beta_grid
radmagratmodel_grid = 0.*beta_grid

for k in range(ngrid_beta):
    lens.gamma = gamma#beta_grid[k] + gamma
    lens.beta = beta_grid[k]
    lens.get_images()

    if np.isfinite(lens.images[0]):
        lens.get_timedelay()
        lens.get_radmag_ratio()
    
        logp_beta[k] = -0.5*(lens.images[0] - xA_obs)**2/xA_err**2 - 0.5*(lens.images[1] - xB_obs)**2/xB_err**2
        logp_beta[k] += -0.5*(lens.radmag_ratio - radmagrat_obs)**2/radmagrat_err**2
        logp_beta[k] += -0.5*(invH0 * lens.timedelay/day * 70. - dt_obs)**2/dt_err**2

        dtmodel_grid[k] = invH0 * lens.timedelay/day * 70.
        xAmodel_grid[k] = lens.images[0]
        xBmodel_grid[k] = lens.images[1]
        radmagratmodel_grid[k] = lens.radmag_ratio
    else:
        logp_beta[k] = -np.inf

logp_beta += -0.5*(beta_grid - mu_beta)**2/sig_beta**2 - np.log(sig_beta)

logp_beta -= logp_beta.max()

p_beta_grid = np.exp(logp_beta)

pylab.plot(beta_grid, p_beta_grid)
pylab.show()

pylab.subplot(2, 2, 1)
pylab.plot(beta_grid, dtmodel_grid)
pylab.axhspan(dt_obs - dt_err, dt_obs + dt_err, color='gray', alpha=0.5)

pylab.subplot(2, 2, 2)
pylab.plot(beta_grid, radmagratmodel_grid)
pylab.axhspan(radmagrat_obs - radmagrat_err, radmagrat_obs + radmagrat_err, color='gray', alpha=0.5)

pylab.subplot(2, 2, 3)
pylab.plot(beta_grid, xAmodel_grid)
pylab.axhspan(xA_obs - xA_err, xA_obs + xA_err, color='gray', alpha=0.5)

pylab.subplot(2, 2, 4)
pylab.plot(beta_grid, xBmodel_grid)
pylab.axhspan(xB_obs - xB_err, xB_obs + xB_err, color='gray', alpha=0.5)

pylab.show()

p_beta_spline = splrep(beta_grid, p_beta_grid)
     
intfunc = lambda t: splint(beta_min, t, p_beta_spline)

norm = intfunc(beta_max)

F = np.random.rand(1) * norm

beta[j, i+1] = brentq(lambda t: intfunc(t) - F, beta_min, beta_max)

lens.gamma = gamma[j, i+1]
lens.beta = beta[j, i+1]

lens.get_images()
lens.get_timedelay()

dt_model[j, i+1] = lens.timedelay / day


