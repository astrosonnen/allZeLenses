import pickle
import h5py
import numpy as np
import pylab
import lens_models
from scipy.interpolate import splrep, splint
from scipy.optimize import brentq


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
lens.gamma = gibbs['gamma'][lensno, step-1]
lens.beta = gibbs['beta'][lensno, step-1]

invH0 = gibbs['invH0'][step]

print lens.gamma, lens.beta, invH0**-1
print gibbs['invH0'][step] * gibbs['dt_model'][lensno, step] * 70., mock['lenses'][lensno].obs_timedelay[0]/day

source_min = max(0., xA_obs - 5.*xA_err - lens.alpha(xA_obs - 5.*xA_err))
source_max = xA_obs + 5.*xA_err - lens.alpha(xA_obs + 5.*xA_err)
ngrid_source = 101
s2_grid = np.linspace(source_min**2, source_max**2, ngrid_source)

logp_s2 = 0.*s2_grid

dtmodel_grid = 0.*s2_grid
xAmodel_grid = 0.*s2_grid
xBmodel_grid = 0.*s2_grid
radmagratmodel_grid = 0.*s2_grid

for k in range(ngrid_source):
    lens.source = s2_grid[k]**0.5
    lens.get_images()

    if np.isfinite(lens.images[0]):
        lens.get_timedelay()
        lens.get_radmag_ratio()
    
        logp_s2[k] = -0.5*(lens.images[0] - xA_obs)**2/xA_err**2 - 0.5*(lens.images[1] - xB_obs)**2/xB_err**2
        logp_s2[k] += -0.5*(lens.radmag_ratio - radmagrat_obs)**2/radmagrat_err**2
        logp_s2[k] += -0.5*(invH0 * lens.timedelay/day * 70. - dt_obs)**2/dt_err**2

        dtmodel_grid[k] = invH0 * lens.timedelay/day * 70.
        xAmodel_grid[k] = lens.images[0]
        xBmodel_grid[k] = lens.images[1]
        radmagratmodel_grid[k] = lens.radmag_ratio
    else:
        logp_s2[k] = -np.inf

logp_s2 -= logp_s2.max()

p_s2_grid = np.exp(logp_s2)

p_s2_spline = splrep(s2_grid, p_s2_grid, k=1)
     
intfunc = lambda t: splint(s2_grid[0], t, p_s2_spline)

norm = intfunc(s2_grid[-1])

F = np.random.rand(1) * norm

s2 = brentq(lambda t: intfunc(t) - F, s2_grid[0], s2_grid[-1])

for k in range(ngrid_source):
    print s2_grid[k], intfunc(s2_grid[k]) / norm

print s2, s2**0.5

pylab.plot(s2_grid, p_s2_grid)
pylab.axvline(s2, linestyle='--', color='k')
pylab.show()

pylab.subplot(2, 2, 1)
pylab.plot(s2_grid, dtmodel_grid)
pylab.axhspan(dt_obs - dt_err, dt_obs + dt_err, color='gray', alpha=0.5)

pylab.subplot(2, 2, 2)
pylab.plot(s2_grid, radmagratmodel_grid)
pylab.axhspan(radmagrat_obs - radmagrat_err, radmagrat_obs + radmagrat_err, color='gray', alpha=0.5)

pylab.subplot(2, 2, 3)
pylab.plot(s2_grid, xAmodel_grid)
pylab.axhspan(xA_obs - xA_err, xA_obs + xA_err, color='gray', alpha=0.5)

pylab.subplot(2, 2, 4)
pylab.plot(s2_grid, xBmodel_grid)
pylab.axhspan(xB_obs - xB_err, xB_obs + xB_err, color='gray', alpha=0.5)

pylab.show()


