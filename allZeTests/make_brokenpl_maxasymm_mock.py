import lens_models
import numpy as np
from allZeTools import statistics
import pickle


mockname = 'mockJ'

day = 24.*3600.

nlens=100

rein_mu = 1.5
rein_sig = 0.3

gamma_mu = 1.95
gamma_sig = 0.15

beta_mu = -0.10
beta_sig = 0.05

radmagrat_err=0.020
imerr=0.01
dt_err=1.

max_asymm = 0.2

h=0.7

# redshift distribution of lenses: uniform between 0.1 and 0.3 (hardcoded)
zds = np.random.rand(nlens)*0.2+0.2

# redshift distribution of sources: some sort of truncated exponential... (hardcoded)
zss = statistics.general_random(lambda z: np.exp(-(np.log(z-0.4))**2), nlens, (0.5, 4.))

# distribution of Einstein radius: Gaussian
reins = np.random.normal(rein_mu, rein_sig, nlens)

# distribution of density slope: Gaussian
gammas = np.random.normal(gamma_mu, gamma_sig, nlens)
betas = np.random.normal(beta_mu, beta_sig, nlens)

# percentage uncertainties on observed radial magnification ratio
radmagrat_errs = np.random.normal(0., radmagrat_err, nlens)

hyperpars = {'h': h, 'rein_mu': rein_mu, 'rein_sig': rein_sig, 'gamma_mu': gamma_mu, 'gamma_sig': gamma_sig, 'beta_mu': beta_mu, 'beta_sig': beta_sig, 'radmagrat_err': radmagrat_err, 'imerr': imerr, 'dt_err': dt_err}

output = {'truth': hyperpars, 'rein_sample': reins, 'gamma_sample': gammas, 'zd_sample': zds, 'zs_sample': zss}

lenses = []
for i in range(nlens):
    lens = lens_models.sps_ein_break(zd=zds[i], zs=zss[i], rein=reins[i], gamma=gammas[i], beta=betas[i], h=h)

    ymax = lens.rein * (1. + max_asymm) - lens.alpha(lens.rein * (1. + max_asymm))

    # source position: uniform distribution in a circle
    ysource = (np.random.rand(1))**0.5*ymax

    lens.source = ysource
    lens.get_images()
    lens.get_radmag_ratio()
    lens.get_timedelay()

    imerrs = np.random.normal(0., imerr, 2)
    lens.obs_images = ((lens.images[0] + imerrs[0], lens.images[1] + imerrs[1]), imerr)
    lens.obs_radmagrat = (lens.radmag_ratio + radmagrat_errs[i], radmagrat_err)
    lens.obs_timedelay = (lens.timedelay + day*np.random.normal(0., dt_err, 1), dt_err*day)

    if lens.images is None:
        df

    lenses.append(lens)

output['lenses'] = lenses

f = open('%s.dat'%mockname, 'w')
pickle.dump(output, f)
f.close()


