import lens_models
import numpy as np
from allZeTools import statistics
import pickle
from sonnentools.cgsconstants import *
from scipy.interpolate import splev
from scipy.stats import truncnorm
import os


grid_dir = os.environ.get('ATL_GRIDDIR')

mockname = 'mockS'

day = 24.*3600.

nlens=100

mstar_mu=11.3
mstar_sig=0.1

mhalo_mu=13.3
mhalo_sig=0.3
mstar_mhalo=0.6

cvir_sig=0.1
cvir_mu=0.877
cvir_beta=-0.094

aimf_mu=-0.1
aimf_sig=0.05

beta_mu = -0.1
beta_sig = 0.

logreff_mu=0.46
logreff_sig = 0.16

mstar_err=0.1
radmagrat_err=0.020
imerr=0.01
dt_err=1.

sigma_err = 10.

max_asymm = 0.2

h=0.7

f = open('/gdrive/projects/cs82_weaklensing/PyBaWL/nfw_re2_s2_grid.dat', 'r')
nfw_re2_s2_spline = pickle.load(f)
f.close()

f = open(grid_dir+'deV_mlgrad_re2_s2_spline.dat', 'r')
deV_mlgrad_s2_spline = pickle.load(f)
f.close()

# redshift distribution of lenses: uniform between 0.1 and 0.3 (hardcoded)
zds = np.random.rand(nlens)*0.2+0.2

# redshift distribution of sources: some sort of truncated exponential... (hardcoded)
zss = statistics.general_random(lambda z: np.exp(-(np.log(z-0.4))**2), nlens, (0.5, 4.))

# distribution of halo masses: Gaussian
mhalos = mhalo_mu + np.random.normal(0., mhalo_sig, nlens)

# distribution of stellar masses: power-law dependence on halo mass + scatter
mstars = mstar_mu + mstar_mhalo*(mhalos - 13.) + np.random.normal(0., mstar_sig, nlens)

# distribution of stellar IMF: Gaussian
aimfs = np.random.normal(aimf_mu, aimf_sig, nlens)

# distribution in M/L gradient
betas = beta_mu * np.ones(nlens)

# SED-fitting stellar masses
mstars_sps = mstars - aimfs

# observed SED-fitting stellar masses
mstars_meas = mstars_sps + np.random.normal(0., mstar_err, nlens)

# percentage uncertainties on observed radial magnification ratio
radmagrat_errs = np.random.normal(0., radmagrat_err, nlens)

# distribution in concentration: using Mass-concentration relation form Maccio et al. 2008 + scatter
logcvirs = cvir_mu + cvir_beta*(mhalos-13.) + np.random.normal(0., cvir_sig, nlens)

# distribution in effective radii: power-law dependence on stellar mass and redshift plus scatter
logreffs = logreff_mu + 0.59*(mstars - 11.) - 0.26*(zds - 0.7) + np.random.normal(0., logreff_sig, nlens)
reffs = 10.**logreffs

sigma_sample = np.zeros(nlens)
sigma_obs = np.zeros(nlens)
sigma_dev = np.random.normal(0., sigma_err, nlens)

hyperpars = {'h': h, 'mstar_mu': mstar_mu, 'mstar_sig': mstar_sig, 'mhalo_mu': mhalo_mu, 'mhalo_sig': mhalo_sig,  \
             'mstar_mhalo': mstar_mhalo, 'cvir_mu': cvir_mu, 'cvir_beta': cvir_beta, 'cvir_sig': cvir_sig, \
             'aimf_mu': aimf_mu, 'aimf_sig': aimf_sig, 'beta_mu': beta_mu, 'beta_sig': beta_sig, 'logreff_mu': logreff_mu, 'mstar_err': mstar_err, \
             'radmagrat_err': radmagrat_err, 'imerr': imerr, 'dt_err': dt_err}

output = {'truth': hyperpars, 'mhalo_sample': mhalos, 'mstar_sample': mstars, 'msps_sample': mstars_sps, \
          'aimf_sample': aimfs, 'beta_sample': betas, 'msps_obs_sample': mstars_meas, 'reff_sample': reffs, 'logcvir_sample': logcvirs, \
          'zd_sample': zds, 'zs_sample': zss}

lenses = []
for i in range(nlens):
    lens = lens_models.NfwGradDev(zd=zds[i], zs=zss[i], mstar=10.**mstars[i], mhalo=10.**mhalos[i], \
                              reff_phys=reffs[i], beta=betas[i], cvir=10.**logcvirs[i], h=h, delta_halo=200.)

    lens.normalize()
    lens.get_rein()
    lens.get_caustic()

    m200tomrs = (np.log(2.) - 0.5)/(np.log(1. + lens.cvir) - lens.cvir/(1. + lens.cvir))

    s2_halo = lens.mhalo * m200tomrs*splev(lens.rs/lens.reff, nfw_re2_s2_spline)/reffs[i]
    s2_bulge = lens.mstar * splev(lens.beta, deV_mlgrad_s2_spline) / reffs[i]

    sigma_sample[i] = (s2_halo + s2_bulge)**0.5

    lens.obs_sigma = (sigma_sample[i] + sigma_dev[i], sigma_err)
    sigma_obs[i] = lens.obs_sigma[0]

    ymax = lens.rein * (1. + max_asymm) - lens.alpha(lens.rein * (1. + max_asymm))

    # source position: uniform distribution in a circle
    ysource = (np.random.rand(1))**0.5*ymax

    lens.source = ysource
    lens.get_images()
    lens.get_radmag_ratio()
    lens.get_timedelay()

    imerrs = np.random.normal(0., imerr, 2)
    lens.obs_images = ((lens.images[0] + imerrs[0], lens.images[1] + imerrs[1]), imerr)
    lens.obs_lmstar = (mstars_meas[i], mstar_err)
    lens.obs_radmagrat = (lens.radmag_ratio + radmagrat_errs[i], radmagrat_err)
    lens.obs_timedelay = (lens.timedelay + day*np.random.normal(0., dt_err, 1), dt_err*day)

    if lens.images is None:
        df

    lenses.append(lens)

output['sigma_sample'] = sigma_sample
output['sigma_obs'] = sigma_obs
output['sigma_err'] = sigma_err

output['lenses'] = lenses

f = open('%s.dat'%mockname, 'w')
pickle.dump(output, f)
f.close()

