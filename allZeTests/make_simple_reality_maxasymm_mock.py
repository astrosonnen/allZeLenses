import lens_models
import numpy as np
from allZeTools import statistics
import pickle


mockname = 'mockK'

day = 24.*3600.

nlens=100

mstar_mu=11.4
mstar_sig=0.1

mhalo_mu=13.3
mhalo_sig=0.3
mstar_mhalo=0.7

cvir_sig=0.1
cvir_mu=0.877
cvir_beta=-0.094

aimf_mu=0.
aimf_sig=0.05

logreff_mu=0.46
logreff_sig = 0.16

mstar_err=0.1
radmagrat_err=0.020
imerr=0.01
dt_err=1.

max_asymm = 0.5

h=0.7

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

hyperpars = {'h': h, 'mstar_mu': mstar_mu, 'mstar_sig': mstar_sig, 'mhalo_mu': mhalo_mu, 'mhalo_sig': mhalo_sig,  \
             'mstar_mhalo': mstar_mhalo, 'cvir_mu': cvir_mu, 'cvir_beta': cvir_beta, 'cvir_sig': cvir_sig, \
             'aimf_mu': aimf_mu, 'aimf_sig': aimf_sig, 'logreff_mu': logreff_mu, 'mstar_err': mstar_err, \
             'radmagrat_err': radmagrat_err, 'imerr': imerr, 'dt_err': dt_err}

output = {'truth': hyperpars, 'mhalo_sample': mhalos, 'mstar_sample': mstars, 'msps_sample': mstars_sps, \
          'aimf_sample': aimfs, 'msps_obs_sample': mstars_meas, 'reff_sample': reffs, 'logcvir_sample': logcvirs, \
          'zd_sample': zds, 'zs_sample': zss}

lenses = []
for i in range(nlens):
    lens = lens_models.NfwDev(zd=zds[i], zs=zss[i], mstar=10.**mstars[i], mhalo=10.**mhalos[i], \
                              reff_phys=reffs[i], cvir=10.**logcvirs[i], h=h)

    lens.normalize()
    lens.get_rein()
    lens.get_caustic()

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

output['lenses'] = lenses

f = open('%s.dat'%mockname, 'w')
pickle.dump(output, f)
f.close()



