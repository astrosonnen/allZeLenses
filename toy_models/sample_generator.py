import lens_models
import numpy as np
from allZeTools import statistics
import pickle

day = 24.*3600.


def simple_reality_sample(nlens=1000, mstar_mu=11.5, mstar_sig=0.1, mhalo_mu=13.0, mhalo_sig=0.3, mstar_mhalo=0.8, c_sig=0.1, \
                          aimf_0=0., aimf_sig=0.05, logreff_0=0.46, mstar_err=0.1, radmagrat_err=0.015, imerr=0.1, \
                          dt_err=5., h70=1.):

    # redshift distribution of lenses: uniform between 0.1 and 0.3 (hardcoded)
    zds = np.random.rand(nlens)*0.2+0.2

    # redshift distribution of sources: some sort of truncated exponential... (hardcoded)
    zss = statistics.general_random(lambda z: np.exp(-(np.log(z-0.4))**2), nlens, (0.5, 4.))

    # distribution of halo masses: Gaussian
    mhalos = mhalo_mu + np.random.normal(0., mhalo_sig, nlens)

    # distribution of stellar masses: power-law dependence on halo mass + scatter
    mstars = mstar_mu + mstar_mhalo*(mhalos - 13.) + np.random.normal(0., mstar_sig, nlens)

    # distribution of stellar IMF: Gaussian
    aimfs = np.random.normal(aimf_0, aimf_sig, nlens)

    # SED-fitting stellar masses
    mstars_sps = mstars - aimfs

    # observed SED-fitting stellar masses
    mstars_meas = mstars_sps + np.random.normal(0., mstar_err, nlens)

    # percentage uncertainties on observed radial magnification ratio
    radmagrat_errs = np.random.normal(0., radmagrat_err, nlens)

    # distribution in concentration: using Mass-concentration relation form Maccio et al. 2008 + scatter
    logcvirs = 0.971 - 0.094*(mhalos-12.) + np.random.normal(0., c_sig, nlens)
    
    # distribution in effective radii: power-law dependence on stellar mass and redshift plus scatter
    logreffs = logreff_0 + 0.59*(mstars - 11.) - 0.26*(zds - 0.7)
    reffs = 10.**logreffs

    hyperpars = {'h70': h70, 'mstar_mu': mstar_mu, 'mstar_sig': mstar_sig, 'mhalo_mu': mhalo_mu, 'mhalo_sig': mhalo_sig,  \
                 'mstar_mhalo': mstar_mhalo, 'c_sig': c_sig, 'aimf_0': aimf_0, 'aimf_sig': aimf_sig, \
                 'logreff_0': logreff_0, 'mstar_err': mstar_err, 'radmagrat_err': radmagrat_err, 'imerr': imerr, \
                 'dt_err': dt_err}

    output = {'truth': hyperpars, 'mhalo_sample': mhalos, 'mstar_sample': mstars, 'msps_sample': mstars_sps, \
              'aimf_sample': aimfs, 'msps_obs_sample': mstars_meas, 'reff_sample': reffs, 'logcvir_sample': logcvirs, \
              'zd_sample': zds, 'zs_sample': zss}

    lenses = []
    for i in range(0, nlens):
        lens = lens_models.NfwDev(zd=zds[i], zs=zss[i], mstar=10.**mstars[i], mhalo=10.**mhalos[i], \
                                  reff_phys=reffs[i], cvir=10.**logcvirs[i], h70=h70)
        lens.normalize()
        lens.get_caustic()

        # source position: uniform distribution in the circle of radius equal to the caustic
        ysource = (np.random.rand(1))**0.5*lens.caustic

        lens.source = ysource
        lens.get_images()
        lens.get_radmag_ratio()
        lens.get_timedelay()

        lens.get_rein()

        imerrs = np.random.normal(0., imerr, 2)
        lens.obs_images = ((lens.images[0] + imerrs[0], lens.images[1] + imerrs[1]), imerr)
        lens.obs_lmstar = (mstars_meas[i], mstar_err)
        lens.obs_radmagrat = (lens.radmag_ratio + radmagrat_errs[i], radmagrat_err)
        lens.obs_timedelay = (lens.timedelay + day*np.random.normal(0., dt_err, 1), dt_err*day)

        if lens.images is None:
            df

        lenses.append(lens)

    output['lenses'] = lenses

    return output


def simple_reality_sample_knownimf_nocvirscat(nlens=1000, mstar_mu=11.5, mstar_sig=0.1, mhalo_mu=13.0, mhalo_sig=0.3, \
                                              mstar_mhalo=0.8, logreff_0=0.46, mstar_err=0.1, radmagrat_err=0.015, \
                                              imerr=0.1, dt_err=5., h70=1.):

    # redshift distribution of lenses: uniform between 0.1 and 0.3 (hardcoded)
    zds = np.random.rand(nlens)*0.2+0.2

    # redshift distribution of sources: some sort of truncated exponential... (hardcoded)
    zss = statistics.general_random(lambda z: np.exp(-(np.log(z-0.4))**2), nlens, (0.5, 4.))

    # distribution of halo masses: Gaussian
    mhalos = mhalo_mu + np.random.normal(0., mhalo_sig, nlens)

    # distribution of stellar masses: power-law dependence on halo mass + scatter
    mstars = mstar_mu + mstar_mhalo*(mhalos - 13.) + np.random.normal(0., mstar_sig, nlens)

    # SED-fitting stellar masses
    mstars_sps = mstars

    # observed SED-fitting stellar masses
    mstars_meas = mstars_sps + np.random.normal(0., mstar_err, nlens)

    # percentage uncertainties on observed radial magnification ratio
    radmagrat_errs = np.random.normal(0., radmagrat_err, nlens)

    # distribution in concentration: using Mass-concentration relation form Maccio et al. 2008
    logcvirs = 0.971 - 0.094*(mhalos-12.)

    # distribution in effective radii: power-law dependence on stellar mass and redshift plus scatter
    logreffs = logreff_0 + 0.59*(mstars - 11.) - 0.26*(zds - 0.7)
    reffs = 10.**logreffs

    hyperpars = {'h70': h70, 'mstar_mu': mstar_mu, 'mstar_sig': mstar_sig, 'mhalo_mu': mhalo_mu, \
                 'mhalo_sig': mhalo_sig, 'mstar_mhalo': mstar_mhalo, 'logreff_0': logreff_0, 'mstar_err': mstar_err, \
                 'radmagrat_err': radmagrat_err, 'imerr': imerr, 'dt_err': dt_err}

    output = {'truth': hyperpars, 'mhalo_sample': mhalos, 'mstar_sample': mstars, 'msps_sample': mstars_sps, \
              'msps_obs_sample': mstars_meas, 'reff_sample': reffs, 'logcvir_sample': logcvirs, \
              'zd_sample': zds, 'zs_sample': zss}

    lenses = []
    for i in range(0, nlens):
        lens = lens_models.NfwDev(zd=zds[i], zs=zss[i], mstar=10.**mstars[i], mhalo=10.**mhalos[i], \
                                  reff_phys=reffs[i], cvir=10.**logcvirs[i], h70=h70)

        lens.normalize()
        lens.get_caustic()

        # source position: uniform distribution in the circle of radius equal to the caustic
        ysource = (np.random.rand(1))**0.5*lens.caustic

        lens.source = ysource
        lens.get_images()
        lens.get_radmag_ratio()
        lens.get_timedelay()

        lens.get_rein()

        imerrs = np.random.normal(0., imerr, 2)
        lens.obs_images = ((lens.images[0] + imerrs[0], lens.images[1] + imerrs[1]), imerr)
        lens.obs_lmstar = (mstars_meas[i], mstar_err)
        lens.obs_radmagrat = (lens.radmag_ratio + radmagrat_errs[i], radmagrat_err)
        lens.obs_timedelay = (lens.timedelay + day*np.random.normal(0., dt_err, 1), dt_err*day)

        if lens.images is None:
            df

        lenses.append(lens)

    output['lenses'] = lenses

    return output

