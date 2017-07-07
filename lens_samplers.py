import lens_models
import numpy as np
import pymc

day = 24.*3600.


def fit_nfwdev_interimprior(lens, nstep=15000, burnin=5000, thin=1, mhalo_prior=(13., 0.5), mstar_prior=(11.5, 0.5), alpha_prior=(0., 0.2), cvir_prior=(0.877, 0.3), max_fcaust=1.):

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, images=lens.images, source=lens.source)

    model_lens.normalize()

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs, mstar_err = lens.obs_lmstar

    radmagrat_obs, radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr, nsig=5.)

    mstar_par = pymc.Normal('lmstar', mu=mstar_prior[0], tau=1./mstar_prior[1]**2, value=np.log10(lens.mstar))
    alpha_par = pymc.Normal('alpha', mu=alpha_prior[0], tau=1./alpha_prior[1]**2, value=0.)
    mhalo_par = pymc.Normal('mhalo', mu=mhalo_prior[0], tau=1./mhalo_prior[1]**2, value=np.log10(lens.mhalo))
    c_par = pymc.Normal('lcvir', mu=cvir_prior[0], tau=1./cvir_prior[1]**2, value=np.log10(lens.cvir))

    @pymc.deterministic()
    def caustic(mstar=mstar_par, mhalo=mhalo_par, cvir=c_par):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_par = pymc.Uniform('s2', lower=0., upper=(max_fcaust*caustic)**2, value=lens.source**2)

    @pymc.deterministic()
    def images(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, cvir=c_par):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()

        model_lens.get_images()
        if len(model_lens.images) < 2:
            return np.inf, -np.inf
        else:
            return model_lens.images

    @pymc.deterministic()
    def image_a(imgs=images):
        return imgs[0]

    @pymc.deterministic()
    def image_b(imgs=images):
        return imgs[1]

    @pymc.deterministic()
    def radmagrat(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, cvir=c_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return model_lens.radmag_ratio

    @pymc.deterministic()
    def timedelay(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, cvir=c_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_timedelay()
            return model_lens.timedelay

    ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
    imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

    msps_logp = pymc.Normal('msps_logp', mu=mstar_par-alpha_par, tau=1./mstar_err**2, value=mstar_obs, observed=True)

    radmag_logp = pymc.Normal('radmag_logp', mu=radmagrat, tau=1./radmagrat_err**2, value=radmagrat_obs, observed=True)

    pars = [mstar_par, mhalo_par, alpha_par, c_par, s2_par, timedelay, image_a, image_b, caustic, radmagrat]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis, [mstar_par, mhalo_par, alpha_par, c_par, s2_par])
    M.sample(nstep, burnin, thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:], 'mhalo':M.trace('mhalo')[:], 'alpha': M.trace('alpha')[:], \
              'lcvir': M.trace('lcvir')[:], 'timedelay':M.trace('timedelay')[:], \
              'source':(M.trace('s2')[:]**0.5).flatten(), 'image_a':M.trace('image_a')[:], \
              'image_b':M.trace('image_b')[:], 'caustic': M.trace('caustic')[:], 'radmagrat': M.trace('radmagrat')[:], 'mhalo_prior': mhalo_prior, 'mstar_prior': mstar_prior, 'alpha_prior': alpha_prior, 'cvir_prior': cvir_prior}

    return outdic


def fit_nfwdev_mstarmhaloprior(lens, nstep=15000, burnin=5000, thin=1, mhalo_prior=(13., 1., 0.5), mstar_prior=(11.5, 0.5), alpha_prior=(0., 0.2), cvir_prior=(0.877, 0.3)):

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, images=lens.images, source=lens.source)

    model_lens.normalize()

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs, mstar_err = lens.obs_lmstar

    radmagrat_obs, radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr, nsig=5.)

    mstar_par = pymc.Normal('lmstar', mu=mstar_prior[0], tau=1./mstar_prior[1]**2, value=np.log10(lens.mstar))
    alpha_par = pymc.Normal('alpha', mu=alpha_prior[0], tau=1./alpha_prior[1]**2, value=0.)
    mhalo_par = pymc.Normal('mhalo', mu=mhalo_prior[0] + mhalo_prior[1]*(mstar_par - 11.5), tau=1./mhalo_prior[2]**2, value=np.log10(lens.mhalo))
    c_par = pymc.Normal('lcvir', mu=cvir_prior[0], tau=1./cvir_prior[1]**2, value=np.log10(lens.cvir))

    @pymc.deterministic()
    def caustic(mstar=mstar_par, mhalo=mhalo_par, cvir=c_par):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_par = pymc.Uniform('s2', lower=0., upper=caustic**2, value=lens.source**2)

    @pymc.deterministic()
    def images(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, cvir=c_par):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()

        model_lens.get_images()
        if len(model_lens.images) < 2:
            return np.inf, -np.inf
        else:
            return model_lens.images

    @pymc.deterministic()
    def image_a(imgs=images):
        return imgs[0]

    @pymc.deterministic()
    def image_b(imgs=images):
        return imgs[1]

    @pymc.deterministic()
    def radmagrat(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, cvir=c_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return model_lens.radmag_ratio

    @pymc.deterministic()
    def timedelay(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, cvir=c_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_timedelay()
            return model_lens.timedelay

    ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
    imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

    msps_logp = pymc.Normal('msps_logp', mu=mstar_par-alpha_par, tau=1./mstar_err**2, value=mstar_obs, observed=True)

    radmag_logp = pymc.Normal('radmag_logp', mu=radmagrat, tau=1./radmagrat_err**2, value=radmagrat_obs, observed=True)

    pars = [mstar_par, mhalo_par, alpha_par, c_par, s2_par, timedelay, image_a, image_b, caustic, radmagrat]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis, [mstar_par, mhalo_par, alpha_par, c_par, s2_par])
    M.sample(nstep, burnin, thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:], 'mhalo':M.trace('mhalo')[:], 'alpha': M.trace('alpha')[:], \
              'lcvir': M.trace('lcvir')[:], 'timedelay':M.trace('timedelay')[:], \
              'source':(M.trace('s2')[:]**0.5).flatten(), 'image_a':M.trace('image_a')[:], \
              'image_b':M.trace('image_b')[:], 'caustic': M.trace('caustic')[:], 'radmagrat': M.trace('radmagrat')[:], 'mhalo_prior': mhalo_prior, 'mstar_prior': mstar_prior, 'alpha_prior': alpha_prior, 'cvir_prior': cvir_prior}

    return outdic


def fit_nfwdev_interimprior_flatcprior(lens, nstep=15000, burnin=5000, thin=1, mhalo_prior=(13., 0.5), mstar_prior=(11.5, 0.5), alpha_prior=(0., 0.2), cvir_prior=(0.5, 1.5)):

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, images=lens.images, source=lens.source)

    model_lens.normalize()

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs, mstar_err = lens.obs_lmstar

    radmagrat_obs, radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr, nsig=5.)

    mstar_par = pymc.Normal('lmstar', mu=mstar_prior[0], tau=1./mstar_prior[1]**2, value=np.log10(lens.mstar))
    alpha_par = pymc.Normal('alpha', mu=alpha_prior[0], tau=1./alpha_prior[1]**2, value=0.)
    mhalo_par = pymc.Normal('mhalo', mu=mhalo_prior[0], tau=1./mhalo_prior[1]**2, value=np.log10(lens.mhalo))
    c_par = pymc.Uniform('lcvir', lower=cvir_prior[0], upper=cvir_prior[1], value=np.log10(lens.cvir))

    @pymc.deterministic()
    def caustic(mstar=mstar_par, mhalo=mhalo_par, cvir=c_par):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_par = pymc.Uniform('s2', lower=0., upper=caustic**2, value=lens.source**2)

    @pymc.deterministic()
    def images(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, cvir=c_par):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()

        model_lens.get_images()
        if len(model_lens.images) < 2:
            return np.inf, -np.inf
        else:
            return model_lens.images

    @pymc.deterministic()
    def image_a(imgs=images):
        return imgs[0]

    @pymc.deterministic()
    def image_b(imgs=images):
        return imgs[1]

    @pymc.deterministic()
    def radmagrat(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, cvir=c_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return model_lens.radmag_ratio

    @pymc.deterministic()
    def timedelay(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, cvir=c_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_timedelay()
            return model_lens.timedelay

    ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
    imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

    msps_logp = pymc.Normal('msps_logp', mu=mstar_par-alpha_par, tau=1./mstar_err**2, value=mstar_obs, observed=True)

    radmag_logp = pymc.Normal('radmag_logp', mu=radmagrat, tau=1./radmagrat_err**2, value=radmagrat_obs, observed=True)

    pars = [mstar_par, mhalo_par, alpha_par, c_par, s2_par, timedelay, image_a, image_b, caustic, radmagrat]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis, [mstar_par, mhalo_par, alpha_par, c_par, s2_par])
    M.sample(nstep, burnin, thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:], 'mhalo':M.trace('mhalo')[:], 'alpha': M.trace('alpha')[:], \
              'lcvir': M.trace('lcvir')[:], 'timedelay':M.trace('timedelay')[:], \
              'source':(M.trace('s2')[:]**0.5).flatten(), 'image_a':M.trace('image_a')[:], \
              'image_b':M.trace('image_b')[:], 'caustic': M.trace('caustic')[:], 'radmagrat': M.trace('radmagrat')[:], 'mhalo_prior': mhalo_prior, 'mstar_prior': mstar_prior, 'alpha_prior': alpha_prior, 'cvir_prior': cvir_prior}

    return outdic


def fit_nfwdev_mhalo_given_mstar_truthprior(lens, truth, nstep=15000, burnin=5000, thin=1):

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, images=lens.images, source=lens.source)

    model_lens.normalize()

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs, mstar_err = lens.obs_lmstar

    radmagrat_obs, radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr, nsig=5.)

    mstar_par = pymc.Normal('lmstar', mu=truth['mstar_mu'], tau=1./truth['mstar_sig']**2, value=np.log10(lens.mstar))
    mhalo_par = pymc.Normal('mhalo', mu=truth['mhalo_mu'] + truth['mhalo_beta']*(mstar_par - 11.5), tau=1./truth['mhalo_sig']**2, value=np.log10(lens.mhalo))
    alpha_par = pymc.Normal('alpha', mu=truth['aimf_mu'], tau=1./truth['aimf_sig']**2, value=truth['aimf_mu'])
    c_par = pymc.Normal('lcvir', mu=truth['cvir_mu'] + truth['cvir_beta']*(mhalo_par-13.), tau=1./truth['cvir_sig']**2, value=np.log10(lens.cvir))

    @pymc.deterministic()
    def caustic(mstar=mstar_par, mhalo=mhalo_par, cvir=c_par):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_par = pymc.Uniform('s2', lower=0., upper=caustic**2, value=lens.source**2)

    @pymc.deterministic()
    def images(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, cvir=c_par):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()

        model_lens.get_images()
        if len(model_lens.images) < 2:
            return np.inf, -np.inf
        else:
            return model_lens.images

    @pymc.deterministic()
    def image_a(imgs=images):
        return imgs[0]

    @pymc.deterministic()
    def image_b(imgs=images):
        return imgs[1]

    @pymc.deterministic()
    def radmagrat(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, cvir=c_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return model_lens.radmag_ratio

    @pymc.deterministic()
    def timedelay(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, cvir=c_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_timedelay()
            return model_lens.timedelay/day

    ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
    imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

    msps_logp = pymc.Normal('msps_logp', mu=mstar_par-alpha_par, tau=1./mstar_err**2, value=mstar_obs, observed=True)

    radmag_logp = pymc.Normal('radmag_logp', mu=radmagrat, tau=1./radmagrat_err**2, value=radmagrat_obs, observed=True)

    pars = [mstar_par, mhalo_par, alpha_par, c_par, s2_par, timedelay, image_a, image_b, caustic, radmagrat]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis, [mstar_par, mhalo_par, alpha_par, c_par, s2_par])
    M.sample(nstep, burnin, thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:], 'mhalo':M.trace('mhalo')[:], 'alpha': M.trace('alpha')[:], \
              'lcvir': M.trace('lcvir')[:], 'timedelay':M.trace('timedelay')[:], \
              'source':(M.trace('s2')[:]**0.5).flatten(), 'image_a':M.trace('image_a')[:], \
              'image_b':M.trace('image_b')[:], 'caustic': M.trace('caustic')[:], 'radmagrat': M.trace('radmagrat')[:]}

    return outdic


def fit_nfwdev_truthprior(lens, truth, nstep=15000, burnin=5000, thin=1):

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, images=lens.images, source=lens.source)

    model_lens.normalize()

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs, mstar_err = lens.obs_lmstar

    radmagrat_obs, radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr, nsig=5.)

    mhalo_par = pymc.Normal('mhalo', mu=truth['mhalo_mu'], tau=1./truth['mhalo_sig']**2, value=np.log10(lens.mhalo))
    mstar_par = pymc.Normal('lmstar', mu=truth['mstar_mu']+truth['mstar_mhalo']*(mhalo_par-13.), \
                            tau=1./truth['mstar_sig']**2, value=np.log10(lens.mstar))
    alpha_par = pymc.Normal('alpha', mu=truth['aimf_mu'], tau=1./truth['aimf_sig']**2, value=0.)
    c_par = pymc.Normal('lcvir', mu=truth['cvir_mu'] + truth['cvir_beta']*(mhalo_par-13.), tau=1./truth['cvir_sig']**2, value=np.log10(lens.cvir))

    @pymc.deterministic()
    def caustic(mstar=mstar_par, mhalo=mhalo_par, cvir=c_par):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_par = pymc.Uniform('s2', lower=0., upper=caustic**2, value=lens.source**2)

    @pymc.deterministic()
    def images(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, cvir=c_par):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()

        model_lens.get_images()
        if len(model_lens.images) < 2:
            return np.inf, -np.inf
        else:
            return model_lens.images

    @pymc.deterministic()
    def image_a(imgs=images):
        return imgs[0]

    @pymc.deterministic()
    def image_b(imgs=images):
        return imgs[1]

    @pymc.deterministic()
    def radmagrat(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, cvir=c_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return model_lens.radmag_ratio

    @pymc.deterministic()
    def timedelay(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, cvir=c_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**cvir

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_timedelay()
            return model_lens.timedelay/day

    ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
    imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

    msps_logp = pymc.Normal('msps_logp', mu=mstar_par-alpha_par, tau=1./mstar_err**2, value=mstar_obs, observed=True)

    radmag_logp = pymc.Normal('radmag_logp', mu=radmagrat, tau=1./radmagrat_err**2, value=radmagrat_obs, observed=True)

    pars = [mstar_par, mhalo_par, alpha_par, c_par, s2_par, timedelay, image_a, image_b, caustic, radmagrat]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis, [mstar_par, mhalo_par, alpha_par, c_par, s2_par])
    M.sample(nstep, burnin, thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:], 'mhalo':M.trace('mhalo')[:], 'alpha': M.trace('alpha')[:], \
              'lcvir': M.trace('lcvir')[:], 'timedelay':M.trace('timedelay')[:], \
              'source':(M.trace('s2')[:]**0.5).flatten(), 'image_a':M.trace('image_a')[:], \
              'image_b':M.trace('image_b')[:], 'caustic': M.trace('caustic')[:], 'radmagrat': M.trace('radmagrat')[:]}

    return outdic


def fit_nfwdev_nocvirscat_noradmagfit_nodtfit_mstarmhaloprior(lens, nstep=15000, burnin=5000, thin=1):

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, images=lens.images, source=lens.source)

    model_lens.normalize()

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs, mstar_err = lens.obs_lmstar

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr, nsig=5.)

    mstar_par = pymc.Uniform('lmstar', lower=10.5, upper=12.5, value=np.log10(lens.mstar))
    alpha_par = pymc.Uniform('alpha', lower=-0.5, upper=0.5, value=0.)
    mhalo_par = pymc.Normal('mhalo', mu=mstar_par + 1.7, tau=1./0.3**2, value=np.log10(lens.mhalo))

    def cfunc(lmhalo):
        return 10.**(0.971 - 0.094*(lmhalo - 12.))

    @pymc.deterministic()
    def caustic(mstar=mstar_par, mhalo=mhalo_par):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_par = pymc.Uniform('s2', lower=0., upper=caustic**2, value=lens.source**2)

    @pymc.deterministic()
    def images(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()

        model_lens.get_images()
        if len(model_lens.images) < 2:
            return np.inf, -np.inf
        else:
            return model_lens.images

    @pymc.deterministic()
    def image_a(imgs=images):
        return imgs[0]

    @pymc.deterministic()
    def image_b(imgs=images):
        return imgs[1]

    @pymc.deterministic()
    def timedelay(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_timedelay()
            return model_lens.timedelay

    ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
    imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

    msps_logp = pymc.Normal('msps_logp', mu=mstar_par-alpha_par, tau=1./mstar_err**2, value=mstar_obs, observed=True)

    pars = [mstar_par, mhalo_par, alpha_par, s2_par, timedelay, image_a, image_b, caustic]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis, [mstar_par, mhalo_par, alpha_par, s2_par])
    M.sample(nstep, burnin, thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:], 'mhalo':M.trace('mhalo')[:], 'alpha': M.trace('alpha')[:], \
              'timedelay':M.trace('timedelay')[:], 'source':(M.trace('s2')[:]**0.5).flatten(), \
              'image_a':M.trace('image_a')[:], 'image_b':M.trace('image_b')[:], 'caustic': M.trace('caustic')[:]}

    return outdic


def fit_nfwdev_nocvirscat_nodtfit(lens, nstep=15000, burnin=5000, thin=1):

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, images=lens.images, source=lens.source)

    model_lens.normalize()

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs, mstar_err = lens.obs_lmstar

    radmagrat_obs, radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr, nsig=5.)

    mstar_par = pymc.Uniform('lmstar', lower=10.5, upper=12.5, value=np.log10(lens.mstar))
    alpha_par = pymc.Uniform('alpha', lower=-0.5, upper=0.5, value=0.)
    mhalo_par = pymc.Uniform('mhalo', lower=12., upper=14., value=np.log10(lens.mhalo))

    def cfunc(lmhalo):
        return 10.**(0.971 - 0.094*(lmhalo - 12.))

    @pymc.deterministic()
    def caustic(mstar=mstar_par, mhalo=mhalo_par):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_par = pymc.Uniform('s2', lower=0., upper=caustic**2, value=lens.source**2)

    @pymc.deterministic()
    def images(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()

        model_lens.get_images()
        if len(model_lens.images) < 2:
            return np.inf, -np.inf
        else:
            return model_lens.images

    @pymc.deterministic()
    def image_a(imgs=images):
        return imgs[0]

    @pymc.deterministic()
    def image_b(imgs=images):
        return imgs[1]

    @pymc.deterministic()
    def radmagrat(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return model_lens.radmag_ratio

    @pymc.deterministic()
    def timedelay(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_timedelay()
            return model_lens.timedelay

    ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
    imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

    msps_logp = pymc.Normal('msps_logp', mu=mstar_par-alpha_par, tau=1./mstar_err**2, value=mstar_obs, observed=True)

    radmag_logp = pymc.Normal('radmag_logp', mu=radmagrat, tau=1./radmagrat_err**2, value=radmagrat_obs, observed=True)

    pars = [mstar_par, mhalo_par, alpha_par, s2_par, timedelay, image_a, image_b, caustic, radmagrat]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis, [mstar_par, mhalo_par, alpha_par, s2_par])
    M.sample(nstep, burnin, thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:], 'mhalo':M.trace('mhalo')[:], 'alpha': M.trace('alpha')[:], \
              'timedelay':M.trace('timedelay')[:], 'source':(M.trace('s2')[:]**0.5).flatten(), \
              'image_a':M.trace('image_a')[:], 'image_b':M.trace('image_b')[:], 'caustic': M.trace('caustic')[:],\
              'radmagrat': M.trace('radmagrat')[:]}

    return outdic


def fit_nfwdev_nocvirscat_noradmagfit_nodtfit(lens, nstep=15000, burnin=5000, thin=1):

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, images=lens.images, source=lens.source)

    model_lens.normalize()

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs, mstar_err = lens.obs_lmstar

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr, nsig=5.)

    mstar_par = pymc.Uniform('lmstar', lower=10.5, upper=12.5, value=np.log10(lens.mstar))
    alpha_par = pymc.Uniform('alpha', lower=-0.5, upper=0.5, value=0.)
    mhalo_par = pymc.Uniform('mhalo', lower=12., upper=14., value=np.log10(lens.mhalo))

    def cfunc(lmhalo):
        return 10.**(0.971 - 0.094*(lmhalo - 12.))

    @pymc.deterministic()
    def caustic(mstar=mstar_par, mhalo=mhalo_par):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_par = pymc.Uniform('s2', lower=0., upper=caustic**2, value=lens.source**2)

    @pymc.deterministic()
    def images(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()

        model_lens.get_images()
        if len(model_lens.images) < 2:
            return np.inf, -np.inf
        else:
            return model_lens.images

    @pymc.deterministic()
    def image_a(imgs=images):
        return imgs[0]

    @pymc.deterministic()
    def image_b(imgs=images):
        return imgs[1]

    @pymc.deterministic()
    def timedelay(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_timedelay()
            return model_lens.timedelay

    ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
    imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

    msps_logp = pymc.Normal('msps_logp', mu=mstar_par-alpha_par, tau=1./mstar_err**2, value=mstar_obs, observed=True)

    pars = [mstar_par, mhalo_par, alpha_par, s2_par, timedelay, image_a, image_b, caustic]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis, [mstar_par, mhalo_par, alpha_par, s2_par])
    M.sample(nstep, burnin, thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:], 'mhalo':M.trace('mhalo')[:], 'alpha': M.trace('alpha')[:], \
              'timedelay':M.trace('timedelay')[:], 'source':(M.trace('s2')[:]**0.5).flatten(), \
              'image_a':M.trace('image_a')[:], 'image_b':M.trace('image_b')[:], 'caustic': M.trace('caustic')[:]}

    return outdic


def fit_nfwdev_dodtfit(lens, nstep=11000, burnin=1000, thin=1):

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, images=lens.images, source=lens.source)

    model_lens.normalize()

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs, mstar_err = lens.obs_lmstar
    dt_obs, dt_err = lens.obs_timedelay

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr, nsig=5.)

    mstar_par = pymc.Uniform('lmstar', lower=10.5, upper=12.5, value=np.log10(lens.mstar))
    alpha_par = pymc.Uniform('alpha', lower=-0.5, upper=0.5, value=0.)
    mhalo_par = pymc.Uniform('mhalo', lower=12., upper=14., value=np.log10(lens.mhalo))
    h70_par = pymc.Uniform('h70', lower=0.5, upper=1.5, value=1.)
    c_par = pymc.Normal('lcvir', mu=0.971 - 0.094*(mhalo_par-12.), tau=1./0.1**2, value=np.log10(lens.cvir))

    @pymc.deterministic()
    def caustic(mstar=mstar_par, mhalo=mhalo_par, c_par=c_par):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**c_par

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_par = pymc.Uniform('s2', lower=0., upper=caustic**2, value=lens.source**2)

    @pymc.deterministic()
    def images(mstar=mstar_par, mhalo=mhalo_par, c_par=c_par, s2=s2_par):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**c_par

        model_lens.normalize()

        model_lens.get_images()
        if len(model_lens.images) < 2:
            return np.inf, -np.inf
        else:
            return model_lens.images

    @pymc.deterministic()
    def image_a(imgs=images):
        return imgs[0]

    @pymc.deterministic()
    def image_b(imgs=images):
        return imgs[1]

    @pymc.deterministic()
    def timedelay(mstar=mstar_par, mhalo=mhalo_par, c_par=c_par, s2=s2_par, h70=h70_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**c_par
        model_lens.h70 = h70

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_timedelay()
            return model_lens.timedelay

    ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
    imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

    msps_logp = pymc.Normal('msps_logp', mu=mstar_par-alpha_par, tau=1./mstar_err**2, value=mstar_obs, observed=True)
    timedelay_logp = pymc.Normal('timedelay_logp', mu=timedelay, tau=1./dt_err**2, value=dt_obs, observed=True)

    pars = [mstar_par, mhalo_par, alpha_par, c_par, h70_par, s2_par, timedelay, image_a, image_b, caustic]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis, [mstar_par, mhalo_par, alpha_par, c_par, h70_par, s2_par])
    M.sample(nstep, burnin, thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:], 'mhalo':M.trace('mhalo')[:], 'lcvir': M.trace('lcvir')[:], \
              'alpha': M.trace('alpha')[:], 'h70':M.trace('h70')[:], 'timedelay':M.trace('timedelay')[:], \
              'source':(M.trace('s2')[:]**0.5).flatten(), 'image_a':M.trace('image_a')[:], \
              'image_b':M.trace('image_b')[:], 'caustic': M.trace('caustic')[:]}

    return outdic


def fit_nfwdev_nocvirscat_dodtfit(lens, nstep=11000, burnin=1000, thin=1):

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, images=lens.images, source=lens.source)

    model_lens.normalize()

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs, mstar_err = lens.obs_lmstar
    dt_obs, dt_err = lens.obs_timedelay

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr, nsig=5.)

    mstar_par = pymc.Uniform('lmstar', lower=10.5, upper=12.5, value=np.log10(lens.mstar))
    alpha_par = pymc.Uniform('alpha', lower=-0.5, upper=0.5, value=0.)
    mhalo_par = pymc.Uniform('mhalo', lower=12., upper=14., value=np.log10(lens.mhalo))
    h70_par = pymc.Uniform('h70', lower=0.5, upper=1.5, value=1.)

    def cfunc(lmhalo):
        return 10.**(0.971 - 0.094*(lmhalo - 12.))

    @pymc.deterministic()
    def caustic(mstar=mstar_par, mhalo=mhalo_par):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_par = pymc.Uniform('s2', lower=0., upper=caustic**2, value=lens.source**2)

    @pymc.deterministic()
    def images(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()

        model_lens.get_images()
        if len(model_lens.images) < 2:
            return np.inf, -np.inf
        else:
            return model_lens.images

    @pymc.deterministic()
    def image_a(imgs=images):
        return imgs[0]

    @pymc.deterministic()
    def image_b(imgs=images):
        return imgs[1]

    @pymc.deterministic()
    def timedelay(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, h70=h70_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)
        model_lens.h70 = h70

        model_lens.normalize()
        model_lens.images = imgs

        if not np.isfinite(imgs[0]):
            return 0.
        else:
            model_lens.get_timedelay()
            return model_lens.timedelay

    ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
    imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

    msps_logp = pymc.Normal('msps_logp', mu=mstar_par-alpha_par, tau=1./mstar_err**2, value=mstar_obs, observed=True)
    timedelay_logp = pymc.Normal('timedelay_logp', mu=timedelay, tau=1./dt_err**2, value=dt_obs, observed=True)

    pars = [mstar_par, mhalo_par, alpha_par, h70_par, s2_par, timedelay, image_a, image_b, caustic]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis, [mstar_par, mhalo_par, alpha_par, h70_par, s2_par])
    M.sample(nstep, burnin, thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:], 'mhalo':M.trace('mhalo')[:], 'alpha': M.trace('alpha')[:], \
              'h70':M.trace('h70')[:], 'timedelay':M.trace('timedelay')[:], \
              'source':(M.trace('s2')[:]**0.5).flatten(), 'image_a':M.trace('image_a')[:], \
              'image_b':M.trace('image_b')[:], 'caustic': M.trace('caustic')[:]}

    return outdic


def fit_nfwdev_knownimf_nocvirscat_dodtfit(lens, nstep=11000, burnin=1000, thin=1):

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, images=lens.images, source=lens.source)

    model_lens.normalize()

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs, mstar_err = lens.obs_lmstar
    dt_obs, dt_err = lens.obs_timedelay

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr, nsig=5.)

    mstar_par = pymc.Uniform('lmstar', lower=10.5, upper=12.5, value=np.log10(lens.mstar))
    mhalo_par = pymc.Uniform('mhalo', lower=12., upper=14., value=np.log10(lens.mhalo))
    h70_par = pymc.Uniforma('h70', lower=0.5, upper=1.5, value=1.)

    def cfunc(lmhalo):
        return 10.**(0.971 - 0.094*(lmhalo - 12.))

    @pymc.deterministic()
    def caustic(mstar=mstar_par, mhalo=mhalo_par):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_par = pymc.Uniform('s2', lower=0., upper=caustic**2, value=lens.source**2)

    @pymc.deterministic()
    def images(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()

        model_lens.get_images()
        if len(model_lens.images) < 2:
            return np.inf, -np.inf
        else:
            return model_lens.images

    @pymc.deterministic()
    def image_a(imgs=images):
        return imgs[0]

    @pymc.deterministic()
    def image_b(imgs=images):
        return imgs[1]

    @pymc.deterministic()
    def timedelay(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, h70=h70_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)
        model_lens.h70 = h70

        model_lens.normalize()
        model_lens.images = imgs

        if len(imgs) < 2:
            return 0.
        else:
            model_lens.get_timedelay()
            return model_lens.timedelay

    ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
    imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

    mstar_logp = pymc.Normal('mstar_logp', mu=mstar_par, tau=1./mstar_err**2, value=mstar_obs, observed=True)
    timedelay_logp = pymc.Normal('timedelay_logp', mu=timedelay, tau=1./dt_err**2, value=dt_obs, observed=True)

    pars = [mstar_par, mhalo_par, h70_par, s2_par, timedelay, image_a, image_b, caustic]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis, [mstar_par, mhalo_par, h70_par, s2_par])
    M.sample(nstep, burnin, thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:], 'mhalo':M.trace('mhalo')[:], 'h70':M.trace('h70')[:], \
              'timedelay':M.trace('timedelay')[:], 'source':(M.trace('s2')[:]**0.5).flatten(), \
              'image_a':M.trace('image_a')[:], 'image_b':M.trace('image_b')[:], 'caustic': M.trace('caustic')[:]}

    return outdic


def fit_nfwdev_knownimf_nocvirscat_nodtfit(lens, nstep=11000, burnin=1000, thin=1):

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, images=lens.images, source=lens.source)

    model_lens.normalize()

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs, mstar_err = lens.obs_lmstar

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr, nsig=5.)

    mstar_par = pymc.Uniform('lmstar', lower=10.5, upper=12.5, value=np.log10(lens.mstar))
    mhalo_par = pymc.Uniform('mhalo', lower=12., upper=14., value=np.log10(lens.mhalo))

    def cfunc(lmhalo):
        return 10.**(0.971 - 0.094*(lmhalo - 12.))

    @pymc.deterministic()
    def caustic(mstar=mstar_par, mhalo=mhalo_par):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_par = pymc.Uniform('s2', lower=0., upper=caustic**2, value=lens.source**2)

    @pymc.deterministic()
    def images(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()

        model_lens.get_images()
        if len(model_lens.images) < 2:
            return np.inf, -np.inf
        else:
            return model_lens.images

    @pymc.deterministic()
    def image_a(imgs=images):
        return imgs[0]

    @pymc.deterministic()
    def image_b(imgs=images):
        return imgs[1]

    @pymc.deterministic()
    def timedelay(mstar=mstar_par, mhalo=mhalo_par, s2=s2_par, imgs=images):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = cfunc(mhalo)

        model_lens.normalize()
        model_lens.images = imgs

        if len(imgs) < 2:
            return 0.
        else:
            model_lens.get_timedelay()
            return model_lens.timedelay

    ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
    imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

    mstar_logp = pymc.Normal('mstar_logp', mu=mstar_par, tau=1./mstar_err**2, value=mstar_obs, observed=True)

    pars = [mstar_par, mhalo_par, s2_par, timedelay, image_a, image_b, caustic]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis, [mstar_par, mhalo_par, s2_par])
    M.sample(nstep, burnin, thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:], 'mhalo':M.trace('mhalo')[:], \
              'timedelay':M.trace('timedelay')[:], 'source':(M.trace('s2')[:]**0.5).flatten(), \
              'image_a':M.trace('image_a')[:], 'image_b':M.trace('image_b')[:], 'caustic': M.trace('caustic')[:]}

    return outdic


def fit_powerlaw_noimerr(lens, nstep=15000, burnin=5000, thin=1):

    model_lens = lens_models.powerlaw(zd=lens.zd, zs=lens.zs, rein=lens.rein, gamma=lens.gamma, images=lens.images, \
                                      source=lens.source)

    xA, xB = lens.images

    gamma_par = pymc.Uniform('gamma', lower=1.2, upper=2.8, value=max(1.2, min(2.8, lens.gamma)))

    @pymc.deterministic()
    def b(gamma=gamma_par):
        return ((3.-gamma)*(xA - xB)/(xA**(2.-gamma) + abs(xB)**(2.-gamma)))**(1./(gamma-1.))

    @pymc.deterministic()
    def source(gamma=gamma_par):

        model_lens.gamma = gamma
        model_lens.b = float(b)

        return xA - model_lens.alpha(xA)

    @pymc.deterministic()
    def timedelay(gamma=gamma_par):

        model_lens.source = float(source)
        model_lens.gamma = gamma

        model_lens.get_timedelay()

        return model_lens.timedelay

    gamma_logp = pymc.Normal('gamma_logp', mu=gamma_par, tau=1./lens.obs_gamma[1]**2, value=lens.obs_gamma[0], observed=True)

    pars = [gamma_par, b, source, timedelay]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis, [gamma_par])
    M.sample(nstep, burnin, thin=thin)

    outdic = {'gamma': M.trace('gamma')[:], 'b': M.trace('b')[:], 'source': M.trace('source')[:], \
              'timedelay': M.trace('timedelay')[:].flatten()}

    return outdic

def fit_powerlaw(lens, nstep=15000, burnin=5000, thin=1):

    xA, xB = lens.images

    xa_obs, xb_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    radmagrat_obs, radmagrat_err = lens.obs_radmagrat

    rein_guess = 0.5*(xA - xB)

    model_lens = lens_models.powerlaw(zd=lens.zd, zs=lens.zs, rein=rein_guess, gamma=2., images=lens.images, source=lens.source)
    model_lens.make_grids(err=imerr, nsig=5.)

    gamma_par = pymc.Uniform('gamma', lower=1.2, upper=2.8, value=2.)
    rein_par = pymc.Uniform('rein', lower=0.2*rein_guess, upper=3.*rein_guess, value=rein_guess)
    s2_par = pymc.Uniform('s2', lower=0., upper=(2.*rein_guess)**2, value=(xA - model_lens.alpha(xA))**2)

    @pymc.deterministic()
    def images(rein=rein_par, gamma=gamma_par, s2=s2_par):
        model_lens.rein = rein
        model_lens.gamma = gamma
        model_lens.source = s2**0.5
        model_lens.get_b_from_rein()

        model_lens.fast_images()

        if len(model_lens.images) >= 2:
            return (model_lens.images[0], model_lens.images[1])
        else:
            return (-np.inf, np.inf)

    @pymc.deterministic()
    def imA(rein=rein_par, gamma=gamma_par, s2=s2_par):
        return float(images[0])

    @pymc.deterministic()
    def imB(rein=rein_par, gamma=gamma_par, s2=s2_par):
        return float(images[1])

    @pymc.deterministic()
    def radmagrat(rein=rein_par, gamma=gamma_par, s2=s2_par):
        model_imA, model_imB = images

        model_lens.rein = rein
        model_lens.gamma = gamma
        model_lens.source = s2**0.5

        model_lens.get_b_from_rein()

        return model_lens.mu_r(model_imA)/model_lens.mu_r(model_imB)

    @pymc.deterministic()
    def timedelay(rein=rein_par, gamma=gamma_par, s2=s2_par):
        model_imA, model_imB = images

        model_lens.rein = rein
        model_lens.gamma = gamma
        model_lens.source = s2**0.5

        model_lens.get_b_from_rein()

        model_lens.images = (model_imA, model_imB)

        model_lens.get_timedelay()

        return model_lens.timedelay

    ima_logp = pymc.Normal('ima_logp', mu=imA, tau=1./imerr**2, value=xa_obs, observed=True)
    imb_logp = pymc.Normal('imb_logp', mu=imB, tau=1./imerr**2, value=xb_obs, observed=True)

    radmag_logp = pymc.Normal('radmag_logp', mu=radmagrat, tau=1./radmagrat_err**2, value=radmagrat_obs, observed=True)

    pars = [gamma_par, rein_par, s2_par]

    M = pymc.MCMC(pars + [images, radmagrat, timedelay])
    M.use_step_method(pymc.AdaptiveMetropolis, pars)
    M.sample(nstep, burnin, thin=thin)

    outdic = {'gamma': M.trace('gamma')[:], 'rein': M.trace('rein')[:], 'source': M.trace('s2')[:]**0.5, \
              'timedelay': M.trace('timedelay')[:].flatten(), 'radmagrat': M.trace('radmagrat')[:], 'images': M.trace('images')[:]}

    return outdic


