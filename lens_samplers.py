import lens_models
import numpy as np
import pymc

day = 24.*3600.


def fit_nfwdev_h70prior(lens, nstep=11000, burnin=1000, thin=1, h70p_mu=1., h70p_sig=0.05):

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
    h70_par = pymc.Normal('h70', mu=h70p_mu, tau=1./h70p_sig**2, value=1.)
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

        if len(imgs) < 2:
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


def fit_nfwdev_nocvirscat_h70prior(lens, nstep=11000, burnin=1000, thin=1, h70p_mu=1., h70p_sig=0.05):

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
    h70_par = pymc.Normal('h70', mu=h70p_mu, tau=1./h70p_sig**2, value=1.)

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


def fit_nfwdev_knownimf_nocvirscat_h70prior(lens, nstep=11000, burnin=1000, thin=1, h70p_mu=1., h70p_sig=0.05):

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
    h70_par = pymc.Normal('h70', mu=h70p_mu, tau=1./h70p_sig**2, value=1.)

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

