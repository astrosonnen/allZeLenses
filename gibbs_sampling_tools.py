import numpy as np
import lens_models
from scipy.interpolate import splrep, splint
from scipy.optimize import brentq

ngrid = 1001
lmstar_grid = np.linspace(10.5, 12.5, ngrid)
lmhalo_grid = np.linspace(12., 14., ngrid)
s_grid = np.linspace(0., 1., ngrid)

eps = 1e-8

def flat_mstar_prior(lmstar):
    x = 0.5*(lmstar - 10.5)
    return 0.5**2*(np.sign(x) + 1.)*(np.sign(1. - x) + 1.)

def flat_mhalo_prior(lmhalo):
    x = 0.5*(lmhalo - 12.)
    return 0.5**2*(np.sign(x) + 1.)*(np.sign(1. - x) + 1.)

def draw_mstar_knownimf(lens, prior_func=flat_mstar_prior):

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, source=lens.source, h70=lens.h70)

    model_lens.normalize()
    model_lens.get_caustic()
    model_lens.get_images()
    model_lens.make_grids(err=lens.obs_images[1], nsig=4.)

    xa_obs, xb_obs = lens.obs_images[0]
    xerr = lens.obs_images[1]
    mstar_obs, mstar_err = lens.obs_lmstar
    dt_obs, dt_err = lens.obs_timedelay

    pmstar_grid = 0.*lmstar_grid

    for i in range(ngrid):
        model_lens.mstar = 10.**lmstar_grid[i]
        model_lens.normalize()
        model_lens.fast_images()
        if len(model_lens.images) > 1:
            model_lens.get_time_delay()
            logp = -0.5*(model_lens.images[0] - xa_obs)**2/xerr**2 - 0.5*(model_lens.images[1] - xb_obs)**2/xerr**2 + \
                   -0.5*(mstar_obs - lmstar_grid[i])**2 -0.5*(model_lens.timedelay - dt_obs)**2/dt_err**2
            pmstar_grid[i] = np.exp(logp)

    pmstar_grid *= prior_func(lmstar_grid)

    pmstar_spline = splrep(lmstar_grid, pmstar_grid)

    def intfunc(m):
        return splint(lmstar_grid[0], m, pmstar_spline)

    norm = intfunc(lmstar_grid[-1])
    t = np.random.rand(1)*norm

    mstar_draw = brentq(lambda m: intfunc(m) - t, lmstar_grid[0], lmstar_grid[-1])

    return mstar_draw


def draw_mhalo_fixedc(lens, prior_func=flat_mhalo_prior):

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, source=lens.source, h70=lens.h70)

    model_lens.normalize()
    model_lens.get_caustic()
    model_lens.get_images()
    model_lens.make_grids(err=lens.obs_images[1], nsig=4.)

    xa_obs, xb_obs = lens.obs_images[0]
    xerr = lens.obs_images[1]
    mstar_obs, mstar_err = lens.obs_lmstar
    dt_obs, dt_err = lens.obs_timedelay

    pmhalo_grid = 0.*lmhalo_grid

    for i in range(ngrid):
        model_lens.mhalo = 10.**lmhalo_grid[i]
        model_lens.cvir = 10.**(0.971-0.094*(lmhalo_grid[i]-12.))
        model_lens.normalize()
        model_lens.fast_images()
        if len(model_lens.images) > 1:
            model_lens.get_time_delay()
            logp = -0.5*(model_lens.images[0] - xa_obs)**2/xerr**2 - 0.5*(model_lens.images[1] - xb_obs)**2/xerr**2 + \
                   -0.5*(model_lens.timedelay - dt_obs)**2/dt_err**2
            pmhalo_grid[i] = np.exp(logp)

    pmhalo_grid[pmhalo_grid != pmhalo_grid] = 0.

    pmhalo_grid *= prior_func(lmhalo_grid)

    pmhalo_spline = splrep(lmhalo_grid, pmhalo_grid)

    def intfunc(m):
        return splint(lmhalo_grid[0], m, pmhalo_spline)

    norm = intfunc(lmhalo_grid[-1])
    t = np.random.rand(1)*norm

    mhalo_draw = brentq(lambda m: intfunc(m) - t, lmhalo_grid[0], lmhalo_grid[-1])

    return mhalo_draw


def draw_sourcepos(lens, prior_func=None):

    # the prior on source position is uniform in the square

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, source=lens.source, h70=lens.h70)

    model_lens.normalize()
    model_lens.get_caustic()
    model_lens.get_images()
    model_lens.make_grids(err=lens.obs_images[1], nsig=4.)

    s2_grid = s_grid*model_lens.caustic**2

    xa_obs, xb_obs = lens.obs_images[0]
    xerr = lens.obs_images[1]
    dt_obs, dt_err = lens.obs_timedelay

    pspos_grid = 0.*s2_grid

    for i in range(ngrid):
        model_lens.source = s2_grid[i]**0.5
        model_lens.fast_images()
        if len(model_lens.images) > 1:
            model_lens.get_time_delay()
            logp = -0.5*(model_lens.images[0] - xa_obs)**2/xerr**2 - 0.5*(model_lens.images[1] - xb_obs)**2/xerr**2 + \
                   -0.5*(model_lens.timedelay - dt_obs)**2/dt_err**2
            pspos_grid[i] = np.exp(logp)

    pspos_grid[pspos_grid != pspos_grid] = 0.

    pspos_spline = splrep(s2_grid, pspos_grid)

    def intfunc(m):
        return splint(s2_grid[0], m, pspos_spline)

    norm = intfunc(s2_grid[-1])
    t = np.random.rand(1)*norm

    s2_draw = brentq(lambda m: intfunc(m) - t, s2_grid[0], s2_grid[-1])

    return s2_draw**0.5


def draw_h70(lens, prior_func=None):

    model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, \
                                    reff_phys=lens.reff_phys, cvir=lens.cvir, source=lens.source, h70=1.)

    model_lens.normalize()
    model_lens.get_caustic()
    model_lens.get_images()
    model_lens.get_time_delay()

    actual_timedelay_draw = np.random.normal(lens.obs_timedelay[0], lens.obs_timedelay[1], 1)

    return model_lens.timedelay / actual_timedelay_draw


def onelens_gibbs_sampling(lens, nstep=1000):

    mstar_chain = np.zeros(nstep)
    mhalo_chain = np.zeros(nstep)
    sourcepos_chain = np.zeros(nstep)
    h70_chain = np.zeros(nstep)

    for i in range(nstep):
        new_mstar = draw_mstar_knownimf(lens)
        lens.mstar = 10.**new_mstar

        new_mhalo = draw_mhalo_fixedc(lens)
        lens.mhalo = 10.**new_mhalo
        lens.cvir = 10.**(0.971-0.094*(new_mhalo-12.))

        new_sourcepos = draw_sourcepos(lens)
        #new_sourcepos = lens.source
        lens.source = new_sourcepos

        new_h70 = draw_h70(lens)
        lens.h70 = new_h70

        print i, new_mstar, new_mhalo, new_sourcepos, new_h70, lens.cvir
        mstar_chain[i] = new_mstar
        mhalo_chain[i] = new_mhalo
        sourcepos_chain[i] = new_sourcepos
        h70_chain[i] = new_h70

    return mstar_chain, mhalo_chain, sourcepos_chain, h70_chain


def draw_sigma_given_mu(mu, sample):

    ns = 1001

    sigma_grid = np.linspace(0., 1., ns)

    def p_sig_given_mu(sig):
        psigma = 0.*sig
        for j in range(ns):
            psigma[j] = np.exp((-0.5*(mu - sample)**2/sig[j]**2 - np.log(sig[j])).sum())
        return psigma

    psigma_grid = p_sig_given_mu(sigma_grid)
    psigma_grid[psigma_grid != psigma_grid] = 0.
    spline = splrep(sigma_grid, psigma_grid)

    def intfunc(sig):
        return splint(sigma_grid[0], sig, spline)

    norm = intfunc(sigma_grid[-1])

    t = np.random.rand(1)*norm

    return brentq(lambda sig: intfunc(sig) - t, sigma_grid[0], sigma_grid[-1])


def hierarchical_gibbs_sampling_knownimf_nocvirscat(lenses, nstep=1000):

    mhalo_mu = 13.
    mhalo_sig = 0.3

    mstar_mu = 11.5
    mstar_sig = 0.1
    mstar_mhalo = 0.8

    h70 = 1.

    nlens = len(lenses)

    s_ind = np.zeros(nlens)
    mhalo_ind = np.zeros(nlens)
    mstar_ind = np.zeros(nlens)
    dt_unith_ind = np.zeros(nlens)
    dt_obs = np.zeros(nlens)
    dt_errs = np.zeros(nlens)

    chain = {'mhalo_mu': np.zeros(nstep), 'mhalo_sig': np.zeros(nstep), 'mstar_mu': np.zeros(nstep), \
             'mstar_mhalo': np.zeros(nstep), 'mstar_sig': np.zeros(nstep), 'h70': np.zeros(nstep)}

    for i in range(nlens):
        s_ind[i] = lenses[i].source
        mhalo_ind[i] = np.log10(lenses[i].mhalo)
        mstar_ind[i] = np.log10(lenses[i].mstar)
        dt_unith_ind[i] = lenses[i].timedelay*h70
        dt_obs[i] = lenses[i].obs_timedelay[0]
        dt_errs[i] = lenses[i].obs_timedelay[1]

    for j in range(nstep):

        print j
        # draws a new value of the mean halo mass
        eff_sig = mhalo_sig/(1.*nlens)**0.5
        eff_mu = mhalo_ind.mean()
        new_mhalo_mu = np.random.normal(eff_mu, eff_sig, 1)

        # draws a new value of the dispersion in halo mass
        new_mhalo_sig = draw_sigma_given_mu(new_mhalo_mu, mhalo_ind)

        # draws a new value of the mean stellar mass
        eff_sig = mstar_sig/(1.*nlens)**0.5
        eff_mu = (mstar_ind - mstar_mhalo*(mhalo_ind - 13.)).mean()
        new_mstar_mu = np.random.normal(eff_mu, eff_sig, 1)

        # draws a new value of the halo mass dependence of the stellar mass
        eff_sig = (((mhalo_ind - 13.)**2/mstar_sig**2).sum())**(-0.5)
        eff_mu = eff_sig**2*((mstar_ind - new_mstar_mu)/(mhalo_ind - 13.)/mstar_sig**2*(mhalo_ind - 13.)**2).sum()
        new_mstar_mhalo = np.random.normal(eff_mu, eff_sig, 1)

        # draws a new value of the dispersion in stellar mass
        new_mstar_sig = draw_sigma_given_mu(new_mstar_mu + new_mstar_mhalo*(mhalo_ind - 13.), mstar_ind)

        # draws a new value of h70
        sig_invh70 = dt_errs/dt_unith_ind
        mu_invh70 = dt_obs/dt_unith_ind

        eff_sig = ((1./sig_invh70**2).sum())**(-0.5)
        eff_mu = eff_sig**2*(mu_invh70/sig_invh70**2).sum()
        new_h70 = (np.random.normal(eff_mu, eff_sig, 1))**-1

        # loops over all ze lenses
        for i in range(nlens):

            # updates the value of h70
            lenses[i].h70 = new_h70

            # defines the prior on stellar mass
            def mstar_pfunc(lmstar):
                mu_here = new_mstar_mu + new_mstar_mhalo*(mhalo_ind[i] - 13.)
                return 1./new_mstar_sig*np.exp(-0.5*(lmstar - mu_here)**2/new_mstar_sig**2)

            # draws a new value of the stellar mass

            new_mstar = draw_mstar_knownimf(lenses[i], mstar_pfunc)
            mstar_ind[i] = new_mstar

            lenses[i].mstar = 10.**new_mstar
            lenses[i].normalize()

            # defines the prior on halo mass
            def mhalo_pfunc(lmhalo):
                return 1./new_mhalo_sig*np.exp(-0.5*(lmhalo - new_mhalo_mu)**2/new_mhalo_sig**2)

            # draws a new value of the halo mass

            new_mhalo = draw_mhalo_fixedc(lenses[i], mhalo_pfunc)
            mhalo_ind[i] = new_mhalo

            lenses[i].mhalo = 10.**new_mhalo
            lenses[i].normalize()

            # draws a new value of the source position
            new_sourcepos = draw_sourcepos(lenses[i])
            lenses[i].source = new_sourcepos

            # updates the model time delay
            lenses[i].get_images()
            lenses[i].get_timedelay()
            dt_unith_ind[i] = lenses[i].timedelay*new_h70

        # updates values of the hyperparameters
        mhalo_mu = new_mhalo_mu
        mhalo_sig = new_mhalo_sig
        mstar_mu = new_mstar_mu
        mstar_mhalo = new_mstar_mhalo
        mstar_sig = new_mstar_sig
        #h70 = new_h70

        chain['mhalo_mu'][j] = mhalo_mu
        chain['mhalo_sig'][j] = mhalo_sig
        chain['mstar_mu'][j] = mstar_mu
        chain['mstar_mhalo'][j] = mstar_mhalo
        chain['mstar_sig'][j] = mstar_sig
        #chain['h70'][j] = h70

    return chain

def hierarchical_gibbs_sampling_cheating(lenses, nstep=1000):

    mserr = 0.1
    mherr = 0.1

    mhalo_mu = 13.
    mhalo_sig = 0.3

    mstar_mu = 11.5
    mstar_sig = 0.1
    mstar_mhalo = 0.8

    h70 = 1.

    nlens = len(lenses)

    s_ind = np.zeros(nlens)
    mhalo_ind = np.zeros(nlens)
    mstar_ind = np.zeros(nlens)
    dt_unith_ind = np.zeros(nlens)
    dt_obs = np.zeros(nlens)
    dt_errs = np.zeros(nlens)

    chain = {'mhalo_mu': np.zeros(nstep), 'mhalo_sig': np.zeros(nstep), 'mstar_mu': np.zeros(nstep), \
             'mstar_mhalo': np.zeros(nstep), 'mstar_sig': np.zeros(nstep)}

    for i in range(nlens):
        s_ind[i] = lenses[i].source
        mhalo_ind[i] = np.log10(lenses[i].mhalo)
        mstar_ind[i] = np.log10(lenses[i].mstar)

    mstar_obs = np.random.normal(mstar_ind, mserr, nlens)
    mhalo_obs = np.random.normal(mhalo_ind, mherr, nlens)

    for j in range(nstep):

        print j
        # draws a new value of the mean halo mass
        eff_sig = mhalo_sig/(1.*nlens)**0.5
        eff_mu = mhalo_ind.mean()
        new_mhalo_mu = np.random.normal(eff_mu, eff_sig, 1)

        # draws a new value of the dispersion in halo mass
        new_mhalo_sig = draw_sigma_given_mu(new_mhalo_mu, mhalo_ind)

        # draws a new value of the mean stellar mass
        eff_sig = mstar_sig/(1.*nlens)**0.5
        eff_mu = (mstar_ind - mstar_mhalo*(mhalo_ind - 13.)).mean()
        new_mstar_mu = np.random.normal(eff_mu, eff_sig, 1)

        # draws a new value of the halo mass dependence of the stellar mass
        eff_sig = (((mhalo_ind - 13.)**2/mstar_sig**2).sum())**(-0.5)
        eff_mu = eff_sig**2*((mstar_ind - new_mstar_mu)/(mhalo_ind - 13.)/mstar_sig**2*(mhalo_ind - 13.)**2).sum()
        new_mstar_mhalo = np.random.normal(eff_mu, eff_sig, 1)

        # draws a new value of the dispersion in stellar mass
        new_mstar_sig = draw_sigma_given_mu(new_mstar_mu + new_mstar_mhalo*(mhalo_ind - 13.), mstar_ind)

        # loops over all ze lenses
        for i in range(nlens):

            # draws a new value of the halo mass
            eff_sig = (1./mherr**2 + 1./new_mhalo_sig**2)**(-0.5)
            eff_mu = eff_sig**2*(mhalo_obs[i]/mherr**2 + new_mhalo_mu/new_mhalo_sig**2)
            new_mhalo = np.random.normal(eff_mu, eff_sig, 1)
            mhalo_ind[i] = new_mhalo

            # draws a new value of the stellar mass
            eff_sig = (1./mserr**2 + 1./new_mstar_sig**2)**(-0.5)
            eff_mu = eff_sig**2*(mstar_obs[i]/mserr**2 + (new_mstar_mu + new_mstar_mhalo*(mhalo_ind[i] - 13.))/new_mstar_sig**2)
            new_mstar = np.random.normal(eff_mu, eff_sig, 1)
            mstar_ind[i] = new_mstar

        # updates values of the hyperparameters
        mhalo_mu = new_mhalo_mu
        mhalo_sig = new_mhalo_sig
        mstar_mu = new_mstar_mu
        mstar_mhalo = new_mstar_mhalo
        mstar_sig = new_mstar_sig

        chain['mhalo_mu'][j] = mhalo_mu
        chain['mhalo_sig'][j] = mhalo_sig
        chain['mstar_mu'][j] = mstar_mu
        chain['mstar_mhalo'][j] = mstar_mhalo
        chain['mstar_sig'][j] = mstar_sig

    return chain


def hierarchical_gibbs_sampling_ubercheat(lenses, nstep=1000):

    mhalo_mu = 13.
    mhalo_sig = 0.3

    mstar_mu = 11.5
    mstar_sig = 0.1
    mstar_mhalo = 0.8

    h70 = 1.

    nlens = len(lenses)

    s_ind = np.zeros(nlens)
    mhalo_ind = np.zeros(nlens)
    mstar_ind = np.zeros(nlens)
    dt_unith_ind = np.zeros(nlens)
    dt_obs = np.zeros(nlens)
    dt_errs = np.zeros(nlens)

    chain = {'mhalo_mu': np.zeros(nstep), 'mhalo_sig': np.zeros(nstep), 'mstar_mu': np.zeros(nstep), \
             'mstar_mhalo': np.zeros(nstep), 'mstar_sig': np.zeros(nstep)}

    for i in range(nlens):
        mhalo_ind[i] = np.log10(lenses[i].mhalo)
        mstar_ind[i] = np.log10(lenses[i].mstar)

    for j in range(nstep):

        print j
        # draws a new value of the mean halo mass
        eff_sig = mhalo_sig/(1.*nlens)**0.5
        eff_mu = mhalo_ind.mean()
        new_mhalo_mu = np.random.normal(eff_mu, eff_sig, 1)

        # draws a new value of the dispersion in halo mass
        new_mhalo_sig = draw_sigma_given_mu(new_mhalo_mu, mhalo_ind)

        # draws a new value of the mean stellar mass
        eff_sig = mstar_sig/(1.*nlens)**0.5
        eff_mu = (mstar_ind - mstar_mhalo*(mhalo_ind - 13.)).mean()
        new_mstar_mu = np.random.normal(eff_mu, eff_sig, 1)

        # draws a new value of the halo mass dependence of the stellar mass
        eff_sig = (((mhalo_ind - 13.)**2/mstar_sig**2).sum())**(-0.5)
        eff_mu = eff_sig**2*((mstar_ind - new_mstar_mu)/(mhalo_ind - 13.)/mstar_sig**2*(mhalo_ind - 13.)**2).sum()
        new_mstar_mhalo = np.random.normal(eff_mu, eff_sig, 1)

        # draws a new value of the dispersion in stellar mass
        new_mstar_sig = draw_sigma_given_mu(new_mstar_mu + new_mstar_mhalo*(mhalo_ind - 13.), mstar_ind)

        # updates values of the hyperparameters
        mhalo_mu = new_mhalo_mu
        mhalo_sig = new_mhalo_sig
        mstar_mu = new_mstar_mu
        mstar_mhalo = new_mstar_mhalo
        mstar_sig = new_mstar_sig

        chain['mhalo_mu'][j] = mhalo_mu
        chain['mhalo_sig'][j] = mhalo_sig
        chain['mstar_mu'][j] = mstar_mu
        chain['mstar_mhalo'][j] = mstar_mhalo
        chain['mstar_sig'][j] = mstar_sig

    return chain

