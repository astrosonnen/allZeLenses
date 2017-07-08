import numpy as np
import pymc
import pickle

day = 24.*3600.


def infer_simple_reality_truthprior(guess, chains, dt_obs, dt_err, nstep=15000, burnin=5000, thin=1):

    from scipy.special import erf

    nlens = len(chains)

    # goes through the chains and thins them to nis samples
    tchains = []
    for i in range(nlens):
        chain = chains[i]

        for par in chain:
            nsamp = len(chain[par])
            keep = thin*np.arange(nsamp/thin)
            chain[par] = chain[par].flatten()[keep]

        bad = chain['timedelay'] != chain['timedelay']
        chain['timedelay'][bad.flatten()] = 0.

        tchains.append(chain)

    #defines the hyper-parameters

    h70 = pymc.Uniform('h70', lower=0.5, upper=1.5, value=1.)

    pars = [h70]

    @pymc.deterministic(name='like')
    def like(h70=h70):

        totlike = 0.

        for i in range(nlens):

            dt_term = 1./dt_err[i]*np.exp(-0.5*(dt_obs[i] - tchains[i]['timedelay']/h70)**2/dt_err[i]**2)

            totlike += np.log(dt_term.mean())

        return totlike

    @pymc.stochastic(observed=True, name='logp')
    def logp(value=0., p=pars):
        return like

    M = pymc.MCMC(pars+[like])
    M.use_step_method(pymc.AdaptiveMetropolis, pars)
    M.sample(nstep, burnin)

    outdic = {}
    for par in pars:
        outdic[str(par)] = M.trace(par)[:]
    outdic['logp'] = M.trace('like')[:]

    return outdic


def infer_simple_reality_interimprior(guess, chains, dt_obs, dt_err, nstep=15000, burnin=5000, thin=1):

    from scipy.special import erf

    nlens = len(chains)

    # goes through the chains and thins them to nis samples
    tchains = []
    for i in range(nlens):
        chain = chains[i]

        for par in chain:
            nsamp = len(chain[par])
            keep = thin*np.arange(nsamp/thin)
            chain[par] = chain[par].flatten()[keep]

        bad = chain['timedelay'] != chain['timedelay']
        chain['timedelay'][bad.flatten()] = 0.

        chain['s2_norm'] = (chain['source']/chain['caustic'])**2

        tchains.append(chain)

    #defines the hyper-parameters

    mhalo_mu = pymc.Uniform('mhalo_mu', lower=12.0, upper=14.0, value=guess['mhalo_mu'])
    mhalo_sig = pymc.Uniform('mhalo_sig', lower=0., upper=1., value=guess['mhalo_sig'])

    mstar_mhalo = pymc.Uniform('mstar_mhalo', lower=0., upper=2., value=guess['mstar_mhalo'])

    mstar_mu = pymc.Uniform('mstar_mu', lower=11., upper=12., value=guess['mstar_mu'])
    mstar_sig = pymc.Uniform('mstar_sig', lower=0., upper=2., value=guess['mstar_sig'])

    aimf_mu = pymc.Uniform('aimf_mu', lower=-0.2, upper=0.2, value=guess['aimf_mu'])
    aimf_sig = pymc.Uniform('aimf_sig', lower=0., upper=0.5, value=guess['aimf_sig'])

    s2_loga = pymc.Uniform('loga', lower=0., upper=1., value=0.)
    s2_logb = pymc.Uniform('logb', lower=-1., upper=1., value=0.)

    h_par = pymc.Uniform('h', lower=0.2, upper=1.5, value=0.7)

    pars = [mhalo_mu, mhalo_sig, mstar_mhalo, mstar_mu, mstar_sig, aimf_mu, aimf_sig, h_par, s2_loga, s2_logb]

    @pymc.deterministic(name='like')
    def like(mhalo_mu=mhalo_mu, mhalo_sig=mhalo_sig, mstar_mhalo=mstar_mhalo, mstar_mu=mstar_mu, mstar_sig=mstar_sig, \
             aimf_mu=aimf_mu, aimf_sig=aimf_sig, h=h_par, s2_loga=s2_loga, s2_logb=s2_logb):

        totlike = 0.

        for i in range(nlens):
            mglob_model = mstar_mu + mstar_mhalo*(tchains[i]['mhalo'] - 13.)

            mh_term = 1./mhalo_sig*np.exp(-(mhalo_mu - tchains[i]['mhalo'])**2/(2.*mhalo_sig**2))
            ms_term = 1./mstar_sig*np.exp(-(mglob_model - tchains[i]['mstar'])**2/(2.*mstar_sig**2))
            a_term = 1./aimf_sig*np.exp(-0.5*(aimf_mu - tchains[i]['alpha'])**2/aimf_sig**2)

            s2_term = beta.pdf(tchains[i]['s2_norm'], 10.**s2_loga, 10.**s2_logb)

            interim_prior = 1./0.3*np.exp(-0.5*(tchains[i]['mhalo'] - 13.)**2/0.5**2)*\
                            1./0.5*np.exp(-0.5*(tchains[i]['mstar'] - 11.5)**2/0.5**2)*\
                            1./0.2*np.exp(-0.5*(tchains[i]['alpha'] - 0.)**2/0.2**2)

            dt_term = 1./dt_err[i]*np.exp(-0.5*(dt_obs[i] - tchains[i]['timedelay']/(h/0.7))**2/dt_err[i]**2)

            term = (mh_term*ms_term*a_term*dt_term*s2_term/interim_prior).mean()
            totlike += np.log(term)

        return totlike

    @pymc.stochastic(observed=True, name='logp')
    def logp(value=0., p=pars):
        return like

    M = pymc.MCMC(pars+[like])
    M.use_step_method(pymc.AdaptiveMetropolis, pars)
    M.sample(nstep, burnin)

    outdic = {}
    for par in pars:
        outdic[str(par)] = M.trace(par)[:]
    outdic['logp'] = M.trace('like')[:]

    return outdic


def infer_simple_reality(guess, chains, dt_obs, dt_err, nstep=15000, burnin=5000, thin=1):

    from scipy.special import erf

    nlens = len(chains)

    # goes through the chains and thins them to nis samples
    tchains = []
    for i in range(nlens):
        chain = chains[i]

        for par in chain:
            nsamp = len(chain[par])
            keep = thin*np.arange(nsamp/thin)
            chain[par] = chain[par].flatten()[keep]

        bad = chain['timedelay'] != chain['timedelay']
        chain['timedelay'][bad.flatten()] = 0.

        tchains.append(chain)

    #defines the hyper-parameters

    mhalo_mu = pymc.Uniform('mhalo_mu', lower=12.0, upper=14.0, value=guess['mhalo_mu'])
    mhalo_sig = pymc.Uniform('mhalo_sig', lower=0., upper=1., value=guess['mhalo_sig'])

    mstar_mhalo = pymc.Uniform('mstar_mhalo', lower=0., upper=2., value=guess['mstar_mhalo'])

    mstar_mu = pymc.Uniform('mstar_mu', lower=11., upper=12., value=guess['mstar_mu'])
    mstar_sig = pymc.Uniform('mstar_sig', lower=0., upper=2., value=guess['mstar_sig'])

    aimf_mu = pymc.Uniform('aimf_mu', lower=-0.2, upper=0.2, value=guess['aimf_mu'])
    aimf_sig = pymc.Uniform('aimf_sig', lower=0., upper=0.5, value=guess['aimf_sig'])

    h70 = pymc.Uniform('h70', lower=0.5, upper=1.5, value=1.)

    pars = [mhalo_mu, mhalo_sig, mstar_mhalo, mstar_mu, mstar_sig, aimf_mu, aimf_sig, h70]

    @pymc.deterministic(name='like')
    def like(mhalo_mu=mhalo_mu, mhalo_sig=mhalo_sig, mstar_mhalo=mstar_mhalo, mstar_mu=mstar_mu, mstar_sig=mstar_sig, \
             aimf_mu=aimf_mu, aimf_sig=aimf_sig, h70=h70):

        totlike = 0.

        for i in range(nlens):
            mglob_model = mstar_mu + mstar_mhalo*(tchains[i]['mhalo'] - 13.)

            mh_term = 1./mhalo_sig*np.exp(-(mhalo_mu - tchains[i]['mhalo'])**2/(2.*mhalo_sig**2))
            ms_term = 1./mstar_sig*np.exp(-(mglob_model - tchains[i]['mstar'])**2/(2.*mstar_sig**2))
            a_term = 1./aimf_sig*np.exp(-0.5*(aimf_mu - tchains[i]['alpha'])**2/aimf_sig**2)

            dt_term = 1./dt_err[i]*np.exp(-0.5*(dt_obs[i] - tchains[i]['timedelay']/h70)**2/dt_err[i]**2)

            #norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/mstar_sig) - erf((mglob_model - 12.5)/2.**0.5/mstar_sig)) * \
            #        0.5*(erf((mhalo_mu - 12.)/2.**0.5/mhalo_sig) - erf((mhalo_mu - 14.)/2.**0.5/mhalo_sig)) * \
            #        0.5*(erf((aimf_mu + 0.5)/2.**0.5/aimf_sig) - erf((aimf_mu - 0.5)/2.**0.5/aimf_sig))
            norms = 1.

            term = (mh_term*ms_term*a_term*dt_term/norms).mean()
            totlike += np.log(term)

        return totlike

    @pymc.stochastic(observed=True, name='logp')
    def logp(value=0., p=pars):
        return like

    M = pymc.MCMC(pars+[like])
    M.use_step_method(pymc.AdaptiveMetropolis, pars)
    M.sample(nstep, burnin)

    outdic = {}
    for par in pars:
        outdic[str(par)] = M.trace(par)[:]
    outdic['logp'] = M.trace('like')[:]

    return outdic


def infer_simple_reality_nocosmo(guess, chains, nstep=11000, burnin=1000, nis=1000):

    from scipy.special import erf

    nlens = len(chains)

    # goes through the chains and thins them to nis samples
    tchains = []
    for i in range(nlens):
        chain = chains[i]
        samp = np.random.choice(np.arange(len(chain)), nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

        tchains.append(chain)

    #defines the hyper-parameters

    mhalo_mu = pymc.Uniform('mhalo_mu', lower=12.0, upper=14.0, value=guess['mhalo_mu'])
    mhalo_sig = pymc.Uniform('mhalo_sig', lower=0., upper=1., value=guess['mhalo_sig'])

    mstar_mhalo = pymc.Uniform('mstar_mhalo', lower=0., upper=2., value=guess['mstar_mhalo'])

    mstar_mu = pymc.Uniform('mstar_mu', lower=11., upper=12., value=guess['mstar_mu'])
    mstar_sig = pymc.Uniform('mstar_sig', lower=0., upper=2., value=guess['mstar_sig'])

    aimf_mu = pymc.Uniform('aimf_mu', lower=-0.2, upper=0.2, value=guess['aimf_mu'])
    aimf_sig = pymc.Uniform('aimf_sig', lower=0., upper=0.5, value=guess['aimf_sig'])

    pars = [mhalo_mu, mhalo_sig, mstar_mhalo, mstar_mu, mstar_sig, aimf_mu, aimf_sig]

    @pymc.deterministic(name='like')
    def like(mhalo_mu=mhalo_mu, mhalo_sig=mhalo_sig, mstar_mhalo=mstar_mhalo, mstar_mu=mstar_mu, mstar_sig=mstar_sig, \
             aimf_mu=aimf_mu, aimf_sig=aimf_sig):

        totlike = 0.

        for i in range(nlens):
            mglob_model = mstar_mu + mstar_mhalo*(tchains[i]['mhalo'] - 13.)

            mh_term = 1./mhalo_sig*np.exp(-(mhalo_mu - tchains[i]['mhalo'])**2/(2.*mhalo_sig**2))
            ms_term = 1./mstar_sig*np.exp(-(mglob_model - tchains[i]['mstar'])**2/(2.*mstar_sig**2))
            a_term = 1./aimf_sig*np.exp(-0.5*(aimf_mu - tchains[i]['alpha'])**2/aimf_sig**2)

            #norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/mstar_sig) - erf((mglob_model - 12.5)/2.**0.5/mstar_sig)) * \
            #        0.5*(erf((mhalo_mu - 12.)/2.**0.5/mhalo_sig) - erf((mhalo_mu - 14.)/2.**0.5/mhalo_sig))
            norms = 1.

            term = (mh_term*ms_term*a_term/norms).mean()
            totlike += np.log(term)

        return totlike

    @pymc.stochastic(observed=True, name='logp')
    def logp(value=0., p=pars):
        return like
     
    M = pymc.MCMC(pars+[like])
    M.use_step_method(pymc.AdaptiveMetropolis, pars)
    M.sample(nstep, burnin)

    outdic = {}
    for par in pars:
        outdic[str(par)] = M.trace(par)[:]
    outdic['logp'] = M.trace('like')[:]

    return outdic


def infer_simple_reality_knownimf(guess, chains, dt_obs, dt_err, nstep=11000, burnin=1000, thin=1):

    from scipy.special import erf

    nlens = len(chains)

    # goes through the chains and thins them to nis samples
    tchains = []
    for i in range(nlens):
        chain = chains[i]

        for par in chain:
            nsamp = len(chain[par])
            keep = thin*np.arange(nsamp/thin)
            chain[par] = chain[par].flatten()[keep]

        bad = chain['timedelay'] != chain['timedelay']
        chain['timedelay'][bad.flatten()] = 0.

        tchains.append(chain)

    #defines the hyper-parameters

    mhalo_mu = pymc.Uniform('mhalo_mu', lower=12.0, upper=14.0, value=guess['mhalo_mu'])
    mhalo_sig = pymc.Uniform('mhalo_sig', lower=0., upper=1., value=guess['mhalo_sig'])

    mstar_mhalo = pymc.Uniform('mstar_mhalo', lower=0., upper=2., value=guess['mstar_mhalo'])

    mstar_mu = pymc.Uniform('mstar_mu', lower=11., upper=12., value=guess['mstar_mu'])
    mstar_sig = pymc.Uniform('mstar_sig', lower=0., upper=2., value=guess['mstar_sig'])

    h70 = pymc.Uniform('h70', lower=0.5, upper=1.5, value=1.)

    pars = [mhalo_mu, mhalo_sig, mstar_mhalo, mstar_mu, mstar_sig, h70]

    @pymc.deterministic(name='like')
    def like(mhalo_mu=mhalo_mu, mhalo_sig=mhalo_sig, mstar_mhalo=mstar_mhalo, mstar_mu=mstar_mu, mstar_sig=mstar_sig, \
             h70=h70):

        totlike = 0.

        for i in range(nlens):
            mglob_model = mstar_mu + mstar_mhalo*(tchains[i]['mhalo'] - 13.)

            mh_term = 1./mhalo_sig*np.exp(-0.5*(mhalo_mu - tchains[i]['mhalo'])**2/mhalo_sig**2)
            ms_term = 1./mstar_sig*np.exp(-0.5*(mglob_model - tchains[i]['mstar'])**2/mstar_sig**2)

            dt_term = 1./dt_err[i]*np.exp(-0.5*(dt_obs[i] - tchains[i]['timedelay']/h70)**2/dt_err[i]**2)

            #norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/mstar_sig) - erf((mglob_model - 12.5)/2.**0.5/mstar_sig)) * \
            #        0.5*(erf((mhalo_mu - 12.)/2.**0.5/mhalo_sig) - erf((mhalo_mu - 14.)/2.**0.5/mhalo_sig))
            norms = 1.

            term = (mh_term*ms_term*dt_term/norms).mean()
            totlike += np.log(term)

        return totlike

    @pymc.stochastic(observed=True, name='logp')
    def logp(value=0., p=pars):
        return like

    M = pymc.MCMC(pars+[like])
    M.use_step_method(pymc.AdaptiveMetropolis, pars)
    M.sample(nstep, burnin)

    outdic = {}
    for par in pars:
        outdic[str(par)] = M.trace(par)[:]
    outdic['logp'] = M.trace('like')[:]

    return outdic


def infer_simple_reality_knownimf_nocosmo(guess, chains, nstep=11000, burnin=1000, thin=1):

    from scipy.special import erf

    nlens = len(chains)

    # goes through the chains and thins them to nis samples
    tchains = []
    for i in range(nlens):
        chain = chains[i]

        for par in chain:
            nsamp = len(chain[par])
            keep = thin*np.arange(nsamp/thin)
            chain[par] = chain[par].flatten()[keep]

        tchains.append(chain)

    #defines the hyper-parameters

    mhalo_mu = pymc.Uniform('mhalo_mu', lower=12.0, upper=14.0, value=guess['mhalo_mu'])
    mhalo_sig = pymc.Uniform('mhalo_sig', lower=0., upper=1., value=guess['mhalo_sig'])

    mstar_mhalo = pymc.Uniform('mstar_mhalo', lower=0., upper=2., value=guess['mstar_mhalo'])

    mstar_mu = pymc.Uniform('mstar_mu', lower=11., upper=12., value=guess['mstar_mu'])
    mstar_sig = pymc.Uniform('mstar_sig', lower=0., upper=2., value=guess['mstar_sig'])

    pars = [mhalo_mu, mhalo_sig, mstar_mhalo, mstar_mu, mstar_sig]

    @pymc.deterministic(name='like')
    def like(mhalo_mu=mhalo_mu, mhalo_sig=mhalo_sig, mstar_mhalo=mstar_mhalo, mstar_mu=mstar_mu, mstar_sig=mstar_sig):

        totlike = 0.

        for i in range(nlens):
            mglob_model = mstar_mu + mstar_mhalo*(tchains[i]['mhalo'] - 13.)

            mh_term = 1./mhalo_sig*np.exp(-(mhalo_mu - tchains[i]['mhalo'])**2/(2.*mhalo_sig**2))
            ms_term = 1./mstar_sig*np.exp(-(mglob_model - tchains[i]['mstar'])**2/(2.*mstar_sig**2))

            #norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/mstar_sig) - erf((mglob_model - 12.5)/2.**0.5/mstar_sig)) * \
            #        0.5*(erf((mhalo_mu - 12.)/2.**0.5/mhalo_sig) - erf((mhalo_mu - 14.)/2.**0.5/mhalo_sig))
            norms = 1.

            term = (mh_term*ms_term/norms).mean()
            totlike += np.log(term)

        return totlike

    @pymc.stochastic(observed=True, name='logp')
    def logp(value=0., p=pars):
        return like

    M = pymc.MCMC(pars+[like])
    M.use_step_method(pymc.AdaptiveMetropolis, pars)
    M.sample(nstep, burnin)

    outdic = {}
    for par in pars:
        outdic[str(par)] = M.trace(par)[:]
    outdic['logp'] = M.trace('like')[:]

    return outdic


def infer_simple_reality_knownimf_nocosmo_analytic(guess, mhalo, err_mhalo, mstar, err_mstar, nstep=11000, burnin=1000):

    nlens = len(mhalo)

    #defines the hyper-parameters

    mhalo_mu = pymc.Uniform('mhalo_mu', lower=12.0, upper=14.0, value=guess['mhalo_mu'])
    mhalo_sig = pymc.Uniform('mhalo_sig', lower=0., upper=1., value=guess['mhalo_sig'])

    mstar_mhalo = pymc.Uniform('mstar_mhalo', lower=0., upper=2., value=guess['mstar_mhalo'])

    mstar_mu = pymc.Uniform('mstar_mu', lower=11., upper=12., value=guess['mstar_mu'])
    mstar_sig = pymc.Uniform('mstar_sig', lower=0., upper=2., value=guess['mstar_sig'])

    pars = [mhalo_mu, mhalo_sig, mstar_mhalo, mstar_mu, mstar_sig]

    @pymc.deterministic(name='like')
    def like(mhalo_mu=mhalo_mu, mhalo_sig=mhalo_sig, mstar_mhalo=mstar_mhalo, mstar_mu=mstar_mu, mstar_sig=mstar_sig):

        totlike = 0.

        from scipy.integrate import dblquad

        for i in range(nlens):

            err_arr = np.array((err_mhalo[i], err_mstar[i]))

            obs_mu = np.array((mhalo[i], mstar[i]))
            obs_cov = np.diag(err_arr**2)
            obs_invcov = np.diag(err_arr**-2)

            pri_mu = np.array((mhalo_mu, mstar_mu + mstar_mhalo*(mhalo_mu - 13.)))
            pri_cov = np.array(((mhalo_sig**2, mstar_mhalo*mhalo_sig**2), (mstar_mhalo*mhalo_sig**2, mstar_sig**2 + \
                                                                           mstar_mhalo**2*mhalo_sig**2)))
            pri_invcov = 1./np.linalg.det(pri_cov)*np.array(((pri_cov[1, 1], -pri_cov[0, 1]), \
                                                             (-pri_cov[1, 0], pri_cov[0, 0])))

            prod_invcov = obs_invcov + pri_invcov
            prod_cov = 1./np.linalg.det(prod_invcov)*np.array(((prod_invcov[1, 1], -prod_invcov[0, 1]), \
                                                             (-prod_invcov[1, 0], prod_invcov[0, 0])))
            prod_mu = np.dot(prod_cov, np.dot(obs_invcov, obs_mu)) + np.dot(prod_cov, np.dot(pri_invcov, pri_mu))

            zc = (2.*np.pi)*np.linalg.det(prod_cov)**0.5*np.linalg.det(obs_cov)**-0.5*np.linalg.det(pri_cov)**-0.5* \
                 np.exp(-0.5*(np.dot(obs_mu.T, np.dot(obs_invcov, obs_mu)) + \
                              np.dot(pri_mu.T, np.dot(pri_invcov, pri_mu)) - \
                              np.dot(prod_mu.T, np.dot(prod_invcov, prod_mu))))

            totlike += np.log(zc)

        return totlike

    @pymc.stochastic(observed=True, name='logp')
    def logp(value=0., p=pars):
        return like

    M = pymc.MCMC(pars+[like])
    M.use_step_method(pymc.AdaptiveMetropolis, pars)
    M.sample(nstep, burnin)

    outdic = {}
    for par in pars:
        outdic[str(par)] = M.trace(par)[:]
    outdic['logp'] = M.trace('like')[:]

    return outdic

