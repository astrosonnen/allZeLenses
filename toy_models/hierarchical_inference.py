import numpy as np
import pymc
import pickle

day = 24.*3600.


def infer_simple_reality_nocosmo(mock, chains, nstep=11000, burnin=1000, nis=1000):

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

    truth = mock['truth']

    mhalo_mu = pymc.Uniform('mhalo_mu', lower=12.0, upper=14.0, value=truth['mhalo_mu'])
    mhalo_sig = pymc.Uniform('mhalo_sig', lower=0., upper=1., value=truth['mhalo_sig'])

    mstar_mhalo = pymc.Uniform('mstar_mhalo', lower=0., upper=2., value=truth['mstar_mhalo'])

    mstar_mu = pymc.Uniform('mstar_mu', lower=11., upper=12., value=truth['mstar_mu'])
    mstar_sig = pymc.Uniform('mstar_sig', lower=0., upper=2., value=truth['mstar_sig'])

    aimf_mu = pymc.Uniform('aimf_mu', lower=-0.2, upper=0.2, value=truth['aimf_mu'])
    aimf_sig = pymc.Uniform('aimf_sig', lower=0., upper=0.5, value=truth['aimf_sig'])

    pars = [mhalo_mu, mhalo_sig, mstar_mhalo, mstar_mu, mstar_sig, aimf_mu, aimf_sig]

    @pymc.deterministic(name='like')
    def like(mhalo_mu=mhalo_mu, mhalo_sig=mhalo_sig, mstar_mhalo=mstar_mhalo, mstar_mu=mstar_mu, mstar_sig=mstar_sig, \
             aimf_mu=aimf_mu, aimf_sig=aimf_sig):

        totlike = 0.

        for i in range(nlens):
            mglob_model = mstar_mu + mstar_mhalo*(tchains[i]['mhalo'] - 13.)

            mhexp = 1./mhalo_sig*np.exp(-(mhalo_mu - tchains[i]['mhalo'])**2/(2.*mhalo_sig**2))
            msexp = 1./mstar_sig*np.exp(-(mglob_model - tchains[i]['mstar'])**2/(2.*mstar_sig**2))
            aexp = 1./aimf_sig*np.exp(-0.5*(aimf_mu - tchains[i]['alpha'])**2/aimf_sig**2)

            norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/mstar_sig) - erf((mglob_model - 12.5)/2.**0.5/mstar_sig)) * \
                    0.5*(erf((mhalo_mu - 12.)/2.**0.5/mhalo_sig) - erf((mhalo_mu - 14.)/2.**0.5/mhalo_sig))

            term = (mhexp*msexp*aexp/norms).mean()
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


def infer_simple_reality_knownimf_nocosmo(mock, chains, nstep=11000, burnin=1000, nis=1000):

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

    truth = mock['truth']

    mhalo_mu = pymc.Uniform('mhalo_mu', lower=12.0, upper=14.0, value=truth['mhalo_mu'])
    mhalo_sig = pymc.Uniform('mhalo_sig', lower=0., upper=1., value=truth['mhalo_sig'])

    mstar_mhalo = pymc.Uniform('mstar_mhalo', lower=0., upper=2., value=truth['mstar_mhalo'])

    mstar_mu = pymc.Uniform('mstar_mu', lower=11., upper=12., value=truth['mstar_mu'])
    mstar_sig = pymc.Uniform('mstar_sig', lower=0., upper=2., value=truth['mstar_sig'])

    pars = [mhalo_mu, mhalo_sig, mstar_mhalo, mstar_mu, mstar_sig]

    @pymc.deterministic(name='like')
    def like(mhalo_mu=mhalo_mu, mhalo_sig=mhalo_sig, mstar_mhalo=mstar_mhalo, mstar_mu=mstar_mu, mstar_sig=mstar_sig):

        totlike = 0.

        for i in range(nlens):
            mglob_model = mstar_mu + mstar_mhalo*(tchains[i]['mhalo'] - 13.)

            mhexp = 1./mhalo_sig*np.exp(-(mhalo_mu - tchains[i]['mhalo'])**2/(2.*mhalo_sig**2))
            msexp = 1./mstar_sig*np.exp(-(mglob_model - tchains[i]['mstar'])**2/(2.*mstar_sig**2))

            #norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/mstar_sig) - erf((mglob_model - 12.5)/2.**0.5/mstar_sig)) * \
            #        0.5*(erf((mhalo_mu - 12.)/2.**0.5/mhalo_sig) - erf((mhalo_mu - 14.)/2.**0.5/mhalo_sig))
            norms = 1.

            term = (mhexp*msexp/norms).mean()
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

