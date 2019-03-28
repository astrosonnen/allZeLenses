import pymc
import h5py
import pickle
import numpy as np


mockname = 'mockO'

day = 24.*3600.

f = open('../paper/%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

chaindir = '/Users/sonnen/allZeChains/'

nlens = 50


parnames = ['beta', 'mdme', 'mstar', 'timedelay']

chains = []
reff = []
dt_obs = []
dt_err = []

for i in range(50):

    chainname = chaindir+'%s_lens_%02d_gnfwdev.hdf5'%(mockname, i)
    print chainname
    chain_file = h5py.File(chainname, 'r')

    chain = {}
    for par in parnames:
        chain[par] = chain_file[par].value.flatten()[100*np.arange(1000)].copy()

    chain['timedelay'] /= day

    bad = chain['timedelay'] != chain['timedelay']
    chain['timedelay'][bad] = 0.

    chains.append(chain)

    chain_file.close()

    #dt_obs.append(mock['lenses'][i].obs_timedelay[0]/day)
    dt_obs.append(mock['lenses'][i].timedelay/day)
    dt_err.append(mock['lenses'][i].obs_timedelay[1]/day)
    #dt_err.append(0.1)

    reff.append(np.log10(mock['lenses'][i].reff_phys))

H0 = pymc.Uniform('H0', lower=50., upper=100., value=70.)

beta_mu = pymc.Uniform('beta_mu', lower=0.5, upper=2., value=1.)
beta_sig = pymc.Uniform('beta_sig', lower=0., upper=1., value=0.1)

mdme_mu = pymc.Uniform('mdme_mu', lower=10., upper=12., value=11.)
mdme_re = pymc.Uniform('mdme_re', lower=0., upper=3., value=1.)
mdme_mstar = pymc.Uniform('mdme_mstar', lower=-1, upper=2., value=0.5)
mdme_sig = pymc.Uniform('mdme_sig', lower=0., upper=2., value=0.3)

mstar_mu = pymc.Uniform('mstar_mu', lower=10., upper=12., value=11.4)
mstar_sig = pymc.Uniform('mstar_sig', lower=0., upper=1., value=0.3)

pars = [H0, beta_mu, beta_sig, mdme_mu, mdme_re, mdme_mstar, mdme_sig, mstar_mu, mstar_sig]

reff_piv = 0.7
mstar_piv = 11.4

@pymc.deterministic(name='like')
def like(p=pars):

    H0, beta_mu, beta_sig, mdme_mu, mdme_re, mdme_mstar, mdme_sig, mstar_mu, mstar_sig = p

    totlike = 0.

    for i in range(nlens):

        mdme_muhere = mdme_mu + mdme_mstar * (chains[i]['mstar'] - mstar_piv) + mdme_re * (reff[i] - reff_piv)

        mdme_term = 1./mdme_sig * np.exp(-0.5*(chains[i]['mdme'] - mdme_muhere)**2/mdme_sig**2)

        beta_term = 1./beta_sig * np.exp(-0.5*(chains[i]['beta'] - beta_mu)**2/beta_sig**2)

        mstar_term = 1./mstar_sig * np.exp(-0.5*(chains[i]['mstar'] - mstar_mu)**2/mstar_sig**2)

        dt_term = 1./dt_err[i]*np.exp(-0.5*(dt_obs[i] - chains[i]['timedelay']/(H0/70.))**2/dt_err[i]**2)

        term = (mstar_term*beta_term*dt_term*mdme_term).mean()
        totlike += np.log(term)

    return totlike

@pymc.stochastic
def logp(value=0, observed=True, p=pars):
    return like

M = pymc.MCMC(pars)
M.use_step_method(pymc.AdaptiveMetropolis, pars)
M.sample(60000, 10000)

output = {}
for par in pars:
    output[str(par)] = M.trace(par)[:]

f = open('%s_gnfwdev_inference.dat'%mockname, 'w')
pickle.dump(output, f)
f.close()


