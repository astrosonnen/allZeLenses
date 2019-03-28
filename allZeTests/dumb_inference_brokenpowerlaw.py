import numpy as np
import pymc
import h5py
import pickle
from scipy.stats import beta


mockname = 'mockC'
chaindir = '/Users/sonnen/allZeChains/'

day = 24.*3600.

f = open(mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

dt_obs = []
dt_err = []

chains = []

parnames = ['gamma', 'beta', 'timedelay']

for i in range(nlens):
    chainname = chaindir+'%s_lens_%02d_brokenpowerlaw.hdf5'%(mockname, i)
    print chainname
    chain_file = h5py.File(chainname, 'r')

    chain = {}
    for par in parnames:
        chain[par] = chain_file[par].value.flatten()#[10*np.arange(1000)].copy()

    chain['gamma_prior'] = (chain_file['gamma_prior']['mu'].value, chain_file['gamma_prior']['sigma'].value)
    chain['beta_prior'] = (chain_file['beta_prior']['mu'].value, chain_file['beta_prior']['sigma'].value)

    chain['timedelay'] /= day

    bad = chain['timedelay'] != chain['timedelay']
    chain['timedelay'][bad] = 0.

    chains.append(chain)

    chain_file.close()

    dt_obs.append(mock['lenses'][i].obs_timedelay[0]/day)
    dt_err.append(mock['lenses'][i].obs_timedelay[1]/day)

H0 = pymc.Uniform('H0', lower=50., upper=100., value=70.)

pars = [H0]

@pymc.deterministic(name='like')
def like(H0=H0):

    totlike = 0.

    for i in range(nlens):

        gamma_prior = 1./chains[i]['gamma_prior'][1]*np.exp(-0.5*(chains[i]['gamma'] - chains[i]['gamma_prior'][0])**2/chains[i]['gamma_prior'][1]**2)
        beta_prior = 1./chains[i]['beta_prior'][1]*np.exp(-0.5*(chains[i]['beta'] - chains[i]['beta_prior'][0])**2/chains[i]['beta_prior'][1]**2)

        interim_prior = gamma_prior * beta_prior

        dt_term = 1./dt_err[i]*np.exp(-0.5*(dt_obs[i] - chains[i]['timedelay']/(H0/70.))**2/dt_err[i]**2)

        term = (dt_term/interim_prior).mean()
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

f = open('%s_dumb_brokenpowerlaw_inference.dat'%mockname, 'w')
pickle.dump(output, f)
f.close()

