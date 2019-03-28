import numpy as np
import pymc
import h5py
import pickle
from scipy.stats import beta


mockname = 'mockG'
chaindir = '/Users/sonnen/allZeChains/'

day = 24.*3600.

f = open(mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

parnames = ['rein', 'gmb', 'beta', 'timedelay', 's2']

chains = []
reffs = []

for i in range(nlens):
    chainname = chaindir+'%s_lens_%02d_brokenpl_gmb.hdf5'%(mockname, i)
    print chainname
    chain_file = h5py.File(chainname, 'r')

    chain = {}
    for par in parnames:
        chain[par] = chain_file[par].value.flatten()#[10*np.arange(1000)].copy()

    chain['timedelay'] /= day

    bad = chain['timedelay'] != chain['timedelay']
    chain['timedelay'][bad] = 0.

    chain['gmb_prior'] = (chain_file['gmb_prior']['mu'].value, chain_file['gmb_prior']['sigma'].value)
    chain['beta_prior'] = (chain_file['beta_prior']['mu'].value, chain_file['beta_prior']['sigma'].value)

    chains.append(chain)

    chain_file.close()

    reffs.append(mock['lenses'][i].reff)


gmb_mu = pymc.Uniform('gmb_mu', lower=1.5, upper=2.5, value=2.)
gmb_sig = pymc.Uniform('gmb_sig', lower=0., upper=1., value=0.2)

beta_mu = pymc.Uniform('beta_mu', lower=-1., upper=1., value=0.)
beta_sig = pymc.Uniform('beta_sig', lower=0., upper=1., value=0.2)
beta_gamdep = pymc.Uniform('beta_gamdep', lower=-3., upper=3., value=1.)

s2_loga = pymc.Uniform('s2_loga', lower=-1., upper=3., value=1.)
s2_bpar = pymc.Uniform('s2_b', lower=0., upper=1., value=0.1)

pars = [gmb_mu, gmb_sig, beta_mu, beta_sig, beta_gamdep, s2_loga, s2_bpar]

reinreff_piv = 0.
gmb_piv = 2.
msps_piv = 11.4
reff_piv = 0.8

@pymc.deterministic(name='like')
def like(p=pars):

    gmb_mu, gmb_sig, beta_mu, beta_sig, beta_gamdep, s2_loga, s2_bpar = p

    totlike = 0.

    tpa = 2.*np.pi*10.**s2_loga

    I = 1./tpa*(tpa*(s2_bpar*np.arctan(tpa*s2_bpar) - (s2_bpar-1.)*np.arctan(tpa*(s2_bpar - 1.))) - 0.5*np.log(1. + (tpa*s2_bpar)**2) + 0.5*np.log(1. + (tpa*(s2_bpar - 1.))**2))
    norm = 1./(0.5*np.pi + I)

    for i in range(nlens):

        gmb_muhere = gmb_mu

        beta_muhere = beta_mu + beta_gamdep * (chains[i]['gmb'] - gmb_piv)

        gmb_term = 1./gmb_sig*np.exp(-(gmb_muhere - chains[i]['gmb'])**2/(2.*gmb_sig**2))
        beta_term = 1./beta_sig*np.exp(-(beta_muhere - chains[i]['beta'])**2/(2.*beta_sig**2))

        s2_term = norm * (0.5*np.pi + np.arctan(tpa*(s2_bpar - chains[i]['s2']/chains[i]['rein']**2)))

        gmb_prior = 1./chains[i]['gmb_prior'][1]*np.exp(-0.5*(chains[i]['gmb'] - chains[i]['gmb_prior'][0])**2/chains[i]['gmb_prior'][1]**2)
        beta_prior = 1./chains[i]['beta_prior'][1]*np.exp(-0.5*(chains[i]['beta'] - chains[i]['beta_prior'][0])**2/chains[i]['beta_prior'][1]**2)

        interim_prior = gmb_prior * beta_prior

        term = (gmb_term*beta_term*s2_term/interim_prior).mean()
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

f = open('%s_knownH0_brokenpl_gmb_inference.dat'%mockname, 'w')
pickle.dump(output, f)
f.close()

