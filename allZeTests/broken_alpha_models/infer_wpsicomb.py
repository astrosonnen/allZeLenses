import numpy as np
import pymc
import h5py
import pickle
from scipy.stats import beta


mockname = 'mockI'
chaindir = '/Users/sonnen/allZeChains/'
mockdir = '/gdrive/projects/allZeLenses/allZeTests/'

day = 24.*3600.

f = open(mockdir+mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

parnames = ['gamma', 'beta', 'timedelay', 's2']

dt_obs = []
dt_err = []

new_err = 0.1

chains = []

for i in range(nlens):
    chainname = chaindir+'%s_lens_%02d_psifit_brokenalpha_flatprior.hdf5'%(mockname, i)
    chain_file = h5py.File(chainname, 'r')

    chain = {}
    for par in parnames:
        chain[par] = chain_file[par].value.flatten()#[10*np.arange(1000)].copy()

    chain['timedelay'] /= day

    bad = chain['timedelay'] != chain['timedelay']
    chain['timedelay'][bad] = 0.

    chains.append(chain)

    chain_file.close()

    #dt_obs.append(mock['lenses'][i].obs_timedelay[0]/day)
    #dt_err.append(mock['lenses'][i].obs_timedelay[1]/day)

    dt_obs.append(np.random.normal(mock['lenses'][i].timedelay/day, new_err, 1))
    dt_err.append(new_err)

    print '%d %3.2f %3.2f %3.2f %3.2f'%(i, chain['gamma'].mean(), chain['gamma'].std(), chain['beta'].mean(), chain['beta'].std())

H0 = pymc.Uniform('H0', lower=40., upper=100., value=70.)

gamma_mu = pymc.Uniform('gamma_mu', lower=1.5, upper=2.5, value=2.)
gamma_sig = pymc.Uniform('gamma_sig', lower=0., upper=1., value=0.2)

beta_mu = pymc.Uniform('beta_mu', lower=-1., upper=1., value=0.)
beta_sig = pymc.Uniform('beta_sig', lower=0., upper=1., value=0.2)
beta_gamdep = pymc.Uniform('beta_gamdep', lower=-2., upper=1., value=0.)

pars = [gamma_mu, gamma_sig, beta_mu, beta_sig, beta_gamdep, H0]

gamma_piv = 2.

@pymc.deterministic(name='like')
def like(p=pars):

    gamma_mu, gamma_sig, beta_mu, beta_sig, beta_gamdep, H0 = p

    totlike = 0.

    for i in range(nlens):

        gamma_muhere = gamma_mu

        beta_muhere = beta_mu + beta_gamdep * (chains[i]['gamma'] - gamma_piv)

        gamma_term = 1./gamma_sig*np.exp(-(gamma_muhere - chains[i]['gamma'])**2/(2.*gamma_sig**2))
        beta_term = 1./beta_sig*np.exp(-(beta_muhere - chains[i]['beta'])**2/(2.*beta_sig**2))

        dt_term = 1./dt_err[i]*np.exp(-0.5*(dt_obs[i] - chains[i]['timedelay']/(H0/70.))**2/dt_err[i]**2)

        term = (gamma_term*beta_term*dt_term).mean()
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

f = open('%s_wpsicomb_inference.dat'%mockname, 'w')
pickle.dump(output, f)
f.close()

