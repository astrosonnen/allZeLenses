import numpy as np
import pymc
import h5py
import pickle
from scipy.stats import beta


mockname = 'mockF'
chaindir = '/Users/sonnen/allZeChains/'

day = 24.*3600.

f = open(mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

parnames = ['rein', 'gamma', 'timedelay', 's2']

dt_obs = []
dt_err = []

chains = []

for i in range(nlens):
    chainname = chaindir+'%s_lens_%02d_powerlaw.hdf5'%(mockname, i)
    print chainname
    chain_file = h5py.File(chainname, 'r')

    chain = {}
    for par in parnames:
        chain[par] = chain_file[par].value.flatten()#[10*np.arange(1000)].copy()

    chain['timedelay'] /= day

    bad = chain['timedelay'] != chain['timedelay']
    chain['timedelay'][bad] = 0.

    chains.append(chain)

    chain_file.close()

    dt_obs.append(mock['lenses'][i].obs_timedelay[0]/day)
    dt_err.append(mock['lenses'][i].obs_timedelay[1]/day)

H0 = pymc.Uniform('H0', lower=50., upper=100., value=70.)

gamma_mu = pymc.Uniform('gamma_mu', lower=1.5, upper=2.5, value=2.)
gamma_sig = pymc.Uniform('gamma_sig', lower=0., upper=1., value=0.2)
gamma_redep = pymc.Uniform('gamma_redep', lower=-1., upper=1., value=0.)

beta_mu = pymc.Uniform('beta_mu', lower=-1., upper=1., value=0.)
beta_sig = pymc.Uniform('beta_sig', lower=0., upper=1., value=0.2)
beta_redep = pymc.Uniform('beta_redep', lower=-1., upper=1., value=0.)
beta_gamdep = pymc.Uniform('beta_gamdep', lower=-1., upper=1., value=0.)

s2_loga = pymc.Uniform('s2_loga', lower=-1., upper=1., value=0.)
s2_bpar = pymc.Uniform('s2_b', lower=0., upper=2., value=0.5)

pars = [gamma_mu, gamma_sig, gamma_redep, beta_mu, beta_sig, beta_redep, beta_gamdep, s2_loga, s2_bpar, H0]

reinreff_piv = 0.
gamma_piv = 2.

@pymc.deterministic(name='like')
def like(p=pars):

    gamma_mu, gamma_sig, gamma_redep, beta_mu, beta_sig, beta_redep, beta_gamdep, s2_loga, s2_bpar, H0 = p

    totlike = 0.

    tpa = 2.*np.pi*10.**s2_loga

    I = 1./tpa*(tpa*(s2_bpar*np.arctan(tpa*s2_bpar) - (s2_bpar-1.)*np.arctan(tpa*(s2_bpar - 1.))) - 0.5*np.log(1. + (tpa*s2_bpar)**2) + 0.5*np.log(1. + (tpa*(s2_bpar - 1.))**2))
    norm = 1./(0.5*np.pi + I)

    for i in range(nlens):

        gamma_muhere = gamma_mu + gamma_redep * (np.log10(chains[i]['rein']/reffs[i]) - reinreff_piv)

        beta_muhere = beta_mu + beta_redep * (np.log10(chains[i]['rein']/reffs[i]) - reinreff_piv) + beta_gamdep * (chains[i]['gamma'] - chains[i]['beta'] - gamma_piv)

        gamma_term = 1./gamma_sig*np.exp(-(gamma_muhere - chains[i]['gamma'] + chains[i]['beta'])**2/(2.*gamma_sig**2))
        beta_term = 1./beta_sig*np.exp(-(beta_muhere - chains[i]['beta'])**2/(2.*beta_sig**2))

        #s2_term = beta.pdf(chains[i]['s2_norm'], 10.**s2_loga, 10.**s2_logb)
        s2_term = norm * (0.5*np.pi + np.arctan(tpa*(s2_bpar - chains[i]['s2']/chains[i]['rein']**2)))

        gamma_prior = 1./chains[i]['gamma_prior'][1]*np.exp(-0.5*(chains[i]['gamma'] - chains[i]['gamma_prior'][0])**2/chains[i]['gamma_prior'][1]**2)
        beta_prior = 1./chains[i]['beta_prior'][1]*np.exp(-0.5*(chains[i]['beta'] - chains[i]['beta_prior'][0])**2/chains[i]['beta_prior'][1]**2)

        interim_prior = gamma_prior * beta_prior

        dt_term = 1./dt_err[i]*np.exp(-0.5*(dt_obs[i] - chains[i]['timedelay']/(H0/70.))**2/dt_err[i]**2)

        term = (gamma_term*beta_term*dt_term*s2_term/interim_prior).mean()
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

f = open('%s_brokenpowerlaw_gmb_inference.dat'%mockname, 'w')
pickle.dump(output, f)
f.close()

