import numpy as np
import pymc
import h5py
import pickle
from scipy.stats import beta


mockname = 'mockE'
chaindir = '/Users/sonnen/allZeChains/'

day = 24.*3600.

f = open(mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])
nlens = 50

parnames = ['gamma', 'rein', 'source', 'caustic', 'timedelay']

dt_obs = []
dt_err = []

chains = []

for i in range(nlens):
    chainname = chaindir+'%s_lens_%02d_interimprior.hdf5'%(mockname, i)
    chain_file = h5py.File(chainname, 'r')

    chain = {}
    for par in parnames:
        chain[par] = chain_file[par].value.flatten()[10*np.arange(1000)].copy()

    chain['timedelay'] /= day

    chain['gamma_prior'] = chain_file['gamma_prior'].value.copy()

    chain['s2_norm'] = (chain['source']/chain['caustic'])**2
    print chain['s2_norm'].mean(), chain['s2_norm'].std()

    chains.append(chain)

    dt_obs.append(mock['lenses'][i].obs_timedelay[0]/day)
    dt_err.append(mock['lenses'][i].obs_timedelay[1]/day)

H0 = pymc.Uniform('H0', lower=50., upper=100., value=70.)
gamma_mu = pymc.Uniform('mu', lower=1.5, upper=2.5, value=2.)
gamma_sig = pymc.Uniform('sig', lower=0., upper=1., value=0.2)

s2_apar = pymc.Uniform('sa', lower=-1., upper=1., value=0.)
s2_bpar = pymc.Uniform('sb', lower=0., upper=1., value=0.5)

pars = [H0, gamma_mu, gamma_sig, s2_apar, s2_bpar]

@pymc.deterministic
def like(H0=H0, mu=gamma_mu, sig=gamma_sig, s2_apar=s2_apar, s2_bpar=s2_bpar):

    sumlogp = 0.

    tpa = 2.*np.pi*10.**s2_apar

    I = 1./tpa*(tpa*(s2_bpar*np.arctan(tpa*s2_bpar) - (s2_bpar-1.)*np.arctan(tpa*(s2_bpar - 1.))) - 0.5*np.log(1. + (tpa*s2_bpar)**2) + 0.5*np.log(1. + (tpa*(s2_bpar - 1.))**2))
    norm = 1./(0.5*np.pi + I)

    for i in range(nlens):
        dt_model = chains[i]['timedelay']/(H0/100./mock['lenses'][i].h)
        dt_term = 1./dt_err[i]*np.exp(-0.5*(dt_model - dt_obs[i])**2/dt_err[i]**2)

        gamma_term = 1./sig * np.exp(-0.5*(chains[i]['gamma'] - mu)**2/sig**2)

        s2_term = norm * (0.5*np.pi + np.arctan(2.*np.pi*10.**s2_apar*(s2_bpar - chains[i]['s2_norm']))) #* (mock['lenses'][i].images[0]/chains[i]['caustic']**2)

        interim_prior = 1./chains[i]['gamma_prior'][1] * np.exp(-0.5*(chains[i]['gamma'] - chains[i]['gamma_prior'][0])**2/chains[i]['gamma_prior'][1]**2)

        integrand = dt_term * gamma_term * s2_term / interim_prior

        sumlogp += np.log(integrand.sum())

    return sumlogp

@pymc.stochastic
def logp(value=0, observed=True, p=pars):

    return like

M = pymc.MCMC(pars)
M.use_step_method(pymc.AdaptiveMetropolis, pars)
M.sample(100000, 0)

output = {}
for par in pars:
    output[str(par)] = M.trace(par)[:]

f = open('%s_wsourcepars_inference.dat'%mockname, 'w')
pickle.dump(output, f)
f.close()

