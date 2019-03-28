import numpy as np
import pymc
import h5py
import pickle


mockname = '/gdrive/projects/allZeLenses/allZeTests/powerlaw_100lenses.dat'
chainname = '/gdrive/projects/allZeLenses/allZeTests/powerlaw_100lenses_chains.hdf5'

day = 24.*3600.

f = open(mockname, 'r')
mock = pickle.load(f)
f.close()

chain_file = h5py.File(chainname, 'r')

nlens = len(mock['lenses'])

dt_obs = []
dt_err = []

dt_chains = []
gamma_chains = []

for i in range(nlens):
    dt_chains.append(chain_file['lens_%04d'%i]['timedelay'].value/day)
    gamma_chains.append(chain_file['lens_%04d'%i]['gamma'].value)

    dt_obs.append(mock['lenses'][i].obs_timedelay[0]/day)
    dt_err.append(mock['lenses'][i].obs_timedelay[1]/day)

H0 = pymc.Uniform('H0', lower=50., upper=100., value=70.)
gamma_mu = pymc.Uniform('mu', lower=1.5, upper=2.5, value=2.)
gamma_sig = pymc.Uniform('sig', lower=0., upper=1., value=0.2)

pars = [H0, gamma_mu, gamma_sig]

@pymc.deterministic
def like(H0=H0, mu=gamma_mu, sig=gamma_sig):

    sumlogp = 0.

    for i in range(nlens):
        dt_model = dt_chains[i]/(H0/100./mock['lenses'][i].h)
        dt_term = 1./dt_err[i]*np.exp(-0.5*(dt_model - dt_obs[i])**2/dt_err[i]**2)

        gamma_term = 1./sig * np.exp(-0.5*(gamma_chains[i] - mu)**2/sig**2)

        integrand = dt_term * gamma_term

        sumlogp += np.log(integrand.sum())

    return sumlogp

@pymc.stochastic
def logp(value=0, observed=True, p=pars):

    return like

M = pymc.MCMC(pars)
M.use_step_method(pymc.AdaptiveMetropolis, pars)
M.sample(11000, 1000)

output = {}
for par in pars:
    output[str(par)] = M.trace(par)[:]

f = open('powerlaw_bhi_inference_chain.dat', 'w')
pickle.dump(output, f)
f.close()

