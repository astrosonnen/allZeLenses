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

for i in range(nlens):
    dt_chains.append(chain_file['lens_%04d'%i]['timedelay'].value/day)
    dt_obs.append(mock['lenses'][i].obs_timedelay[0]/day)
    dt_err.append(mock['lenses'][i].obs_timedelay[1]/day)

H0 = pymc.Uniform('H0', lower=50., upper=100., value=70.)

@pymc.deterministic
def like(H0=H0):

    sumlogp = 0.

    for i in range(nlens):
        dt_model = dt_chains[i]/(H0/100./mock['lenses'][i].h)
        dt_term = 1./dt_err[i]*np.exp(-0.5*(dt_model - dt_obs[i])**2/dt_err[i]**2)
        sumlogp += np.log(dt_term.sum())

    return sumlogp

@pymc.stochastic
def logp(value=0, observed=True, H0=H0):

    return like

M = pymc.MCMC([H0])
M.sample(11000, 1000)

f = open('dumb_inference_chain.dat', 'w')
pickle.dump(M.trace('H0')[:], f)
f.close()

