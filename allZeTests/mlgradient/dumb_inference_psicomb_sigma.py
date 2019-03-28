import numpy as np
import pymc
import h5py
import pickle
from scipy.stats import beta


mockname = 'mockL'
mockdir = '/gdrive/projects/allZeLenses/allZeTests/'
chaindir = '/Users/sonnen/allZeChains/'

day = 24.*3600.

f = open(mockdir+mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])
nlens = 60

dt_obs = []
dt_err = []

chains = []

parnames = ['timedelay']

for i in range(nlens):
    chainname = chaindir+'%s_lens_%02d_psifit_nfwdev_wsigma_flatprior.hdf5'%(mockname, i)
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

pars = [H0]

@pymc.deterministic(name='like')
def like(H0=H0):

    totlike = 0.

    for i in range(nlens):

        dt_term = 1./dt_err[i]*np.exp(-0.5*(dt_obs[i] - chains[i]['timedelay']/(H0/70.))**2/dt_err[i]**2)

        totlike += np.log(dt_term.mean())

    return totlike

@pymc.stochastic
def logp(value=0, observed=True, p=pars):
    return like

M = pymc.MCMC(pars)
M.use_step_method(pymc.AdaptiveMetropolis, pars)
M.sample(20000, 10000)

output = {}
for par in pars:
    output[str(par)] = M.trace(par)[:]

f = open('%s_psicomb_sigma_dumb_inference.dat'%mockname, 'w')
pickle.dump(output, f)
f.close()

