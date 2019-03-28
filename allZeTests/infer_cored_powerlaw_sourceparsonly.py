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

    chains.append(chain)

    dt_obs.append(mock['lenses'][i].obs_timedelay[0]/day)
    dt_err.append(mock['lenses'][i].obs_timedelay[1]/day)

s2_apar = pymc.Uniform('sa', lower=-1., upper=1., value=0.)
s2_bpar = pymc.Uniform('sb', lower=0., upper=1., value=0.5)

pars = [s2_apar, s2_bpar]

@pymc.deterministic
def like(s2_apar=s2_apar, s2_bpar=s2_bpar):

    sumlogp = 0.

    tpa = 2.*np.pi*10.**s2_apar

    I = 1./tpa*(tpa*(s2_bpar*np.arctan(tpa*s2_bpar) - (s2_bpar-1.)*np.arctan(tpa*(s2_bpar - 1.))) - 0.5*np.log(1. + (tpa*s2_bpar)**2) + 0.5*np.log(1. + (tpa*(s2_bpar - 1.))**2))
    norm = 1./(0.5*np.pi + I)

    for i in range(nlens):
        s2_term = norm * (0.5*np.pi + np.arctan(2.*np.pi*10.**s2_apar*(s2_bpar - chains[i]['s2_norm']))) * (mock['lenses'][i].images[0] / chains[i]['caustic'])**2

        integrand = s2_term

        sumlogp += np.log(integrand.sum())
        #s2_term = norm * (0.5*np.pi + np.arctan(2.*np.pi*10.**s2_apar*(s2_bpar - (mock['lenses'][i].source/mock['lenses'][i].caustic)**2)))
        #sumlogp += np.log(s2_term)

    return sumlogp

@pymc.stochastic
def logp(value=0, observed=True, p=pars):

    return like

M = pymc.MCMC(pars+[like])
M.use_step_method(pymc.AdaptiveMetropolis, pars)
M.sample(20000, 0)

output = {}
for par in pars:
    output[str(par)] = M.trace(par)[:]
output['logp'] = M.trace('like')[:]

f = open('%s_sourceparsonly_inference.dat'%mockname, 'w')
pickle.dump(output, f)
f.close()

