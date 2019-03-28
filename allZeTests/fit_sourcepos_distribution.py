import numpy as np
import pymc
import h5py
import pickle
from scipy.stats import beta


mockname = 'mockF'

f = open(mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

source = np.zeros(nlens)
rein = np.zeros(nlens)

for i in range(nlens):
    source[i] = mock['lenses'][i].source
    rein[i] = mock['lenses'][i].rein

s2_loga = pymc.Uniform('s2_loga', lower=-1., upper=2., value=1.)
s2_bpar = pymc.Uniform('s2_b', lower=0., upper=1., value=0.1)

pars = [s2_loga, s2_bpar]

@pymc.deterministic(name='like')
def like(p=pars):

    s2_loga, s2_bpar = p

    totlike = 0.

    tpa = 2.*np.pi*10.**s2_loga

    I = 1./tpa*(tpa*(s2_bpar*np.arctan(tpa*s2_bpar) - (s2_bpar-1.)*np.arctan(tpa*(s2_bpar - 1.))) - 0.5*np.log(1. + (tpa*s2_bpar)**2) + 0.5*np.log(1. + (tpa*(s2_bpar - 1.))**2))
    norm = 1./(0.5*np.pi + I)

    s2_term = norm * (0.5*np.pi + np.arctan(tpa*(s2_bpar - (source/rein)**2)))
    
    totlike = np.log(s2_term).sum()

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

f = open('%s_sourcepos_inference.dat'%mockname, 'w')
pickle.dump(output, f)
f.close()

