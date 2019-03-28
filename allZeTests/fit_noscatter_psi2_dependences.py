import numpy as np
import pymc
import pickle
import pylab
from plotters import cornerplot


mockname = 'mockI'

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

psi2 = np.zeros(nlens)
rein = np.zeros(nlens)

eps = 1e-4
for i in range(nlens):
    lens = mock['lenses'][i]

    psi2[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    rein[i] = lens.rein * lens.arcsec2kpc

mu_par = pymc.Uniform('mu', lower=-1., upper=1., value=0.)
sig_par = pymc.Uniform('sig', lower=0., upper=1., value=0.2)
mstar_dep = pymc.Uniform('beta', lower=-2., upper=2., value=0.)
reff_dep = pymc.Uniform('xi', lower=-2., upper=2., value=0.)
rein_dep = pymc.Uniform('b', lower=-2., upper=2., value=0.)

pars = [mu_par, sig_par, mstar_dep, reff_dep, rein_dep]

mstar = mock['mstar_sample']
mstar_piv = mstar.mean()

reff = np.log10(mock['reff_sample'])
reff_piv = reff.mean()

rein_piv = rein.mean()

@pymc.deterministic
def like(p=pars):

    mu, sig, beta, xi, b = p

    muhere = mu + beta * (mstar - mstar_piv) + xi * (reff - reff_piv) + b * (rein - rein_piv)

    logp = -0.5*(muhere - psi2)**2/sig**2 - np.log(sig)

    return logp.sum()

@pymc.stochastic
def logp(p=pars, value=0., observed=True):
    return like

M = pymc.MCMC(pars)
M.sample(11000, 1000)

cp = []
for par in pars:
    cp.append({'data': M.trace(par)[:], 'label': str(par)})

cornerplot(cp, color='r')
pylab.show()




