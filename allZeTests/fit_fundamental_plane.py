import numpy as np
import pickle
import pylab
import emcee
from scipy.stats import truncnorm
import h5py


f = open('../paper/mockP.dat', 'r')
mock = pickle.load(f)
f.close()

mstar_piv = 11.4
lmstar = mock['mstar_sample']
lsigma = np.log10(mock['sigma_sample'])
lreff = np.log10(mock['reff_sample'])

nstep = 500

reff_mu = {'name': 'reff_mu', 'lower': 0., 'upper': 2., 'guess': 0.9, 'step': 0.03}
reff_sig = {'name': 'reff_sig', 'lower': 0., 'upper': 1., 'guess': 0.2, 'step': 0.03}
reff_mstar_dep = {'name': 'reff_mstar_dep', 'lower': 0., 'upper': 1., 'guess': 0.2, 'step': 0.03}

sigma_mu = {'name': 'sigma_mu', 'lower': 1., 'upper': 3., 'guess': 2.33, 'step': 0.1}
sigma_sig = {'name': 'sigma_sig', 'lower': 0., 'upper': 1., 'guess': 0.05, 'step': 0.03}
sigma_mstar_dep = {'name': 'sigma_mstar_dep', 'lower': -3., 'upper': 3., 'guess': 0.3, 'step': 0.03}
sigma_reff_dep = {'name': 'sigma_reff_dep', 'lower': -3., 'upper': 3., 'guess': 0.3, 'step': 0.03}

pars = [reff_mu, reff_sig, reff_mstar_dep, sigma_mu, sigma_sig, sigma_mstar_dep, sigma_reff_dep]

npars = len(pars)

nwalkers = 6*npars

bounds = []
for par in pars:
    bounds.append((par['lower'], par['upper']))

start = []
for i in range(nwalkers):
    tmp = np.zeros(npars)
    for j in range(npars):
        a, b = (bounds[j][0] - pars[j]['guess'])/pars[j]['step'], (bounds[j][1] - pars[j]['guess'])/pars[j]['step']
        p0 = truncnorm.rvs(a, b, size=1)*pars[j]['step'] + pars[j]['guess']
        tmp[j] = p0

    start.append(tmp)

def logprior(p):
    for i in range(npars):
        if p[i] < bounds[i][0] or p[i] > bounds[i][1]:
            return -1e300
    return 0.

def logpfunc(p):

    lprior = logprior(p)
    if lprior < 0.:
        return -1e300

    reff_mu, reff_sig, reff_mstar_dep, sigma_mu, sigma_sig, sigma_mstar_dep, sigma_reff_dep = p

    reff_muhere = reff_mu + reff_mstar_dep * (lmstar - mstar_piv)
    sigma_muhere = sigma_mu + sigma_mstar_dep * (lmstar - mstar_piv) + sigma_reff_dep * (lreff - reff_muhere)

    reff_like = -0.5*(lreff - reff_muhere)**2/reff_sig**2 - np.log10(reff_sig)
    sigma_like = -0.5*(lsigma - sigma_muhere)**2/sigma_sig**2 - np.log10(sigma_sig)

    return (reff_like + sigma_like).sum()

sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc, threads=1)

print "Sampling"

sampler.run_mcmc(start, nstep)

ml = sampler.lnprobability.argmax()
for n in range(npars):
    print '%s %4.3f %4.3f'%(pars[n]['name'], sampler.chain[:, -100:, n].mean(), sampler.chain[:, -100:, n].std())

output = h5py.File('mockP_mstar_reff_sigma_rel_inference.hdf5', 'w')
output.create_dataset('logp', data=sampler.lnprobability)
for n in range(npars):
    output.create_dataset(pars[n]['name'], data=sampler.chain[:, :, n])

