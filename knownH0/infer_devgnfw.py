import numpy as np
from scipy.interpolate import splrep, splev, splint
import pickle
import emcee
import sys
from scipy.stats import truncnorm
import h5py
import lens_models
from scipy.optimize import minimize, brentq
from mass_profiles import gNFW as gNFW_profile
from allZeTools import cgsconstants as cgs


mockname = 'devgnfw_A'

nstep = 500

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

mchab_err = 0.1

nlens = len(mock['lmstar_samp'])

grids_file = h5py.File('%s_devgnfw_perfectobs_grids.hdf5'%mockname, 'r')

lmstar_max = []
lmstar_grids = []
lmdm5_grids = []
gammadm_grids = []
beta_grids = []
Jdet_grids = []
toskip = []

for i in range(nlens):
    group = grids_file['lens_%03d'%i]
    lmstar_max.append(group['lmstar_max'].value)
    lmstar_grids.append(group['lmstar_grid'].value.copy())
    lmdm5_grids.append(group['lmdm5_grid'].value.copy())
    beta_grids.append(group['beta_grid'].value.copy())
    gammadm_grids.append(group['gammadm_grid'].value.copy())
    Jdet_grids.append(group['Jdet_grid'].value.copy())
    if np.isnan(group['Jdet_grid'].value.sum()):
        toskip.append(i)

print toskip

mstar_piv = 11.6

mstar_mu = {'name': 'mstar_mu', 'lower': 10., 'upper': 13., 'guess': 11.5, 'step': 0.01}
mstar_sig = {'name': 'mstar_sig', 'lower': 0., 'upper': 1., 'guess': 0.3, 'step': 0.01}

aimf_mu = {'name': 'aimf_mu', 'lower': -0.2, 'upper': 0.3, 'guess': 0.1, 'step': 0.01}
aimf_sig = {'name': 'aimf_sig', 'lower': 0., 'upper': 1., 'guess': 0.1, 'step': 0.01}

mdm5_mu = {'name': 'mdm5_mu', 'lower': 10., 'upper': 13., 'guess': 11., 'step': 0.01}
mdm5_sig = {'name': 'mdm5_sig', 'lower': 0., 'upper': 2., 'guess': 0.3, 'step': 0.01}
mdm5_beta = {'name': 'mdm5_beta', 'lower': -1., 'upper': 3., 'guess': 0.3, 'step': 0.01}

gammadm_mu = {'name': 'gammadm_mu', 'lower': 0.2, 'upper': 2.8, 'guess': 1.3, 'step': 0.01}
gammadm_sig = {'name': 'gammadm_sig', 'lower': 0., 'upper': 1., 'guess': 0.1, 'step': 0.01}

pars = [mstar_mu, mstar_sig, aimf_mu, aimf_sig, mdm5_mu, mdm5_sig, mdm5_beta, gammadm_mu, gammadm_sig]

npars = len(pars)

nwalkers = 6*npars

bounds = []
for par in pars:
    bounds.append((par['lower'], par['upper']))

def logprior(p):
    for i in range(npars):
        if p[i] < bounds[i][0] or p[i] > bounds[i][1]:
            return -1e300
    return 0.

def logpfunc(p):

    lprior = logprior(p)
    if lprior < 0.:
        return -1e300

    mstar_mu, mstar_sig, aimf_mu, aimf_sig, mdm5_mu, mdm5_sig, mdm5_beta, gammadm_mu, gammadm_sig = p

    logp = 0.
    for i in range(nlens):

        mstar_mu_eff = aimf_mu + mock['lmchab_obs'][i]
        mstar_sig_eff = (mchab_err**2 + aimf_sig**2)**0.5

        aimf_integral = 1./(2.*np.pi)**0.5/mstar_sig_eff * np.exp(-0.5*(lmstar_grids[i] - mstar_mu_eff)**2/mstar_sig_eff**2)
        mstar_term = 1./(2.*np.pi)**0.5/mstar_sig * np.exp(-0.5*(lmstar_grids[i] - mstar_mu)**2/mstar_sig**2)

        mdm5_muhere = mdm5_mu + mdm5_beta * (lmstar_grids[i] - mstar_piv)
        mdm5_term = 1./(2.*np.pi)**0.5/mdm5_sig*np.exp(-0.5*(lmdm5_grids[i] - mdm5_muhere)**2/mdm5_sig**2)

        gammadm_term = 1./(2.*np.pi)**0.5/gammadm_sig*np.exp(-0.5*(gammadm_grids[i] - gammadm_mu)**2/gammadm_sig**2)
        beta_term = beta_grids[i]

        if not i in toskip:
            integrand_grid = aimf_integral * mstar_term * mdm5_term * gammadm_term * beta_term / np.abs(Jdet_grids[i])
            integrand_grid[lmstar_grids[i] >= lmstar_max[i]] = 0.
            integrand_spline = splrep(lmstar_grids[i], integrand_grid)
            integral = splint(lmstar_grids[i][0], lmstar_max[i], integrand_spline)

            logp += np.log(integral)

    if logp != logp:
        return -1e300

    #logp += -0.5*(gammadm_mu - 1.3)**2/0.03**2 # Gaussian prior on gammadm

    return logp

sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc, threads=1)

start = []
if len(sys.argv) > 1:
    print 'using last step of %s to initialize walkers'%sys.argv[1]
    f = open(workdir+'/%s'%sys.argv[1], 'r')
    startfile = pickle.load(f)
    f.close()

    for i in range(nwalkers):
        tmp = np.zeros(npars)
        for n in range(npars):
            tmp[n] = startfile[pars[n]['name']][i, -1]
        start.append(tmp)

else:
    for i in range(nwalkers):
        tmp = np.zeros(npars)
        for j in range(npars):
            a, b = (bounds[j][0] - pars[j]['guess'])/pars[j]['step'], (bounds[j][1] - pars[j]['guess'])/pars[j]['step']
            p0 = truncnorm.rvs(a, b, size=1)*pars[j]['step'] + pars[j]['guess']
            tmp[j] = p0

        start.append(tmp)

print "Sampling"

sampler.run_mcmc(start, nstep)

output_file = h5py.File('%s_devgnfw_inference.hdf5'%mockname, 'w')
output_file.create_dataset('logp', data = sampler.lnprobability)
for n in range(npars):
    output_file.create_dataset(pars[n]['name'], data = sampler.chain[:, :, n])

