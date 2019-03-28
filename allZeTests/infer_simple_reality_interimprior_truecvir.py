import numpy as np
import pymc
import h5py
import pickle


mockname = 'mockA'
chaindir = '/Users/sonnen/allZeChains/'

day = 24.*3600.

f = open(mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

parnames = ['mhalo', 'alpha', 'mstar', 'timedelay']

dt_obs = []
dt_err = []

chains = []

for i in range(nlens):
    chainname = chaindir+'%s_lens_%02d_interimprior_truecvir.hdf5'%(mockname, i)
    print chainname
    chain_file = h5py.File(chainname, 'r')

    chain = {}
    for par in parnames:
        chain[par] = chain_file[par].value.flatten().copy()

    chain['timedelay'] /= day

    bad = chain['timedelay'] != chain['timedelay']
    chain['timedelay'][bad] = 0.

    chains.append(chain)

    chain_file.close()

    dt_obs.append(mock['lenses'][i].obs_timedelay[0]/day)
    dt_err.append(mock['lenses'][i].obs_timedelay[1]/day)

guess = mock['truth']

H0 = pymc.Uniform('H0', lower=50., upper=100., value=70.)

mhalo_mu = pymc.Uniform('mhalo_mu', lower=12.0, upper=14.0, value=guess['mhalo_mu'])
mhalo_sig = pymc.Uniform('mhalo_sig', lower=0., upper=1., value=guess['mhalo_sig'])

mstar_mhalo = pymc.Uniform('mstar_mhalo', lower=0., upper=2., value=guess['mstar_mhalo'])

mstar_mu = pymc.Uniform('mstar_mu', lower=11., upper=12., value=guess['mstar_mu'])
mstar_sig = pymc.Uniform('mstar_sig', lower=0., upper=2., value=guess['mstar_sig'])

aimf_mu = pymc.Uniform('aimf_mu', lower=-0.2, upper=0.2, value=guess['aimf_mu'])
aimf_sig = pymc.Uniform('aimf_sig', lower=0., upper=0.5, value=guess['aimf_sig'])

pars = [mhalo_mu, mhalo_sig, mstar_mhalo, mstar_mu, mstar_sig, aimf_mu, aimf_sig, H0]

@pymc.deterministic(name='like')
def like(mhalo_mu=mhalo_mu, mhalo_sig=mhalo_sig, mstar_mhalo=mstar_mhalo, mstar_mu=mstar_mu, mstar_sig=mstar_sig, \
             aimf_mu=aimf_mu, aimf_sig=aimf_sig, H0=H0):

    totlike = 0.

    for i in range(nlens):
        mglob_model = mstar_mu + mstar_mhalo*(chains[i]['mhalo'] - 13.)

        mh_term = 1./mhalo_sig*np.exp(-(mhalo_mu - chains[i]['mhalo'])**2/(2.*mhalo_sig**2))
        ms_term = 1./mstar_sig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*mstar_sig**2))
        a_term = 1./aimf_sig*np.exp(-0.5*(aimf_mu - chains[i]['alpha'])**2/aimf_sig**2)

        interim_prior = 1./0.3*np.exp(-0.5*(chains[i]['mhalo'] - 13.)**2/0.5**2)*\
                        1./0.5*np.exp(-0.5*(chains[i]['mstar'] - 11.5)**2/0.5**2)*\
                        1./0.2*np.exp(-0.5*(chains[i]['alpha'] - 0.)**2/0.2**2)

        dt_term = 1./dt_err[i]*np.exp(-0.5*(dt_obs[i] - chains[i]['timedelay']/(H0/70.))**2/dt_err[i]**2)

        term = (mh_term*ms_term*a_term*dt_term/interim_prior).mean()
        totlike += np.log(term)

    return totlike

@pymc.stochastic
def logp(value=0, observed=True, p=pars):
    return like

M = pymc.MCMC(pars)
M.use_step_method(pymc.AdaptiveMetropolis, pars)
M.sample(22000, 2000)

output = {}
for par in pars:
    output[str(par)] = M.trace(par)[:]

f = open('%s_interimprior_truecvir_inference.dat'%mockname, 'w')
pickle.dump(output, f)
f.close()

