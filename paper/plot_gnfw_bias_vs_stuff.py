import numpy as np
import pylab
import lens_models
import pickle
import h5py
#import pymc
#from matplotlib import rc
#rc('text', usetex=True)


mockname = 'mockP'

#chaindir = '/Users/sonnen/allZeChains/'
chaindir = '/Users/alesonny/allZeChains/'

fsize = 20
tsize = 16

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

psi1_mock = np.zeros(nlens)
psi2_mock = np.zeros(nlens)
psi3_mock = np.zeros(nlens)

rein = np.zeros(nlens)

ind_H0 = np.zeros(nlens)
H0_err = np.zeros(nlens)

eps = 1e-4

taylor_fit = np.zeros(nlens)
taylor_err = np.zeros(nlens)
taylor_true = np.zeros(nlens)
taylor_ml = np.zeros(nlens)

beta_fit = np.zeros(nlens)
rein_reff = np.zeros(nlens)
rs_reff = np.zeros(nlens)

asymm = np.zeros(nlens)

ml_H0 = np.zeros(nlens)

psi1_ml = np.zeros(nlens)
psi2_ml = np.zeros(nlens)
psi3_ml = np.zeros(nlens)

sigma_obs = np.zeros(nlens)
sigma_std = np.zeros(nlens)

nup = 0
ndw = 0

dt_chains = []
dt_obs = []
dt_err = []


for i in range(nlens):

    #chain_file = h5py.File(chaindir+'%s_lens_%02d_gnfwdev.hdf5'%(mockname, i), 'r')
    chain_file = h5py.File(chaindir+'%s_lens_%02d_gnfwdev_perfect_obs.hdf5'%(mockname, i), 'r')
    #chain_file = h5py.File(chaindir+'%s_lens_%02d_gnfwdev_perfect_impos.hdf5'%(mockname, i), 'r')

    lens = mock['lenses'][i]
    psi1_mock[i] = lens.rein
    psi2_mock[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    psi3_mock[i] = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

    dtheta1 = lens.images[0] - lens.rein
    taylor_true[i] = 2.*psi1_mock[i]*(1. - psi2_mock[i]) * dtheta1 * (1. + 0.5*psi3_mock[i]/(1. - psi2_mock[i])*dtheta1)

    rein[i] = lens.rein
    rs_reff[i] = lens.rs/lens.reff

    #dt_model = chain_file['timedelay'].value.copy()[100*np.arange(1000)]
    dt_model = chain_file['timedelay'].value.copy()[10*np.arange(1000)]

    dt_obs_sample = np.random.normal(lens.obs_timedelay[0], lens.obs_timedelay[1], len(dt_model))

    ml = chain_file['logp'].value.argmax()

    ml_H0[i] = 70. * chain_file['timedelay'][ml] / lens.timedelay

    psi1_ml[i] = chain_file['rein'][ml]
    psi2_ml[i] = chain_file['psi2'][ml]
    psi3_ml[i] = chain_file['psi3'][ml]

    dtheta1 = lens.images[0] - chain_file['rein'][ml]
    taylor_ml[i] = 2.*psi1_ml[i]*(1. - psi2_ml[i]) * dtheta1 * (1. + 0.5*psi3_ml[i]/(1. - psi2_ml[i])*dtheta1)

    H0_chain = 70. * dt_model / lens.timedelay #dt_obs_sample
    #H0_chain = 70. * dt_model / dt_obs_sample

    dt_chains.append(dt_model)
    dt_obs.append(lens.timedelay)
    dt_err.append(lens.obs_timedelay[1])

    if H0_chain.mean() - H0_chain.std() > 70.:
       nup += 1
    if H0_chain.mean() + H0_chain.std() < 70.:
       ndw += 1

    ind_H0[i] = H0_chain.mean()
    H0_err[i] = H0_chain.std()

    beta_fit[i] = chain_file['beta'].value.mean()

    rein_reff[i] = lens.rein/lens.reff

    asymm[i] = (lens.images[0] + lens.images[1])/(lens.images[0] - lens.images[1])

    sigma_obs[i] = lens.obs_sigma[0]
    sigma_std[i] = chain_file['sigma'].value.std()

    psit_true = psi3_mock[i]/(1. - psi2_mock[i])
    psit_ml = psi3_ml[i]/(1. - psi2_ml[i])
    #print '%d, H0: %2.1f, rs/Re: %2.1f, rein/Re: %2.1f beta: %3.2f, asymm: %3.2f, sigma_off: %3.2f, rmu_off: %3.2f'%(i, ml_H0[i], rs_reff[i], rein_reff[i], chain_file['beta'][ml], asymm[i], chain_file['sigma'][ml]/chain_file['sigma_true'].value - 1., chain_file['radmagratio'][ml]/lens.radmag_ratio - 1.)
    print '%d, H0: %2.1f, rein/reff: %2.1f, psi2_off: %3.2f, psi3_off: %3.2f, psit_off: %3.2f, sigma_off: %d'%(i, ml_H0[i], rein_reff[i], psi2_ml[i] - psi2_mock[i], psi1_ml[i]*psi3_ml[i] - psi1_mock[i]*psi3_mock[i], psit_ml - psit_true, chain_file['sigma'][ml] - mock['sigma_sample'][i])
    #print '%d, H0: %2.1f +/- %2.1f, sigma_true: %d, sigma_obs: %d, sigma_fit: %d +/- %d'%(i, ind_H0[i], H0_err[i], mock['sigma_sample'][i], sigma_obs[i], chain_file['sigma'].value.mean(), chain_file['sigma'].value.std())

good = (rein > 0.5)

print ml_H0[good].mean(), ml_H0[good].std()

pylab.scatter((taylor_ml/taylor_true)[good], ml_H0[good])
pylab.show()

pylab.scatter(rein_reff[good], ml_H0[good])
pylab.show()

pylab.scatter(rein_reff[good], (psi2_ml - psi2_mock)[good])
pylab.show()

pylab.scatter((psi2_ml - psi2_mock)[good], ml_H0[good])
pylab.show()

pylab.scatter(rs_reff[good], ml_H0[good])
pylab.show()

