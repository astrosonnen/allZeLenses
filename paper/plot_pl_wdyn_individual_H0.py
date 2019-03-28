import numpy as np
import pylab
import lens_models
import pickle
import h5py
from matplotlib import rc
rc('text', usetex=True)


mockname = 'mockQ'

fsize = 20
tsize = 16

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

chain_file = h5py.File('%s_powerlaw_impos_dyn_chains.hdf5'%mockname, 'r')

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

for i in range(nlens):
    print i
    lens = mock['lenses'][i]
    psi1_mock[i] = lens.rein
    psi2_mock[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    psi3_mock[i] = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

    rein[i] = lens.rein

    dt_model = chain_file['lens_%02d'%i]['timedelay'].value.copy()[10*np.arange(1000)]

    dt_obs_sample = np.random.normal(lens.obs_timedelay[0], lens.obs_timedelay[1], len(dt_model))
    H0_chain = 70. * dt_model / dt_obs_sample

    ind_H0[i] = H0_chain.mean()
    H0_err[i] = H0_chain.std()

    """
    dtheta1 = lens.images[0] - lens.rein

    taylor_true[i] = 2.*psi1_mock[i]*(1. - psi2_mock[i]) * dtheta1 * (1. + 0.5*psi3_mock[i]/(1. - psi2_mock[i])*dtheta1)

    taylor_true[i] = 2.*psi1_mock[i]*(1. - psi2_mock[i]) * dtheta1 * (1. + 0.5*psi3_mock[i]/(1. - psi2_mock[i])*dtheta1)

    gamma = chain_file['lens_%02d'%i]['gamma'].value
    psi1 = chain_file['lens_%02d'%i]['rein'].value
    psi2 = 2.-gamma
    psi3 = (2.-gamma)*(1.-gamma)/psi1

    taylor_pl = 2.*psi1*(1. - psi2) * dtheta1 * (1. + 0.5*psi3/(1. - psi2)*dtheta1)

    taylor_fit[i] = taylor_pl.mean()
    taylor_err[i] = taylor_pl.std()
    """

H0_mean = (ind_H0/H0_err**2).sum()/(1./H0_err**2).sum()
H0_meanerr = ((1./H0_err**2).sum())**-0.5

xi = psi1_mock * psi3_mock - psi2_mock * (psi2_mock - 1.)
xo = psi3_mock/(1.-psi2_mock) - psi2_mock

psi2_diff = psi2_mock - 0.5*(1. - (1. + 4.*psi3_mock*psi1_mock)**0.5)

print H0_mean, H0_meanerr

fig = pylab.figure()
pylab.subplots_adjust(left=0.15, right=0.99, bottom=0.12, top=0.99)
ax1 = fig.add_subplot(1, 1, 1)

ax1.axhspan(H0_mean - H0_meanerr, H0_mean + H0_meanerr, color='gray')
ax1.errorbar(psi2_diff, ind_H0, yerr=H0_err, fmt='+', color='b')
ax1.axhline(70., linestyle='--', color='k')
pylab.ylim(20., 120.)
pylab.xticks(fontsize=tsize)
pylab.yticks(fontsize=tsize)

ax1.set_xlabel("$\\xi = \psi'' - \psi''_{\mathrm{PL}}(\psi',\psi''')$", fontsize=fsize)
ax1.set_ylabel("$H_0$", fontsize=fsize)
pylab.title('Power-law model. Fit to image positions and $\sigma_{e2}$', fontsize=tsize)
#pylab.savefig('pl_impos_dyn_individual_H0.eps')
pylab.show()



