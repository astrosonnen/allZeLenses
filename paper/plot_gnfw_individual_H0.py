import numpy as np
import pylab
import lens_models
import pickle
import h5py
from matplotlib import rc
rc('text', usetex=True)


mockname = 'mockO'

chaindir = '/Users/sonnen/allZeChains/'

fsize = 20
tsize = 16

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])
nlens = 50

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

sigma_obs = np.zeros(nlens)

nup = 0
ndw = 0
for i in range(nlens):

    chain_file = h5py.File(chaindir+'%s_lens_%02d_gnfwdev.hdf5'%(mockname, i), 'r')

    lens = mock['lenses'][i]
    psi1_mock[i] = lens.rein
    psi2_mock[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    psi3_mock[i] = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

    rein[i] = lens.rein

    dt_model = chain_file['timedelay'].value.copy()[100*np.arange(1000)]

    dt_obs_sample = np.random.normal(lens.obs_timedelay[0], lens.obs_timedelay[1], len(dt_model))

    H0_chain = 70. * dt_model / lens.timedelay #dt_obs_sample
    #H0_chain = 70. * dt_model / dt_obs_sample

    ind_H0[i] = H0_chain.mean()
    H0_err[i] = H0_chain.std()

    sigma_obs[i] = lens.obs_sigma[0]

    print '%d, %2.1f, %2.1f'%(i, ind_H0[i], H0_err[i])

    if H0_chain.mean() - H0_chain.std() > 70.:
        nup += 1
    if H0_chain.mean() + H0_chain.std() < 70.:
        ndw += 1

H0_err[H0_err < 1.] = 1000000.
H0_mean = (ind_H0/H0_err**2).sum()/(1./H0_err**2).sum()
H0_meanerr = ((1./H0_err**2).sum())**-0.5

print H0_mean, H0_meanerr
print nup, ndw

xi = psi1_mock * psi3_mock - psi2_mock * (psi2_mock - 1.)
xo = psi3_mock/(1.-psi2_mock) - psi2_mock

psi2_diff = psi2_mock - 0.5*(1. - (1. + 4.*psi3_mock*psi1_mock)**0.5)

fig = pylab.figure()
pylab.subplots_adjust(left=0.15, right=0.99, bottom=0.12, top=0.99)
ax1 = fig.add_subplot(1, 1, 1)

ax1.axhspan(H0_mean - H0_meanerr, H0_mean + H0_meanerr, color='gray')
ax1.errorbar(psi2_diff, ind_H0, yerr=H0_err, fmt='+', color='b')
ax1.axhline(70., linestyle='--', color='k')
pylab.ylim(0., 140.)
pylab.xticks(fontsize=tsize)
pylab.yticks(fontsize=tsize)

yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

ax1.set_xlabel("$\\xi = \psi'' - \psi''_{\mathrm{PL}}(\psi',\psi''')$", fontsize=fsize)
ax1.set_ylabel("$H_0$", fontsize=fsize)
#pylab.text(-0.1, 30., 'Composite model. Fit to image positions, $r_{\mu_r}$, and $\sigma_{e2}$', fontsize=tsize)
pylab.text(-0.08, 25., 'truth: de Vauc. + NFW', fontsize=fsize)
#pylab.text(-0.08, 25., 'truth: de Vauc. w/ $M_*/L$ gradient + NFW', fontsize=fsize)
pylab.text(-0.08, 15., 'model: de Vauc. + gNFW', fontsize=fsize)
pylab.text(-0.08, 5., 'constraints: image positions, $r_\mu$, $\sigma_{e2}$', fontsize=fsize)

fig = pylab.gcf()
fig.set_size_inches(6, 8)
#pylab.savefig('gnfw_individual_H0_mock2.eps')
pylab.show()

pylab.scatter(sigma_obs/mock['sigma_sample'][:nlens], ind_H0)
pylab.show()

pylab.hist(sigma_obs/mock['sigma_sample'][:nlens])
pylab.show()
