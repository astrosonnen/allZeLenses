import numpy as np
import pylab
import lens_models
import pickle
import h5py
from matplotlib import rc
rc('text', usetex=True)


mockname = 'mockP'

max_asymm = 0.5

chaindir = '/Users/alesonny/allZeChains/'

fsize = 20
tsize = 16

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

psi1_mock = []
psi2_mock = []
psi3_mock = []

rein = []

ind_H0 = []
H0_err = []

eps = 1e-4

for i in range(nlens):

    chain_file = h5py.File(chaindir+'%s_lens_%02d_gnfwdev.hdf5'%(mockname, i), 'r')

    lens = mock['lenses'][i]
    asymm = (lens.images[0] + lens.images[1])/(lens.images[0] - lens.images[1])

    dt_model = chain_file['timedelay'].value.copy()[100*np.arange(1000)]
  
    dt_obs_sample = np.random.normal(lens.obs_timedelay[0], lens.obs_timedelay[1], len(dt_model))
    
    H0_chain = 70. * dt_model / dt_obs_sample
    err = H0_chain.std()
 
    if asymm < max_asymm and err > 1.:

        psi1_mock.append(lens.rein)
        psi2_mock.append((lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps))
        psi3_mock.append((lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2)
    
   
        ind_H0.append(H0_chain.mean())
        H0_err.append(H0_chain.std())

        print '%d %2.1f %2.1f'%(i, H0_chain.mean(), H0_chain.std())

ind_H0 = np.array(ind_H0)
H0_err = np.array(H0_err)

psi1_mock = np.array(psi1_mock)
psi2_mock = np.array(psi2_mock).reshape(psi1_mock.shape)
psi3_mock = np.array(psi3_mock).reshape(psi1_mock.shape)

H0_mean = (ind_H0/H0_err**2).sum()/(1./H0_err**2).sum()
H0_meanerr = ((1./H0_err**2).sum())**-0.5

print H0_mean, H0_meanerr

xi = psi1_mock * psi3_mock - psi2_mock * (psi2_mock - 1.)
xo = psi3_mock/(1.-psi2_mock) - psi2_mock

psi2_diff = psi2_mock - 0.5*(1. - (1. + 4.*psi3_mock*psi1_mock)**0.5)

fig = pylab.figure()
pylab.subplots_adjust(left=0.15, right=0.99, bottom=0.12, top=0.99)
ax1 = fig.add_subplot(1, 1, 1)

ax1.axhspan(H0_mean - H0_meanerr, H0_mean + H0_meanerr, color='gray')
ax1.errorbar(psi2_diff, ind_H0, yerr=H0_err, fmt='+', color='b')
ax1.axhline(70., linestyle='--', color='k')
pylab.ylim(20., 120.)
pylab.xticks(fontsize=tsize)
pylab.yticks(fontsize=tsize)

yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

ax1.set_xlabel("$\\xi = \psi'' - \psi''_{\mathrm{PL}}(\psi',\psi''')$", fontsize=fsize)
ax1.set_ylabel("$H_0$", fontsize=fsize)
pylab.text(-0.1, 30., 'Composite model. Fit to image positions, $r_{\mu_r}$, and $\sigma_{e2}$', fontsize=tsize)
#pylab.savefig('gnfw_individual_H0.eps')
pylab.show()


