import numpy as np
import pylab
import lens_models
import pickle
import h5py
from matplotlib import rc, patches
rc('text', usetex=True)


mockname = 'mockP'

chaindir = '/Users/sonnen/allZeChains/'
#chaindir = '/Users/alesonny/allZeChains/'

fsize = 20
tsize = 16

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

psi1_mock = np.zeros(nlens)
psi2_mock = np.zeros(nlens)
psi3_mock = np.zeros(nlens)

psi2_inferred = np.zeros(nlens)

rein = np.zeros(nlens)

ind_H0 = np.zeros(nlens)

eps = 1e-4

for i in range(nlens):

    chain_file = h5py.File(chaindir+'%s_lens_%02d_gnfwdev_perfect_obs.hdf5'%(mockname, i), 'r')

    lens = mock['lenses'][i]
    psi1_mock[i] = lens.rein
    psi2_mock[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    psi3_mock[i] = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

    rein[i] = lens.rein

    ml = chain_file['logp'].value.argmax()

    psi2_inferred[i] = chain_file['psi2'][ml]

    ind_H0[i] = 70. * chain_file['timedelay'][ml] / lens.timedelay

good = rein > 0.5

mean_H0 = ind_H0[good].mean()
std_H0 = ind_H0[good].std()

print mean_H0, std_H0

xi = psi1_mock * psi3_mock - psi2_mock * (psi2_mock - 1.)
xo = psi3_mock/(1.-psi2_mock) - psi2_mock

psi2_diff = psi2_mock - 0.5*(1. - (1. + 4.*psi3_mock*psi1_mock)**0.5)
print psi2_diff[good].min()

fig = pylab.figure()
pylab.subplots_adjust(left=0.12, right=0.99, bottom=0.14, top=0.99)
ax1 = fig.add_subplot(1, 1, 1)

ax1.axhspan(mean_H0 - std_H0, mean_H0 + std_H0, color='gray', alpha=0.5)
ax1.scatter((psi2_inferred - psi2_mock)[good], ind_H0[good], color='b')
ax1.axhline(70., linestyle='--', color='k')

rect = patches.Rectangle((-0.02, 68.), 0.04, 4., edgecolor='k', linewidth=1, facecolor='none')

ax1.add_patch(rect)

pylab.xlim(-0.2, 0.15)
pylab.ylim(50., 90.)
pylab.xticks(fontsize=tsize)
pylab.yticks(fontsize=tsize)

yticks = ax1.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
yticks[-1].label1.set_visible(False)

xticks = ax1.xaxis.get_major_ticks()
xticks[-1].label1.set_visible(False)
xticks[0].label1.set_visible(False)
#xticks[1].label1.set_visible(False)

ax1.set_xlabel("$\psi''^{\mathrm{(fit)}} - \psi''$", fontsize=fsize)
ax1.set_ylabel("$H_0$", fontsize=fsize)
#pylab.text(-0.1, 30., 'Composite model. Fit to image positions, $r_{\mu_r}$, and $\sigma_{e2}$', fontsize=tsize)
pylab.text(-0.15, 59., 'truth: de Vauc. + NFW', fontsize=fsize)
pylab.text(-0.15, 56., 'model: de Vauc. + gNFW (fixed $r_s$)', fontsize=fsize)
pylab.text(-0.15, 53., 'constraints: image positions, $r_\mu$, $\sigma_{e2}$', fontsize=fsize)

pylab.axes([0.15, 0.65, 0.4, 0.3])
pylab.axhspan(mean_H0 - std_H0, mean_H0 + std_H0, color='gray', alpha=0.5)
pylab.scatter((psi2_inferred - psi2_mock)[good], ind_H0[good], color='b')
pylab.axhline(70., linestyle='--', color='k')
pylab.xlim(-0.02, 0.02)
pylab.ylim(68., 72.)
pylab.xticks(())
pylab.yticks(())

pylab.savefig('gnfw_individual_H0.pdf')
#pylab.show()

