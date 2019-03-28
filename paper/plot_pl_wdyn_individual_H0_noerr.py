import numpy as np
import pylab
import lens_models
import pickle
import h5py
from matplotlib import rc
rc('text', usetex=True)


mockname = 'mockP'

fsize = 20
tsize = 16

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

psi1_mock = np.zeros(nlens)
psi2_mock = np.zeros(nlens)
psi3_mock = np.zeros(nlens)

ind_H0 = np.zeros(nlens)
old_H0 = np.zeros(nlens)

fit_file = h5py.File('%s_powerlaw_perfectobs_wdyn.hdf5'%mockname, 'r')

old_fit_file = h5py.File('%s_powerlaw_perfectobs_lensing.hdf5'%mockname, 'r')

eps = 1e-4

for i in range(nlens):
    print i
    lens = mock['lenses'][i]
    psi1_mock[i] = lens.rein
    psi2_mock[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    psi3_mock[i] = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

    ind_H0[i] = 70. * fit_file['timedelay'][i] / lens.timedelay 
    old_H0[i] = 70. * old_fit_file['timedelay'][i] / lens.timedelay 

xi = psi1_mock * psi3_mock - psi2_mock * (psi2_mock - 1.)
xo = psi3_mock/(1.-psi2_mock) - psi2_mock

psi2_diff = psi2_mock - 0.5*(1. - (1. + 4.*psi3_mock*psi1_mock)**0.5)

psi2_inferred = fit_file['psi2'].value.copy()

gamma_fit = fit_file['gamma'].value.copy()

good = (gamma_fit > 1.5) & (gamma_fit < 2.5) & (psi1_mock > 0.5)
print psi2_diff[good].min()

mean_H0 = ind_H0[good].mean()
std_H0 = ind_H0[good].std()

print mean_H0, std_H0

fig = pylab.figure()
pylab.subplots_adjust(left=0.12, right=0.99, bottom=0.14, top=0.99)
ax1 = fig.add_subplot(1, 1, 1)

ax1.axhspan(mean_H0 - std_H0, mean_H0 + std_H0, color='gray', alpha=0.5)
#ax1.scatter(psi2_diff[good], ind_H0[good], color='b')
ax1.scatter(psi2_inferred[good] - psi2_mock[good], ind_H0[good], color='b')
ax1.axhline(70., linestyle='--', color='k')
pylab.ylim(50., 90.)
pylab.xlim(-0.2, 0.15)
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
#pylab.title('Power-law model. Fit to image positions and $r_{\mu_r}$', fontsize=tsize)
pylab.text(-0.15, 59., 'truth: de Vauc. + NFW', fontsize=fsize)
pylab.text(-0.15, 56., 'model: power-law', fontsize=fsize)
pylab.text(-0.15, 53., 'constraints: image positions, $\sigma_{e2}$', fontsize=fsize)

pylab.savefig('pl_impos_dyn_individual_H0.pdf')
#pylab.show()



