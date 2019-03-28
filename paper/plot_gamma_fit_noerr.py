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

rmur_taylor_true = np.zeros(nlens)
rmur_taylor_fit = np.zeros(nlens)
rmur_true = np.zeros(nlens)

dtheta1 = np.zeros(nlens)
dtheta1_fit = np.zeros(nlens)

fit_file = h5py.File('%s_powerlaw_perfectobs_lensing.hdf5'%mockname, 'r')
gamma_fit = fit_file['gamma'].value.copy()
psicomb_fit = fit_file['psi3'].value / (1. - fit_file['psi2'].value)

asymm = np.zeros(nlens)

eps = 1e-4
for i in range(nlens):
    lens = mock['lenses'][i]
    psi1_mock[i] = lens.rein
    psi2_mock[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    psi3_mock[i] = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

    asymm[i] = (lens.images[0] + lens.images[1])/(lens.images[0] - lens.images[1])

    dtheta1[i] = lens.images[0] - lens.rein
    dtheta1_fit[i] = lens.images[0] - fit_file['psi1'][i]

    rmur_true[i] = lens.radmag_ratio

psicomb_from_rmur = 1./dtheta1 * (-1. + rmur_true**0.5)

rmur_taylor_true = 1. + 2.*psi3_mock/(1.-psi2_mock) * dtheta1 + (psi3_mock/(1. - psi2_mock))**2 * dtheta1**2

rmur_taylor_fit = 1. + 2.*psicomb_fit * dtheta1 + psicomb_fit**2 * dtheta1**2

good = (gamma_fit > 1.5) & (gamma_fit < 2.5) & (psi1_mock > 0.5)
print good.sum()

fig = pylab.figure()
pylab.subplots_adjust(left=0.15, right=0.99, bottom=0.13, top=0.99)
ax1 = fig.add_subplot(1, 1, 1)

ax1.scatter((2. - psi2_mock)[good], gamma_fit[good], color='b')
xlim = pylab.xlim()
xs = np.linspace(xlim[0], xlim[1])
pylab.plot(xs, xs, linestyle='--', color='k')
pylab.xticks(fontsize=tsize)
pylab.yticks(fontsize=tsize)

ax1.set_xlabel("$2-\psi''$", fontsize=fsize)
ax1.set_ylabel("$\gamma$", fontsize=fsize)
pylab.savefig('gamma_fit.eps')
pylab.show()

psit_mock = psi3_mock/(1.-psi2_mock)

fig = pylab.figure()
pylab.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)
pylab.scatter(psit_mock[good], psicomb_fit[good], color='b')
xlim = pylab.xlim()
xs = np.linspace(xlim[0], xlim[1])
pylab.plot(xs, xs, linestyle='--', color='k')
pylab.xlabel("$\psi'''/(1-\psi'')$", fontsize=fsize)
pylab.ylabel("$\psi'''_{\mathrm{PL}}/(1-\psi''_{\mathrm{PL}})$", fontsize=fsize)
pylab.xticks(fontsize=tsize)
pylab.yticks(fontsize=tsize)
pylab.savefig('psicomb.eps')
pylab.show()

#pylab.scatter(asymm[good], psicomb_fit[good] - psit_mock[good])
#pylab.show()

