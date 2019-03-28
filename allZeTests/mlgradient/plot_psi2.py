import pylab
import pickle
import numpy as np
import h5py


mockname = 'mockL'

f = open('../%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])
nlens = 80

chaindir = '/Users/alesonny/allZeChains/'

psi2_fit = np.zeros(nlens)
psi2_err = np.zeros(nlens)
psi2_true = np.zeros(nlens)

eps = 1e-4
for i in range(nlens):

    chain = h5py.File(chaindir+'%s_lens_%02d_psifit_nfwdev_wsigma_flatprior.hdf5'%(mockname, i), 'r')

    psi2_fit[i] = chain['psi2'].value.mean()
    psi2_err[i] = chain['psi2'].value.std()

    lens = mock['lenses'][i]

    psi2_true[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)

pylab.errorbar(psi2_true, psi2_fit, yerr=psi2_err, fmt='+')
xlim = pylab.xlim()
xs = np.linspace(xlim[0], xlim[1])
pylab.plot(xs, xs, linestyle='--', color='k')
pylab.show()

