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

mstar_fit = np.zeros(nlens)
mstar_err = np.zeros(nlens)
mstar_true = mock['mstar_sample'][:nlens]

for i in range(nlens):

    chain = h5py.File(chaindir+'%s_lens_%02d_psifit_nfwdev_wsigma_flatprior.hdf5'%(mockname, i), 'r')

    mstar_fit[i] = chain['mstar'].value.mean()
    mstar_err[i] = chain['mstar'].value.std()

pylab.errorbar(mstar_true, mstar_fit, yerr=mstar_err, fmt='+')
xlim = pylab.xlim()
xs = np.linspace(xlim[0], xlim[1])
pylab.plot(xs, xs, linestyle='--', color='k')
pylab.show()

