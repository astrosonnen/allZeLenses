import h5py
import pickle
import numpy as np
import pylab


mockname = 'mockI'

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

chaindir = '/Users/sonnen/allZeChains/'

nlens = len(mock['lenses'])

asymm = np.zeros(nlens)
H0 = np.zeros(nlens)
H0_err = np.zeros(nlens)

pl_H0 = np.zeros(nlens)
pl_err = np.zeros(nlens)

for i in range(nlens):
    lens = mock['lenses'][i]
    asymm[i] = (lens.images[0] + lens.images[1])/(lens.images[0] - lens.images[1])
    
    chain = h5py.File(chaindir+'%s_lens_%02d_brokenpowerlaw_wH0par.hdf5'%(mockname, i), 'r')

    H0[i] = np.median(chain['H0'].value)
    H0_err[i] = chain['H0'].value.std()

    chain.close()

    pl_chain = h5py.File(chaindir+'%s_lens_%02d_powerlaw_wH0par.hdf5'%(mockname, i), 'r')

    pl_H0[i] = np.median(pl_chain['H0'].value)
    pl_err[i] = pl_chain['H0'].value.std()

    pl_chain.close()

H0_mean = (H0/H0_err**2).sum() / (1./H0_err**2).sum()
H0_meanerr = (1./H0_err**2).sum()**-0.5
print H0_mean, H0_meanerr

pl_H0_mean = (pl_H0/pl_err**2).sum() / (1./pl_err**2).sum()
pl_H0_meanerr = (1./pl_err**2).sum()**-0.5
print pl_H0_mean, pl_H0_meanerr

pylab.errorbar(asymm, H0, yerr=H0_err, fmt='+')
pylab.errorbar(asymm + 0.001, pl_H0, yerr=pl_err, fmt='+', color='r')
pylab.axhline(70., linestyle='--', color='k')
pylab.show()


