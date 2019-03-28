import h5py
import pickle
import numpy as np
import pylab


mockname = 'mockF'

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

chaindir = '/Users/alesonny/allZeChains/'

nlens = len(mock['lenses'])

asymm = np.zeros(nlens)
H0 = np.zeros(nlens)
H0_err = np.zeros(nlens)

gamma = np.zeros(nlens)
gamma_err = np.zeros(nlens)

beta = np.zeros(nlens)
beta_err = np.zeros(nlens)

for i in range(nlens):
    lens = mock['lenses'][i]
    asymm[i] = (lens.images[0] + lens.images[1])/(lens.images[0] - lens.images[1])
    
    chain = h5py.File(chaindir+'%s_lens_%02d_psifit_brokenpl_flatprior.hdf5'%(mockname, i), 'r')

    H0[i] = np.median(lens.timedelay / chain['timedelay'].value)
    H0_err[i] = (lens.timedelay / chain['timedelay'].value).std()
    print H0[i], H0_err[i]

    gamma[i] = np.median(chain['gamma'].value)
    gamma_err[i] = chain['gamma'].value.std()

    beta[i] = np.median(chain['beta'].value)
    beta_err[i] = chain['beta'].value.std()

    chain.close()

H0_mean = (H0/H0_err**2).sum() / (1./H0_err**2).sum()
H0_meanerr = (1./H0_err**2).sum()**-0.5
print 70.*H0_mean, 70.*H0_meanerr

pylab.errorbar(gamma, beta, yerr=beta_err, xerr=gamma_err, fmt='+')
pylab.show()

pylab.errorbar(gamma-beta, beta, yerr=beta_err, xerr=gamma_err, fmt='+')
pylab.show()

