import h5py
import pylab
import numpy as np
import pickle


mockname = 'mockA'
chaindir = '/Users/alesonny/allZeChains/'

f = open(mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

gamma_err = np.zeros(nlens)
asymm = np.zeros(nlens)

for i in range(nlens):
    chain_file = h5py.File(chaindir+'%s_lens_%02d_powerlaw.hdf5'%(mockname, i), 'r')
    gamma_err[i] = chain_file['gamma'].value.std()

    lens = mock['lenses'][i]
    asymm[i] = (lens.images[0] + lens.images[1]) / (lens.images[0] - lens.images[1])

    print i, gamma_err[i], asymm[i]
    chain_file.close()

pylab.scatter(asymm, gamma_err)
pylab.show()

