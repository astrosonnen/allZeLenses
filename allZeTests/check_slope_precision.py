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
precis = np.zeros(nlens)
dt_precis = np.zeros(nlens)

for i in range(nlens):
    lens = mock['lenses'][i]
    asymm[i] = (lens.images[0] + lens.images[1])/(lens.images[0] - lens.images[1])
    
    chain = h5py.File(chaindir+'%s_lens_%02d_powerlaw.hdf5'%(mockname, i), 'r')

    precis[i] = chain['gamma'].value.std()

    dt_precis[i] = chain['timedelay'].value.std() / chain['timedelay'].value.mean()

pylab.scatter(asymm, precis)
pylab.show()

pylab.scatter(asymm, dt_precis)
pylab.show()

