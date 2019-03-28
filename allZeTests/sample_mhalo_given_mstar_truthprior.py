import lens_samplers
from toy_models import sample_generator
import pylab
import pickle
import h5py


mockname = 'mockB'
chaindir = '/Users/sonnen/allZeChains/'

f = open(mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

for i in range(nlens):
    print i
    lens = mock['lenses'][i]
    print 'sampling lens %d...'%i
    chain = lens_samplers.fit_nfwdev_mhalo_given_mstar_truthprior(lens, mock['truth'], nstep=11000, burnin=1000, thin=1)
    
    chain_file = h5py.File(chaindir+'%s_lens_%02d_truthprior.hdf5'%(mockname, i), 'w')

    for par in chain:
        chain_file.create_dataset(par, data=chain[par])

    chain_file.close()


