import lens_samplers
from toy_models import sample_generator
import pylab
import pickle
import h5py
import os


mockname = 'mockA'
chaindir = '/Users/sonnen/allZeChains/'

cvir_prior = (0.877+0.3, 0.1)

f = open(mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

for i in range(nlens):
    print i
    lens = mock['lenses'][i]
    chainname = chaindir+'%s_lens_%02d_wrongcprior.hdf5'%(mockname, i)
    if not os.path.isfile(chainname):
        print 'sampling lens %d...'%i
        chain_file = h5py.File(chainname, 'w')
        chain = lens_samplers.fit_nfwdev_interimprior(lens, nstep=110000, burnin=10000, thin=10, cvir_prior=cvir_prior)

        for par in chain:
            chain_file.create_dataset(par, data=chain[par])

        chain_file.close()

