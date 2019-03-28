import lens_samplers
from toy_models import sample_generator
import pylab
import pickle
import h5py


# generates a sample of lenses, then samples the posterior probability distribution for each object assuming flat priors on gamma and a Gaussian prior on h (interim prior).

nlens = 100

mockname = 'powerlaw_%dlenses_B.dat'%nlens

mock = sample_generator.powerlaw_lenses(nlens=nlens)

f = open(mockname, 'w')
pickle.dump(mock, f)
f.close()

day = 24.*3600.

chain_file = h5py.File('powerlaw_%dlenses_B_chains.hdf5'%nlens, 'w')

for i in range(nlens):
    print i
    lens = mock['lenses'][i]
    print 'sampling lens %d...'%i
    chain = lens_samplers.fit_powerlaw_noimerr(lens, nstep=110000, burnin=10000, thin=10)
    
    chain_group = chain_file.create_group('lens_%04d'%i)
    for par in chain:
        chain_group.create_dataset(par, data=chain[par], dtype='f')

