import lens_samplers
from toy_models import sample_generator
import pylab
import pickle

# generates a sample of lenses, then samples the posterior probability distribution for each object assuming flat priors on mstar and mhalo and a Gaussian prior on h70 (interim prior).

nlens = 100

mockname = 'mock_D_%dlenses.dat'%nlens

mock = sample_generator.simple_reality_sample_nocvirscat(nlens=nlens)

chains = []
i = 0
for lens in mock['lenses']:
    print 'sampling lens %d...'%i
    chain = lens_samplers.fit_nfwdev_nocvirscat_nodtfit(lens, nstep=15000, burnin=5000)
    chains.append(chain)
    i += 1

f = open(mockname, 'w')
pickle.dump((mock, chains), f)
f.close()

