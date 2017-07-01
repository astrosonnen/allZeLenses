import lens_samplers
from toy_models import sample_generator
import pylab
import pickle

# generates a sample of lenses, then samples the posterior probability distribution for each object assuming flat priors on mstar and mhalo and a Gaussian prior on h70 (interim prior).

nlens = 100

mockname = 'nfw_%dlenses_rmrerr0015_normr.dat'%nlens

mock = sample_generator.simple_reality_sample(nlens=nlens, mstar_mu=11.4, aimf_mu=-0.05, radmagrat_err=0.015)

chains = []
i = 0
for lens in mock['lenses']:
    print 'sampling lens %d...'%i
    chain = lens_samplers.fit_nfwdev_interimprior(lens, nstep=110000, burnin=10000, thin=10)
    chains.append(chain)
    i += 1

f = open(mockname, 'w')
pickle.dump((mock, chains), f)
f.close()

