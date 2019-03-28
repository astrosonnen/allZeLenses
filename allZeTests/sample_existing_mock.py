import lens_samplers
import pylab
import pickle

# generates a sample of lenses, then samples the posterior probability distribution for each object assuming flat priors on mstar and mhalo and a Gaussian prior on h70 (interim prior).

mockname = 'simple_reality_100lenses.dat'
newname = 'simple_reality_100lenses_winterimprior.dat'

f = open(mockname, 'r')
mock, oldchains = pickle.load(f)
f.close()

chains = []
i = 0
for lens in mock['lenses']:
    print 'sampling lens %d...'%i
    chain = lens_samplers.fit_nfwdev_nocvirscat_noradmagfit_nodtfit_mstarmhaloprior(lens, nstep=110000, burnin=10000, thin=10)
    chains.append(chain)
    i += 1

f = open(newname, 'w')
pickle.dump((mock, chains), f)
f.close()

