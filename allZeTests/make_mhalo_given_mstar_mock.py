import lens_samplers
from toy_models import sample_generator
import pylab
import pickle

# generates a sample of lenses, then samples the posterior probability distribution for each object assuming flat priors on mstar and mhalo and a Gaussian prior on h70 (interim prior).

nlens = 100

mockname = 'mockB.dat'

mock = sample_generator.nfw_mhalo_given_mstar(nlens=nlens, mstar_mu=11.5, aimf_mu=-0.1, radmagrat_err=0.050)

f = open(mockname, 'w')
pickle.dump(mock, f)
f.close()

