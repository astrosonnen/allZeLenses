import lens_samplers
from toy_models import sample_generator
import pylab
import pickle

# generates a sample of lenses, then samples the posterior probability distribution for each object assuming flat priors on mstar and mhalo and a Gaussian prior on h70 (interim prior).

nlens = 100

mockname = 'mockC.dat'

mock = sample_generator.simple_reality_sample(nlens=nlens, mstar_mu=11.4, aimf_mu=-0.05, radmagrat_err=0.020, max_fcaust=0.5)

f = open(mockname, 'w')
pickle.dump(mock, f)
f.close()

