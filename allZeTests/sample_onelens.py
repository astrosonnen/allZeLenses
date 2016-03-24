import lens_samplers
from toy_models import sample_generator
import pylab
import pickle
import corner

# generates a sample of lenses, then tries to recover the hyperparameters with a Gibbs sampling MC, assuming the individual values of mstar, mhalo are known exactly.

mock = sample_generator.simple_reality_sample(nlens=1)
lens = mock['lenses'][0]

chain = lens_samplers.fit_nfwdev_h70prior(lens, h70p_mu=1., h70p_sig=0.05)

f = open('sample_onelens_chain.dat', 'w')
pickle.dump((lens, chain), f)
f.close()

