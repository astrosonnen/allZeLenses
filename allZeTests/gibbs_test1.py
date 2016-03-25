import lens_models
import lens_samplers
from toy_models import sample_generator
import gibbs_sampling_tools
import numpy as np
import pylab
import pickle
from plotters import cornerplot

# generates a sample of lenses, then tries to recover the hyperparameters with a Gibbs sampling MC, assuming the individual values of mstar, mhalo are known exactly.

mock = sample_generator.simple_reality_sample(nlens=100)

chain = gibbs_sampling_tools.hierarchical_gibbs_sampling_ubercheat(mock['lenses'], nstep=10000)

f = open('gibbs_test1_chain.dat', 'w')
pickle.dump(chain, f)
f.close()

cp = []
for par in chain:
    cp.append({'data': chain[par], 'label': par, 'value': mock['truth'][par]})

cornerplot(cp, color='g')
pylab.show()

