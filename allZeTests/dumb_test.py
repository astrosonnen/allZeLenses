import pickle
from toy_models import hierarchical_inference
from plotters import cornerplot
import pylab
from toy_models import sample_generator
import numpy as np

nrun = 5
nlens = 20

for i in range(nrun):
    mock = sample_generator.simple_reality_sample_knownimf_nocvirscat(nlens=nlens)

    chains = []
    mserrs = np.random.normal(0., 0.1, nlens)
    mherrs = np.random.normal(0., 0.1, nlens)
    for j in range(nlens):
	chain = {}
	chain['mstar'] = np.random.normal(mock['mstar_sample'][j] + mserrs[j], 0.1, 10000)
	chain['mhalo'] = np.random.normal(mock['mhalo_sample'][j] + mherrs[j], 0.1, 10000)
	chains.append(chain)

    chain = hierarchical_inference.infer_simple_reality_knownimf_nocosmo(mock, chains, nstep=20000, burnin=10000)
    
    cp = []
    for par in chain:
        if par != 'logp':
	    cp.append({'data': chain[par], 'label': par, 'value': mock['truth'][par]})
    
    cornerplot(cp, color='c')
    pylab.show()


