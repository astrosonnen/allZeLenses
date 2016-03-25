import pickle
from toy_models import hierarchical_inference
from plotters import cornerplot, probcontour
import pylab
from toy_models import sample_generator
import numpy as np

nrun = 5
nlens = 100

for i in range(nrun):
    mock = sample_generator.simple_reality_sample_knownimf_nocvirscat(nlens=nlens)

    truth = mock['truth']

    mserrs = np.random.normal(0., 0.1, nlens)
    mherrs = np.random.normal(0., 0.1, nlens)

    chain = hierarchical_inference.infer_simple_reality_knownimf_nocosmo_analytic(mock, mock['mhalo_sample'] + mherrs, 0.1*np.ones(nlens), mock['mstar_sample'] + mserrs, 0.1*np.ones(nlens), nstep=20000, burnin=10000)
    
    cp = []
    for par in chain:
        if par != 'logp':
	    cp.append({'data': chain[par], 'label': par, 'value': mock['truth'][par]})
    
    cornerplot(cp, color='c')
    pylab.show()


