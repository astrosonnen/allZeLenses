import pickle
from toy_models import hierarchical_inference

mockname = 'mock_C_100lenses.dat'

f = open(mockname, 'r')
mock, chains = pickle.load(f)
f.close()

lenses = mock['lenses']
dt_obs = []
dt_err = []
for lens in lenses:
    dt_obs.append(lens.obs_timedelay[0])
    dt_err.append(lens.obs_timedelay[1])

#chain = hierarchical_inference.infer_simple_reality_nocosmo(mock, chains)
chain = hierarchical_inference.infer_simple_reality_knownimf(mock['truth'], chains, dt_obs, dt_err, thin=10, nstep=50000, burnin=10000)

outname=mockname.replace('.dat', '_inference.dat')

f = open(outname, 'w')
pickle.dump(chain, f)
f.close()

