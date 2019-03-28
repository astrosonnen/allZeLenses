import pickle
from toy_models import hierarchical_inference

mockname = 'simple_reality_100lenses.dat'

f = open(mockname, 'r')
mock, chains = pickle.load(f)
f.close()

lenses = mock['lenses']
dt_obs = []
dt_err = []
for lens in lenses:
    dt_obs.append(lens.obs_timedelay[0])
    dt_err.append(lens.obs_timedelay[1])

newchains = []
for chain in chains:
    for par in chain:
	chain[par] = chain[par][:1000]
    newchains.append(chain)

chain = hierarchical_inference.infer_simple_reality(mock['truth'], chains, dt_obs, dt_err, thin=1, nstep=50000, burnin=10000)

outname=mockname.replace('.dat', '_Nind1e4_inference.dat')

f = open(outname, 'w')
pickle.dump(chain, f)
f.close()

