import pickle
from toy_models import hierarchical_inference

mockname = 'mock_A_20lenses.dat'

f = open(mockname, 'r')
mock, chains = pickle.load(f)
f.close()

#chain = hierarchical_inference.infer_simple_reality_nocosmo(mock, chains)
chain = hierarchical_inference.infer_simple_reality_knownimf_nocosmo(mock, chains)

outname=mockname.replace('.dat', '_inference.dat')

f = open(outname, 'w')
pickle.dump(chain, f)
f.close()

