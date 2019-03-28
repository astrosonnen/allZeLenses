from toy_models import sample_generator
import pickle
import numpy as np


# generates large mock, but keeps only lenses with image configuration not too asymmetric

max_asymm = 0.5
nstart = 1000
nlens = 100

mockname = 'mockE.dat'

mock = sample_generator.cored_powerlaw_lenses(nlens=nstart)

keep = np.zeros(nstart, dtype=bool)

new_lenses = []

for i in range(nstart):
    asymm = (mock['lenses'][i].images[0] + mock['lenses'][i].images[1])/(mock['lenses'][i].images[0] - mock['lenses'][i].images[1])
    if asymm < max_asymm:
        keep[i] = True
        new_lenses.append(mock['lenses'][i])

new_mock = {}
new_mock['truth'] = mock['truth']

keys = ['rein_sample', 'gamma_sample', 'zd_sample', 'zs_sample']

for key in keys:
    new_mock[key] = mock[key][keep][:nlens]

new_mock['lenses'] = new_lenses[:nlens]

f = open(mockname, 'w')
pickle.dump(new_mock, f)
f.close()


