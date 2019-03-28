import numpy as np
import pylab
import pickle


mockname = 'mockI'

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

ng = 101
gamma = np.linspace(1.5, 2.5, ng)

psi1 = 1.
psi2 = 2. - gamma
psi3 = (2. - gamma)*(1. - gamma)/psi1

ngal = len(mock['lenses'])

psi1_mock = np.zeros(ngal)
psi2_mock = np.zeros(ngal)
psi3_mock = np.zeros(ngal)

eps = 1e-4
for i in range(ngal):
    lens = mock['lenses'][i]
    psi1_mock[i] = lens.rein
    psi2_mock[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    psi3_mock[i] = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

xi_mock = psi1_mock * psi3_mock - psi2_mock * (psi2_mock - 1.)

for i in range(ngal):
    print i, xi_mock[i]

pylab.plot(psi2, psi1 * psi3)
pylab.scatter(psi2_mock, psi1_mock * psi3_mock, color='r')
pylab.show()

