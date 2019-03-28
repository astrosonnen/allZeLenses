import numpy as np
import pylab
import lens_models
from toy_models import sample_generator
import pickle


mockname = 'mockI'

fsize = 24

ng = 101
gamma = np.linspace(1.5, 2.5, ng)

psi1 = 1.
psi2 = 2. - gamma
psi3 = (2. - gamma)*(1. - gamma)/psi1

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

psi1_mock = np.zeros(nlens)
psi2_mock = np.zeros(nlens)
psi3_mock = np.zeros(nlens)

eps = 1e-4
for i in range(nlens):
    lens = mock['lenses'][i]
    psi1_mock[i] = lens.rein
    psi2_mock[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    psi3_mock[i] = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

fig = pylab.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twiny()

ax1.plot(psi2, psi1*psi3, label='Power-law')
xlim = ax1.set_xlim(psi2[-1], psi2[0])

ax1.scatter(psi2_mock, psi1_mock*psi3_mock, color='r', label='deV + NFW')
ax1.legend(loc = 'upper right', scatterpoints=1, fontsize=fsize)
ax1.set_xlabel("$\psi''$", fontsize=fsize)
ax1.set_ylabel("$\psi'\psi'''$", fontsize=fsize)
ax2.set_xlim(gamma[0], gamma[-1])
ax2.set_xlabel('$\gamma$', fontsize=fsize)
#pylab.savefig('psi3_composite_vs_powerlaw.png')
pylab.show()

# now plots psi2, psi3 as a function of stuff

rein_sample = np.zeros(nlens)
reff_ang = np.zeros(nlens)
for i in range(nlens):
    rein_sample[i] = mock['lenses'][i].rein
    reff_ang[i] = mock['lenses'][i].reff

pylab.subplot(2, 2, 1)
pylab.scatter(mock['mstar_sample'] - 2.*np.log10(mock['reff_sample']), psi2_mock)
pylab.ylabel("$\psi''$")

pylab.subplot(2, 2, 2)
pylab.scatter(rein_sample/reff_ang, psi2_mock)
pylab.ylabel("$\psi''$")

pylab.subplot(2, 2, 3)
pylab.scatter(mock['mstar_sample'] - 2.*np.log10(mock['reff_sample']), psi1_mock*psi3_mock)
pylab.ylabel("$\psi'\psi'''$")

pylab.subplot(2, 2, 4)
pylab.scatter(rein_sample/reff_ang, psi1_mock*psi3_mock)
pylab.ylabel("$\psi'\psi'''$")
pylab.show()


