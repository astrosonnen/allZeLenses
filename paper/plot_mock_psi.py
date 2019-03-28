import numpy as np
import pylab
import lens_models
import pickle
from matplotlib import rc
rc('text', usetex=True)


mockname = 'mockP'

fsize = 16

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
pylab.subplots_adjust(left=0.15, right=0.99, bottom=0.12, top=0.9)
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twiny()

ax1.plot(psi2, psi1*psi3, label='Power-law')
ax1.tick_params(labelsize=fsize)
xlim = ax1.set_xlim(psi2[-1], psi2[0])

ax1.scatter(psi2_mock, psi1_mock*psi3_mock, color='r', label='Mock')
ax1.legend(loc = 'upper right', scatterpoints=1, fontsize=fsize)
ax1.set_xlabel("$\psi''$", fontsize=fsize)
ax1.set_ylabel("$\psi'\psi'''$", fontsize=fsize)
ax2.set_xlim(gamma[0], gamma[-1])
ax2.set_xlabel('$\gamma$', fontsize=fsize)
ax2.tick_params(labelsize=fsize)
pylab.savefig('psi_plot.eps')
pylab.show()


