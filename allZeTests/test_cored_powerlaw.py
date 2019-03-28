import numpy as np
import pylab
from scipy.integrate import quad
import lens_models


# TEST 1: is m(theta) the integral of 2*kappa*theta?

gamma = 1.8

lens = lens_models.cored_powerlaw(gamma=gamma, rc=0.1)

ngrid = 101
theta = np.logspace(-2., 0., ngrid)

mint = np.zeros(ngrid)

for i in range(ngrid):
    mint[i] = 2.*quad(lambda x: lens.kappa(x)*x, 0., theta[i])[0]

pylab.loglog(theta, lens.m(theta))
pylab.loglog(theta, mint, linestyle='--')
pylab.show()

# TEST 2: is alpha the gradient of the potential?

ngrid = 1001
theta = np.linspace(0.1, 1., ngrid)

dt = theta[1] - theta[0]

gradpot = (lens.lenspot(theta[2:]) - lens.lenspot(theta[:-2]))/(2.*dt)

pylab.plot(theta, lens.alpha(theta))
pylab.plot(theta[1:-1], gradpot, linestyle='--')
pylab.show()

