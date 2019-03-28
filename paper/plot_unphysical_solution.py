import pylab
import numpy as np


gamma = np.linspace(1., 3.)

psiii = 2. - gamma
psiiii = (2.-gamma)*(1.-gamma)

psit = psiiii/(1.-psiii)

da = 0.1

term = 1. + 2.*psit*da - psit**2*da**2

unphys = 1./psit * (-1. - term**0.5)

pylab.plot(gamma, term)
pylab.show()

pylab.plot(psit, unphys)
pylab.show()

