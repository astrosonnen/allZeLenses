import numpy as np
import pylab
import lens_models
from scipy.integrate import quad


lens = lens_models.broken_alpha_powerlaw(rein=1.4, gamma=2.1, beta=-0.2)

nx = 201
x = np.linspace(0.1, 2.1, nx)

# is the derivative of the potential equal to the deflection angle?

eps = 1e-4

psi_deriv = (lens.lenspot(x+eps) - lens.lenspot(x-eps))/(2.*eps)

pylab.plot(x, lens.alpha(x))
pylab.plot(x, psi_deriv, linestyle='--')
pylab.show()

# is m the integral of kappa within the circle?

m_num = 0.*x

for i in range(nx):
    m_num[i] = 2. * quad(lambda x: x * lens.kappa(x), 0., x[i])[0]

pylab.plot(x, lens.m(x))
pylab.plot(x, m_num, linestyle='--')
pylab.show()

