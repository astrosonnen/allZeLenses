import numpy as np
import pylab
import lens_models


lens = lens_models.powerlaw(rein=1., gamma=1.8)

lens.get_caustic()

nr = 1001

theta = np.linspace(lens.radcrit, lens.rein, nr)

pylab.subplot(2, 1, 1)
pylab.plot(theta, lens.mu_r(theta) * lens.mu_t(theta))
pylab.ylim(-2, 0)

pylab.subplot(2, 1, 2)
pylab.plot(theta, lens.ddetA(theta))
pylab.ylim(-1, 1)

pylab.show()
