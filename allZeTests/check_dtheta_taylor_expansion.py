import numpy as np
import lens_models
import pylab


lens = lens_models.powerlaw(rein=1., gamma=1.9)
lens.get_caustic()

psi1 = lens.rein
psi2 = 2. - lens.gamma
psi3 = (2. - lens.gamma)*(1 - lens.gamma)/lens.rein

nr = 99

eps = 0.01 * lens.caustic

#source = np.linspace(eps, lens.caustic - eps, nr)
source = np.logspace(-4., -0.001, nr) * lens.caustic

imA = 0.*source
imB = 0.*source

for i in range(nr):
    lens.source = source[i]
    lens.get_images(xtol=1e-8)
    imA[i] = lens.images[0]
    imB[i] = lens.images[1]

dtheta1 = imA - lens.rein
dtheta2 = imB + lens.rein

dtheta2_expand = dtheta1 - psi3/(1. - psi2)*dtheta1**2

precis = dtheta2_expand - dtheta2

pylab.plot(dtheta1, dtheta2)
pylab.plot(dtheta1, dtheta2_expand)
xlim = pylab.xlim()
xs = np.linspace(xlim[0], xlim[1])
pylab.plot(xs, xs, linestyle='--', color='k')
pylab.show()

pylab.loglog(dtheta1, abs(precis))
pylab.loglog(dtheta1, dtheta1**3)
pylab.loglog(dtheta1, dtheta1**2)
pylab.show()

