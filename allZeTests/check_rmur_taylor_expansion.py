import numpy as np
import lens_models
import pylab
from sonnentools import cgsconstants as cgs


lens = lens_models.powerlaw(rein=1., gamma=1.9)
lens.get_caustic()

psi1 = lens.rein
psi2 = 2. - lens.gamma
psi3 = (2. - lens.gamma)*(1 - lens.gamma)/lens.rein

nr = 99

eps = 0.01 * lens.caustic

#source = np.linspace(eps, lens.caustic - eps, nr)
source = np.logspace(-3., -0.001, nr) * lens.caustic

imA = 0.*source
imB = 0.*source
alpha1 = 0.*source
alpha2 = 0.*source

rmur = 0.*source

for i in range(nr):
    lens.source = source[i]
    lens.get_images(xtol=1e-8)
    imA[i] = lens.images[0]
    imB[i] = lens.images[1]
    lens.get_radmag_ratio()
    rmur[i] = lens.radmag_ratio

dtheta1 = imA - lens.rein
dtheta2 = imB + lens.rein

rmur_2order = 1. + 2.*psi3/(1. - psi2) * dtheta1 
rmur_3order = rmur_2order + (psi3/(1.-psi2))**2 * dtheta1**2

precis_2 = abs(rmur_2order - rmur)
precis_3 = abs(rmur_3order - rmur)

pylab.plot(dtheta1, rmur)
pylab.plot(dtheta1, rmur_2order)
pylab.plot(dtheta1, rmur_3order)
pylab.show()

pylab.loglog(dtheta1, dtheta1**3)
pylab.loglog(dtheta1, precis_2)
pylab.loglog(dtheta1, precis_3)
pylab.show()


