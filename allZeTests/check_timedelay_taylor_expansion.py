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

timedelay = 0.*source
potdiff = 0.*source
a2diff = 0.*source

for i in range(nr):
    lens.source = source[i]
    lens.get_images(xtol=1e-8)
    imA[i] = lens.images[0]
    imB[i] = lens.images[1]
    lens.get_timedelay()
    timedelay[i] = lens.timedelay
    potdiff[i] = lens.lenspot(lens.images[1]) - lens.lenspot(lens.images[0])
    a2diff[i] = lens.alpha(lens.images[1])**2 - lens.alpha(lens.images[0])**2
    alpha1[i] = lens.alpha(lens.images[0])
    alpha2[i] = lens.alpha(lens.images[1])

timedelay /= lens.Dt/cgs.c*cgs.arcsec2rad**2

dtheta1 = imA - lens.rein
dtheta2 = imB + lens.rein

dt_2order = -2.*psi1*(psi2 - 1.)*dtheta1
dt_3order = -2.*psi1*(psi2 - 1.)*dtheta1 - psi1*psi3*dtheta1**2

precis_2 = abs(dt_2order - timedelay)
precis_3 = abs(dt_3order - timedelay)

pylab.plot(dtheta1, timedelay)
pylab.plot(dtheta1, dt_2order)
pylab.plot(dtheta1, dt_3order)
pylab.show()

pylab.loglog(dtheta1, precis_2)
pylab.loglog(dtheta1, precis_3)
pylab.loglog(dtheta1, dtheta1**3)
pylab.show()

df


