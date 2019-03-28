import numpy as np
import pylab
import lens_models


# test 1: I see a double lens. I can measure image positions exactly. I can constrain a combination of Einstein radius and slope. Is this curve consistent with my analytical estimate based on Taylor expansion of the lens equation to 2nd order in deltatheta?

lens = lens_models.powerlaw(zd=0.3, zs=1., rein=1., gamma=2.)

ng = 101
gamma_grid = np.linspace(1.5, 2.5, ng)

xA = 1.01
lens.source = xA - lens.alpha(xA)
lens.get_images(xtol=1e-8)
xB = lens.images[1]

print lens.images

psi2_grid = 2. - gamma_grid

psi1_grid = 0. * gamma_grid

mur_ratio_grid = 0. * gamma_grid

for i in range(ng):
    psi1_grid[i] = ((xA - xB)/(xA**(2.-gamma_grid[i]) + abs(xB)**(2.-gamma_grid[i])))**(1./(gamma_grid[i] - 1.))
    lens.gamma = gamma_grid[i]
    lens.rein = psi1_grid[i]
    lens.get_b_from_rein()
    lens.source = xA - lens.alpha(xA)
    lens.get_images(xtol=1e-8)
    mur_ratio_grid[i] = lens.mu_r(xA)/lens.mu_r(xB)

psi3_grid = psi2_grid * (psi2_grid - 1.) / psi1_grid
psi3_grid = (2. - gamma_grid)*(1. - gamma_grid)/psi1_grid
a_grid = psi3_grid/(1. - psi2_grid)
#a_grid = -psi2_grid/psi1_grid

psi1_inferred_taylor = 1./(a_grid) * (-(1. - a_grid*xA) + ((1 - a_grid*xA)**2 - a_grid * (a_grid * xA**2 - xA + xB))**0.5)

dtheta1 = xA - psi1_grid
dtheta2 = xB + psi1_grid

#pylab.plot(xB + psi1_grid, xA - psi1_grid - a_grid*(xA - psi1_grid)**2)
#pylab.plot(xB + 2.*psi1_grid, xA - a_grid*(xA - psi1_grid)**2)
#pylab.plot(xB + 2.*psi1_grid, xA - a_grid*xA**2 - a_grid*psi1_grid**2 + 2.*a_grid*xA*psi1_grid)
pylab.plot(psi1_grid, psi1_inferred_taylor)
pylab.show()

# now let's verify the Taylor expansion of the magnification ratio
pylab.plot(mur_ratio_grid, 1. + 2.*a_grid*dtheta1 + a_grid**2 * dtheta1**2)
pylab.show()


