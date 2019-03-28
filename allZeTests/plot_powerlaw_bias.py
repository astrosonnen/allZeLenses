import numpy as np
import pylab
import lens_models
from toy_models import sample_generator
from sonnentools import cgsconstants as cgs
from scipy.interpolate import splrep, splev
from scipy.optimize import brentq


fsize = 20

ngal = 100
mock = sample_generator.simple_reality_sample(nlens=ngal, aimf_mu=-0.1)

psi1_true = np.zeros(ngal)
psi2_true = np.zeros(ngal)
psi3_true = np.zeros(ngal)

dt_true = np.zeros(ngal)
dt_3rdorder = np.zeros(ngal)
dt_powerlaw = np.zeros(ngal)

gamma_fit = np.zeros(ngal)
psi1_powerlaw = np.zeros(ngal)
psi2_powerlaw = np.zeros(ngal)
psi3_powerlaw = np.zeros(ngal)

ng = 101
gamma_grid = np.linspace(1.5, 2.5, ng)

eps = 1e-4
for i in range(ngal):
    lens = mock['lenses'][i]
    lens.get_rein()
    #lens.source = 0.1*lens.caustic
    lens.source = np.random.rand(1)**0.5 * 0.5*lens.caustic
    lens.get_images()
    lens.get_radmag_ratio()
    lens.get_timedelay()

    xA, xB = (lens.images[0], lens.images[1])

    psi1_true[i] = lens.rein
    psi2_true[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    psi3_true[i] = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

    dt_true[i] = lens.timedelay/lens.Dt*cgs.c/(1. + lens.zd) / cgs.arcsec2rad**2

    powerlaw = lens_models.powerlaw(zd=lens.zd, zs=lens.zs, rein=lens.rein)
    powerlaw.images = (xA, xB)

    # makes a grid of radial magnification ratio as a function of power-law slope
    radmagrat_grid = np.zeros(ng)
    for j in range(ng):
        powerlaw.gamma = gamma_grid[j]
        powerlaw.b = ((3.-powerlaw.gamma)*(xA - xB)/(xA**(2.-powerlaw.gamma) + abs(xB)**(2.-powerlaw.gamma)))**(1./(powerlaw.gamma-1.))
        radmagrat_grid[j] = powerlaw.mu_r(xA)/powerlaw.mu_r(xB)

    # finds gamma that produces the same radial magnification ratio as the truth
    radmagrat_spline = splrep(gamma_grid, radmagrat_grid)

    gamma_fit[i] = brentq(lambda g: splev(g, radmagrat_spline) - lens.radmag_ratio, 1.5, 2.5)
    
    powerlaw.gamma = gamma_fit[i]
    powerlaw.b = ((3.-powerlaw.gamma)*(xA - xB)/(xA**(2.-powerlaw.gamma) + abs(xB)**(2.-powerlaw.gamma)))**(1./(powerlaw.gamma-1.))

    powerlaw.get_rein_from_b()

    psi1_powerlaw[i] = float(powerlaw.rein)
    psi2_powerlaw[i] = 2. - powerlaw.gamma
    psi3_powerlaw[i] = (2. - powerlaw.gamma)*(1. - powerlaw.gamma)/powerlaw.rein

    powerlaw.source = xA - powerlaw.alpha(xA)

    powerlaw.get_images(xtol=1e-8)
    powerlaw.get_radmag_ratio()

    powerlaw.get_timedelay()

    dt_powerlaw[i] = powerlaw.timedelay/powerlaw.Dt*cgs.c/(1.+powerlaw.zd)/cgs.arcsec2rad**2

xi = psi1_true*psi3_true - psi2_true*(psi2_true - 1.)

dt_diff = (dt_powerlaw - dt_true) / dt_true

pylab.scatter(xi, dt_diff, color='r')
xlim = pylab.xlim()
ylim = pylab.ylim()
xs = np.linspace(xlim[0], xlim[1])
pylab.plot(xs, xs, linestyle='--', color='k')
pylab.xlim(xlim[0], xlim[1])
pylab.ylim(ylim[0], ylim[1])
pylab.xlabel('$\\xi$', fontsize=fsize)
pylab.ylabel('$(\Delta t^{\mathrm{PL}} - \Delta t^{\mathrm{true}})/\Delta t^{\mathrm{true}}$', fontsize=fsize)
pylab.axhline(0., linestyle='--', color='k')
pylab.savefig('powerlaw_bias_vs_xi.png')
pylab.show()

