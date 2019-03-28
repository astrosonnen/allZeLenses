import numpy as np
import ndinterp
from scipy.interpolate import splrep
import lens_models
import pickle
from scipy.optimize import minimize
import pylab


ng = 1601
gamma_min = 1.2
gamma_max = 2.8

gamma_grid = np.linspace(gamma_min, gamma_max, ng)

nb = 2001
beta_min = -1.
beta_max = 1.
beta_grid = np.linspace(beta_min, beta_max, nb)

gamma_spline = splrep(gamma_grid, np.arange(ng))
beta_spline = splrep(beta_grid, np.arange(nb))

axes = {0: gamma_spline, 1: beta_spline}

psi2 = np.zeros((ng, nb))

psi3 = np.zeros((ng, nb))

lens = lens_models.broken_alpha_powerlaw(rein=1.)

for i in range(ng):
    lens.gamma = gamma_grid[i]
    for j in range(nb):
        lens.beta = beta_grid[j]
        psi2[i, j] = lens.psi2()
        psi3[i, j] = lens.psi3()

print psi2.min(), psi2.max()
print psi3.min(), psi3.max()

psi2_ndinterp = ndinterp.ndInterp(axes, psi2, order=1)
psi3_ndinterp = ndinterp.ndInterp(axes, psi3, order=1)

start = np.array((2., 0.))
bounds = np.array(((gamma_min, gamma_max), (beta_min, beta_max)))
scale_free_bounds = 0.*bounds
scale_free_bounds[:, 1] = 1.

scale_free_guess = (start - bounds[:, 0])/(bounds[:, 1] - bounds[:, 0])

eps = 1e-5

minimizer_kwargs = dict(method="L-BFGS-B", bounds=scale_free_bounds, tol=eps)

gammas_grid = np.linspace(1.4, 2.6, 11)
betas_grid = np.linspace(-0.8, 0.8, 101)

psi1 = 1.

def min_func(x, psi2, psi3):
    p = x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
            
    return (psi2_ndinterp.eval(p.reshape((1, 2))) - psi2)**2 + (psi3_ndinterp.eval(p.reshape((1, 2)))/psi1 - psi3)**2

def get_gammabeta_from_psi23(psi2, psi3):
    res = minimize(min_func, scale_free_guess, args=(psi2, psi3), bounds=scale_free_bounds)

    gamma, beta = res.x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    return (gamma, beta)

nconst_psi2 = 11
const_psi2_grid = np.linspace(-0.5, 0.5, nconst_psi2)
npsi3 = 81
psi3_grid = np.linspace(-0.4, 0.4, npsi3)

for i in range(nconst_psi2):

    gamma_grid = 0. * psi3_grid
    beta_grid = 0. * psi3_grid

    dont_plot = np.zeros(npsi3, dtype=bool)

    for j in range(npsi3):
        g, b = get_gammabeta_from_psi23(const_psi2_grid[i], psi3_grid[j])

        scale_free_point = (np.array((g, b)) - bounds[:, 0])/(bounds[:, 1] - bounds[:, 0])

        if min_func(scale_free_point, const_psi2_grid[i], psi3_grid[j]) > eps:
            dont_plot[j] = True
       
        gamma_grid[j] = g
        beta_grid[j] = b

    do_plot = np.logical_not(dont_plot)

    pylab.plot(gamma_grid[do_plot], beta_grid[do_plot], color='gray')

nconst_psi3 = 11
const_psi3_grid = np.linspace(-0.4, 0.4, nconst_psi3)
npsi2 = 81
psi2_grid = np.linspace(-0.5, 0.5, npsi2)

for i in range(nconst_psi3):

    gamma_grid = 0. * psi2_grid
    beta_grid = 0. * psi2_grid

    dont_plot = np.zeros(npsi2, dtype=bool)

    for j in range(npsi2):
        g, b = get_gammabeta_from_psi23(psi2_grid[j], const_psi3_grid[i])

        scale_free_point = (np.array((g, b)) - bounds[:, 0])/(bounds[:, 1] - bounds[:, 0])

        if min_func(scale_free_point, psi2_grid[j], const_psi3_grid[i]) > eps:
            dont_plot[j] = True
       
        gamma_grid[j] = g
        beta_grid[j] = b

    do_plot = np.logical_not(dont_plot)

    pylab.plot(gamma_grid[do_plot], beta_grid[do_plot], color='gray')

pylab.show()


