import numpy as np
from lensingtools import sersic
from scipy.optimize import brentq
from scipy.integrate import dblquad


# How do we integrate over the individual lens parameters when the likelihood is a delta function? What is the correct normalization?
# My calculations suggest that it's the Jacobian matrix of the coordinate transformation between model parameters and observables, evaluated at the observed data points. Let's verify this numerically with Gaussian integrals with arbitrarily small widths.

# example: de Vaucouleurs lens with known density profile and no dark matter, two image positions known exactly.

# observed image positions
xA_obs = 1.2
xB_obs = -0.8

reff = 1.
nser = 4.

def mfunc(x):
    return sersic.M2d(x, nser, reff)

# first let's find the solution
mstar = (xA_obs - xB_obs)/(mfunc(xA_obs) + mfunc(-xB_obs))
lmstar = np.log10(mstar)
beta = 0.5*(xA_obs + xB_obs - (xA_obs - xB_obs)*(mfunc(xA_obs) - mfunc(-xB_obs))/(mfunc(xA_obs) + mfunc(-xB_obs)))

dlmstar = 1e-4
dbeta = 1e-4

lmstar_up = lmstar + dlmstar
lmstar_dw = lmstar - dlmstar

xA_lmstar_up = brentq(lambda x: x - 10.**lmstar_up * mfunc(x) - beta, xA_obs, 1.1*xA_obs)
xA_lmstar_dw = brentq(lambda x: x - 10.**lmstar_dw * mfunc(x) - beta, 0.9*xA_obs, xA_obs)

dxA_dlmstar = (xA_lmstar_up - xA_lmstar_dw)/(2.*dlmstar)

print dxA_dlmstar

xB_lmstar_up = brentq(lambda x: x + 10.**lmstar_up * mfunc(-x) - beta, 1.1*xB_obs, xB_obs)
xB_lmstar_dw = brentq(lambda x: x + 10.**lmstar_dw * mfunc(-x) - beta, xB_obs, 0.9*xB_obs)

dxB_dlmstar = (xB_lmstar_up - xB_lmstar_dw)/(2.*dlmstar)

print dxB_dlmstar

beta_up = beta + dbeta
beta_dw = beta - dbeta

xA_beta_up = brentq(lambda x: x - 10.**lmstar * mfunc(x) - beta_up, xA_obs, 1.1*xA_obs)
xA_beta_dw = brentq(lambda x: x - 10.**lmstar * mfunc(x) - beta_dw, 0.9*xA_obs, xA_obs)

dxA_dbeta = (xA_beta_up - xA_beta_dw)/(2.*dbeta)

print dxA_dbeta

xB_beta_up = brentq(lambda x: x + 10.**lmstar * mfunc(-x) - beta_up, xB_obs, 0.9*xB_obs)
xB_beta_dw = brentq(lambda x: x + 10.**lmstar * mfunc(-x) - beta_dw, 1.1*xB_obs, xB_obs)

dxB_dbeta = (xB_beta_up - xB_beta_dw)/(2.*dbeta)

print dxB_dbeta

J = np.array(((dxA_dlmstar, dxA_dbeta), (dxB_dlmstar, dxB_dbeta)))

detJ = np.linalg.det(J)

# now with the Gaussian

xA_err = 0.01
xB_err = 0.01

def zerofuncA(x, lmstar, beta):
    return x - 10.**lmstar * mfunc(x) - beta

xA_min = xA_obs + 0.1
xA_max = xA_obs - 0.1

def xA_func(lmstar, beta):
    if zerofuncA(xA_min, lmstar, beta) * zerofuncA(xA_max, lmstar, beta) > 0.:
        return -np.inf
    else:
        return brentq(lambda x: zerofuncA(x, lmstar, beta), xA_min, xA_max) 

def zerofuncB(x, lmstar, beta):
    return x + 10.**lmstar * mfunc(-x) - beta

xB_min = xB_obs - 0.1
xB_max = xB_obs + 0.1

def xB_func(lmstar, beta):
    if zerofuncB(xB_min, lmstar, beta) * zerofuncB(xB_max, lmstar, beta) > 0.:
        return +np.inf
    else:
        return brentq(lambda x: zerofuncB(x, lmstar, beta), xB_min, xB_max) 

lmstar_min = lmstar - 0.1
lmstar_max = lmstar + 0.1

beta_min = beta - 0.1
beta_max = beta + 0.1

nsamp = 10000
lmstar_samp = np.random.rand(nsamp) * (lmstar_max - lmstar_min) + lmstar_min
beta_samp = np.random.rand(nsamp) * (beta_max - beta_min) + beta_min

xA_samp = np.zeros(nsamp)
xB_samp = np.zeros(nsamp)

for i in range(nsamp):
    xA_samp[i] = xA_func(lmstar_samp[i], beta_samp[i])
    xB_samp[i] = xB_func(lmstar_samp[i], beta_samp[i])

likelihood_samp = 1./(2.*np.pi)/xA_err/xB_err * np.exp(-0.5*(xA_obs - xA_samp)**2/xA_err**2) * np.exp(-0.5*(xB_obs - xB_samp)**2/xB_err**2)

print likelihood_samp.sum()/float(nsamp)*(lmstar_max - lmstar_min)*(beta_max - beta_min)

