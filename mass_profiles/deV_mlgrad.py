#definitions of the Sersic profile

import numpy as np
from scipy.special import gamma as gfunc
import pickle
from scipy.integrate import quad
from scipy.interpolate import splev
import os


grid_dir = os.environ.get('ATL_GRIDDIR')

f = open(grid_dir+'deV_mlgrad_M2d_grid.dat','r')
M2d_grid = pickle.load(f)
f.close()

f = open(grid_dir+'deV_mlgrad_norm_spline.dat','r')
norm_spline = pickle.load(f)
f.close()

f = open(grid_dir+'deV_mlgrad_lenspot_grid.dat','r')
pot_grid = pickle.load(f)
f.close()

ndeV = 4.

def b(n):
    return 2*n - 1./3. + 4/405./n + 46/25515/n**2

def L(n,Re):
    return Re**2*2*np.pi*n/b(n)**(2*n)*gfunc(2*n)

def I(R, Re, beta):
    return np.exp(-b(ndeV)*(R/Re)**(1./ndeV))/L(ndeV,Re) * (R/Re)**beta * splev(beta, norm_spline)

def deproject(r, Re, beta):
    deriv = lambda R: -b(ndeV)/ndeV*(R/Re)**(1/ndeV)/R*I(R, ndeV, Re, beta) + beta/Re * I(R, ndeV, Re, beta) / (R/Re)

    rho = -1/np.pi*quad(lambda R: deriv(R)/np.sqrt(R**2 - r**2), r, np.inf)[0]
    return rho

def rho(r, Re, beta):
    r = np.atleast_1d(r)
    out = 0.*r
    for i in range(0,len(r)):
        out[i] = deproject(r[i], Re, beta)
    return out

def fast_M2d(R, Re, beta):
    R = np.atleast_1d(R)
    beta = np.atleast_1d(beta)
    length = max(len(beta), len(R))
    sample = np.array([beta*np.ones(length),R/Re*np.ones(length)]).reshape((2,length)).T
    M2d = M2d_grid.eval(sample)
    return M2d

def fast_lenspot(R, Re, beta):
    R = np.atleast_1d(R)
    beta = np.atleast_1d(beta)
    length = max(len(beta), len(R))
    sample = np.array([beta*np.ones(length),R/Re*np.ones(length)]).reshape((2,length)).T
    pot = pot_grid.eval(sample)
    return pot

