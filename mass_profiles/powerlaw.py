import numpy as np
from scipy.special import gamma as gfunc
#to get quantities in physical units, multiply by rho0

def rho(r,gamma):
    return 1./r**gamma

def Sigma(R,gamma):
    return R**(1-gamma)*np.sqrt(np.pi)*gfunc((gamma-1.)/2.)/gfunc(gamma/2.)
  
def M2d(R,gamma):
    return 2*np.pi*np.sqrt(np.pi)/(3.-gamma)*gfunc((gamma-1.)/2.)/gfunc(gamma/2.)*R**(3-gamma)

def M3d(r,gamma):
    return 4*pi/(3.-gamma)*r**(3-gamma)

def lenspot(r,gamma):
    #m = 2*np.sqrt(pi)/(3.-gamma)*gfunc((gamma-1.)/2.)/gfunc(gamma/2.)*R**(3-gamma)
    #alpha = 2*np.sqrt(pi)/(3.-gamma)*gfunc((gamma-1.)/2.)/gfunc(gamma/2.)*R**(2-gamma)
    return 2*np.sqrt(np.pi)/(3.-gamma)**2*gfunc((gamma-1.)/2.)/gfunc(gamma/2.)*r**(3-gamma)


def getrho0(Rein,Menc,gamma):
    return Menc/M2d(Rein,gamma)

