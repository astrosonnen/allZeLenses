#to get quantities in physical units, everything should be multiplied by 4*pi*rhoc*rs**3=M200/M3d(c)

import numpy as np


def gfunc(x):
    x = np.atleast_1d(x)
    g = x*.0
    arr = x[x<1]
    g[x<1] = np.log(arr/2) + 1/np.sqrt(1 - arr**2)*np.arccosh(1/arr)
    arr = x[x==1]
    g[x==1] = 1 + np.log(0.5)
    arr = x[x>1]
    g[x>1] = np.log(arr/2) + 1/np.sqrt(arr**2-1)*np.arccos(1/arr)
    return g

def Ffunc(x):
    x = np.atleast_1d(x)
    c1 = x<1
    c2 = x==1
    c3 = x>1
    x[c1] = 1/(x[c1]**2-1)*(1 - 1/np.sqrt(1-x[c1]**2)*np.arccosh(1/x[c1]))
    x[c2] = 1/3.
    x[c3] = 1/(x[c3]**2-1)*(1 - 1/np.sqrt(x[c3]**2-1)*np.arccos(1/x[c3]))
    return x

def hfunc(x):
    x = np.atleast_1d(x)
    h = x*.0
    arr = x[x<1]
    h[x<1] = np.log(arr/2)**2 - np.arccosh(1/arr)**2
    arr = x[x>=1]
    h[x>=1] = np.log(arr/2)**2 - np.arccos(1/arr)**2
    return h


def rho(r,rs):
    return 1./(r/rs)/(1 + r/rs)**2/(4*np.pi*rs**3)

def M3d(r,rs):
    return np.log(1 + r/rs) - r/(r+rs)

def Sigma(r,rs):
    return Ffunc(r/rs)/(2*np.pi*rs**2)

def M2d(r,rs):
    return gfunc(r/rs)

def lenspot(r,rs):
    return hfunc(r/rs)/(2.*np.pi)

def avgSigma(r,rs):
    return gfunc(r/rs)/r**2/np.pi

def kappa(r,rs): #kappa, in units of kappa_s = rho_c*rs/S_cr = M200/(4*pi*rs**2*M3d(c))/S_cr
    return 2*Ffunc(r/rs)

def gamma(r,rs): #shear, in units of kappa_s = rho_c*rs/S_cr = M200/(4*pi*rs**2*M3d(c))/S_cr
    return 4.*gfunc(r/rs) - 2*Ffunc(r/rs)

def c200(M200): #mass-concentration relation from Maccio' et al. 2008
    return 10**(0.830 - 0.098*np.log10(M200/10.**12))

def cvir(Mvir):
    return 10**(0.971 - 0.094*np.log10(Mvir/10.**12))

def rs_from_Mvir(Mvir,z,Delta=93.5):
    #calculates the scale radius from the concentration and Mvir
    from cosmology import rhoc
    from sonnentools.cgsconstants import M_Sun,kpc
    cfunc = lambda M: 10**(0.971 - 0.094*log10(M/10.**12))
    cvir = cfunc(Mvir)
    rs = (Mvir*M_Sun*3/Delta/(4*pi)/cvir**3/rhoc(z))**(1/3.)
    return rs/kpc

