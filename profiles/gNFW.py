from numpy import *
from scipy.integrate import quad
import pickle

f = open('/data3/sonnen/Python/SL2S/gNFW_rs30kpc_M2d_ndinterp.dat','r')
M2d_ndinterp = pickle.load(f)
f.close()

f = open('/data3/sonnen/Python/SL2S/gNFW_rs50kpc_M3d_ndinterp.dat','r')
M3d_ndinterp = pickle.load(f)
f.close()

def rho(r,rs,beta):
    return 1/r**beta/(1 + r/rs)**(3-beta)

def Sigma(R,rs,beta):
    Rs = atleast_1d(R)
    out = 0.*Rs
    norm = 0.5*rs**(beta-1.)
    for i in range(0,len(Rs)):
        R = Rs[i]
        out[i] = (R/rs)**(1-beta)*quad(lambda theta: sin(theta)*(sin(theta) + R/rs)**(beta-3),0.,pi/2.)[0]
    return out/norm


def M2d(R,rs,beta):
    Rs = atleast_1d(R)
    out = 0.*Rs
    for i in range(0,len(Rs)):
        R = Rs[i]
        out[i] = 2*pi*quad(lambda x: Sigma(x,rs,beta)*x,0.,R)[0]
    return out


def lenspot(R,rs,beta):
    Rs = atleast_1d(R)
    out = 0.*Rs
    for i in range(0,len(Rs)):
        R = Rs[i]
        out[i] = 2*pi*quad(lambda x: Sigma(x,rs,beta)*x*log(x),0.,R)[0]
    return out


def M3d(r,rs,beta):
    r = atleast_1d(r)
    out = 0.*r
    for i in range(0,len(r)):
        out[i] = 4*pi*quad(lambda x: rho(x,rs,beta)*x**2,0.,r[i])[0]
    return out

def fast_M2d(R,rs,beta):
    lam = rs/30.
    R = atleast_1d(R)
    beta = atleast_1d(beta)
    length = max(len(beta),len(R))
    sample = array([beta*ones(length),R/lam*ones(length)]).reshape((2,length)).T
    M2d = M2d_ndinterp.eval(sample)*lam**(3-beta)*(R/lam)**(3.-beta)
    return M2d

def fast_M3d(r,rs,beta):
    lam = rs/50.
    r = atleast_1d(r)
    beta = atleast_1d(beta)
    length = max(len(beta),len(r))
    sample = array([beta*ones(length),r/lam*ones(length)]).reshape((2,length)).T
    M3d = M3d_ndinterp.eval(sample)*lam**(3-beta)*(r/lam)**(3.-beta)
    return M3d

