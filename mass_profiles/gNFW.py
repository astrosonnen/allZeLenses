import numpy as np
from scipy.integrate import quad
import pickle
import os
from scipy.interpolate import splrep

#calculates density profiles, projected mass densities, projected enclosed masses, 3d enclosed masses for generalized-NFW profiles.

rsgrid = 50.

grid_dir = '/setri6/sonnen/allZeLenses/'


def rho(r,rs,beta):
    return 1./r**beta/(1. + r/rs)**(3.-beta)


def Sigma(R,rs,beta):
    Rs = np.atleast_1d(R)
    out = 0.*Rs
    norm = 0.5*rs**(beta-1.)
    for i in range(0,len(Rs)):
        R = Rs[i]
        out[i] = (R/rs)**(1-beta)*quad(lambda theta: np.sin(theta)*(np.sin(theta) + R/rs)**(beta-3),0.,np.pi/2.)[0]
    return out/norm


def M2d(R,rs,beta):
    Rs = np.atleast_1d(R)
    out = 0.*Rs
    for i in range(0,len(Rs)):
        R = Rs[i]
        out[i] = 2*np.pi*quad(lambda x: Sigma(x,rs,beta)*x,0.,R)[0]
    return out


def lenspot(R,rs,beta):
    Rs = np.atleast_1d(R)
    out = 0.*Rs
    for i in range(0,len(Rs)):
        R = Rs[i]
        out[i] = 2*quad(lambda x: Sigma(x,rs,beta)*x*np.log(R/x),0.,R)[0]
    return out


def M3d(r,rs,beta):
    r = np.atleast_1d(r)
    out = 0.*r
    for i in range(0,len(r)):
        out[i] = 4*np.pi*quad(lambda x: rho(x,rs,beta)*x**2,0.,r[i])[0]
    return out

def fast_M2d(R,rs,beta):
    lam = rs/50.
    R = np.atleast_1d(R)
    beta = np.atleast_1d(beta)
    length = max(len(beta),len(R))
    sample = np.array([beta*np.ones(length),R/lam*np.ones(length)]).reshape((2,length)).T
    M2d = M2d_grid.eval(sample)*lam**(3-beta)*(R/lam)**(3.-beta)
    return M2d

def fast_M3d(r,rs,beta):
    lam = rs/50.
    r = np.atleast_1d(r)
    beta = np.atleast_1d(beta)
    length = max(len(beta),len(r))
    sample = np.array([beta*np.ones(length),r/lam*np.ones(length)]).reshape((2,length)).T
    M3d = M3d_grid.eval(sample)*lam**(3-beta)*(r/lam)**(3.-beta)
    return M3d

def fast_lenspot(R,rs,beta):
    lam = rs/50.
    R = np.atleast_1d(R)
    beta = np.atleast_1d(beta)
    length = max(len(beta),len(R))
    sample = np.array([beta*np.ones(length),R/lam*np.ones(length)]).reshape((2,length)).T
    pot = pot_grid.eval(sample)*lam**(3-beta)*(R/lam)**(3.-beta)
    return pot


def make_M2dRbetam3_grid(Nr=100,Nb=28,Rmin=0.1,Rmax=100.):
    #this code calculates the quantity M2d(R,rs=rsgrid,beta)*R**(3-beta) on a grid of values of R between Rmin and Rmax, and values of the inner slope beta between 0.1 and 2.8.
    #the reason for the multiplication by R**(3-beta) is to make interpolation easier by having a function as flat as possible.

    print 'calculating grid of enclosed projected masses...'
    import ndinterp
    reins = np.logspace(np.log10(Rmin),np.log10(Rmax),Nr)
    spl_rein = splrep(reins,np.arange(Nr))
    betas = np.linspace(0.1,2.8,Nb)
    spl_beta = splrep(betas,np.arange(Nb))
    axes = {0:spl_beta,1:spl_rein}

    R,B = np.meshgrid(reins,betas)
    M2d_grid = np.empty((Nb,Nr))
    
    for i in range(0,Nb):
        print 'inner slope %4.2f'%betas[i]
        for j in range(0,Nr):
            M2d_grid[i,j] = M2d(reins[j],rsgrid,betas[i])
    thing = ndinterp.ndInterp(axes,M2d_grid*R**(B-3.),order=3)
    f = open('gNFW_rs%d_M2d_grid.dat'%int(rsgrid),'w')
    pickle.dump(thing,f)
    f.close()

def make_lenspotRbetam3_grid(Nr=30,Nb=28,Rmin=0.1,Rmax=100.):
    #this code calculates the psi(R,rs=rsgrid,beta)*R**(3-beta) on a grid of values of R between Rmin and Rmax, and values of the inner slope beta between 0.1 and 2.8.
    #the reason for the multiplication by R**(3-beta) is to make interpolation easier by having a function as flat as possible.

    print 'calculating grid of lensing potential...'
    import ndinterp
    reins = np.logspace(np.log10(Rmin),np.log10(Rmax),Nr)
    spl_rein = splrep(reins,np.arange(Nr))
    betas = np.linspace(0.1,2.8,Nb)
    spl_beta = splrep(betas,np.arange(Nb))
    axes = {0:spl_beta,1:spl_rein}

    R,B = np.meshgrid(reins,betas)
    pot_grid = np.empty((Nb,Nr))
    
    for i in range(0,Nb):
        print 'inner slope %4.2f'%betas[i]
        for j in range(0,Nr):
            pot_grid[i,j] = lenspot(reins[j],rsgrid,betas[i])
    thing = ndinterp.ndInterp(axes,pot_grid*R**(B-3.),order=3)
    f = open('gNFW_rs%d_lenspotRbetam3_grid.dat'%int(rsgrid),'w')
    pickle.dump(thing,f)
    f.close()


def make_M3d_grid():
    import ndinterp
    pass


if not os.path.isfile(grid_dir+'gNFW_rs%d_M2dRbetam3_grid.dat'%int(rsgrid)):
    make_M2dRbetam3_grid()

if not os.path.isfile(grid_dir+'gNFW_rs%d_lenspotRbetam3_grid.dat'%int(rsgrid)):
    make_lenspotRbetam3_grid()


f = open(grid_dir+'gNFW_rs%d_M2dRbetam3_grid.dat'%rsgrid,'r')
M2d_grid = pickle.load(f)
f.close()

f = open(grid_dir+'gNFW_rs%d_lenspotRbetam3_grid.dat'%rsgrid,'r')
pot_grid = pickle.load(f)
f.close()


