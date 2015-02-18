#definitions of the Sersic profile

import numpy as np
from scipy.special import gamma as gfunc
import pickle
from scipy.integrate import quad
from scipy.interpolate import splrep
import os

grid_dir = '/setri6/sonnen/allZeLenses/'

def b(n):
    return 2*n - 1./3. + 4/405./n + 46/25515/n**2

def L(n,Re):
    return Re**2*2*np.pi*n/b(n)**(2*n)*gfunc(2*n)

def I(R,n,Re):
    return np.exp(-b(n)*(R/Re)**(1./n))/L(n,Re)

def M2d(R,n,Re):
    R = np.atleast_1d(R)
    out = 0.*R
    for i in range(0,len(R)):
        out[i] = 2*np.pi*quad(lambda r: r*I(r,n,Re),0.,R[i])[0]
    return out

def lenspot(R,n,Re):
    R = np.atleast_1d(R)
    out = 0.*R
    for i in range(0,len(R)):
        out[i] = 2*quad(lambda r: r*I(r,n,Re)*np.log(R[i]/r),0.,R[i])[0]
    return out

def fast_M2d(R,n,Re):
    R = np.atleast_1d(R)
    n = np.atleast_1d(n)
    length = max(len(n),len(R))
    sample = np.array([n*np.ones(length),R/Re*np.ones(length)]).reshape((2,length)).T
    M2d = M2d_grid.eval(sample)
    return M2d

def fast_lenspot(R,n,Re):
    R = np.atleast_1d(R)
    n = np.atleast_1d(n)
    length = max(len(n),len(R))
    sample = np.array([n*np.ones(length),R/Re*np.ones(length)]).reshape((2,length)).T
    pot = pot_grid.eval(sample)
    return pot


def deproject(r,n,Re):
    #should return an array with values of rho at radius r
    deriv = lambda R: -b(n)/n*(R/Re)**(1/n)/R*I(R,n,Re)
    
    rho = -1/np.pi*quad(lambda R: deriv(R)/sqrt(R**2 - r**2),r,np.inf)[0]
    return rho


def rho(r,n,Re):
    r = np.atleast_1d(r)
    out = 0.*r
    for i in range(0,len(r)):
        out[i] = deproject(r[i],n,Re)
    return out

def make_M2d_grid(Nr=100,Nn=15,Rmin=0.01,Rmax=10.):
    #this code calculates the quantity M2d(R,rs=rsgrid,beta)*R**(3-beta) on a grid of values of R between Rmin and Rmax, and values of the inner slope beta between 0.1 and 2.8.
    #the reason for the multiplication by R**(3-beta) is to make interpolation easier by having a function as flat as possible.

    print 'calculating grid of enclosed projected masses...'
    import ndinterp
    reins = np.logspace(np.log10(Rmin),np.log10(Rmax),Nr)
    spl_rein = splrep(reins,np.arange(Nr))
    ns = np.linspace(1.,8.,Nn)
    spl_n = splrep(ns,np.arange(Nn))
    axes = {0:spl_n,1:spl_rein}

    R,B = np.meshgrid(reins,ns)
    M2d_grid = np.empty((Nn,Nr))
    
    for i in range(0,Nn):
        print 'sersic index %4.2f'%ns[i]
        for j in range(0,Nr):
            M2d_grid[i,j] = M2d(reins[j],ns[i],1.)
    thing = ndinterp.ndInterp(axes,M2d_grid,order=3)
    f = open('sersic_M2d_grid.dat','w')
    pickle.dump(thing,f)
    f.close()

def make_lenspot_grid(Nr=100,Nn=15,Rmin=0.01,Rmax=10.):
    #this code calculates the psi(R,rs=rsgrid,beta)*R**(3-beta) on a grid of values of R between Rmin and Rmax, and values of the inner slope beta between 0.1 and 2.8.
    #the reason for the multiplication by R**(3-beta) is to make interpolation easier by having a function as flat as possible.

    print 'calculating grid of lensing potential...'
    import ndinterp
    reins = np.logspace(np.log10(Rmin),np.log10(Rmax),Nr)
    spl_rein = splrep(reins,np.arange(Nr))
    ns = np.linspace(1.0,8.0,Nn)
    spl_n = splrep(ns,np.arange(Nn))
    axes = {0:spl_n,1:spl_rein}

    R,B = np.meshgrid(reins,ns)
    pot_grid = np.empty((Nn,Nr))
    
    for i in range(0,Nn):
        print 'sersic index %4.2f'%ns[i]
        for j in range(0,Nr):
            pot_grid[i,j] = lenspot(reins[j],ns[i],1.)
    thing = ndinterp.ndInterp(axes,pot_grid,order=3)
    f = open('sersic_lenspot_grid.dat','w')
    pickle.dump(thing,f)
    f.close()


def make_M3d_grid():
    import ndinterp
    pass


if not os.path.isfile(grid_dir+'sersic_M2d_grid.dat'):
    make_M2d_grid()

if not os.path.isfile(grid_dir+'sersic_lenspot_grid.dat'):
    make_lenspot_grid()


f = open(grid_dir+'sersic_M2d_grid.dat','r')
M2d_grid = pickle.load(f)
f.close()

f = open(grid_dir+'sersic_lenspot_grid.dat','r')
pot_grid = pickle.load(f)
f.close()


