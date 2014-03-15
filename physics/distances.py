import numpy
from numpy import *
from scipy import integrate
from sonnentools.cgsconstants import *

def Hz(z,H0=70.,omegaM=0.3,omegaL=0.7,omegar=0.):
    omegak = 1 - omegaM - omegaL - omegar
    return H0*sqrt(omegaL + omegaM*(1+z)**3 + omegar*(1+z)**4 + omegak*(1+z)**2)

def Hz_w(z,H0=70.,omegaM=0.3,omegaL=0.7,omegar=0.,w=lambda z: -1.):
    omegak = 1 - omegaM - omegaL - omegar
    exponent = lambda z: 3*integrate.quad(lambda x: (1 + w(x))/(1+x),0.,z)[0]
    return H0*sqrt(omegaM*(1+z)**3 + omegar*(1+z)**4 + omegak*(1+z)**2 + omegaL*exp(exponent(z)))

def comovd(z1,z2=0.,H0=70.,omegaM=0.3,omegaL=0.7,w=-1.,omegar=0.):
    omegak = 1 - omegaM - omegaL - omegar
    if z1>z2:
        z1,z2 = z2,z1
    I = integrate.quad(lambda z: 1./sqrt(omegaL*exp(3.*(1+w)) + omegaM*(1+z)**3 + omegar*(1+z)**4 + omegak*(1+z)**2),z1,z2)
    return c/(H0*10.**5)*Mpc*I[0]
    
def comovV(z1,z2=0.,H0=70.,omegaM=0.3,omegaL=0.7,omegar=0.):
    omegak = 1 - omegaM - omegaL - omegar
    if z1>z2:
        z1,z2 = z2,z1
    cd1 = comovd(z1,H0=70.,omegaM=omegaM,omegaL=omegaL,omegar=omegar)
    cd2 = comovd(z2,H0=70.,omegaM=omegaM,omegaL=omegaL,omegar=omegar)
    #returns the angular diameter disance in cm (default)
    if omegak==0.:
        V1 = 4/3.*pi*cd1**3
        V2 = 4/3.*pi*cd2**3
        return V2 - V1
    else:
        if omegak > 0.:
            R = c/H0/10.**5*Mpc/sqrt(omegak)
            V1 = 2*pi*R**3*(-cd1/R + 0.5*sinh(2*cd1/R))
            V2 = 2*pi*R**3*(-cd2/R + 0.5*sinh(2*cd2/R))
            return V2 - V1
        else:
            R = c/H0/10.**5*Mpc/sqrt(-omegak)
            V1 = 2*pi*R**3*(cd1/R - 0.5*sin(2*cd1/R))
            V2 = 2*pi*R**3*(cd2/R - 0.5*sin(2*cd2/R))
            return V2 - V1


def Dang(z1,z2=0.,H0=70.,omegaM=0.3,omegaL=0.7,w=-1.,omegar=0.,units='cgs'):
    omegak = 1 - omegaM - omegaL - omegar
    if z1>z2:
        z1,z2 = z2,z1
    cd = comovd(z1,z2,H0=H0,omegaM=omegaM,omegaL=omegaL,w=w,omegar=omegar)
    #returns the angular diameter disance in cm (default)
    if omegak==0.:
        D = cd/(1+z2)
    else:
        if omegak > 0.:
            D = c/H0/10.**5*Mpc/sqrt(abs(omegak))*sinh(sqrt(omegak)*H0/c*10.**5/Mpc*cd)/(1+z2)
        else:
            D = c/H0/10.**5*Mpc/sqrt(abs(omegak))*sin(sqrt(-omegak)*H0/c*10.**5/Mpc*cd)/(1+z2)


        
    if units != 'cgs':
        if units =='Mpc':
            D = D/Mpc
        elif units =='MKS':
            D = D/100.
    return D

def Dlum(z1,z2=0.,H0=70.,omegaM=0.3,omegaL=0.7,units='cgs'):
    omegak = 1 - omegaM - omegaL
    if z1>z2:
        z1,z2 = z2,z1
    cd = comovd(z1,z2,H0=70.,omegaM=omegaM,omegaL=omegaL)
    #returns the angular diameter disance in cm (default)
    if omegak==0.:
        D = cd*(1+z2)
    else:
        if omegak > 0.:
            D = c/H0/10.**5*Mpc/sqrt(abs(omegak))*sinh(sqrt(omegak)*H0/c*10.**5/Mpc*cd)*(1+z2)
        else:
            D = c/H0/10.**5*Mpc/sqrt(abs(omegak))*sin(sqrt(-omegak)*H0/c*10.**5/Mpc*cd)*(1+z2)


        
    if units != 'cgs':
        if units =='Mpc':
            D = D/Mpc
        elif units =='MKS':
            D = D/100.
    return D

def DL(z,H0=70.,omegaM=0.3,omegaL=0.7,units='cgs'):
    return comovd(z,H0=H0,omegaM=omegaM,omegaL=omegaL)*(1.+z)

def uniage(z,H0=70.,omegaM=0.3,omegaL=0.7,omegar=0.):
    omegak = 1 - omegaM - omegaL - omegar
    fint = lambda z: 1./(1+z)/numpy.sqrt(omegaL + omegaM*(1+z)**3 + omegar*(1+z)**4 + omegak*(1+z)**2)
    return integrate.quad(fint,z,inf)[0]/H0/yr*Mpc/10.**5


def lookback(z,H0=70.,omegaM=0.3,omegaL=0.7):
    fint = lambda z: 1./(1+z)/numpy.sqrt(omegaL + omegaM*(1+z)**3)
    return integrate.quad(fint,0.,z)[0]/H0/yr*Mpc/10.**5


def rhoc(z,H0=70.,omegaM=0.3,omegaL=0.7):
    return 3*(H0*10**5/Mpc)**2/(8.*pi*G)*(omegaM*(1+z)**3 + omegaL)


