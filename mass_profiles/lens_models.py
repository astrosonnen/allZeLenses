import numpy as np
import pickle
import sersic as sersic_profile, gNFW as gNFW_profile
from scipy.optimize import brentq
from physics import cgsconstants as cgs, distances

#pseudo-elliptical sersic profile model. 
#The lens is elliptical in the deflection angle, NOT in projected density.
#takes a normalization constant, an effective radius, the axis ratio of the iso-deflection contours, and the PA of the major axis with respect to the x axis.

#Conversions to 3d quantities are done by "sphericizing" the mass profile in the following way: the spherical equivalent of a pseudo-elliptical sersic profile is defined by the circular sersic profile that has the same value of M2d(R)/R.

#norm is the total mass of the spherical equivalent profile.


class sersic:
    
    def __init__(self,norm=1.,reff=1.,n=4.,q=1.,PA=0.,x0=0.,y0=0.):
        self.norm = norm
        self.reff = reff
        self.n = n
        self.q = q
        self.PA = PA
        self.x0 = x0
        self.y0 = y0


    def kappa(self,x,y):
        #calculating kappa for a lens with an elliptical lensing potential is non trivial
        return None

    def m(self,r):
        #calculating the enclosed mass for a lens with an elliptical lensing potential is non trivial
        return None

    def lenspot(self,x):
        xl = (x[0] - self.x0)*np.cos(self.PA) + (x[1] - self.y0)*np.sin(self.PA)
        yl = -(x[0] - self.x0)*np.sin(self.PA) + (x[1] - self.y0)*np.cos(self.PA)
        r = (xl**2*self.q + yl**2/self.q)**0.5
        return self.norm*sersic.fast_lenspot(r,self.reff,self.n)

    def alpha(self,x):
        xl = (x[0] - self.x0)*np.cos(self.PA) + (x[1] - self.y0)*np.sin(self.PA)
        yl = -(x[0] - self.x0)*np.sin(self.PA) + (x[1] - self.y0)*np.cos(self.PA)
        r = (xl**2*self.q + yl**2/self.q)**0.5
        mod = self.norm*sersic.fast_M2d(r,self.reff,self.n)/r/np.pi
        alphaxl = mod*self.q/r*xl
        alphayl = mod/self.q/r*yl
        return (alphaxl*np.cos(-self.PA) + alphayl*np.sin(-self.PA),-alphaxl*np.sin(-self.PA) + alphayl*np.cos(-self.PA))


#pseudo-elliptical generalized-NFW profile model. 
#The lens is elliptical in the deflection angle, NOT in projected density.
#takes a normalization constant, a scale radius, a power-law index of the inner slope, the axis ratio of the iso-deflection contours, and the PA of the major axis with respect to the x axis.

#Conversions to 3d quantities are done by "sphericizing" the mass profile in the following way: the spherical equivalent of a pseudo-elliptical gNFW profile is defined by the spherical gNFW profile that has the same value of M2d(R)/R.


class gNFW:
    
    def __init__(self,norm=1.,rs=50.,beta=1.,q=1.,PA=0.,x0=0.,y0=0.):
        self.norm = norm
        self.rs = rs
        self.beta = beta
        self.q = q
        self.PA = PA
        self.x0 = x0
        self.y0 = y0

    def check_grids(self):
        pass


    def kappa(self,x,y):
        #calculating kappa for a lens with an elliptical lensing potential is non trivial
        return None

    def m(self,r):
        #calculating the enclosed mass for a lens with an elliptical lensing potential is non trivial
        return None

    def lenspot(self,x):
        xl = (x[0] - self.x0)*np.cos(self.PA) + (x[1] - self.y0)*np.sin(self.PA)
        yl = -(x[0] - self.x0)*np.sin(self.PA) + (x[1] - self.y0)*np.cos(self.PA)
        r = (xl**2*self.q + yl**2/self.q)**0.5
        return self.norm*gNFW.fast_lenspot(r,self.rs,self.beta)

    def alpha(self,x):
        xl = (x[0] - self.x0)*np.cos(self.PA) + (x[1] - self.y0)*np.sin(self.PA)
        yl = -(x[0] - self.x0)*np.sin(self.PA) + (x[1] - self.y0)*np.cos(self.PA)
        r = (xl**2*self.q + yl**2/self.q)**0.5
        mod = self.norm*gNFW.fast_M2d(r,self.rs,self.beta)/r/np.pi
        alphaxl = mod*self.q/r*xl
        alphayl = mod/self.q/r*yl
        return (alphaxl*np.cos(-self.PA) + alphayl*np.sin(-self.PA),-alphaxl*np.sin(-self.PA) + alphayl*np.cos(-self.PA))


class kappa_sheet:
    
    def __init__(self,kappa=0.):
        self.k = kappa

    def kappa(self,x,y):
        return self.k

    def m(self,r):
        return self.k*r**2

    def lenspot(self,x):
        return 0.5*self.k*(x[0]**2 + x[1]**2)

    def alpha(self,x):
        return (self.k*x[0],self.k*x[1])


class spherical_cow:
    
    def __init__(self,zd=0.3,zs=2.,mstar=11.,mdm=11.,reff_phys=1.,n=4.,rs_phys=50.,gamma=1.,kext=0.,images=None,source=0.):
        self.zd = zd
        self.zs = zs
        self.bulge = None
        self.halo = None
        self.mstar = mstar
        self.mdm = mdm
        self.reff_phys = reff_phys
        self.n = n
        self.rs_phys = rs_phys
        self.gamma = gamma
        self.kext = kext
        self.caustic = None
        self.radcrit = None
        self.source = source
        self.images = images
        self.timedelay = None

        self.S_cr = cgs.c**2/(4.*np.pi*cgs.G)*distances.Dang(0.,self.zs)*distances.Dang(0.,self.zd)/distances.Dang(self.zd,self.zs)/cgs.M_Sun*cgs.arcsec2rad**2
        arcsec2kpc = cgs.arcsec2rad*distances.Dang(self.zd)/cgs.kpc
        self.rs = self.rs_phys/arcsec2kpc
        self.reff = self.reff_phys/arcsec2kpc
        self.Dt = distances.Dang(self.zd)*distances.Dang(self.zs)/distances.Dang(self.zd,self.zs)/cgs.c*(1. + self.zd)/cgs.c
 
    def normalize(self):
        self.halo = 10.**self.mdm/gNFW_profile.M3d(self.reff,self.rs,self.gamma)/self.S_cr
        self.bulge = 10.**self.mstar/self.S_cr

    def kappa(self,r):
        return self.halo*gNFW_profile.Sigma(r,self.rs,self.gamma) + self.bulge*sersic_profile.I(r,self.n,self.reff) + self.kext

    def m(self,r):
        return self.halo*gNFW_profile.fast_M2d(r,self.rs,self.gamma)/np.pi + self.bulge*sersic_profile.fast_M2d(r,self.n,self.reff)/np.pi + self.kext*r**2

    def lenspot(self,r):
        return self.halo*gNFW_profile.fast_lenspot(r,self.rs,self.gamma) + self.bulge*sersic_profile.fast_lenspot(r,self.n,self.reff) + 0.5*self.kext*r**2

    def alpha(self,x):
        r = abs(x)
        return self.halo*gNFW_profile.fast_M2d(r,self.rs,self.gamma)/x/np.pi + self.bulge*sersic_profile.fast_M2d(r,self.n,self.reff)/x/np.pi + self.kext*r

    def get_caustic(self):

        rmin = self.rs/500.
        rmax = 10.*self.reff

        radial_invmag = lambda r: 2.*self.kappa(r) - self.m(r)/r**2 - 1.

        if radial_invmag(rmin)*radial_invmag(rmax) > 0.:
            rcrit = rmin
        else:
            rcrit = brentq(radial_invmag,rmin,rmax)

        ycaust = -(rcrit - self.alpha(rcrit))
        self.caustic = ycaust
        self.radcrit = rcrit


    def get_images(self):

        rmin = self.rs/500.
        rmax = 10.*self.reff

        imageeq = lambda r: r - self.alpha(r) - self.source
        if imageeq(self.radcrit)*imageeq(rmax) >= 0.:
            pass
        elif self.source <= self.caustic:
            xA = brentq(imageeq,self.radcrit,rmax)
            xB = brentq(imageeq,-rmax,-self.radcrit)
            self.images = (xA,xB)

        elif source > self.caustic:
            x = brentq(imageeq,self.radcrit,rmax)
            self.images = (x)
        else:
            pass

    def get_time_delay(self):
        self.timedelay = -self.Dt*(0.5*(self.images[0]**2 - self.images[1]**2) - self.images[0]*self.source + self.images[1]*self.source - self.lenspot(self.images[0]) + self.lenspot(-self.images[1]))


