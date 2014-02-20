import numpy as np
import pickle
import sersic, gNFW

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


class spherical_cow:
    
    def __init__(self,bulge=1.,halo=1.,reff=1.,n=4.,rs=50.,beta=1.):
        self.bulge = bulge
        self.halo = halo
        self.reff = reff
        self.n = n
        self.rs = rs
        self.beta = beta

    def kappa(self,x,y):
        return None

    def m(self,r):
        return None

    def lenspot(self,r):
        return self.halo*gNFW.fast_lenspot(r,self.rs,self.beta) + self.bulge*sersic(r,self.n,self.reff)

    def alpha(self,x):
        r = abs(x)
        mod = self.halo*gNFW.fast_M2d(r,self.rs,self.beta)/x/np.pi + self.bulge*sersic.fast_M2d(r,self.n,self.reff)/x/np.pi
        return 



