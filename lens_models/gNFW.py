import numpy as np
from scipy.integrate import quad
import pickle
from mass_profiles import gNFW_spheroid

#pseudo-elliptical generalized-NFW profile model. 
#The lens is elliptical in the deflection angle, NOT in projected density.
#takes a normalization constant, a scale radius, a power-law index of the inner slope, the axis ratio of the iso-deflection contours, and the PA of the major axis with respect to the x axis.

#Conversions to 3d quantities are done by "sphericizing" the mass profile in the following way: the spherical equivalent of a pseudo-elliptical gNFW profile is defined by the spherical gNFW profile that has the same value of M2d(R)/R.


class gNFW:
    
    def __init__(self,norm=None,rs=None,beta=None,q=None,PA=None):
        self.norm = norm
        self.rs = rs
        self.beta = beta
        self.q = q
        self.PA = PA

    def check_grids(self):
        pass


    def kappa(self,x,y):
        #calculating kappa for a lens with an elliptical lensing potential is non trivial
        return None

    def m(self,r):
        #calculating the enclosed mass for a lens with an elliptical lensing potential is non trivial
        return None

    def lenspot(self,x,y):
        xl = x*np.cos(self.PA) + y*np.sin(self.PA)
        yl = -x*np.sin(self.PA) + y*np.cos(self.PA)
        r = (xl**2*self.q + yl**2/self.q)**0.5
        return self.norm*gNFW_spheroid.fast_lenspot(r,self.rs,self.beta)

    def alpha(self,x,y):
        xl = x*np.cos(self.PA) + y*np.sin(self.PA)
        yl = -x*np.sin(self.PA) + y*np.cos(self.PA)
        r = (xl**2*self.q + yl**2/self.q)**0.5
        return self.norm*gNFW_spheroid.fast_M2d(r,self.rs,self.beta)/r
        

if __name__ == "__main__":
    main()

def main():
    print 'main'

