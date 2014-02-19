import numpy as np
from scipy.integrate import quad
import pickle
from mass_profiles import gNFW_spheroid

#pseudo-elliptical generalized-NFW profile model. 
#The lens is elliptical in the deflection angle, NOT in projected density.
#takes a normalization constant, a scale radius, a power-law index of the inner slope, the axis ratio of the iso-deflection contours, and the PA of the major axis with respect to the x axis.

#Conversions to 3d quantities are done by "sphericizing" the mass profile in the following way: the spherical equivalent of a pseudo-elliptical gNFW profile is defined by the spherical gNFW profile that has the same value of M2d(R)/R.


class gNFW:
    
    def __init__(self,norm=None,rs=None,beta=None,q=None,PA=None,x0=0.,y0=0.):
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
        return self.norm*gNFW_spheroid.fast_lenspot(r,self.rs,self.beta)

    def alpha(self,x):
        xl = (x[0] - self.x0)*np.cos(self.PA) + (x[1] - self.y0)*np.sin(self.PA)
        yl = -(x[0] - self.x0)*np.sin(self.PA) + (x[1] - self.y0)*np.cos(self.PA)
        r = (xl**2*self.q + yl**2/self.q)**0.5
        mod = self.norm*gNFW_spheroid.fast_M2d(r,self.rs,self.beta)/r/np.pi
        alphaxl = mod*self.q/r*xl
        alphayl = mod/self.q/r*yl
        return (alphaxl*np.cos(-self.PA) + alphayl*np.sin(-self.PA),-alphaxl*np.sin(-self.PA) + alphayl*np.cos(-self.PA))


def main():
#checks if deflection angle and lensing potential are consistent with each other.
    import pylab
    import numpy as np

    lens = gNFW(norm=1.,rs=50.,beta=1.3,q=0.8,PA=0.)
    #lens = gNFW(norm=1.,rs=50.,beta=1.3,q=1.,PA=0.)

    xs = np.linspace(1.1,11.,100)
    dx = xs[1] - xs[0]
    lenspot = lens.lenspot((xs,0.*xs))
    lensfd = (lenspot[2:] - lenspot[:-2])/(2.*dx)
    alpha = lens.alpha((xs,0.*xs))
    pylab.plot(xs[1:-1],lensfd)
    pylab.plot(xs[1:-1],alpha[0][1:-1])
    pylab.show()
    



if __name__ == "__main__":
    main()


