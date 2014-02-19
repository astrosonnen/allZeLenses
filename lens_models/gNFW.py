from numpy import *
from scipy.integrate import quad
import pickle

#pseudo-elliptical generalized-NFW profile + elliptical de Vaucouleurs model. 
#The gNFW component is elliptical in the deflection angle, NOT in projected density.
#takes a normalization constant, a scale radius, a power-law index of the inner slope, the axis ratio of the iso-deflection contours, and the PA of the major axis with respect to the x axis.

#Conversions to 3d quantities are done by "sphericizing" the mass profile in the following way: the spherical equivalent of a pseudo-elliptical gNFW profile is defined by the spherical gNFW profile that has the same enclosed projected mass within r_equiv, which by default is set to a tenth of the scale radius.

#QUESTION: does the equivalent spherical profile depend on the value of r_equiv that we pick?

#normalization can be defined in two ways: either with the norm keyword, or with the m_equiv keyword. If m_equiv is not None, the mass profile will automatically renormalize itself so that the projected mass within r_equiv is equal to m_equiv.

class gNFW:
    
    def __init__(self,norm=None,rs=None,beta=None,q=None,PA=None,m_equiv=None,r_equiv=None):
        self.norm = norm
        self.rs = rs
        self.beta = beta
        self.q = q
        self.PA = PA
        self.m_equiv = m_equiv
        self.r_equiv = r_equiv
        if self.r_equiv is None and self.rs is not None:
            self.r_equiv = 0.1*self.rs
        if m_equiv is not None:
            self.normalize()

    def normalize(self):
        self.norm = 1.
        norm = m_equiv/self.M2d(r_equiv,rs,beta)
        self.norm = norm

    def check_grids(self):
        pass

    def rho(self,r):
        return self.norm/r**self.beta/(1 + r/self.rs)**(3-self.beta)

    def Sigma(self,R):
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
            out[i] = 2*quad(lambda x: Sigma(x,rs,beta)*x*log(R/x),0.,R)[0]
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

