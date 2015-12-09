import numpy as np
import pickle
from mass_profiles import sersic as sersic_profile, gNFW as gNFW_profile, powerlaw,NFW as NFW_profile
from scipy.optimize import brentq
from allZeLenses.tools import cgsconstants as cgs, cosmology

#pseudo-elliptical sersic profile model. 
#The lens is elliptical in the deflection angle, NOT in projected density.
#takes a normalization constant, an effective radius, the axis ratio of the iso-deflection contours, and the PA of the major axis with respect to the x axis.

#Conversions to 3d quantities are done by "sphericizing" the mass profile in the following way: the spherical equivalent of a pseudo-elliptical sersic profile is defined by the circular sersic profile that has the same value of M2d(R)/R.

#norm is the total mass of the spherical equivalent profile.

eps = 1e-15

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


class sps:
    
    def __init__(self,zd=0.3,zs=2.,rein=1.,m5=None,gamma=2.,kext=0.,images=[],source=0.):
        self.zd = zd
        self.zs = zs
        self.rein = rein
        self.m5 = m5
        self.gamma = gamma
        self.kext = kext
        #self.bq12 = 4.*np.pi*(sigmav/cgs.c*1e5)**2*cgs.rad2arcsec
        self.caustic = None
        self.radcrit = None
        self.source = source
        self.images = images
        self.timedelay = None
        self.grids = None
        self.radmag_ratio = None

        self.S_cr = cgs.c**2/(4.*np.pi*cgs.G)*cosmology.Dang(0.,self.zs)*cosmology.Dang(0.,self.zd)/cosmology.Dang(self.zd,self.zs)/cgs.M_Sun*cgs.arcsec2rad**2
        self.arcsec2kpc = cgs.arcsec2rad*cosmology.Dang(self.zd)/cgs.kpc
        self.Dt = cosmology.Dang(self.zd)*cosmology.Dang(self.zs)/cosmology.Dang(self.zd,self.zs)/cgs.c*(1. + self.zd)/cgs.c
 
    def normalize(self):
        self.norm = self.rein/powerlaw.M2d(self.rein,self.gamma)/self.S_cr

    def normalize_m5(self):
        self.norm = self.m5/powerlaw.M2d(5./self.arcsec2kpc,self.gamma)/self.S_cr

    def kappa(self,x):
        return self.norm*powerlaw.Sigma(abs(x),self.gamma)

    def m(self,x):
        return self.norm*powerlaw.M2d(abs(x),self.gamma)/np.pi + self.kext*x**2

    def lenspot(self,r):
        return self.norm*powerlaw.lenspot(r,self.gamma) + 0.5*self.kext*r**2

    def alpha(self,x):
        r = abs(x)
        return self.norm*powerlaw.M2d(r,self.gamma)/x/np.pi + self.kext*x

    def get_caustic(self):

        rmin = 0.01
        rmax = 10.

        radial_invmag = lambda r: 2.*self.kappa(r) - self.m(r)/r**2 - 1.

        if radial_invmag(rmin)*radial_invmag(rmax) > 0.:
            rcrit = rmin
        else:
            rcrit = brentq(radial_invmag,rmin,rmax)

        ycaust = -(rcrit - self.alpha(rcrit))
        self.caustic = ycaust
        self.radcrit = rcrit


    def get_images(self):

        rmin = 0.01
        rmax = 10.

        imageeq = lambda r: r - self.alpha(r) - self.source
        if imageeq(self.radcrit)*imageeq(rmax) >= 0. or imageeq(-rmax)*imageeq(-self.radcrit) >=0.:
            self.images = []
        else:
            xA = brentq(imageeq,self.radcrit,rmax,xtol=1e-4)
            xB = brentq(imageeq,-rmax,-self.radcrit,xtol=1e-4)
            self.images = [xA,xB]

    def get_time_delay(self):
        self.timedelay = -self.Dt*(0.5*(self.images[0]**2 - self.images[1]**2) - self.images[0]*self.source + self.images[1]*self.source - self.lenspot(self.images[0]) + self.lenspot(-self.images[1]))

    def get_radmag_ratio(self):
        radmag_A = (1. + self.m(self.images[0])/self.images[0]**2 - 2.*self.kappa(self.images[0]))**(-1)
        radmag_B = (1. + self.m(self.images[1])/self.images[1]**2 - 2.*self.kappa(self.images[1]))**(-1)
        self.radmag_ratio = radmag_A/radmag_B


class sps_ang:
    
    def __init__(self,zd=0.3,zs=2.,rein=1.,gamma=2.,kext=0.,images=[],source=0.):
        self.zd = zd
        self.zs = zs
        self.rein = rein
        self.gamma = gamma
        self.kext = kext
        self.caustic = None
        self.radcrit = None
        self.source = source
        self.images = images
        self.timedelay = None
        self.grids = None
        self.radmag_ratio = None

        self.arcsec2kpc = cgs.arcsec2rad*cosmology.Dang(self.zd)/cgs.kpc
        self.Dt = cosmology.Dang(self.zd)*cosmology.Dang(self.zs)/cosmology.Dang(self.zd,self.zs)/cgs.c*(1. + self.zd)/cgs.c
 
    def kappa(self,x):
        return (3. - self.gamma)/2.*abs(x/self.rein)**(1.-self.gamma)

    def m(self,x):
        return self.rein**2*(abs(x)/self.rein)**(3.-self.gamma) + self.kext*x**2

    def lenspot(self,x):
        return self.rein**2/(3.-self.gamma)*(abs(x)/self.rein)**(3.-self.gamma) + 0.5*self.kext*x**2

    def alpha(self,x):
        r = abs(x)
        return self.m(x)/x + self.kext*x

    def get_caustic(self):

        rmin = 0.01
        rmax = 10.

	if self.gamma > 2.:
	    self.caustic = np.inf
	    self.radcrit = 0.

	elif self.gamma == 2.:
	    self.caustic = self.rein
	    self.radcrit = 0.

	else:
	    self.radcrit = self.rein*(1./(2.-self.gamma))**(1./(1.-self.gamma))
	    self.caustic = -(self.radcrit - self.alpha(self.radcrit))


    def get_images(self, xmax=10.):

	self.get_caustic()

	if self.source < min(self.caustic, xmax):
	    imageeq = lambda r: r - self.alpha(r) - self.source
	    xA = brentq(imageeq, self.rein, xmax, xtol=1e-4)
	    xB = brentq(imageeq, -self.rein, -max(self.radcrit, eps), xtol=1e-4)
	    self.images = (xA, xB)
	else:
	    self.images = (-99., 99.)


    def get_time_delay(self):
        self.timedelay = -self.Dt*(0.5*(self.images[0]**2 - self.images[1]**2) - self.images[0]*self.source + self.images[1]*self.source - self.lenspot(self.images[0]) + self.lenspot(-self.images[1]))

    def get_radmag_ratio(self):
        radmag_A = (1. + self.m(self.images[0])/self.images[0]**2 - 2.*self.kappa(self.images[0]))**(-1)
        radmag_B = (1. + self.m(self.images[1])/self.images[1]**2 - 2.*self.kappa(self.images[1]))**(-1)
        self.radmag_ratio = radmag_A/radmag_B


class sps_mst: #spherical power-law profile with a mass-sheet transformation
    
    def __init__(self,zd=0.3,zs=2.,rein=1.,gamma=2.,lmst=1.,images=[],source=0.):
        self.zd = zd
        self.zs = zs
        self.rein = rein
        self.gamma = gamma
        self.lmst = lmst
        self.caustic = None
        self.radcrit = None
        self.source = source
        self.images = images
        self.timedelay = None
        self.grids = None
        self.radmag_ratio = None

        self.arcsec2kpc = cgs.arcsec2rad*cosmology.Dang(self.zd)/cgs.kpc
        self.Dt = cosmology.Dang(self.zd)*cosmology.Dang(self.zs)/cosmology.Dang(self.zd,self.zs)/cgs.c*(1. + self.zd)/cgs.c
 
    def kappa(self,x):
        return self.lmst*(3. - self.gamma)/2.*abs(x/self.rein)**(1.-self.gamma) + 1 - self.lmst

    def m(self,x):
        return self.lmst*self.rein**2*(abs(x)/self.rein)**(3.-self.gamma) + (1.- self.lmst)*x**2

    def lenspot(self,x):
        return self.lmst*self.rein**2/(3.-self.gamma)*(abs(x)/self.rein)**(3.-self.gamma) + 0.5*(1. - self.lmst)*x**2

    def alpha(self,x):
        r = abs(x)
        return self.m(x)/x #+ (1. - self.lmst)*x

    def get_caustic(self):

        rmin = eps
        rmax = 10.

        radial_invmag = lambda r: 2.*self.kappa(r) - self.m(r)/r**2 - 1.

        if radial_invmag(rmin)*radial_invmag(rmax) > 0.:
            rcrit = rmin
        else:
            rcrit = brentq(radial_invmag,rmin,rmax)

        ycaust = -(rcrit - self.alpha(rcrit))
        self.caustic = ycaust
        self.radcrit = rcrit


    def get_images(self, xmax=10., xtol=1e-4):

	self.get_caustic()

	if self.source < min(self.caustic, xmax):
	    imageeq = lambda r: r - self.alpha(r) - self.source
	    xA = brentq(imageeq, self.rein, xmax, xtol=xtol)
	    xB = brentq(imageeq, -self.rein, -max(self.radcrit, eps), xtol=xtol)
	    self.images = (xA, xB)
	else:
	    self.images = (-99., 99.)


    def get_time_delay(self):
        self.timedelay = -self.Dt*(0.5*(self.images[0]**2 - self.images[1]**2) - self.images[0]*self.source + self.images[1]*self.source - self.lenspot(self.images[0]) + self.lenspot(-self.images[1]))

    def get_radmag_ratio(self):
        radmag_A = (1. + self.m(self.images[0])/self.images[0]**2 - 2.*self.kappa(self.images[0]))**(-1)
        radmag_B = (1. + self.m(self.images[1])/self.images[1]**2 - 2.*self.kappa(self.images[1]))**(-1)
        self.radmag_ratio = radmag_A/radmag_B


class spherical_cow:
    
    def __init__(self,zd=0.3,zs=2.,mstar=1e11,mdm5=1e11,reff_phys=1.,n=4.,rs_phys=50.,gamma=1.,kext=0.,images=[],source=0.,obs_images=None,obs_lmstar=None,obs_radmagrat=None):
        self.zd = zd
        self.zs = zs
        self.bulge = None
        self.halo = None
        self.mstar = mstar
        self.mdm5 = mdm5
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
        self.grids = None
        self.rein_grid = None
        self.radmag_ratio = None
        self.dyndic = None
        self.rein = None
        self.veldisp = None
        self.imA = None
        self.imB = None
        self.obs_lmstar = obs_lmstar
        self.obs_radmagrat = obs_radmagrat
        self.obs_images = obs_images

        self.S_cr = cgs.c**2/(4.*np.pi*cgs.G)*cosmology.Dang(0.,self.zs)*cosmology.Dang(0.,self.zd)/cosmology.Dang(self.zd,self.zs)/cgs.M_Sun*cgs.arcsec2rad**2
        self.arcsec2kpc = cgs.arcsec2rad*cosmology.Dang(self.zd)/cgs.kpc
        self.rs = self.rs_phys/self.arcsec2kpc
        self.reff = self.reff_phys/self.arcsec2kpc
        self.Dt = cosmology.Dang(self.zd)*cosmology.Dang(self.zs)/cosmology.Dang(self.zd,self.zs)/cgs.c*(1. + self.zd)/cgs.c
 
    def normalize(self):
        self.halo = self.mdm5/gNFW_profile.fast_M2d(5./self.arcsec2kpc,self.rs,self.gamma)/self.S_cr
        self.bulge = self.mstar/self.S_cr

    def kappa(self,x):
        return self.halo*gNFW_profile.Sigma(abs(x),self.rs,self.gamma) + self.bulge*sersic_profile.I(abs(x),self.n,self.reff) + self.kext

    def m(self,x):
        return self.halo*gNFW_profile.fast_M2d(abs(x),self.rs,self.gamma)/np.pi + self.bulge*sersic_profile.fast_M2d(abs(x),self.n,self.reff)/np.pi + self.kext*x**2

    def lenspot(self,r):
        return self.halo*gNFW_profile.fast_lenspot(r,self.rs,self.gamma) + self.bulge*sersic_profile.fast_lenspot(r,self.n,self.reff) + 0.5*self.kext*r**2

    def alpha(self,x):
        r = abs(x)
        return self.halo*gNFW_profile.fast_M2d(r,self.rs,self.gamma)/x/np.pi + self.bulge*sersic_profile.fast_M2d(r,self.n,self.reff)/x/np.pi + self.kext*x

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


    def get_rein(self):

        rmin = self.rs/500.
        rmax = 10.*self.reff

        tangential_invmag = lambda r: self.m(r)/r**2 - 1.

        tcrit = brentq(tangential_invmag,rmin,rmax)

        self.rein = tcrit


    def get_images(self):

        rmin = self.rs/500.
        rmax = 10.*self.reff

        imageeq = lambda r: r - self.alpha(r) - self.source
        if imageeq(self.radcrit)*imageeq(rmax) >= 0. or imageeq(-rmax)*imageeq(-self.radcrit) >=0.:
            self.images = []
        else:
        #elif self.source <= self.caustic:
            xA = brentq(imageeq,self.radcrit,rmax,xtol=1e-4)
            xB = brentq(imageeq,-rmax,-self.radcrit,xtol=1e-4)
            self.images = [xA,xB]
        """
        elif self.source > self.caustic:
            x = brentq(imageeq,self.radcrit,rmax)
            self.images = [x]
        else:
            self.images = []
        """

    def get_time_delay(self):
        self.timedelay = -self.Dt*(0.5*(self.images[0]**2 - self.images[1]**2) - self.images[0]*self.source + self.images[1]*self.source - self.lenspot(self.images[0]) + self.lenspot(-self.images[1]))

    def make_grids(self,err=0.01,nsig=3.):
        tol = 0.1*err
        x0A = self.images[0] - nsig*err
        x0A = x0A - x0A%tol
        x1A = self.images[0] + nsig*err
        x1A = x1A - x1A%tol
        gridA = np.arange(x0A,x1A,tol)

        x0B = self.images[1] - nsig*err
        x0B = x0B - x0B%tol
        x1B = -max(self.rs/500.,-(self.images[1] + nsig*err))#,self.radcrit)
        x1B = x1B - x1B%tol
        gridB = np.arange(x0B,x1B,tol)

        self.grids = (gridA,gridB)

    def make_rein_grid(self,err=0.01,nsigma=5.):
        tol = 0.1*err
        r0 = max(self.rs/500.,self.rein - nsigma*err)
        r1 = self.rein + nsigma*err
        grid = np.arange(r0,r1,tol)

        self.rein_grid = grid


    def fast_images(self):

        self.images = []
        for grid in self.grids:
            dx = grid[1] - grid[0]
            ys = grid - self.alpha(grid)
            diff = (self.source - ys[:-1])*(self.source - ys[1:])
            found = diff <= 0.
            for x in grid[:-1][found]:
                self.images.append(x+0.5*dx)


    def fast_imA(self):

        grid = self.grids[0]
        dx = grid[1] - grid[0]
        ys = grid - self.alpha(grid)
        diff = (self.source - ys[:-1])*(self.source - ys[1:])
        found = diff <= 0.
        for x in grid[:-1][found]:
            self.imA = x + 0.5*dx

    def fast_imB(self):

        grid = self.grids[1]
        dx = grid[1] - grid[0]
        ys = grid - self.alpha(grid)
        diff = (self.source - ys[:-1])*(self.source - ys[1:])
        found = diff <= 0.
        for x in grid[:-1][found]:
            self.imB = x + 0.5*dx


    def fast_rein(self):

        dr = self.rein_grid[1] - self.rein_grid[0]
        ys = self.rein_grid - self.alpha(self.rein_grid)
        
        rein = abs(ys).argmin()
        
        self.rein = self.rein_grid[rein]


    def get_radmag_ratio(self):
        radmag_A = (1. + self.m(self.images[0])/self.images[0]**2 - 2.*self.kappa(self.images[0]))**(-1)
        radmag_B = (1. + self.m(self.images[1])/self.images[1]**2 - 2.*self.kappa(self.images[1]))**(-1)
        self.radmag_ratio = radmag_A/radmag_B


class PIEMD_Keeton:
    def __init__(self,sigmav=200.,x0=0.,y0=0.,rc=0.01,q=1.,PA=0.):
        self.rc = rc
        self.q = q
        self.s = self.q**0.5*self.rc
        self.sigmav = sigmav
        self.x0 = x0
        self.y0 = y0
        self.PA = PA
        self.bq12 = 4.*np.pi*(self.sigmav/cgs.c*1e5)**2*cgs.rad2arcsec

    def update(self):
        self.bq12 = 4.*np.pi*(self.sigmav/cgs.c*1e5)**2*cgs.rad2arcsec

    def alpha(self,x):

        xl = (x[0] - self.x0)*np.cos(self.PA) + (x[1] - self.y0)*np.sin(self.PA)
        yl = -(x[0] - self.x0)*np.sin(self.PA) + (x[1] - self.y0)*np.cos(self.PA)

        psi = (self.q**2*(self.s**2 + xl**2) + yl**2)**0.5

        alphaxl = self.bq12*(self.q/(1.-self.q**2))**0.5*np.arctan(np.sqrt(1.-self.q**2)*xl/(psi + self.s))
        alphayl = self.bq12*(self.q/(1.-self.q**2))**0.5*np.arctanh(np.sqrt(1.-self.q**2)*yl/(psi + self.q**2*self.s))

        return (alphaxl*np.cos(-self.PA) + alphayl*np.sin(-self.PA),-alphaxl*np.sin(-self.PA) + alphayl*np.cos(-self.PA))
        

class PIEMD_glee:
    def __init__(self,E0=1.,x0=0.,y0=0.,w=1.,q=1.,PA=0.,scale=False):
        self.w = w
        if scale:
            self.E0 = E0/(np.sqrt(E0**2 + w**2) - w)
        else:
            self.E0 = E0

        #self.rc = rc
        self.q = q
        self.rein = None
        self.x0 = x0
        self.y0 = y0
        self.PA = PA
        self.eps = (1.-self.q)/(1+self.q)
        self.omega = None
        self.amp = None
        #self.update()
        self.theta = self.PA/180.*np.pi
        

    def update(self):
        self.eps = (1 - self.q)/(1 + self.q)
        self.omega = self.rc/np.sqrt((1 + self.eps)*(1-self.eps))
        self.amp = 1.
        self.amp = self.rein**2/self.m(self.rein)
        self.E0 = 2*self.amp/np.sqrt((1+self.eps)*(1-self.eps))
        self.theta = self.PA/180.*np.pi

    def kappa(self,x):
        req2 = (x[0]-self.x0)**2/(1+self.eps)**2 + (x[1]-self.y0)**2/(1-self.eps)**2
        return self.E0/2./np.sqrt(self.w**2 + req2)

    def m(self,r):
        return 2*self.amp*self.rc*((1. + (r/self.rc)**2)**0.5 - 1.)

    def alpha(self,x):
        x1,x2 = x[0] - self.x0,x[1] - self.y0
        x1p = np.cos(-self.theta)*x1 - np.sin(-self.theta)*x2
        x2p = np.sin(-self.theta)*x1 + np.cos(-self.theta)*x2

        alpha = (1 - self.eps**2)*self.E0/2j/np.sqrt(self.eps)*np.log(((1-self.eps)/(1+self.eps)*x1p - (1+self.eps)/(1-self.eps)*1j*x2p + 2j*np.sqrt(self.eps)*np.sqrt(self.w**2 + x1p**2/(1+self.eps)**2 + x2p**2/(1-self.eps)**2))/(x1p - x2p*1j + 2j*self.w*np.sqrt(self.eps)))
        alpha *= np.exp(self.theta*1j)
        
        return (np.real(alpha),np.imag(alpha))

    def lenspot(self,x):
        x1,x2 = x[0] - self.x0,x[1] - self.y0
        x1p = np.cos(-self.theta)*x1 - np.sin(-self.theta)*x2
        x2p = np.sin(-self.theta)*x1 + np.cos(-self.theta)*x2
        cosphi = x1p/(x1p**2 + x2p**2)**0.5
        sinphi = x2p/(x1p**2 + x2p**2)**0.5

        req2 = x1p**2/(1+self.eps)**2 + x2p**2/(1-self.eps)**2
        eta = -0.5*np.arcsinh(2.*self.eps**0.5/(1.-self.eps)*sinphi) + 0.5*1j*np.arcsin(2.*self.eps**0.5/(1.+self.eps)*cosphi)
        zeta = 0.5*np.log((req2**0.5 + (req2 + self.w**2)**0.5)/self.w)

        Kstar = np.sinh(2.*eta)*np.log(np.cosh(eta)**2/(np.cosh(eta+zeta)*np.cosh(eta-zeta))) + np.sinh(2.*zeta)*np.log(np.cosh(eta+zeta)/np.cosh(eta-zeta))

        return 0.5*self.E0*self.w*(1.-self.eps**2)/req2**0.5/self.eps**0.5*np.imag((x1p - 1j*x2p)*Kstar)



class PIEMD:
    def __init__(self,rein=1.,x0=0.,y0=0.,rc=1.,q=1.,PA=0.):
        self.rc = rc
        self.q = q
        self.rein = rein
        self.x0 = x0
        self.y0 = y0
        self.PA = PA
        self.eps = None
        self.omega = None
        self.amp = None
        self.E0 = None
        self.update()

    def update(self):
        self.eps = (1 - self.q)/(1 + self.q)
        self.omega = self.rc/np.sqrt((1 + self.eps)*(1-self.eps))
        self.amp = 1.
        self.amp = self.rein**2/self.m(self.rein)
        self.E0 = 2*self.amp/np.sqrt((1+self.eps)*(1-self.eps))
        self.theta = self.PA/180.*np.pi

    def kappa(self,x):
        req2 = (x[0]-self.x0)**2/(1+self.eps) + (x[1]-self.y0)**2/(1+self.eps)
        return self.E0/2./np.sqrt(self.omega**2 + req2)

    def m(self,r):
        return 2*self.amp*self.rc*((1. + (r/self.rc)**2)**0.5 - 1.)

    def alpha(self,x):
        x1,x2 = x[0] - self.x0,x[1] - self.y0
        x1p = np.cos(-self.theta)*x1 - np.sin(-self.theta)*x2
        x2p = np.sin(-self.theta)*x1 + np.cos(-self.theta)*x2

        alpha = (1 - self.eps**2)*self.E0/2j/np.sqrt(self.eps)*np.log(((1-self.eps)/(1+self.eps)*x1p - (1+self.eps)/(1-self.eps)*1j*x2p + 2j*np.sqrt(self.eps)*np.sqrt(self.omega**2 + x1p**2/(1+self.eps)**2 + x2p**2/(1-self.eps)**2))/(x1p - x2p*1j + 2j*self.omega*np.sqrt(self.eps)))
        alpha *= np.exp(self.theta*1j)
        
        return (np.real(alpha),np.imag(alpha))



class nfw:
    def __init__(self,zd=0.3,mvir=1e11,rvir=100,rs=10,x0=0.,y0=0.):
        self.zd = zd
        self.mvir = mvir
        self.rvir = rvir
        self.rs = rs
        
        self.kpc2arcsec = cgs.rad2arcsec*cgs.kpc/cosmology.Dang(self.zd)
        self.arcsec2kpc = self.kpc2arcsec**-1
        self.K = self.mvir/NFW.M3d(self.rvir,self.rs)
        self.x0 = x0
        self.y0 = y0

    def normalize(self):
        self.K = self.mvir/NFW.M3d(self.rvir,self.rs)

    def alpha(self,x):
        r = ((x[0]-self.x0)**2 + (x[1]-self.y0)**2)**0.5
        menc = self.K*NFW.M2d(r*self.arcsec2kpc,self.rs)*cgs.M_Sun
        defl = 4.*cgs.G/cgs.c**2*menc/(r*self.arcsec2kpc*cgs.kpc)*cgs.rad2arcsec
        return (defl*(x[0]-self.x0)/r,defl*(x[1]-self.y0)/r)


class shear:
    def __init__(self,mag=0.,PA=0.,x0=0.,y0=0.):
        self.mag = mag
        self.PA = PA
        self.theta = self.PA/180.*np.pi
        self.gamma1 = self.mag*np.cos(2.*self.theta)
        self.gamma2 = self.mag*np.sin(2.*self.theta)
        self.x0 = x0
        self.y0 = y0

    def alpha(self,x):
        return (self.gamma1*(x[0]-self.x0) + self.gamma2*(x[1]-self.y0),-self.gamma1*(x[1]-self.y0) + self.gamma2*(x[0]-self.x0))

    def lenspot(self,x):
        return 0.5*self.gamma1*((x[0]-self.x0)**2 - (x[1]-self.y0)**2) + self.gamma2*(x[0]-self.x0)*(x[1]-self.y0)



class nfw_deV:
    
    def __init__(self,zd=0.3,zs=2.,mstar=1e11,mhalo=1e13,reff_phys=1.,cvir=5.,images=[],source=0.,obs_images=None,obs_lmstar=None,obs_radmagrat=None,Delta_halo=93.5):
        self.zd = zd
        self.zs = zs
        self.bulge = None
        self.halo = None
        self.mstar = mstar
        self.mhalo = mhalo
        self.Delta_halo = Delta_halo
        self.reff_phys = reff_phys
        self.caustic = None
        self.radcrit = None
        self.source = source
        self.images = images
        self.timedelay = None
        self.grids = None
        self.radmag_ratio = None
        self.dyndic = None
        self.rein = None
        self.veldisp = None
        self.imA = None
        self.imB = None
        self.rvir = None
        self.gammap = None
        self.obs_lmstar = obs_lmstar
        self.obs_radmagrat = obs_radmagrat
        self.obs_images = obs_images

        self.S_cr = cgs.c**2/(4.*np.pi*cgs.G)*cosmology.Dang(0.,self.zs)*cosmology.Dang(0.,self.zd)/cosmology.Dang(self.zd,self.zs)/cgs.M_Sun*cgs.arcsec2rad**2
        self.arcsec2kpc = cgs.arcsec2rad*cosmology.Dang(self.zd)/cgs.kpc
        self.reff = self.reff_phys/self.arcsec2kpc
        self.cvir = cvir
        self.Dt = cosmology.Dang(self.zd)*cosmology.Dang(self.zs)/cosmology.Dang(self.zd,self.zs)/cgs.c*(1. + self.zd)/cgs.c
        self.rhoc = cosmology.rhoc(self.zd)
 
    def normalize(self):
        self.rvir = (self.mhalo*cgs.M_Sun*3./self.Delta_halo/(4.*np.pi)/self.rhoc)**(1/3.)/cgs.kpc/self.arcsec2kpc
        self.rs = self.rvir/self.cvir
        self.halo = self.mhalo/NFW_profile.M3d(self.rvir,self.rs)/self.S_cr
        self.bulge = self.mstar/self.S_cr

    def kappa(self,x):
        return self.halo*NFW_profile.Sigma(abs(x),self.rs) + self.bulge*sersic_profile.I(abs(x),4.,self.reff)

    def m(self,x):
        return self.halo*NFW_profile.M2d(abs(x),self.rs)/np.pi + self.bulge*sersic_profile.fast_M2d(abs(x),4.,self.reff)/np.pi

    def lenspot(self,r):
        return self.halo*NFW_profile.lenspot(r,self.rs) + self.bulge*sersic_profile.fast_lenspot(r,4.,self.reff)

    def alpha(self,x):
        r = abs(x)
        return self.halo*NFW_profile.M2d(r,self.rs)/x/np.pi + self.bulge*sersic_profile.fast_M2d(r,4.,self.reff)/x/np.pi

    def get_caustic(self):

        rmin = self.reff/50.
        rmax = 10.*self.reff

        radial_invmag = lambda r: 2.*self.kappa(r) - self.m(r)/r**2 - 1.

        if radial_invmag(rmin)*radial_invmag(rmax) > 0.:
            rcrit = rmin
        else:
            rcrit = brentq(radial_invmag,rmin,rmax)

        ycaust = -(rcrit - self.alpha(rcrit))
        self.caustic = ycaust
        self.radcrit = rcrit


    def get_rein(self):

        eps = 1e-4

        tangential_invmag = lambda r: self.m(r)/r**2 - 1.

        #tcrit = brentq(tangential_invmag,-self.images[1],self.images[0])
        tcrit = brentq(tangential_invmag, self.reff/50., 10.*self.reff)

        self.rein = tcrit


    def get_images(self):

        rmin = self.reff/50.
        rmax = 10.*self.reff

	#self.get_rein()

        imageeq = lambda r: r - self.alpha(r) - self.source
        if imageeq(self.radcrit)*imageeq(rmax) >= 0. or imageeq(-rmax)*imageeq(-self.radcrit) >=0.:
            self.images = (-99., 99.)
        else:
            xA = brentq(imageeq, rmin, rmax, xtol=1e-4)
            xB = brentq(imageeq, -rmax, -self.radcrit, xtol=1e-4)
            self.images = (xA, xB)


    def get_time_delay(self):
        self.timedelay = -self.Dt*(0.5*(self.images[0]**2 - self.images[1]**2) - self.images[0]*self.source + self.images[1]*self.source - self.lenspot(self.images[0]) + self.lenspot(-self.images[1]))

    def make_grids(self,err=0.01,nsig=3.):
        tol = 0.1*err
        x0A = self.images[0] - nsig*err
        x0A = x0A - x0A%tol
        x1A = self.images[0] + nsig*err
        x1A = x1A - x1A%tol
        gridA = np.arange(x0A,x1A,tol)

        x0B = self.images[1] - nsig*err
        x0B = x0B - x0B%tol
        x1B = -max(self.rs/500.,-(self.images[1] + nsig*err))#,self.radcrit)
        x1B = x1B - x1B%tol
        gridB = np.arange(x0B,x1B,tol)

        self.grids = (gridA,gridB)

    def make_rein_grid(self,err=0.01,nsigma=5.):
        tol = 0.1*err
        r0 = max(self.rs/500.,self.rein - nsigma*err)
        r1 = self.rein + nsigma*err
        grid = np.arange(r0,r1,tol)

        self.rein_grid = grid


    def fast_images(self):

        self.images = []
        for grid in self.grids:
            dx = grid[1] - grid[0]
            ys = grid - self.alpha(grid)
            diff = (self.source - ys[:-1])*(self.source - ys[1:])
            found = diff <= 0.
            for x in grid[:-1][found]:
                self.images.append(x+0.5*dx)


    def fast_imA(self):

        grid = self.grids[0]
        dx = grid[1] - grid[0]
        ys = grid - self.alpha(grid)
        diff = (self.source - ys[:-1])*(self.source - ys[1:])
        found = diff <= 0.
        for x in grid[:-1][found]:
            self.imA = x + 0.5*dx

    def fast_imB(self):

        grid = self.grids[1]
        dx = grid[1] - grid[0]
        ys = grid - self.alpha(grid)
        diff = (self.source - ys[:-1])*(self.source - ys[1:])
        found = diff <= 0.
        for x in grid[:-1][found]:
            self.imB = x + 0.5*dx


    def fast_rein(self):

        dr = self.rein_grid[1] - self.rein_grid[0]
        ys = self.rein_grid - self.alpha(self.rein_grid)
        
        rein = abs(ys).argmin()
        
        self.rein = self.rein_grid[rein]


    def get_radmag_ratio(self):
        radmag_A = (1. + self.m(self.images[0])/self.images[0]**2 - 2.*self.kappa(self.images[0]))**(-1)
        radmag_B = (1. + self.m(self.images[1])/self.images[1]**2 - 2.*self.kappa(self.images[1]))**(-1)
        self.radmag_ratio = radmag_A/radmag_B

    def get_gammap(self):
        self.gammap = 3. - (4.*np.pi*self.reff**3)*(self.mhalo*NFW_profile.rho(self.reff,self.rs)/NFW_profile.M3d(self.rvir,self.rs) + self.mstar*sersic_profile.rho(self.reff,4.,self.reff))/(self.mhalo*NFW_profile.M3d(self.reff,self.rs)/NFW_profile.M3d(self.rvir,self.rs) + self.mstar*0.4135)


class nfw_sersic:
    
    def __init__(self,zd=0.3,zs=2.,mstar=1e11,mhalo=1e13,reff_phys=1.,n=4.,cvir=5.,images=[],source=0.,obs_images=None,obs_lmstar=None,obs_radmagrat=None,Delta_halo=93.5):
        self.zd = zd
        self.zs = zs
        self.bulge = None
        self.halo = None
        self.mstar = mstar
        self.mhalo = mhalo
        self.Delta_halo = Delta_halo
        self.reff_phys = reff_phys
        self.n = n
        self.caustic = None
        self.radcrit = None
        self.source = source
        self.images = images
        self.timedelay = None
        self.grids = None
        self.radmag_ratio = None
        self.dyndic = None
        self.rein = None
        self.veldisp = None
        self.imA = None
        self.imB = None
        self.rvir = None
        self.gammap = None
        self.obs_lmstar = obs_lmstar
        self.obs_radmagrat = obs_radmagrat
        self.obs_images = obs_images

        self.S_cr = cgs.c**2/(4.*np.pi*cgs.G)*cosmology.Dang(0.,self.zs)*cosmology.Dang(0.,self.zd)/cosmology.Dang(self.zd,self.zs)/cgs.M_Sun*cgs.arcsec2rad**2
        self.arcsec2kpc = cgs.arcsec2rad*cosmology.Dang(self.zd)/cgs.kpc
        self.reff = self.reff_phys/self.arcsec2kpc
        self.cvir = cvir
        self.Dt = cosmology.Dang(self.zd)*cosmology.Dang(self.zs)/cosmology.Dang(self.zd,self.zs)/cgs.c*(1. + self.zd)/cgs.c
        self.rhoc = cosmology.rhoc(self.zd)
 
    def normalize(self):
        self.rvir = (self.mhalo*cgs.M_Sun*3./self.Delta_halo/(4.*np.pi)/self.rhoc)**(1/3.)/cgs.kpc/self.arcsec2kpc
        self.rs = self.rvir/self.cvir
        self.halo = self.mhalo/NFW_profile.M3d(self.rvir,self.rs)/self.S_cr
        self.bulge = self.mstar/self.S_cr

    def kappa(self,x):
        return self.halo*NFW_profile.Sigma(abs(x),self.rs) + self.bulge*sersic_profile.I(abs(x),self.n,self.reff)

    def m(self,x):
        return self.halo*NFW_profile.M2d(abs(x),self.rs)/np.pi + self.bulge*sersic_profile.fast_M2d(abs(x),self.n,self.reff)/np.pi

    def lenspot(self,r):
        return self.halo*NFW_profile.lenspot(r,self.rs) + self.bulge*sersic_profile.fast_lenspot(r,self.n,self.reff)

    def alpha(self,x):
        r = abs(x)
        return self.halo*NFW_profile.M2d(r,self.rs)/x/np.pi + self.bulge*sersic_profile.fast_M2d(r,self.n,self.reff)/x/np.pi

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


    def get_rein(self):

        eps = 1e-4

        tangential_invmag = lambda r: self.m(r)/r**2 - 1.

        #tcrit = brentq(tangential_invmag,-self.images[1],self.images[0])
        tcrit = brentq(tangential_invmag,eps,10.*self.reff)

        self.rein = tcrit


    def get_images(self):

        rmin = self.rs/500.
        rmax = 10.*self.reff

        imageeq = lambda r: r - self.alpha(r) - self.source
        if imageeq(self.radcrit)*imageeq(rmax) >= 0. or imageeq(-rmax)*imageeq(-self.radcrit) >=0.:
            self.images = []
        else:
            xA = brentq(imageeq,self.radcrit,rmax,xtol=1e-4)
            xB = brentq(imageeq,-rmax,-self.radcrit,xtol=1e-4)
            self.images = [xA,xB]


    def get_time_delay(self):
        self.timedelay = -self.Dt*(0.5*(self.images[0]**2 - self.images[1]**2) - self.images[0]*self.source + self.images[1]*self.source - self.lenspot(self.images[0]) + self.lenspot(-self.images[1]))

    def make_grids(self,err=0.01,nsig=3.):
        tol = 0.1*err
        x0A = self.images[0] - nsig*err
        x0A = x0A - x0A%tol
        x1A = self.images[0] + nsig*err
        x1A = x1A - x1A%tol
        gridA = np.arange(x0A,x1A,tol)

        x0B = self.images[1] - nsig*err
        x0B = x0B - x0B%tol
        x1B = -max(self.rs/500.,-(self.images[1] + nsig*err))#,self.radcrit)
        x1B = x1B - x1B%tol
        gridB = np.arange(x0B,x1B,tol)

        self.grids = (gridA,gridB)

    def make_rein_grid(self,err=0.01,nsigma=5.):
        tol = 0.1*err
        r0 = max(self.rs/500.,self.rein - nsigma*err)
        r1 = self.rein + nsigma*err
        grid = np.arange(r0,r1,tol)

        self.rein_grid = grid


    def fast_images(self):

        self.images = []
        for grid in self.grids:
            dx = grid[1] - grid[0]
            ys = grid - self.alpha(grid)
            diff = (self.source - ys[:-1])*(self.source - ys[1:])
            found = diff <= 0.
            for x in grid[:-1][found]:
                self.images.append(x+0.5*dx)


    def fast_imA(self):

        grid = self.grids[0]
        dx = grid[1] - grid[0]
        ys = grid - self.alpha(grid)
        diff = (self.source - ys[:-1])*(self.source - ys[1:])
        found = diff <= 0.
        for x in grid[:-1][found]:
            self.imA = x + 0.5*dx

    def fast_imB(self):

        grid = self.grids[1]
        dx = grid[1] - grid[0]
        ys = grid - self.alpha(grid)
        diff = (self.source - ys[:-1])*(self.source - ys[1:])
        found = diff <= 0.
        for x in grid[:-1][found]:
            self.imB = x + 0.5*dx


    def fast_rein(self):

        dr = self.rein_grid[1] - self.rein_grid[0]
        ys = self.rein_grid - self.alpha(self.rein_grid)
        
        rein = abs(ys).argmin()
        
        self.rein = self.rein_grid[rein]


    def get_radmag_ratio(self):
        radmag_A = (1. + self.m(self.images[0])/self.images[0]**2 - 2.*self.kappa(self.images[0]))**(-1)
        radmag_B = (1. + self.m(self.images[1])/self.images[1]**2 - 2.*self.kappa(self.images[1]))**(-1)
        self.radmag_ratio = radmag_A/radmag_B


