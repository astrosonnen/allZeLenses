import numpy as np
from mass_profiles import sersic as sersic_profile, NFW as NFW_profile
from scipy.optimize import brentq
from allZeTools import cgsconstants as cgs
from cosmolopy import distance, density

default_cosmo = {'omega_M_0': 0.3, 'omega_lambda_0': 0.7, 'omega_k_0': 0., 'h': 0.7}

eps = 1e-15


class NfwDev:
    
    def __init__(self, zd=0.3, zs=2., h70=1.0, mstar=1e11, mhalo=1e13, reff_phys=1., cvir=5., images=[], source=0., \
                 obs_images=None, obs_lmstar=None, obs_radmagrat=None, obs_timedelay=None, delta_halo=93.5):

        self.zd = zd
        self.zs = zs
        self.h70 = h70
        self.bulge = None
        self.halo = None
        self.mstar = mstar
        self.mhalo = mhalo
        self.delta_halo = delta_halo
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
        self.imA = None
        self.imB = None
        self.rvir = None
        self.rs = None
        self.gammap = None
        self.obs_lmstar = obs_lmstar
        self.obs_radmagrat = obs_radmagrat
        self.obs_images = obs_images
        self.obs_timedelay = obs_timedelay

        self.ds = distance.angular_diameter_distance(self.zs, **default_cosmo)
        self.dds = distance.angular_diameter_distance(self.zs, self.zd, **default_cosmo)
        self.dd = distance.angular_diameter_distance(self.zd, **default_cosmo)

        self.S_cr = cgs.c**2/(4.*np.pi*cgs.G)*self.ds*self.dd/self.dds*cgs.Mpc/cgs.M_Sun*cgs.arcsec2rad**2
        self.arcsec2kpc = cgs.arcsec2rad*self.dd*1000.
        self.reff = self.reff_phys/self.arcsec2kpc
        self.cvir = cvir
        self.Dt = self.dd*self.ds/self.dds*cgs.Mpc/cgs.c*(1. + self.zd)/cgs.c
        self.rhoc = density.cosmo_densities(**default_cosmo)[0]*distance.e_z(self.zd, **default_cosmo)**2
 
    def normalize(self):
        self.rvir = (self.mhalo*3./self.delta_halo/(4.*np.pi)/self.rhoc)**(1/3.)*1000./self.arcsec2kpc
        self.rs = self.rvir/self.cvir
        self.halo = self.mhalo/NFW_profile.M3d(self.rvir, self.rs)/self.S_cr
        self.bulge = self.mstar/self.S_cr

    def kappa(self, x):
        return self.halo*NFW_profile.Sigma(abs(x), self.rs) + self.bulge*sersic_profile.I(abs(x), 4., self.reff)

    def m(self, x):
        return self.halo*NFW_profile.M2d(abs(x), self.rs)/np.pi + \
               self.bulge*sersic_profile.fast_M2d(abs(x), 4., self.reff)/np.pi

    def lenspot(self, r):
        return self.halo*NFW_profile.lenspot(r, self.rs) + self.bulge*sersic_profile.fast_lenspot(r, 4., self.reff)

    def alpha(self, x):
        r = abs(x)
        return self.halo*NFW_profile.M2d(r, self.rs)/x/np.pi + \
               self.bulge*sersic_profile.fast_M2d(r, 4., self.reff)/x/np.pi

    def get_caustic(self):

        rmin = self.reff/50.
        rmax = 10.*self.reff

        radial_invmag = lambda r: 2.*self.kappa(r) - self.m(r)/r**2 - 1.

        if radial_invmag(rmin)*radial_invmag(rmax) > 0.:
            rcrit = rmin
        else:
            rcrit = brentq(radial_invmag, rmin, rmax)

        ycaust = -(rcrit - self.alpha(rcrit))
        self.caustic = ycaust
        self.radcrit = rcrit

    def get_rein(self):

        tangential_invmag = lambda r: self.m(r)/r**2 - 1.

        # tcrit = brentq(tangential_invmag,-self.images[1], self.images[0])
        tcrit = brentq(tangential_invmag, self.reff/50., 10.*self.reff)

        self.rein = tcrit

    def get_images(self):

        rmin = self.reff/50.
        rmax = 10.*self.reff

        imageeq = lambda r: r - self.alpha(r) - self.source
        if imageeq(self.radcrit)*imageeq(rmax) >= 0. or imageeq(-rmax)*imageeq(-self.radcrit) >= 0.:
            self.images = (-99., 99.)
        else:
            xa = brentq(imageeq, rmin, rmax, xtol=1e-4)
            xb = brentq(imageeq, -rmax, -self.radcrit, xtol=1e-4)
            self.images = (xa, xb)

    def get_timedelay(self):
        self.timedelay = -self.Dt*(0.5*(self.images[0]**2 - self.images[1]**2) - self.images[0]*self.source + \
                                   self.images[1]*self.source - self.lenspot(self.images[0]) + \
                                   self.lenspot(-self.images[1]))/self.h70

    def make_grids(self, err=0.01, nsig=3.):
        tol = 0.1*err
        x0A = self.images[0] - nsig*err
        x0A = x0A - x0A%tol
        x1A = self.images[0] + nsig*err
        x1A = x1A - x1A%tol
        gridA = np.arange(x0A, x1A, tol)

        x0B = self.images[1] - nsig*err
        x0B = x0B - x0B%tol
        x1B = -max(self.rs/500.,-(self.images[1] + nsig*err))#, self.radcrit)
        x1B = x1B - x1B%tol
        gridB = np.arange(x0B,x1B,tol)

        self.grids = (gridA,gridB)

    def make_rein_grid(self,err=0.01,nsigma=5.):
        tol = 0.1*err
        r0 = max(self.rs/500., self.rein - nsigma*err)
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
        self.gammap = 3. - (4.*np.pi*self.reff**3)*(self.mhalo*NFW_profile.rho(self.reff, self.rs)/NFW_profile.M3d(self.rvir, self.rs) + self.mstar*sersic_profile.rho(self.reff,4., self.reff))/(self.mhalo*NFW_profile.M3d(self.reff, self.rs)/NFW_profile.M3d(self.rvir, self.rs) + self.mstar*0.4135)

