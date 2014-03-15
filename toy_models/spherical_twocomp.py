import os,om10
from om10 import make_twocomp_lenses
import numpy as np
from mass_profiles import gNFW,sersic,lens_models
from physics.distances import Dang
from physics import cgsconstants
from scipy.optimize import brentq
from scipy.misc import derivative
import pylab

def Sigma_cr(zd,zs,H0=70.,omegaM=0.3,omegaL=0.7):
    #critical density in Solar masses per square arcsecond
    S = cgsconstants.c**2/(4*np.pi*cgsconstants.G)*Dang(0.,zs,H0,omegaM,omegaL=omegaL)*Dang(0.,zd,H0,omegaM,omegaL=omegaL)/Dang(zd,zs,H0,omegaM,omegaL=omegaL)/cgsconstants.M_Sun*cgsconstants.arcsec2rad**2
    return S

db = om10.DB(catalog=os.path.expandvars("$OM10_DIR/data/qso_mock.fits"))
db.select_random(maglim=23.3,IQ=0.75,Nlens=1000)
mstars,logreffs = make_twocomp_lenses.assign_stars(db)
mdms,gammas = make_twocomp_lenses.assign_halos(db,mstars,logreffs)

count = 0
lenses = []
#defines the lenses. Calculates image positions.
for i in range(0,db.Nlenses):
    arcsec2kpc = cgsconstants.arcsec2rad*Dang(db.sample.ZLENS[i])/cgsconstants.kpc
    reff = 10.**logreffs[i]/arcsec2kpc
    S_cr = Sigma_cr(db.sample.ZLENS[i],db.sample.ZSRC[i])
    #bulge = lens_models.sersic(norm=10.**mstars[i]/S_cr,reff=reff,n=4.)

    rs = 10.*reff
    norm = 10.**mdms[i]/gNFW.M3d(reff,rs,gammas[i])/S_cr
    #halo = lens_models.gNFW(norm=norm,rs=rs,beta=gammas[i])

    rmin_halo = rs/50.*0.1
    rmin_bulge = 0.01*reff
    rmax_halo = rs/50.*100.
    rmax_bulge = 10.*reff 
    rmin = rmin_halo
    rmax = rmax_bulge

    lens = lens_models.spherical_cow(bulge=10.**mstars[i]/S_cr,halo=norm,reff=reff,n=4.,rs=rs,beta=gammas[i])
    #finds the radial critical curve and caustic

    radial_invmag = lambda r: 2.*lens.kappa(r) - lens.m(r)/r**2 - 1.

    if radial_invmag(rmin)*radial_invmag(rmax) > 0.:
        rcrit = rmin
    else:
        rcrit = brentq(radial_invmag,rmin,rmax)

    ycaust = -(rcrit - lens.alpha(rcrit))

    rsrc = (db.sample.XSRC[i]**2 + db.sample.YSRC[i]**2)**0.5

    imageeq = lambda r: float(r - lens.alpha(r) - rsrc)

    if rsrc > ycaust or imageeq(rcrit)*imageeq(rmax) >= 0.:
        print 'source is not multiply imaged'
        rsrc = float(np.random.rand(1)*ycaust)
        count += 1

    if imageeq(rcrit)*imageeq(rmax) >= 0. or ycaust < 0.:
        """
        print 'crap',rmin_bulge,rmin_halo,reff,rs
        bulge = lens_models.spherical_cow(bulge=10.**mstars[i]/S_cr,halo=0.,reff=reff,n=4.,rs=rs,beta=gammas[i])
        halo = lens_models.spherical_cow(bulge=0.,halo=norm,reff=reff,n=4.,rs=rs,beta=gammas[i])
        xs = np.linspace(-5,5,1001)
        #pylab.plot(xs,xs - lens.alpha(xs))
        pylab.plot(xs,xs - bulge.alpha(xs),color='r')
        pylab.plot(xs,xs - halo.alpha(xs),color='g')
        #pylab.scatter((xA,xB),(rsrc,rsrc),color='r')
        pylab.axvline(-rcrit,linestyle=':',color='k')
        pylab.axvline(rcrit,linestyle=':',color='k')
        pylab.axvline(rmin_bulge,color='k')
        pylab.axvline(rmin_halo,color='k')
        pylab.show()

        df
        """
        pass

    else:
        xA = brentq(lambda r: r - lens.alpha(r) - rsrc,rcrit,rmax)
        xB = brentq(lambda r: r - lens.alpha(r) - rsrc,-rcrit,-rmax)

        lens.sources.append(rsrc)
        lens.images.append((xA,xB))

        lenses.append(lens) 

    print rsrc,xA,xB
