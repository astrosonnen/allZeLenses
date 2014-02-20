import os,om10
from om10 import make_twocomp_lenses
import numpy as np
from lens_models import gNFW,sersic
from mass_profiles import gNFW_spheroid,sersic_spheroid
from cosmology.distances import Dang
from cosmology import cgsconstants
from scipy.optimize import brentq
from scipy.misc import derivative

def Sigma_cr(zd,zs,H0=70.,omegaM=0.3,omegaL=0.7):
    #critical density in Solar masses per square arcsecond
    S = c**2/(4*pi*G)*Dang(0.,zs,H0,omegaM,omegaL=omegaL)*Dang(0.,zd,H0,omegaM,omegaL=omegaL)/Dang(zd,zs,H0,omegaM,omegaL=omegaL)/cgsconstants.M_sun*cgsconstants.arcsec2rad**2
    return S

db = om10.DM(catalog=os.path.expandvars("$OM10_DIR/data/qso_mock.fits"))
db.select_random(maglim=23.3,IQ=0.75,Nlens=1000)
mstars,logreffs = make_twocomp_lenses.assign_stars(db)
mdms,gammas = make_twocomp_lenses.assign_halos(db)


#defines the lenses. Calculates image positions.
bulges = []
halos = []
for i in range(0,db.Nlenses):
    arcsec2kpc = arcsec2rad*Dang(db.sample.ZLENS[i])/cgsconstants.kpc
    reff = 10.**logreffs[i]/arcsec2kpc
    S_cr = Sigma_cr(db.sample.ZLENS[i],db.sample.ZSRC[i])
    bulge = sersic.sersic(norm=10.**mstars[i]/S_cr,reff=reff,n=4.)

    rs = 10.*reff
    norm = 10.**mdms[i]/gNFW_spheroid.M3d(reff,rs,gammas[i])/S_cr
    halo = gNFW.gNFW(norm=norm,rs=rs,beta=gammas[i])

    rmin_halo = rs/50.*0.1
    rmin_bulge = 0.01*reff
    rmax_halo = rs/50.*100.
    rmax_bulge = 10.*reff 
    rmin = rmin_halo
    rmax = rmax_bulge

    bulges.append(bulge)
    halos.append(halo)

    #finds the radial critical curve and caustic

    def radial_eigen(r):
                



