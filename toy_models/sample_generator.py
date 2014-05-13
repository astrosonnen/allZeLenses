from allZeLenses import mass_profiles,tools
import os,om10
from om10 import make_twocomp_lenses
import numpy as np
from allZeLenses.mass_profiles import gNFW,sersic,lens_models
from allZeLenses.tools.distances import Dang
from allZeLenses.tools import cgsconstants
from scipy.optimize import brentq
from scipy.misc import derivative
import pymc
import pickle
from tools.statistics import general_random


day = 24.*3600.


def om10_to_spherical_cows(Nlens=1000,maglim=23.3,IQ=0.75):

    db = om10.DB(catalog=os.path.expandvars("$OM10_DIR/data/qso_mock.fits"))
    db.select_random(maglim=23.3,IQ=0.75,Nlens=1000)
    mstars,logreffs = make_twocomp_lenses.assign_stars(db)
    mdms,gammas = make_twocomp_lenses.assign_halos(db,mstars,logreffs)

    lenses = []
    for i in range(0,db.Nlenses):

        reff = 10.**logreffs[i]
        lens = lens_models.spherical_cow(zd=db.sample.ZLENS[i],zs=db.sample.ZSRC[i],mstar=10.**mstars[i],mdm=10.**mdms[i],reff_phys=reff,n=4.,rs_phys=10.*reff,gamma=gammas[i])
        lens.source = (db.sample.XSRC[i]**2 + db.sample.YSRC[i]**2)**0.5

        #finds the radial critical curve and caustic
        lens.normalize()
        lens.get_caustic()

        if lens.caustic > 0.:

            if lens.source > lens.caustic:
                lens.source = np.random.rand(1)*lens.caustic

            #calculate image positions
            lens.get_images()

            if lens.images is not None:
                lenses.append(lens)
 
    return lenses


def fisher_price_sample(Nlens=1000,mmu=11.5,msig=0.3,mdm_0=10.5,mdm_sig=0.1,gamma_0=1.5,gamma_sig=0.1,logreff_0=0.46,mstar_err=0.1,radmagrat_err=0.015,outname='fisher_price_mock_sample.dat'):

    zds = np.random.rand(Nlens)*0.2+0.2

    zss = general_random(lambda z: np.exp(-(np.log(z-0.4))**2),Nlens,(0.5,4.))

    mstars = np.random.normal(mmu,msig,Nlens)
    mstars_meas = mstars + np.random.normal(0.,mstar_err,Nlens)
    radmag_errs = np.random.normal(0.,radmagrat_err,Nlens)

    mdms = mdm_0 + (mstars - 11.5) + np.random.normal(0.,mdm_sig,Nlens)
    gammas = gamma_0 + np.random.normal(0.,gamma_sig,Nlens)
    logreffs = logreff_0 + 0.59*(mstars - 11.) -0.26*(zds - 0.7)
    reffs = 10.**logreffs

    lenses = []
    for i in range(0,Nlens):
        lens = mp.lens_models.spherical_cow(zd=zds[i],zs=zss[i],mstar=10.**mstars[i],mdm=10.**mdms[i],reff_phys = reffs[i],rs_phys=10.*reffs[i],gamma=gammas[i])
        lens.normalize()
        lens.get_caustic()

        ysource = np.random.rand(1)*lens.caustic

        lens.source = ysource
        lens.get_images()
        lens.get_radmag_ratio()

        if lens.images is None:
            df

        lenses.append(lens)

    f = open(outname,'w')
    pickle.dump((lenses,mstars_meas,radmagrat_errs),f)
    f.close()

