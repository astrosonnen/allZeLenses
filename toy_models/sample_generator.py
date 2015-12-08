from allZeLenses import mass_profiles,tools,lens_models
import os,om10
from om10 import make_twocomp_lenses
import numpy as np
from allZeLenses.mass_profiles import gNFW,sersic
from allZeLenses.tools.distances import Dang
from allZeLenses.tools import cgsconstants,statistics
from scipy.optimize import brentq
from scipy.misc import derivative
import pymc
import pickle


day = 24.*3600.

def simple_reality_sample(Nlens=1000,mmu=11.5,msig=0.1,mhalo_0=13.0,mhalo_sig=0.5,mstar_mhalo=0.5,c_sig=0.1,aimf_0=0.,aimf_sig=0.05,logreff_0=0.46,mstar_err=0.1,radmagrat_err=0.015,imerr=0.1,outname='fisher_price_mock_sample.dat'):

    zds = np.random.rand(Nlens)*0.2+0.2

    zss = statistics.general_random(lambda z: np.exp(-(np.log(z-0.4))**2),Nlens,(0.5,4.))

    mhalos = mhalo_0 + np.random.normal(0.,mhalo_sig,Nlens)

    mstars = mmu + mstar_mhalo*(mhalos - 13.) + np.random.normal(0.,msig,Nlens)

    aimfs = np.random.normal(aimf_0,aimf_sig,Nlens)

    mstars_sps = mstars - aimfs

    mstars_meas = mstars_sps + np.random.normal(0.,mstar_err,Nlens)


    radmagrat_errs = np.random.normal(0.,radmagrat_err,Nlens)


    logcvirs = 0.971 - 0.094*(mhalos-12.) + np.random.normal(0.,c_sig,Nlens)
    

    logreffs = logreff_0 + 0.59*(mstars - 11.) -0.26*(zds - 0.7)
    reffs = 10.**logreffs

    lenses = []
    for i in range(0,Nlens):
        lens = lens_models.nfw_deV(zd=zds[i],zs=zss[i],mstar=10.**mstars[i],mhalo=10.**mhalos[i],reff_phys = reffs[i],cvir=10.**logcvirs[i])
        lens.normalize()
        lens.get_caustic()

        ysource = (np.random.rand(1))**0.5*lens.caustic

        lens.source = ysource
        lens.get_images()
        lens.get_radmag_ratio()

        lens.get_rein()

        imerrs = np.random.normal(0.,imerr,2)
        lens.obs_images = ((lens.images[0] + imerrs[0],lens.images[1] + imerrs[1]),imerr)
        lens.obs_lmstar = (mstars_meas[i],mstar_err)
        lens.obs_radmagrat = (lens.radmag_ratio + radmagrat_errs[i],radmagrat_err)

        if lens.images is None:
            df

        lenses.append(lens)

    f = open(outname,'w')
    pickle.dump(lenses,f)
    f.close()


def simple_reality_knownimf_sample(Nlens=1000,mmu=11.5,msig=0.1,mhalo_0=13.0,mhalo_sig=0.5,mstar_mhalo=0.5,c_sig=0.1,logreff_0=0.46,mstar_err=0.1,radmagrat_err=0.015,imerr=0.1,outname='fisher_price_mock_sample.dat'):

    zds = np.random.rand(Nlens)*0.2+0.2

    zss = statistics.general_random(lambda z: np.exp(-(np.log(z-0.4))**2),Nlens,(0.5,4.))

    mhalos = mhalo_0 + np.random.normal(0.,mhalo_sig,Nlens)

    mstars = mmu + mstar_mhalo*(mhalos - 13.) + np.random.normal(0.,msig,Nlens)

    mstars_meas = mstars + np.random.normal(0.,mstar_err,Nlens)


    radmagrat_errs = np.random.normal(0.,radmagrat_err,Nlens)


    logcvirs = 0.971 - 0.094*(mhalos-12.) + np.random.normal(0.,c_sig,Nlens)
    

    logreffs = logreff_0 + 0.59*(mstars - 11.) -0.26*(zds - 0.7)
    reffs = 10.**logreffs

    lenses = []
    for i in range(0,Nlens):
        lens = lens_models.nfw_deV(zd=zds[i],zs=zss[i],mstar=10.**mstars[i],mhalo=10.**mhalos[i],reff_phys = reffs[i],cvir=10.**logcvirs[i])
        lens.normalize()
        lens.get_caustic()

        ysource = (np.random.rand(1))**0.5*lens.caustic

        lens.source = ysource
        lens.get_images()
        lens.get_radmag_ratio()
        print i,lens.images

        lens.get_rein()

        imerrs = np.random.normal(0.,imerr,2)
        lens.obs_images = ((lens.images[0] + imerrs[0],lens.images[1] + imerrs[1]),imerr)
        lens.obs_lmstar = (mstars_meas[i],mstar_err)
        lens.obs_radmagrat = (lens.radmag_ratio + radmagrat_errs[i],radmagrat_err)

        if lens.images is None:
            df

        lenses.append(lens)

    f = open(outname,'w')
    pickle.dump(lenses,f)
    f.close()


def simple_reality_knownimf_fixedc_sample(Nlens=1000,mmu=11.5,msig=0.1,mhalo_0=13.0,mhalo_sig=0.5,mstar_mhalo=0.5,c_sig=0.1,logreff_0=0.46,mstar_err=0.1,radmagrat_err=0.015,imerr=0.1,outname='fisher_price_mock_sample.dat'):

    zds = np.random.rand(Nlens)*0.2+0.2

    zss = statistics.general_random(lambda z: np.exp(-(np.log(z-0.4))**2),Nlens,(0.5,4.))

    mhalos = mhalo_0 + np.random.normal(0.,mhalo_sig,Nlens)

    mstars = mmu + mstar_mhalo*(mhalos - 13.) + np.random.normal(0.,msig,Nlens)

    mstars_meas = mstars + np.random.normal(0.,mstar_err,Nlens)


    radmagrat_errs = np.random.normal(0.,radmagrat_err,Nlens)


    logcvirs = 0.971 - 0.094*(mhalos-12.)
    

    logreffs = logreff_0 + 0.59*(mstars - 11.) -0.26*(zds - 0.7)
    reffs = 10.**logreffs

    lenses = []
    for i in range(0,Nlens):
        lens = lens_models.nfw_deV(zd=zds[i],zs=zss[i],mstar=10.**mstars[i],mhalo=10.**mhalos[i],reff_phys = reffs[i],cvir=10.**logcvirs[i])
        lens.normalize()
        lens.get_caustic()

        ysource = (np.random.rand(1))**0.5*lens.caustic

        lens.source = ysource
        lens.get_images()
        lens.get_radmag_ratio()
        print i,lens.images

        lens.get_rein()

        imerrs = np.random.normal(0.,imerr,2)
        lens.obs_images = ((lens.images[0] + imerrs[0],lens.images[1] + imerrs[1]),imerr)
        lens.obs_lmstar = (mstars_meas[i],mstar_err)
        lens.obs_radmagrat = (lens.radmag_ratio + radmagrat_errs[i],radmagrat_err)

        if lens.images is None:
            df

        lenses.append(lens)

    f = open(outname,'w')
    pickle.dump(lenses,f)
    f.close()


def simple_reality_supersimple_sample(Nlens=1000,mmu=11.5,msig=0.1,mhalo_0=13.0,mhalo_sig=0.5,mstar_mhalo=0.5,c_sig=0.1,logreff_0=0.46,mstar_err=0.1,radmagrat_err=0.015,imerr=0.1,outname='fisher_price_mock_sample.dat'):

    zds = 0.3*np.ones(Nlens)

    zss = 2.*np.ones(Nlens)

    mhalos = mhalo_0 + np.random.normal(0.,mhalo_sig,Nlens)

    mstars = mmu + mstar_mhalo*(mhalos - 13.) + np.random.normal(0.,msig,Nlens)

    mstars_meas = mstars + np.random.normal(0.,mstar_err,Nlens)


    radmagrat_errs = np.random.normal(0.,radmagrat_err,Nlens)


    logcvirs = 0.971 - 0.094*(mhalos-12.)
    

    reffs = 5.*np.ones(Nlens)

    lenses = []
    for i in range(0,Nlens):
        lens = lens_models.nfw_deV(zd=zds[i],zs=zss[i],mstar=10.**mstars[i],mhalo=10.**mhalos[i],reff_phys = reffs[i],cvir=10.**logcvirs[i])
        lens.normalize()
        lens.get_caustic()

        ysource = (np.random.rand(1))**0.5*lens.caustic

        print i,mstars[i],mhalos[i]
        lens.source = ysource
        lens.get_images()
        lens.get_radmag_ratio()

        lens.get_rein()

        imerrs = np.random.normal(0.,imerr,2)
        lens.obs_images = ((lens.images[0] + imerrs[0],lens.images[1] + imerrs[1]),imerr)
        lens.obs_lmstar = (mstars_meas[i],mstar_err)
        lens.obs_radmagrat = (lens.radmag_ratio + radmagrat_errs[i],radmagrat_err)

        if lens.images is None:
            df

        lenses.append(lens)

    f = open(outname,'w')
    pickle.dump(lenses,f)
    f.close()


def simple_reality_knownimf_nomhdep_sample(Nlens=1000,mmu=11.5,msig=0.1,mhalo_0=13.0,mhalo_sig=0.5,c_sig=0.1,logreff_0=0.46,mstar_err=0.1,radmagrat_err=0.015,imerr=0.1,outname='fisher_price_mock_sample.dat'):

    zds = np.random.rand(Nlens)*0.2+0.2

    zss = statistics.general_random(lambda z: np.exp(-(np.log(z-0.4))**2),Nlens,(0.5,4.))

    mhalos = mhalo_0 + np.random.normal(0.,mhalo_sig,Nlens)

    mstars = mmu + np.random.normal(0.,msig,Nlens)

    mstars_meas = mstars + np.random.normal(0.,mstar_err,Nlens)


    radmagrat_errs = np.random.normal(0.,radmagrat_err,Nlens)


    logcvirs = 0.971 - 0.094*(mhalos-12.) + np.random.normal(0.,c_sig,Nlens)
    

    logreffs = logreff_0 + 0.59*(mstars - 11.) -0.26*(zds - 0.7)
    reffs = 10.**logreffs

    lenses = []
    for i in range(0,Nlens):
        lens = lens_models.nfw_deV(zd=zds[i],zs=zss[i],mstar=10.**mstars[i],mhalo=10.**mhalos[i],reff_phys = reffs[i],cvir=10.**logcvirs[i])
        lens.normalize()
        lens.get_caustic()

        ysource = (np.random.rand(1))**0.5*lens.caustic

        lens.source = ysource
        lens.get_images()
        lens.get_radmag_ratio()

        lens.get_rein()

        imerrs = np.random.normal(0.,imerr,2)
        lens.obs_images = ((lens.images[0] + imerrs[0],lens.images[1] + imerrs[1]),imerr)
        lens.obs_lmstar = (mstars_meas[i],mstar_err)
        lens.obs_radmagrat = (lens.radmag_ratio + radmagrat_errs[i],radmagrat_err)

        if lens.images is None:
            df

        lenses.append(lens)

    f = open(outname,'w')
    pickle.dump(lenses,f)
    f.close()



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


def fisher_price_sample(Nlens=1000,mmu=11.5,msig=0.3,mdm5_0=10.8,mdm5_sig=0.1,mdm_mstar=0.,gamma_0=1.0,gamma_sig=0.1,logreff_0=0.46,mstar_err=0.1,radmagrat_err=0.015,outname='fisher_price_mock_sample.dat'):

    zds = np.random.rand(Nlens)*0.2+0.2

    zss = statistics.general_random(lambda z: np.exp(-(np.log(z-0.4))**2),Nlens,(0.5,4.))

    mstars = np.random.normal(mmu,msig,Nlens)
    mstars_meas = mstars + np.random.normal(0.,mstar_err,Nlens)
    radmagrat_errs = np.random.normal(0.,radmagrat_err,Nlens)

    mdms = mdm5_0 + mdm_mstar*(mstars - 11.5) + np.random.normal(0.,mdm5_sig,Nlens)
    gammas = gamma_0 + np.random.normal(0.,gamma_sig,Nlens)
    logreffs = logreff_0 + 0.59*(mstars - 11.) -0.26*(zds - 0.7)
    reffs = 10.**logreffs

    lenses = []
    mstar_sample = []
    radmagrat_sample = []
    for i in range(0,Nlens):
        lens = lens_models.spherical_cow(zd=zds[i],zs=zss[i],mstar=10.**mstars[i],mdm5=10.**mdms[i],reff_phys = reffs[i],rs_phys=10.*reffs[i],gamma=gammas[i])
        lens.normalize()
        lens.get_caustic()

        ysource = (np.random.rand(1))**0.5*lens.caustic

        lens.source = ysource
        lens.get_images()
        lens.get_radmag_ratio()

        if lens.images is None:
            df

        lenses.append(lens)
        mstar_sample.append((mstars_meas[i],mstar_err))
        radmagrat_sample.append((lens.radmag_ratio + radmagrat_errs[i],radmagrat_err))

    f = open(outname,'w')
    pickle.dump((lenses,mstar_sample,radmagrat_sample),f)
    f.close()



def simple_powerlaw_sample(Nlens=1000,gmu=2.1,gsig=0.18,remu=1.5,resig=0.3,radmagrat_err=0.015,outname='simple_powerlaw.dat'):

    reins = np.random.normal(remu,resig,Nlens)
    gammas = np.random.normal(gmu,gsig,Nlens)

    lenses = []
    radmagrat_sample = []
    for i in range(0,Nlens):
        lens = lens_models.sps(zd=0.3,zs=1.,gamma=gammas[i])
        lens.normalize()
        lens.get_caustic()

        ysource = np.random.rand(1)*lens.caustic

        lens.source = ysource
        lens.get_images()
        lens.get_radmag_ratio()

        if lens.images is None:
            df

        lenses.append(lens)
        mstar_sample.append((mstars_meas[i],mstar_err))
        radmagrat_sample.append((lens.radmag_ratio + radmagrat_errs[i],radmagrat_err))

    f = open(outname,'w')
    pickle.dump((lenses,mstar_sample,radmagrat_sample),f)
    f.close()


def very_simple(Nlens=1000,mmu=11.5,msig=0.3,mdm5_0=10.8,mdm5_sig=0.1,logreff_0=0.46,mstar_err=0.1,radmagrat_err=0.015,outname='very_simple_mock_sample.dat'):

    #zds = np.random.rand(Nlens)*0.2+0.2
    zds = 0.3*np.ones(Nlens)
    zss = 1.5*np.ones(Nlens)

    #zss = statistics.general_random(lambda z: np.exp(-(np.log(z-0.4))**2),Nlens,(0.5,4.))

    mstars = np.random.normal(mmu,msig,Nlens)
    mstars_meas = mstars + np.random.normal(0.,mstar_err,Nlens)
    radmagrat_errs = np.random.normal(0.,radmagrat_err,Nlens)

    mdms = mdm5_0 + np.random.normal(0.,mdm5_sig,Nlens)

    logreffs = logreff_0 + 0.59*(mstars - 11.) -0.26*(zds - 0.7)
    reffs = 10.**logreffs

    lenses = []
    mstar_sample = []
    radmagrat_sample = []
    for i in range(0,Nlens):
        lens = lens_models.spherical_cow(zd=zds[i],zs=zss[i],mstar=10.**mstars[i],mdm5=10.**mdms[i],reff_phys = reffs[i],rs_phys=10.*reffs[i],gamma=1.)
        lens.normalize()
        lens.get_caustic()

        ysource = (np.random.rand(1))**0.5*lens.caustic

        lens.source = ysource
        lens.get_images()
        lens.get_rein()
        lens.get_radmag_ratio()

        if lens.images is None:
            df

        lenses.append(lens)
        mstar_sample.append((mstars_meas[i],mstar_err))
        radmagrat_sample.append((lens.radmag_ratio + radmagrat_errs[i],radmagrat_err))

    f = open(outname,'w')
    pickle.dump((lenses,mstar_sample,radmagrat_sample),f)
    f.close()



