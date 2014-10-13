from allZeLenses import mass_profiles,physics
import os,om10
from om10 import make_twocomp_lenses
import numpy as np
from allZeLenses.mass_profiles import gNFW,sersic,lens_models
from allZeLenses.physics.distances import Dang
from allZeLenses.physics import cgsconstants
from scipy.optimize import brentq
from scipy.misc import derivative
import pymc
import pickle

day = 24.*3600.


def make_sample(Nlens=1000,maglim=23.3,IQ=0.75):

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

def fit_sps_exactrein(lens,radmagrat_meas,N=11000,burnin=1000,gammaup=2.8):

    model_lens = lens_models.sps(zd=lens.zd,zs=lens.zs,rein=1.,gamma=2.,kext=lens.kext,images=lens.images)
    model_lens.normalize()

    xA,xB = lens.images

    gamma_var = pymc.Uniform('gamma',lower=1.01,upper=gammaup,value=model_lens.gamma)
    #kext_var = pymc.Uniform('kext',lower=-0.1,upper=0.1,value=lens.kext)


    @pymc.deterministic()
    def timedelay(gamma=gamma_var):#,kext=kext_var):

        model_lens.gamma = gamma
        #model_lens.kext = kext

        norm = (xA - xB)/(model_lens.alpha(xA) - model_lens.alpha(xB))
        model_lens.norm *= norm

        model_lens.source = xA - model_lens.alpha(xA)

        model_lens.get_time_delay()
        return float(model_lens.timedelay)


    @pymc.deterministic()
    def radmag_ratio(gamma=gamma_var):#,kext=kext_var):

        model_lens.gamma = gamma
        #model_lens.kext = kext

        norm = (xA - xB)/(model_lens.alpha(xA) - model_lens.alpha(xB))
        model_lens.norm *= norm

        model_lens.source = xA - model_lens.alpha(xA)

        model_lens.get_radmag_ratio()
        return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def like(gamma=gamma_var):#,kext=kext_var):
        model_lens.gamma = gamma
        #model_lens.kext = kext

        model_lens.normalize()

        return -0.5*(float(radmag_ratio) - radmagrat_meas[0])**2/radmagrat_meas[1]**2

        
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,gamma=gamma_var):#,kext=kext_var):
        return like

    pars = [gamma_var,timedelay,like,radmag_ratio]#,kext_var]

    M = pymc.MCMC(pars)
    M.isample(N,burnin)

    outdic = {'gamma':M.trace('gamma')[:],'timedelay':M.trace('timedelay')[:],'logp':M.trace('like')[:],'radmag_ratio':M.trace('radmag_ratio')[:]}#,'kext':M.trace('kext')[:]}

    return outdic


def fit_spherical_cow_exactrein(lens,mstar_meas,radmagrat_meas,N=11000,burnin=1000,gammaup=2.2,imerr=0.1,thin=1): #fits a spherical cow model to image position and stellar mass data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mdm=lens.mdm,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)
    model_bulge = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=1.,mdm=0.,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=0.,images=lens.images,source=lens.source)
    model_halo = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=0.,mdm=1.,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)

    model_lens.normalize()
    model_bulge.normalize()
    model_halo.normalize()

    xA,xB = lens.images

    mstar_max = (xA - xB)/(model_bulge.alpha(xA) - model_bulge.alpha(xB))

    mdm_var = pymc.Uniform('mdm',lower=9.,upper=12.5,value=np.log10(lens.mdm))
    gamma_var = pymc.Uniform('gamma',lower=0.2,upper=gammaup,value=lens.gamma)


    @pymc.deterministic()
    def mstar(mdm=mdm_var,gamma=gamma_var):

        model_halo.mdm = 10.**mdm
        model_halo.gamma = gamma
        model_halo.normalize()

        mmstar = ((xA - xB) - (model_halo.alpha(xA) - model_halo.alpha(xB)))/(model_bulge.alpha(xA) - model_bulge.alpha(xB))

        return mmstar

    @pymc.deterministic()
    def timedelay(mdm=mdm_var,gamma=gamma_var):

        model_lens.mstar = float(mstar)
        model_lens.gamma = gamma
        model_lens.mdm = 10.**mdm

        model_lens.normalize()

        model_lens.source = xA - model_lens.alpha(xA)

        model_lens.get_time_delay()
        return float(model_lens.timedelay)


    @pymc.deterministic()
    def radmag_ratio(mdm=mdm_var,gamma=gamma_var):

        model_lens.mstar = float(mstar)
        model_lens.gamma = gamma
        model_lens.mdm = 10.**mdm

        model_lens.normalize()

        model_lens.source = xA - model_lens.alpha(xA)

        model_lens.get_radmag_ratio()
        return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def like(mdm=mdm_var,gamma=gamma_var):
        model_lens.mstar = float(mstar)
        model_lens.gamma = gamma
        model_lens.mdm = 10.**mdm

        model_lens.normalize()

        if float(mstar) < 0.:
            return -1e300
        else:
            return 1.#-0.5*(np.log10(float(mstar)) - mstar_meas[0])**2/mstar_meas[1]**2 #- 0.5*(float(radmag_ratio) - radmagrat_meas[0])**2/radmagrat_meas[1]**2

        
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mdm=mdm_var,gamma=gamma_var):
        return like

    pars = [mdm_var,gamma_var,mstar,timedelay,like,radmag_ratio]

    M = pymc.MCMC(pars)
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':np.log10(M.trace('mstar')[:]),'mdm':M.trace('mdm')[:],'gamma':M.trace('gamma')[:],'timedelay':M.trace('timedelay')[:],'logp':M.trace('like')[:],'radmag_ratio':M.trace('radmag_ratio')[:]}

    return outdic


def fit_spherical_cow_wkext(lens,mstar_meas,radmagrat_meas,N=11000,burnin=1000,gammaup=2.2,imerr=0.1,thin=1): #fits a spherical cow model to image position and stellar mass data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mdm=lens.mdm,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)
    model_bulge = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=1.,mdm=0.,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=0.,images=lens.images,source=lens.source)
    model_halo = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=0.,mdm=1.,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)

    model_lens.normalize()
    model_bulge.normalize()
    model_halo.normalize()

    xA,xB = lens.images

    mstar_max = (xA - xB)/(model_bulge.alpha(xA) - model_bulge.alpha(xB))

    mdm_var = pymc.Uniform('mdm',lower=9.,upper=12.5,value=np.log10(lens.mdm))
    gamma_var = pymc.Uniform('gamma',lower=0.2,upper=gammaup,value=lens.gamma)
    kext_var = pymc.Uniform('kext',lower=-0.1,upper=0.1,value=lens.kext)


    @pymc.deterministic()
    def mstar(mdm=mdm_var,gamma=gamma_var,kext=kext_var):

        model_halo.mdm = 10.**mdm
        model_halo.gamma = gamma
        model_halo.kext = kext
        model_halo.normalize()

        mmstar = ((xA - xB) - (model_halo.alpha(xA) - model_halo.alpha(xB)))/(model_bulge.alpha(xA) - model_bulge.alpha(xB))

        return mmstar

    @pymc.deterministic()
    def timedelay(mdm=mdm_var,gamma=gamma_var,kext=kext_var):

        model_lens.mstar = float(mstar)
        model_lens.gamma = gamma
        model_lens.mdm = 10.**mdm
        model_lens.kext = kext

        model_lens.normalize()

        model_lens.source = xA - model_lens.alpha(xA)

        model_lens.get_time_delay()
        return float(model_lens.timedelay)


    @pymc.deterministic()
    def radmag_ratio(mdm=mdm_var,gamma=gamma_var,kext=kext_var):

        model_lens.mstar = float(mstar)
        model_lens.gamma = gamma
        model_lens.mdm = 10.**mdm
        model_lens.kext = kext

        model_lens.normalize()

        model_lens.source = xA - model_lens.alpha(xA)

        model_lens.get_radmag_ratio()
        return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def like(mdm=mdm_var,gamma=gamma_var,kext=kext_var):
        model_lens.mstar = float(mstar)
        model_lens.gamma = gamma
        model_lens.mdm = 10.**mdm
        model_lens.kext = kext

        model_lens.normalize()

        if float(mstar) < 0.:
            return -1e300
        else:
            return -0.5*(np.log10(float(mstar)) - mstar_meas[0])**2/mstar_meas[1]**2 - 0.5*(float(radmag_ratio) - radmagrat_meas[0])**2/radmagrat_meas[1]**2

        
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mdm=mdm_var,gamma=gamma_var,kext=kext_var):
        return like

    pars = [mdm_var,gamma_var,mstar,timedelay,like,radmag_ratio,kext_var]

    M = pymc.MCMC(pars)
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':np.log10(M.trace('mstar')[:]),'mdm':M.trace('mdm')[:],'gamma':M.trace('gamma')[:],'timedelay':M.trace('timedelay')[:],'logp':M.trace('like')[:],'radmag_ratio':M.trace('radmag_ratio')[:],'kext':M.trace('kext')[:]}

    return outdic



def slow_fit_spherical_cow(lens,mstar_meas,N=11000,burnin=1000,gammaup=2.2,imerr=0.1,thin=1): #fits a spherical cow model to image position and stellar mass data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mdm=lens.mdm,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)
    model_bulge = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=1.,mdm=0.,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)
    model_halo = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=0.,mdm=1.,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)

    model_lens.normalize()
    model_bulge.normalize()
    model_halo.normalize()

    xA,xB = lens.images

    model_lens.get_caustic()

    mstar_max = (xA - xB)/(model_bulge.alpha(xA) - model_bulge.alpha(xB))
    mdm_max = (xA - xB)/(model_halo.alpha(xA) - model_halo.alpha(xB))

    mstar_var = pymc.Uniform('lmstar',lower=min(lens.mstar,mstar_meas[0]) - 5*mstar_meas[1],upper=np.log10(mstar_max)+0.1,value=np.log10(lens.mstar))
    gamma_var = pymc.Uniform('gamma',lower=0.2,upper=gammaup,value=lens.gamma)
    mdm_var = pymc.Uniform('mdm',lower=min(9.5,np.log10(lens.mdm)-0.2),upper=np.log10(mdm_max)+0.1,value=np.log10(lens.mdm))
    s_var = pymc.Uniform('s',lower=0.,upper=2.*lens.caustic,value=lens.source)

    @pymc.deterministic()
    def images(mstar=mstar_var,gamma=gamma_var,mdm=mdm_var,s=s_var):

        model_lens.source = s
        model_lens.mstar = 10.**mstar
        model_lens.gamma = gamma
        model_lens.mdm = 10.**mdm

        model_lens.normalize()

        model_lens.get_images()

        return model_lens.images


    @pymc.deterministic()
    def timedelay(mstar=mstar_var,gamma=gamma_var,mdm=mdm_var,s=s_var,imgs=images):

        model_lens.source = s
        model_lens.mstar = 10.**mstar
        model_lens.gamma = gamma
        model_lens.mdm = 10.**mdm

        model_lens.normalize()

        if len(imgs) < 2:
            return 0.
        else:
            model_lens.images = imgs
            model_lens.get_time_delay()
            return float(model_lens.timedelay)


    @pymc.deterministic()
    def like(mstar=mstar_var,gamma=gamma_var,mdm=mdm_var,s=s_var,imgs=images):

        if len(imgs) < 2:
            return -1e300
        else:
            imA = imgs[0]
            imB = imgs[1]
            loglike = -0.5*(imA - xA)**2/imerr**2 - 0.5*(imB - xB)**2/imerr**2
            loglike += -0.5*(mstar - mstar_meas[0])**2/mstar_meas[1]**2
            return loglike

        
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mstar=mstar_var,gamma=gamma_var,mdm=mdm_var,s=s_var,imgs=images):
        return like

    pars = [mstar_var,gamma_var,mdm_var,s_var,timedelay,like,images]

    M = pymc.MCMC(pars)
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mdm':M.trace('mdm')[:],'gamma':M.trace('gamma')[:],'timedelay':M.trace('timedelay')[:],'logp':M.trace('like')[:],'source':M.trace('s')[:],'images':M.trace('images')[:]}

    return outdic



def fit_spherical_cow(lens,mstar_meas,N=11000,burnin=1000,gammaup=2.2,imerr=0.1,thin=1): #fits a spherical cow model to image position and stellar mass data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mdm=lens.mdm,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)
    model_bulge = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=1.,mdm=0.,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)
    model_halo = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=0.,mdm=1.,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)

    model_lens.normalize()
    model_bulge.normalize()
    model_halo.normalize()

    xA,xB = lens.images

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr)


    mstar_max = (xA - xB)/(model_bulge.alpha(xA) - model_bulge.alpha(xB))
    mdm_max = (xA - xB)/(model_halo.alpha(xA) - model_halo.alpha(xB))

    mstar_var = pymc.Uniform('lmstar',lower=min(lens.mstar,mstar_meas[0]) - 5*mstar_meas[1],upper=np.log10(mstar_max)+0.1,value=np.log10(lens.mstar))
    gamma_var = pymc.Uniform('gamma',lower=0.2,upper=gammaup,value=lens.gamma)
    mdm_var = pymc.Uniform('mdm',lower=min(9.5,np.log10(lens.mdm)-0.2),upper=np.log10(mdm_max)+0.1,value=np.log10(lens.mdm))
    s_var = pymc.Uniform('s',lower=0.,upper=2.*lens.caustic,value=lens.source)

    @pymc.deterministic()
    def images(mstar=mstar_var,gamma=gamma_var,mdm=mdm_var,s=s_var):

        model_lens.source = s
        model_lens.mstar = 10.**mstar
        model_lens.gamma = gamma
        model_lens.mdm = 10.**mdm

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            return model_lens.images


    @pymc.deterministic()
    def timedelay(mstar=mstar_var,gamma=gamma_var,mdm=mdm_var,s=s_var):

        model_lens.source = s
        model_lens.mstar = 10.**mstar
        model_lens.gamma = gamma
        model_lens.mdm = 10.**mdm

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            model_lens.get_time_delay()
            return float(model_lens.timedelay)


    @pymc.deterministic()
    def like(mstar=mstar_var,gamma=gamma_var,mdm=mdm_var,s=s_var):
        model_lens.mstar = 10.**mstar
        model_lens.gamma = gamma
        model_lens.mdm = 10.**mdm
        model_lens.source = s

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images) < 2:
            return -1e300
        else:
            imA = model_lens.images[0]
            imB = model_lens.images[1]
            loglike = -0.5*(imA - xA)**2/imerr**2 - 0.5*(imB - xB)**2/imerr**2
            return loglike -0.5*(mstar - mstar_meas[0])**2/mstar_meas[1]**2

        
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mstar=mstar_var,gamma=gamma_var,mdm=mdm_var,s=s_var):
        return like

    pars = [mstar_var,gamma_var,mdm_var,s_var,timedelay,like,images]

    M = pymc.MCMC(pars)
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mdm':M.trace('mdm')[:],'gamma':M.trace('gamma')[:],'timedelay':M.trace('timedelay')[:],'logp':M.trace('like')[:],'source':M.trace('s')[:],'images':M.trace('images')[:]}

    return outdic



def hierarchical_inference(lenses,deltaT=5.,Nlens=100,N=11000,burnin=1000,outname=None):

    chaindir = '/home/sonnen/allZeLenses/mcmc_chains/'
    toyname = 'toy0'

    logS0 = 11.5 - np.log10(5.)

    chains = []
    t_meas = []
    reffs = []
    zd = []

    for i in range(0,Nlens):
        f = open(chaindir+toyname+'_lens_%03d.dat'%i,'r')
        chain = pickle.load(f)
        f.close()

        chain['timedelay'] /= day
        if chain['timedelay'].mean() != chain['timedelay'].mean():
            print i
            df

        chains.append(chain)

        lens = lenses[i]
        lens.get_time_delay()

        time_meas = (lens.timedelay/day + np.random.normal(0.,deltaT,1),deltaT)

        t_meas.append(time_meas)

        zd.append(lens.zd)
        reffs.append(np.log10(lens.reff_phys))


    #defines the hyper-parameters

    cvar = pymc.Uniform('c',lower=0.,upper=2,value=1.3)
    gmstar = pymc.Uniform('gmstar',lower=-2.,upper=3.,value=0.0)
    gsstar = pymc.Uniform('gsstar',lower=-2.,upper=3.,value=0.0)
    scatterg = pymc.Uniform('sg',lower=0.,upper=2.,value=0.1)

    fvar = pymc.Uniform('f',lower=8.,upper=13.,value=10.5)
    mmstar = pymc.Uniform('mmstar',lower=-2.,upper=2.,value=0.6)
    msstar = pymc.Uniform('msstar',lower=-2.,upper=2.,value=-1.0)
    scatterm = pymc.Uniform('sm',lower=0.,upper=3.,value=0.2)

    mmu = pymc.Uniform('mmu',lower=10.,upper=13.,value=11.5)
    msig = pymc.Uniform('msig',lower=0.,upper=2.,value=0.3)
    mz = pymc.Uniform('mz',lower=-2,upper=2.,value=0.)

    rmu = pymc.Uniform('rmu',lower=0.,upper=1.,value=0.5)
    rsig = pymc.Uniform('rsig',lower=0.,upper=1.,value=0.21)
    rz = pymc.Uniform('rz',lower=-1,upper=1.,value=-0.26)
    rm = pymc.Uniform('rm',lower=0.,upper=1.,value=0.59)

    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    pars = [cvar,scatterg,fvar,scatterm,mmu,msig,mz,gmstar,mmstar,gsstar,msstar,H0,rmu,rsig,rz,rm]


    @pymc.deterministic(name='likeall')
    def likeall(c=cvar,s1=scatterg,f=fvar,s2=scatterm,mmu=mmu,msig=msig,mz=mz,gmstar=gmstar,mmstar=mmstar,gsstar=gsstar,msstar=msstar,H0=H0,rmu=rmu,rsig=rsig,rz=rz,rm=rm):

        totlike = 0.

        for i in range(0,Nlens):
            gmodel = gmstar*(chains[i]['mstar'] - 11.5) + gsstar*(chains[i]['mstar'] - 2.*reffs[i] - logS0) + c
            mmodel = mmstar*(chains[i]['mstar'] - 11.5) + msstar*(chains[i]['mstar'] - 2.*reffs[i] - logS0) + f
            mglob_model = mz*(zd[i] - 0.5) + mmu
            rglob_model = rz*(zd[i] - 0.5) + rmu + rm*(chains[i]['mstar'] - 11.5)

            gexp = 1./s1*np.exp(-(gmodel - chains[i]['gamma'])**2/(2.*s1**2))
            mexp = 1./s2*np.exp(-(mmodel - chains[i]['mdm'])**2/(2.*s2**2))
            msexp = 1./msig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*msig**2))
            rsexp = 1./rsig*np.exp(-(rglob_model - reffs[i])**2/(2.*rsig**2))

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))

            totlike += np.log((gexp*mexp*msexp*rsexp*dtexp).mean())

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,c=cvar,s1=scatterg,f=fvar,s2=scatterm,mmu=mmu,msig=msig,mz=mz,gmstar=gmstar,mmstar=mmstar,gsstar=gsstar,msstar=msstar,H0=H0,rmu=rmu,rsig=rsig,rz=rz,rm=rm):
        return likeall
     
    M = pymc.MCMC(pars+[likeall])
    #M.use_step_method(pymc.AdaptiveMetropolis,pars,cov=diag(0.0000001*ones(len(pars))))
    M.use_step_method(pymc.AdaptiveMetropolis,pars)
    M.isample(N,burnin)

    outdic = {}
    for par in pars:
        outdic[str(par)] = M.trace(par)[:]
    outdic['logp'] = M.trace('likeall')[:]

    if outname is None:
        outname = chaindir+toyname+'_hierarch.dat'

    f = open(outname,'w')
    pickle.dump(outdic,f)
    f.close()


def hierarchical_wkext(lenses,deltaT=5.,Nlens=100,N=11000,burnin=1000,outname=None):

    from scipy.special import gamma as gfunc,gammainc

    chaindir = '/home/sonnen/allZeLenses/mcmc_chains/'
    toyname = 'kext'

    chains = []
    t_meas = []
    reffs = []
    zd = []

    if type(Nlens) == type(1):
        lenslim = (0,Nlens)
    else:
        lenslim = Nlens
        Nlens = lenslim[1] - lenslim[0]

    print 'sampling lenses %d to %d'%lenslim

    for i in range(lenslim[0],lenslim[1]):
        f = open(chaindir+toyname+'_lens_%03d.dat'%i,'r')
        chain = pickle.load(f)
        f.close()

        for par in chain:
            chain[par] = chain[par].flatten()

        chain['timedelay'] /= day
        if chain['timedelay'].mean() != chain['timedelay'].mean():
            print i
            df

        chains.append(chain)

        lens = lenses[i]
        lens.get_time_delay()

        time_meas = (lens.timedelay/day + np.random.normal(0.,deltaT,1),deltaT)
        #time_meas = (lens.timedelay/day*(1. + np.random.normal(0.,0.05,1)),deltaT)

        t_meas.append(time_meas)

        zd.append(lens.zd)
        reffs.append(np.log10(lens.reff_phys))


    #defines the hyper-parameters

    cvar = pymc.Uniform('c',lower=0.,upper=2,value=1.5)
    scatterg = pymc.Uniform('sg',lower=0.,upper=2.,value=0.1)

    fvar = pymc.Uniform('f',lower=8.,upper=13.,value=10.5)
    mmstar = pymc.Uniform('mmstar',lower=-2.,upper=2.,value=1.0)
    scatterm = pymc.Uniform('sm',lower=0.,upper=3.,value=0.1)

    mmu = pymc.Uniform('mmu',lower=10.,upper=13.,value=11.5)
    msig = pymc.Uniform('msig',lower=0.,upper=2.,value=0.3)

    #kextk_var = pymc.Uniform('kk',lower=0.,upper=10.,value=3.)
    #theta_var = pymc.Uniform('theta',lower=0.,upper=1.,value=0.03)

    kmu = pymc.Uniform('kmu',lower=-0.1,upper=0.1,value=0.)
    ksig = pymc.Uniform('ksig',lower=0.,upper=1.,value=0.03)

    """
    rmu = pymc.Uniform('rmu',lower=0.,upper=1.,value=0.5)
    rsig = pymc.Uniform('rsig',lower=0.,upper=1.,value=0.21)
    rz = pymc.Uniform('rz',lower=-1,upper=1.,value=-0.26)
    rm = pymc.Uniform('rm',lower=0.,upper=1.,value=0.59)
    """

    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    #pars = [cvar,scatterg,fvar,scatterm,mmu,msig,mmstar,H0,kextk_var,theta_var]
    pars = [cvar,scatterg,fvar,scatterm,mmu,msig,mmstar,H0,kmu,ksig]


    @pymc.deterministic(name='likeall')
    #def likeall(c=cvar,s1=scatterg,f=fvar,s2=scatterm,mmu=mmu,msig=msig,mmstar=mmstar,H0=H0,kextk=kextk_var,theta=theta_var):
    def likeall(c=cvar,s1=scatterg,f=fvar,s2=scatterm,mmu=mmu,msig=msig,mmstar=mmstar,H0=H0,kmu=kmu,ksig=ksig):

        totlike = 0.

        #knorm = gammainc(kextk,0.2/theta)

        for i in range(0,Nlens):
            gmodel = c
            mmodel = mmstar*(chains[i]['mstar'] - 11.5) + f
            mglob_model = mmu
            #rglob_model = rz*(zd[i] - 0.5) + rmu + rm*(chains[i]['mstar'] - 11.5)

            gexp = 1./s1*np.exp(-(gmodel - chains[i]['gamma'])**2/(2.*s1**2))
            mexp = 1./s2*np.exp(-(mmodel - chains[i]['mdm'])**2/(2.*s2**2))
            msexp = 1./msig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*msig**2))
            #rsexp = 1./rsig*np.exp(-(rglob_model - reffs[i])**2/(2.*rsig**2))

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))
            #kfunc = 1./gfunc(kextk)/theta**kextk*(chains[i]['kext'] + 0.1)**(kextk - 1.)*np.exp(-(chains[i]['kext']+0.1)/theta)/knorm
            kfunc = 1./ksig*np.exp(-0.5*(chains[i]['kext'] - kmu)**2/ksig**2)

            #totlike += np.log((gexp*mexp*msexp).mean())
            totlike += np.log((gexp*mexp*msexp*dtexp*kfunc).mean())

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    #def logp(value=0.,c=cvar,s1=scatterg,f=fvar,s2=scatterm,mmu=mmu,msig=msig,mmstar=mmstar,H0=H0,kextk=kextk_var,theta=theta_var):
    def logp(value=0.,c=cvar,s1=scatterg,f=fvar,s2=scatterm,mmu=mmu,msig=msig,mmstar=mmstar,H0=H0,kmu=kmu,ksig=ksig):
        return likeall
     
    M = pymc.MCMC(pars+[likeall])
    #M.use_step_method(pymc.AdaptiveMetropolis,pars,cov=diag(0.0000001*ones(len(pars))))
    M.use_step_method(pymc.AdaptiveMetropolis,pars)
    M.isample(N,burnin)

    outdic = {}
    for par in pars:
        outdic[str(par)] = M.trace(par)[:]
    outdic['logp'] = M.trace('likeall')[:]

    if outname is None:
        outname = chaindir+toyname+'_hierarch_%d-%d.dat'%(lenslim[0],lenslim[1]-1)

    f = open(outname,'w')
    pickle.dump(outdic,f)
    f.close()


def hierarchical_strong_prior(lenses,deltaT=5.,Nlens=100,N=11000,burnin=1000,outname=None):

    from scipy.special import gamma as gfunc,gammainc

    chaindir = '/home/sonnen/allZeLenses/mcmc_chains/'
    toyname = 'fisher_price'

    chains = []
    t_meas = []
    reffs = []
    zd = []

    if type(Nlens) == type(1):
        lenslim = (0,Nlens)
    else:
        lenslim = Nlens
        Nlens = lenslim[1] - lenslim[0]

    print 'sampling lenses %d to %d'%lenslim

    for i in range(lenslim[0],lenslim[1]):
        f = open(chaindir+toyname+'_lens_%03d.dat'%i,'r')
        chain = pickle.load(f)
        f.close()

        for par in chain:
            chain[par] = chain[par].flatten()

        chain['timedelay'] /= day
        if chain['timedelay'].mean() != chain['timedelay'].mean():
            print i
            df

        chains.append(chain)

        lens = lenses[i]
        lens.get_time_delay()

        time_meas = (lens.timedelay/day + np.random.normal(0.,deltaT,1),deltaT)
        #time_meas = (lens.timedelay/day*(1. + np.random.normal(0.,0.05,1)),deltaT)

        t_meas.append(time_meas)

        zd.append(lens.zd)
        reffs.append(np.log10(lens.reff_phys))


    #defines the hyper-parameters

    c = 1.5
    s1 = 0.1
    f = 10.5
    mmstar = 1.
    s2 = 0.1

    mmu = pymc.Uniform('mmu',lower=10.,upper=13.,value=11.5)
    msig = pymc.Uniform('msig',lower=0.,upper=2.,value=0.3)

    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    pars = [mmu,msig,H0]


    @pymc.deterministic(name='likeall')
    def likeall(mmu=mmu,msig=msig,H0=H0):

        totlike = 0.

        for i in range(0,Nlens):
            gmodel = c
            mmodel = mmstar*(chains[i]['mstar'] - 11.5) + f
            mglob_model = mmu

            gexp = 1./s1*np.exp(-(gmodel - chains[i]['gamma'])**2/(2.*s1**2))
            mexp = 1./s2*np.exp(-(mmodel - chains[i]['mdm'])**2/(2.*s2**2))

            msexp = 1./msig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*msig**2))

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))

            totlike += np.log((gexp*mexp*msexp*dtexp).mean())

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mmu=mmu,msig=msig,H0=H0):
        return likeall
     
    M = pymc.MCMC(pars+[likeall])
    M.use_step_method(pymc.AdaptiveMetropolis,pars)
    M.isample(N,burnin)

    outdic = {}
    for par in pars:
        outdic[str(par)] = M.trace(par)[:]
    outdic['logp'] = M.trace('likeall')[:]

    if outname is None:
        outname = chaindir+toyname+'_hierarch_%d-%d.dat'%(lenslim[0],lenslim[1]-1)

    f = open(outname,'w')
    pickle.dump(outdic,f)
    f.close()


def hierarchical_fisher_price(lenses,lensname='fisher_price_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,outname=None):

    from scipy.special import gamma as gfunc,gammainc

    chaindir = '/home/sonnen/allZeLenses/mcmc_chains/'

    chains = []
    t_meas = []
    reffs = []
    zd = []

    if type(Nlens) == type(1):
        lenslim = (0,Nlens)
    else:
        lenslim = Nlens
        Nlens = lenslim[1] - lenslim[0]

    print 'sampling lenses %d to %d'%lenslim

    for i in range(lenslim[0],lenslim[1]):
        f = open(chaindir+lensname+'_%03d.dat'%i,'r')
        chain = pickle.load(f)
        f.close()

        for par in chain:
            chain[par] = chain[par].flatten()

        chain['timedelay'] /= day
        if chain['timedelay'].mean() != chain['timedelay'].mean():
            print i
            df

        chains.append(chain)

        lens = lenses[i]
        lens.get_time_delay()

        time_meas = (lens.timedelay/day + np.random.normal(0.,deltaT,1),deltaT)
        #time_meas = (lens.timedelay/day*(1. + np.random.normal(0.,0.05,1)),deltaT)

        t_meas.append(time_meas)

        zd.append(lens.zd)
        reffs.append(np.log10(lens.reff_phys))


    #defines the hyper-parameters

    cvar = pymc.Uniform('c',lower=0.,upper=2,value=1.0)
    scatterg = pymc.Uniform('sg',lower=0.,upper=2.,value=0.1)

    fvar = pymc.Uniform('f',lower=8.,upper=13.,value=10.3)
    mmstar = pymc.Uniform('mmstar',lower=-2.,upper=2.,value=1.0)
    scatterm = pymc.Uniform('sm',lower=0.,upper=3.,value=0.1)

    mmu = pymc.Uniform('mmu',lower=10.,upper=13.,value=11.5)
    msig = pymc.Uniform('msig',lower=0.,upper=2.,value=0.3)

    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    pars = [cvar,scatterg,fvar,scatterm,mmu,msig,mmstar,H0]


    @pymc.deterministic(name='likeall')
    def likeall(c=cvar,s1=scatterg,f=fvar,s2=scatterm,mmu=mmu,msig=msig,mmstar=mmstar,H0=H0):

        totlike = 0.

        for i in range(0,Nlens):
            gmodel = c
            mmodel = mmstar*(chains[i]['mstar'] - 11.5) + f
            mglob_model = mmu

            gexp = 1./s1*np.exp(-(gmodel - chains[i]['gamma'])**2/(2.*s1**2))
            mexp = 1./s2*np.exp(-(mmodel - chains[i]['mdm'])**2/(2.*s2**2))
            msexp = 1./msig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*msig**2))

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))

            totlike += np.log((gexp*mexp*msexp*dtexp).mean())

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,c=cvar,s1=scatterg,f=fvar,s2=scatterm,mmu=mmu,msig=msig,mmstar=mmstar,H0=H0):
        return likeall
     
    M = pymc.MCMC(pars+[likeall])
    M.use_step_method(pymc.AdaptiveMetropolis,pars)
    M.isample(N,burnin)

    outdic = {}
    for par in pars:
        outdic[str(par)] = M.trace(par)[:]
    outdic['logp'] = M.trace('likeall')[:]

    if outname is None:
        outname = chaindir+toyname+'_hierarch_%d-%d.dat'%(lenslim[0],lenslim[1]-1)

    f = open(outname,'w')
    pickle.dump(outdic,f)
    f.close()


def hierarchical_controcazzi(lenses,deltaT=5.,Nlens=100,N=11000,burnin=1000,outname=None):

    from scipy.special import erf
    from scipy.integrate import quad

    gup = 2.2
    gdw = 0.2
    mup = 12.5
    mdw = 9.5

    chaindir = '/home/sonnen/allZeLenses/mcmc_chains/'
    toyname = 'nfw'

    chains = []
    t_meas = []
    reffs = []
    zd = []

    if type(Nlens) == type(1):
        lenslim = (0,Nlens)
    else:
        lenslim = Nlens
        Nlens = lenslim[1] - lenslim[0]

    print 'sampling lenses %d to %d'%lenslim

    for i in range(lenslim[0],lenslim[1]):
        print i
        f = open(chaindir+toyname+'_lens_%03d.dat'%i,'r')
        chain = pickle.load(f)
        f.close()

        for par in chain:
            chain[par] = chain[par].flatten()

        chain['timedelay'] /= day
        if chain['timedelay'].mean() != chain['timedelay'].mean():
            print i
            df

        chains.append(chain)

        lens = lenses[i]
        lens.get_time_delay()

        time_meas = (lens.timedelay/day + np.random.normal(0.,deltaT,1),deltaT)

        t_meas.append(time_meas)

        zd.append(lens.zd)
        reffs.append(np.log10(lens.reff_phys))


    #defines the hyper-parameters

    cvar = pymc.Uniform('c',lower=0.,upper=2,value=1.0)
    scatterg = pymc.Uniform('sg',lower=0.,upper=2.,value=0.1)

    fvar = pymc.Uniform('f',lower=8.,upper=13.,value=10.3)
    mmstar = pymc.Uniform('mmstar',lower=-2.,upper=2.,value=1.0)
    scatterm = pymc.Uniform('sm',lower=0.,upper=3.,value=0.1)

    mmu = pymc.Uniform('mmu',lower=10.,upper=13.,value=11.5)
    msig = pymc.Uniform('msig',lower=0.,upper=2.,value=0.3)


    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    pars = [cvar,scatterg,fvar,scatterm,mmu,msig,mmstar,H0]


    @pymc.deterministic(name='likeall')
    def likeall(c=cvar,s1=scatterg,f=fvar,s2=scatterm,mmu=mmu,msig=msig,mmstar=mmstar,H0=H0):

        totlike = 0.

        for i in range(0,Nlens):
            gmodel = c
            mmodel = mmstar*(chains[i]['mstar'] - 11.5) + f
            mglob_model = mmu

            gexp = 1./s1*np.exp(-(gmodel - chains[i]['gamma'])**2/(2.*s1**2))
            mexp = 1./s2*np.exp(-(mmodel - chains[i]['mdm'])**2/(2.*s2**2))
            msexp = 1./msig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*msig**2))

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))

            totlike += np.log((gexp*mexp*msexp*dtexp).mean())

        #now calculates the normalization of the hyperdistribution.
        mu_mdm_func = lambda mstar: mmstar*(mstar - 11.5) + f
        hyperfunc = lambda mstar: 1./(2.*np.pi)**0.5/msig*np.exp(-0.5*(mstar - mmu)**2/msig**2)*0.5*(erf((gup - c)/2.**0.5/s1) - erf((gdw - c)/2.**0.5/s1))*0.5*(erf((mup - mu_mdm_func(mstar))/2.**0.5/s2) - erf((mdw - mu_mdm_func(mstar))/2.**0.5/s2))
        norm = quad(hyperfunc,10.5,12.5)[0]

        totlike -= Nlens*np.log(norm)

        if totlike!=totlike:
            return -1e300

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,c=cvar,s1=scatterg,f=fvar,s2=scatterm,mmu=mmu,msig=msig,mmstar=mmstar,H0=H0):
        return likeall
     
    M = pymc.MCMC(pars+[likeall])
    M.use_step_method(pymc.AdaptiveMetropolis,pars)
    M.isample(N,burnin)

    outdic = {}
    for par in pars:
        outdic[str(par)] = M.trace(par)[:]
    outdic['logp'] = M.trace('likeall')[:]

    if outname is None:
        outname = chaindir+toyname+'_hierarch_controcazzi_%d-%d.dat'%(lenslim[0],lenslim[1]-1)

    f = open(outname,'w')
    pickle.dump(outdic,f)
    f.close()

def legit_hierarchical(lenses,mstar_meas,mstar_err=0.1,deltaT=5.,Nlens=100,N=11000,burnin=1000,outname=None):

    from scipy.special import erf
    from scipy.integrate import quad

    gup = 2.2
    gdw = 0.2
    mup = 12.5
    mdw = 9.5

    chaindir = '/home/sonnen/allZeLenses/mcmc_chains/'
    toyname = 'fisher_price'

    t_meas = []
    reffs = []
    zd = []
    model_lenses = []
    model_bulges = []
    model_halos = []
    xAs = []
    xBs = []
    mstar_maxs = []

    if type(Nlens) == type(1):
        lenslim = (0,Nlens)
    else:
        lenslim = Nlens
        Nlens = lenslim[1] - lenslim[0]

    print 'sampling lenses %d to %d'%lenslim

    for i in range(lenslim[0],lenslim[1]):

        lens = lenses[i]
        lens.get_time_delay()

        time_meas = (lens.timedelay/day + np.random.normal(0.,deltaT,1),deltaT)

        t_meas.append(time_meas)

        zd.append(lens.zd)
        reffs.append(np.log10(lens.reff_phys))

        model_lens = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mdm=lens.mdm,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)
        model_bulge = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=1.,mdm=0.,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)
        model_halo = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=0.,mdm=1.,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)

        model_lens.normalize()
        model_bulge.normalize()
        model_halo.normalize()

        xA,xB = lens.images

        mstar_max = (xA - xB)/(model_bulge.alpha(xA) - model_bulge.alpha(xB))

        model_lenses.append(model_lens)
        model_bulges.append(model_bulge)
        model_halos.append(model_halo)

        xAs.append(xA)
        xBs.append(xB)

        mstar_maxs.append(np.log10(mstar_max))


    #defines the hyper-parameters

    cvar = pymc.Uniform('c',lower=0.,upper=2,value=1.3)
    scatterg = pymc.Uniform('sg',lower=0.,upper=2.,value=0.1)

    fvar = pymc.Uniform('f',lower=8.,upper=13.,value=10.5)
    mmstar = pymc.Uniform('mmstar',lower=-2.,upper=2.,value=1.0)
    scatterm = pymc.Uniform('sm',lower=0.,upper=3.,value=0.1)

    mmu = pymc.Uniform('mmu',lower=10.,upper=13.,value=11.5)
    msig = pymc.Uniform('msig',lower=0.,upper=2.,value=0.3)


    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    pars = [cvar,scatterg,fvar,scatterm,mmu,msig,mmstar,H0]


    def mdm(model_bulge,model_halo,mstar,gamma,xA,xB):
        model_bulge.mstar = 10.**mstar
        model_halo.gamma = gamma
        model_bulge.normalize()
        model_halo.normalize()

        mmdm = ((xA - xB) - (model_bulge.alpha(xA) - model_bulge.alpha(xB)))/(model_halo.alpha(xA) - model_halo.alpha(xB))

        return np.log10(mmdm)

    def timedelay(model_lens,mstar,gamma,mdm,xA,xB):
        model_lens.mstar = 10.**mstar
        model_lens.gamma = gamma
        model_lens.mdm = 10.**mdm
        model_lens.normalize()

        model_lens.source = xA - model_lens.alpha(xA)

        model_lens.get_time_delay()

        return model_lens.timedelay/day

    from scipy.integrate import dblquad

    @pymc.deterministic(name='likeall')
    def likeall(c=cvar,s1=scatterg,f=fvar,s2=scatterm,mmu=mmu,msig=msig,mmstar=mmstar,H0=H0):

        totlike = 0.

        for i in range(0,Nlens):

            gmodel = c

            integrand = lambda mstar,gammadm: 1./(s1*s2*msig*mstar_err*t_meas[i][1])*np.exp(-0.5*(mstar - mmu)**2/msig**2)*np.exp(-0.5*(gammadm - c)**2/s1**2)*np.exp(-0.5*(mdm(model_bulges[i],model_halos[i],mstar,gammadm,xAs[i],xBs[i]) - mmstar*(mstar - 11.5) - f)**2/s2**2)*np.exp(-0.5*(mstar - mstar_meas[i])**2/mstar_err**2)*np.exp(-0.5*(timedelay(model_lenses[i],mstar,gammadm,mdm(model_bulges[i],model_halos[i],mstar,gammadm,xAs[i],xBs[i]),xAs[i],xBs[i])*70./H0 - t_meas[i][0])**2/t_meas[i][1]**2)

            integral = dblquad(integrand,0.2,2.2,lambda g: 10.5, lambda g: mstar_maxs[i])[0]
            print integral

            totlike += np.log10(integral)

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,c=cvar,s1=scatterg,f=fvar,s2=scatterm,mmu=mmu,msig=msig,mmstar=mmstar,H0=H0):
        return likeall
     
    M = pymc.MCMC(pars+[likeall])
    M.use_step_method(pymc.AdaptiveMetropolis,pars)
    M.isample(N,burnin)

    outdic = {}
    for par in pars:
        outdic[str(par)] = M.trace(par)[:]
    outdic['logp'] = M.trace('likeall')[:]

    if outname is None:
        outname = chaindir+toyname+'_hierarch_legit_%d-%d.dat'%(lenslim[0],lenslim[1]-1)

    f = open(outname,'w')
    pickle.dump(outdic,f)
    f.close()


def hierarchical_powerlaw(lenses,mstar_meas,deltaT=5.,Nlens=100,N=11000,burnin=1000,outname=None):

    from scipy.special import gamma as gfunc,gammainc

    chaindir = '/home/sonnen/allZeLenses/mcmc_chains/'
    toyname = 'powerlaw_nfw'

    chains = []
    t_meas = []
    reffs = []
    zd = []

    logr0 = np.log10(5.)

    if type(Nlens) == type(1):
        lenslim = (0,Nlens)
    else:
        lenslim = Nlens
        Nlens = lenslim[1] - lenslim[0]

    print 'sampling lenses %d to %d'%lenslim

    for i in range(lenslim[0],lenslim[1]):
        f = open(chaindir+toyname+'_lens_%03d.dat'%i,'r')
        chain = pickle.load(f)
        f.close()

        for par in chain:
            chain[par] = chain[par].flatten()

        chain['mstar'] = np.random.normal(mstar_meas[i],0.1,len(chain['gamma']))
        chain['timedelay'] /= day
        if chain['timedelay'].mean() != chain['timedelay'].mean():
            print i
            df

        chains.append(chain)

        lens = lenses[i]
        lens.get_time_delay()

        time_meas = (lens.timedelay/day + np.random.normal(0.,deltaT,1),deltaT)
        print chain['timedelay'].mean(),lens.timedelay/day
        #time_meas = (lens.timedelay/day*(1. + np.random.normal(0.,0.05,1)),deltaT)

        t_meas.append(time_meas)

        zd.append(lens.zd)
        reffs.append(np.log10(lens.reff_phys))


    #defines the hyper-parameters

    cvar = pymc.Uniform('c',lower=1.5,upper=2.5,value=2.)
    gmstar = pymc.Uniform('gm',lower=-1.,upper=2.,value=0.4)
    greff = pymc.Uniform('gr',lower=-2.,upper=2.,value=-0.8)
    scatterg = pymc.Uniform('sg',lower=0.,upper=2.,value=0.1)

    mmu = pymc.Uniform('mmu',lower=10.,upper=13.,value=11.5)
    msig = pymc.Uniform('msig',lower=0.,upper=2.,value=0.3)

    H0 = pymc.Uniform('H0',lower=40,upper=100.,value=70.)

    pars = [cvar,gmstar,greff,scatterg,mmu,msig,H0]


    @pymc.deterministic(name='likeall')
    def likeall(c=cvar,gm=gmstar,gr=greff,s1=scatterg,mmu=mmu,msig=msig,H0=H0):

        totlike = 0.

        for i in range(0,Nlens):
            gmodel = c + gm*(chains[i]['mstar'] - 11.5) + gr*(reffs[i] - logr0)
            mglob_model = mmu

            gexp = 1./s1*np.exp(-(gmodel - chains[i]['gamma'])**2/(2.*s1**2))
            msexp = 1./msig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*msig**2))

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))
            totlike += np.log((gexp*msexp*dtexp).mean())

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    #def logp(value=0.,c=cvar,s1=scatterg,f=fvar,s2=scatterm,mmu=mmu,msig=msig,mmstar=mmstar,H0=H0,kextk=kextk_var,theta=theta_var):
    def logp(value=0.,c=cvar,gm=gmstar,gr=greff,s1=scatterg,mmu=mmu,msig=msig,H0=H0):
        return likeall
     
    M = pymc.MCMC(pars+[likeall])
    #M.use_step_method(pymc.AdaptiveMetropolis,pars,cov=diag(0.0000001*ones(len(pars))))
    M.use_step_method(pymc.AdaptiveMetropolis,pars)
    M.isample(N,burnin)

    outdic = {}
    for par in pars:
        outdic[str(par)] = M.trace(par)[:]
    outdic['logp'] = M.trace('likeall')[:]

    if outname is None:
        outname = chaindir+toyname+'_hierarch_%d-%d.dat'%(lenslim[0],lenslim[1]-1)

    f = open(outname,'w')
    pickle.dump(outdic,f)
    f.close()


