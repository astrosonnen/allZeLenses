from allZeLenses import mass_profiles,tools,lens_models
import os,om10
from om10 import make_twocomp_lenses
import numpy as np
from allZeLenses.mass_profiles import gNFW,sersic
from allZeLenses.tools.distances import Dang
from allZeLenses.tools import cgsconstants
from scipy.optimize import brentq
from scipy.misc import derivative
import pymc
import pickle
import emcee

day = 24.*3600.



def fit_spherical_cow_exactrein(lens,mstar_meas,radmagrat_meas,N=11000,burnin=1000,gammaup=2.2,thin=1): #fits a spherical cow model to image position, stellar mass and ratio of radial magnification data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).
#image positions are assumed to be known exactly.

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
            return -0.5*(np.log10(float(mstar)) - mstar_meas[0])**2/mstar_meas[1]**2 - 0.5*(float(radmag_ratio) - radmagrat_meas[0])**2/radmagrat_meas[1]**2

        
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mdm=mdm_var,gamma=gamma_var):
        return like

    pars = [mdm_var,gamma_var,mstar,timedelay,like,radmag_ratio]

    M = pymc.MCMC(pars)
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':np.log10(M.trace('mstar')[:]),'mdm':M.trace('mdm')[:],'gamma':M.trace('gamma')[:],'timedelay':M.trace('timedelay')[:],'logp':M.trace('like')[:],'radmag_ratio':M.trace('radmag_ratio')[:]}

    return outdic



def fit_spherical_cow_wkext(lens,mstar_meas,radmagrat_meas,N=11000,burnin=1000,gammaup=2.2,imerr=0.1,thin=1): #fits a spherical cow model to image position, stellar mass and ratio of radial magnification data. Does NOT fit the time-delay (that's done later in the hierarchical inference step). Has k_ext as free parameter.

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



def fit_spherical_cow(lens,mstar_meas,N=11000,burnin=1000,gammaup=2.2,imerr=0.1,thin=1): #fits a spherical cow model to image position and stellar mass data. Does NOT fit the time-delay (that's done later in the hierarchical inference step). Sampling is not efficient and gives biased results.

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



def fit_sps_exactrein(lens,radmagrat_meas,N=11000,burnin=1000,gammaup=2.8):
#fits a spherical power-law profile to image positions, assumed to be known perfectly, and ratio of radial magnification data.

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


def emcee_spherical_cow_exactrein(lens,mstar_meas,radmagrat_meas,N=11000,burnin=1000,nwalkers=16,gammaup=2.2,thin=1): #fits a spherical cow model to image position, stellar mass and ratio of radial magnification data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).
#image positions are assumed to be known exactly.

    model_lens = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mdm=lens.mdm,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)
    model_bulge = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=1.,mdm=0.,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=0.,images=lens.images,source=lens.source)
    model_halo = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=0.,mdm=1.,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)

    model_lens.normalize()
    model_bulge.normalize()
    model_halo.normalize()

    xA,xB = lens.images

    mstar_max = (xA - xB)/(model_bulge.alpha(xA) - model_bulge.alpha(xB))


    def mstar(mdm,gamma):

        model_halo.mdm = 10.**mdm
        model_halo.gamma = gamma
        model_halo.normalize()

        mmstar = ((xA - xB) - (model_halo.alpha(xA) - model_halo.alpha(xB)))/(model_bulge.alpha(xA) - model_bulge.alpha(xB))

        return mmstar

    def timedelay(mdm,gamma):

        model_lens.mstar = mstar(mdm,gamma)
        model_lens.gamma = gamma
        model_lens.mdm = 10.**mdm

        model_lens.normalize()

        model_lens.source = xA - model_lens.alpha(xA)

        model_lens.get_time_delay()
        return model_lens.timedelay


    def radmag_ratio(mdm,gamma):

        model_lens.mstar = mstar(mdm,gamma)
        model_lens.gamma = gamma
        model_lens.mdm = 10.**mdm

        model_lens.normalize()

        model_lens.source = xA - model_lens.alpha(xA)

        model_lens.get_radmag_ratio()
        return model_lens.radmag_ratio


    def logp(mass):
        mdm,gamma = mass
        if mdm < 9.5 or gamma < 0.2 or gamma > gammaup:
            return -1e300
        mmstar = mstar(mdm,gamma)
        model_lens.mstar = mmstar
        model_lens.gamma = gamma
        model_lens.mdm = 10.**mdm

        model_lens.normalize()

        if mmstar < 0.:
            return -1e300
        else:
            return -0.5*(np.log10(mmstar) - mstar_meas[0])**2/mstar_meas[1]**2 

        
    sampler = emcee.EnsembleSampler(nwalkers,2,logp)

    start = []
    for i in range(nwalkers):
        mdm0 = np.log10(lens.mdm) + 0.03*np.random.rand(1)
        gamma0 = lens.gamma + 0.03*np.random.rand(1)
        start.append(np.array((mdm0,gamma0)).reshape(2,))

    print 'sampling...'
    sampler.run_mcmc(start,N)

    samples = sampler.chain[:,burnin:,:].reshape((-1,2))

    ntot = nwalkers*(N-burnin)
    #now goes through the chain and recalculates stellar masses and time delays...
    mstars = np.empty(ntot)
    timedelays = 0.*mstars

    print 'recalculating stuff...'
    for i in range(0,ntot):
        mstars[i] = mstar(samples[i,0],samples[i,1])        
        timedelays[i] = timedelay(samples[i,0],samples[i,1])        



    outdic = {'mstar':np.log10(mstars),'mdm':samples[:,0],'gamma':samples[:,1],'timedelay':timedelays}

    return outdic



