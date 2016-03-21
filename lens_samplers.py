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
#import emcee

day = 24.*3600.

def fit_nfw_deV(lens,N=11000,burnin=1000,thin=1): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mhalo=lens.mhalo,reff_phys=lens.reff_phys,cvir=lens.cvir,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images
    xA_obs,xB_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs,mstar_err = lens.obs_lmstar
    radmagrat_obs,radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr,nsig=5.)


    mstar_var = pymc.Uniform('lmstar',lower=10.5,upper=12.5,value=np.log10(lens.mstar))
    mhalo_var = pymc.Uniform('mhalo',lower=11.,upper=14.5,value=np.log10(lens.mhalo))
    c_var = pymc.Uniform('lcvir',lower=0.,upper=10.,value=np.log10(lens.cvir))

    @pymc.deterministic()
    def caustic(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_var = pymc.Uniform('s2',lower=0.,upper=caustic**2,value=lens.source**2)

    @pymc.deterministic()
    def imageA(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return np.inf
        else:
            return model_lens.images[0]


    @pymc.deterministic()
    def imageB(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return -np.inf
        else:
            return model_lens.images[1]

    @pymc.deterministic()
    def radmag_ratio(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2:
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def timedelay(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            model_lens.get_time_delay()
            return model_lens.timedelay


    imA_logp = pymc.Normal('imA_logp',mu=imageA,tau=1./imerr**2,value=xA_obs,observed=True)
    imB_logp = pymc.Normal('imB_logp',mu=imageB,tau=1./imerr**2,value=xB_obs,observed=True)

    #mstar_logp = pymc.Normal('mstar_logp',mu=mstar_var,tau=1./mstar_err**2,value=mstar_obs,observed=True)

    radmagrat_logp = pymc.Normal('radmagrat_logp',mu=radmag_ratio,tau=1./radmagrat_err**2,value=radmagrat_obs,observed=True)

    pars = [mstar_var,mhalo_var,c_var,s2_var,timedelay,radmag_ratio,imageA,imageB]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mhalo_var,c_var,s2_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mhalo':M.trace('mhalo')[:],'lcvir':M.trace('lcvir')[:],'timedelay':M.trace('timedelay')[:],'source':M.trace('s2')[:]**0.5,'imageA':M.trace('imageA')[:],'imageB':M.trace('imageB')[:],'radmagrat':M.trace('radmag_ratio')[:]}

    return outdic


def fit_nfw_deV_cprior(lens,N=11000,burnin=1000,thin=1): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mhalo=lens.mhalo,reff_phys=lens.reff_phys,cvir=lens.cvir,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images
    xA_obs,xB_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs,mstar_err = lens.obs_lmstar
    radmagrat_obs,radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr,nsig=5.)


    mstar_var = pymc.Uniform('lmstar',lower=10.5,upper=12.5,value=np.log10(lens.mstar))
    mhalo_var = pymc.Uniform('mhalo',lower=11.,upper=14.5,value=np.log10(lens.mhalo))
    c_var = pymc.Normal('lcvir',mu=0.971 - 0.094*(mhalo_var-12.),tau=1./0.1**2,value=np.log10(lens.cvir))

    @pymc.deterministic()
    def caustic(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_var = pymc.Uniform('s2',lower=0.,upper=caustic**2,value=lens.source**2)

    @pymc.deterministic()
    def imageA(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return np.inf
        else:
            return model_lens.images[0]


    @pymc.deterministic()
    def imageB(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return -np.inf
        else:
            return model_lens.images[1]

    @pymc.deterministic()
    def radmag_ratio(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2:
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def timedelay(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            model_lens.get_time_delay()
            return model_lens.timedelay


    imA_logp = pymc.Normal('imA_logp',mu=imageA,tau=1./imerr**2,value=xA_obs,observed=True)
    imB_logp = pymc.Normal('imB_logp',mu=imageB,tau=1./imerr**2,value=xB_obs,observed=True)

    #mstar_logp = pymc.Normal('mstar_logp',mu=mstar_var,tau=1./mstar_err**2,value=mstar_obs,observed=True)

    radmagrat_logp = pymc.Normal('radmagrat_logp',mu=radmag_ratio,tau=1./radmagrat_err**2,value=radmagrat_obs,observed=True)

    pars = [mstar_var,mhalo_var,c_var,s2_var,timedelay,radmag_ratio,imageA,imageB]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mhalo_var,c_var,s2_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mhalo':M.trace('mhalo')[:],'lcvir':M.trace('lcvir')[:],'timedelay':M.trace('timedelay')[:],'source':M.trace('s2')[:]**0.5,'imageA':M.trace('imageA')[:],'imageB':M.trace('imageB')[:],'radmagrat':M.trace('radmag_ratio')[:]}

    return outdic


def fit_nfw_deV_fixedc(lens,N=11000,burnin=1000,thin=1): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mhalo=lens.mhalo,reff_phys=lens.reff_phys,cvir=lens.cvir,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images
    xA_obs,xB_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs,mstar_err = lens.obs_lmstar
    radmagrat_obs,radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr,nsig=5.)


    mstar_var = pymc.Uniform('lmstar',lower=10.5,upper=12.5,value=np.log10(lens.mstar))
    mhalo_var = pymc.Uniform('mhalo',lower=11.,upper=14.5,value=np.log10(lens.mhalo))
    c_var = pymc.Normal('lcvir',mu=0.971 - 0.094*(mhalo_var-12.),tau=1./0.1**2,value=np.log10(lens.cvir))

    @pymc.deterministic()
    def caustic(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**(0.971-0.094*(mhalo-12.))

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_var = pymc.Uniform('s2',lower=0.,upper=caustic**2,value=lens.source**2)

    @pymc.deterministic()
    def imageA(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**(0.971-0.094*(mhalo-12.))

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return -1e300
        else:
            return model_lens.images[0]


    @pymc.deterministic()
    def imageB(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**(0.971-0.094*(mhalo-12.))

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return -1e300
        else:
            return model_lens.images[1]

    @pymc.deterministic()
    def radmag_ratio(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**(0.971-0.094*(mhalo-12.))

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2:
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def timedelay(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**(0.971-0.094*(mhalo-12.))

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            model_lens.get_time_delay()
            return model_lens.timedelay


    imA_logp = pymc.Normal('imA_logp',mu=imageA,tau=1./imerr**2,value=xA_obs,observed=True)
    imB_logp = pymc.Normal('imB_logp',mu=imageB,tau=1./imerr**2,value=xB_obs,observed=True)

    #mstar_logp = pymc.Normal('mstar_logp',mu=mstar_var,tau=1./mstar_err**2,value=mstar_obs,observed=True)

    radmagrat_logp = pymc.Normal('radmagrat_logp',mu=radmag_ratio,tau=1./radmagrat_err**2,value=radmagrat_obs,observed=True)

    pars = [mstar_var,mhalo_var,c_var,s2_var,timedelay,radmag_ratio,imageA,imageB]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mhalo_var,c_var,s2_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mhalo':M.trace('mhalo')[:],'lcvir':M.trace('lcvir')[:],'timedelay':M.trace('timedelay')[:],'source':M.trace('s2')[:]**0.5,'imageA':M.trace('imageA')[:],'imageB':M.trace('imageB')[:],'radmagrat':M.trace('radmag_ratio')[:]}

    return outdic


def fit_nfw_deV_fixedc_noradmag(lens,N=11000,burnin=1000,thin=1): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mhalo=lens.mhalo,reff_phys=lens.reff_phys,cvir=lens.cvir,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images
    xA_obs,xB_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs,mstar_err = lens.obs_lmstar
    radmagrat_obs,radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr,nsig=5.)


    mstar_var = pymc.Uniform('lmstar',lower=10.5,upper=12.5,value=np.log10(lens.mstar))
    mhalo_var = pymc.Uniform('mhalo',lower=11.,upper=14.5,value=np.log10(lens.mhalo))
    c_var = pymc.Normal('lcvir',mu=0.971 - 0.094*(mhalo_var-12.),tau=1./0.1**2,value=np.log10(lens.cvir))

    @pymc.deterministic()
    def caustic(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**(0.971-0.094*(mhalo-12.))

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_var = pymc.Uniform('s2',lower=0.,upper=caustic**2,value=lens.source**2)

    @pymc.deterministic()
    def imageA(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**(0.971-0.094*(mhalo-12.))

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return np.inf
        else:
            return model_lens.images[0]


    @pymc.deterministic()
    def imageB(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**(0.971-0.094*(mhalo-12.))

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return -np.inf
        else:
            return model_lens.images[1]

    @pymc.deterministic()
    def radmag_ratio(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**(0.971-0.094*(mhalo-12.))

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2:
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def timedelay(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**(0.971-0.094*(mhalo-12.))

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            model_lens.get_time_delay()
            return model_lens.timedelay


    imA_logp = pymc.Normal('imA_logp',mu=imageA,tau=1./imerr**2,value=xA_obs,observed=True)
    imB_logp = pymc.Normal('imB_logp',mu=imageB,tau=1./imerr**2,value=xB_obs,observed=True)

    #mstar_logp = pymc.Normal('mstar_logp',mu=mstar_var,tau=1./mstar_err**2,value=mstar_obs,observed=True)

    #radmagrat_logp = pymc.Normal('radmagrat_logp',mu=radmag_ratio,tau=1./radmagrat_err**2,value=radmagrat_obs,observed=True)

    pars = [mstar_var,mhalo_var,c_var,s2_var,timedelay,radmag_ratio,imageA,imageB]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mhalo_var,c_var,s2_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mhalo':M.trace('mhalo')[:],'lcvir':M.trace('lcvir')[:],'timedelay':M.trace('timedelay')[:],'source':M.trace('s2')[:]**0.5,'imageA':M.trace('imageA')[:],'imageB':M.trace('imageB')[:],'radmagrat':M.trace('radmag_ratio')[:]}

    return outdic


def fit_nfw_deV_cprior_noradmag(lens,N=11000,burnin=1000,thin=1): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mhalo=lens.mhalo,reff_phys=lens.reff_phys,cvir=lens.cvir,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images
    xA_obs,xB_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs,mstar_err = lens.obs_lmstar
    radmagrat_obs,radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr,nsig=5.)


    mstar_var = pymc.Uniform('lmstar',lower=10.5,upper=12.5,value=np.log10(lens.mstar))
    mhalo_var = pymc.Uniform('mhalo',lower=11.,upper=14.5,value=np.log10(lens.mhalo))
    c_var = pymc.Normal('lcvir',mu=0.971 - 0.094*(mhalo_var-12.),tau=1./0.1**2,value=np.log10(lens.cvir))

    @pymc.deterministic()
    def caustic(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_var = pymc.Uniform('s2',lower=0.,upper=caustic**2,value=lens.source**2)

    @pymc.deterministic()
    def imageA(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return np.inf
        else:
            return model_lens.images[0]


    @pymc.deterministic()
    def imageB(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return -np.inf
        else:
            return model_lens.images[1]

    @pymc.deterministic()
    def radmag_ratio(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2:
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def timedelay(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            model_lens.get_time_delay()
            return model_lens.timedelay


    imA_logp = pymc.Normal('imA_logp',mu=imageA,tau=1./imerr**2,value=xA_obs,observed=True)
    imB_logp = pymc.Normal('imB_logp',mu=imageB,tau=1./imerr**2,value=xB_obs,observed=True)

    #mstar_logp = pymc.Normal('mstar_logp',mu=mstar_var,tau=1./mstar_err**2,value=mstar_obs,observed=True)

    #radmagrat_logp = pymc.Normal('radmagrat_logp',mu=radmag_ratio,tau=1./radmagrat_err**2,value=radmagrat_obs,observed=True)

    pars = [mstar_var,mhalo_var,c_var,s2_var,timedelay,radmag_ratio,imageA,imageB]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mhalo_var,c_var,s2_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mhalo':M.trace('mhalo')[:],'lcvir':M.trace('lcvir')[:],'timedelay':M.trace('timedelay')[:],'source':M.trace('s2')[:]**0.5,'imageA':M.trace('imageA')[:],'imageB':M.trace('imageB')[:],'radmagrat':M.trace('radmag_ratio')[:]}

    return outdic


def fit_nfw_deV_fixedc_rein(lens,N=11000,burnin=1000,thin=1): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    from scipy.optimize import brentq

    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mhalo=lens.mhalo,reff_phys=lens.reff_phys,cvir=lens.cvir,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images
    xA_obs,xB_obs = lens.obs_images[0]
    rein_obs = lens.rein
    imerr = lens.obs_images[1]

    mstar_obs,mstar_err = lens.obs_lmstar
    radmagrat_obs,radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr,nsig=5.)

    eps = 1e-4
    rmax = 10.*lens.reff
    model_lens.images = (rmax,-eps)

    mstar_var = pymc.Uniform('lmstar',lower=10.5,upper=12.5,value=np.log10(lens.mstar))
    mhalo_var = pymc.Uniform('mhalo',lower=11.,upper=14.5,value=np.log10(lens.mhalo))

    @pymc.deterministic()
    def rein(mstar=mstar_var,mhalo=mhalo_var):

        lcvir = 0.971 - 0.094*(mhalo-12.)
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()
        
        tangential_invmag = lambda r: model_lens.m(r)/r**2 - 1.
        if tangential_invmag(eps)*tangential_invmag(rmax) >= 0.:
            return 1e10
        else:
            tcrit = brentq(tangential_invmag,eps,rmax,xtol=0.01*imerr)
            return tcrit


    rein_logp = pymc.Normal('rein_logp',mu=rein,tau=2./imerr**2,value=rein_obs,observed=True)

    pars = [mstar_var,mhalo_var,rein]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mhalo_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mhalo':M.trace('mhalo')[:],'rein':M.trace('rein')[:]}

    return outdic


def grid_nfw_deV_fixedc_rein(lens,N=11000,burnin=1000,thin=1,Ngrid=101): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    from scipy.optimize import brentq

    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mhalo=lens.mhalo,reff_phys=lens.reff_phys,cvir=lens.cvir,images=lens.images,source=lens.source)

    model_lens.normalize()

    rein_obs = lens.rein
    imerr = lens.obs_images[1]

    mstar_obs,mstar_err = lens.obs_lmstar

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr,nsig=5.)

    eps = 0.1
    rmax = 10.*lens.reff
    model_lens.images = (rmax,-eps)

    mstar_grid = np.linspace(10.5,12.5,Ngrid)
    mhalo_grid = np.linspace(12.,14.,Ngrid)

    MS,MH = np.meshgrid(mstar_grid,mhalo_grid)

    grid = np.zeros((Ngrid,Ngrid))

    for i in range(0,Ngrid):
        for j in range(0,Ngrid):
            lcvir = 0.971 - 0.094*(mhalo_grid[i]-12.)
            model_lens.mstar = 10.**mstar_grid[j]
            model_lens.mhalo = 10.**mhalo_grid[i]
            model_lens.cvir = 10.**lcvir

            model_lens.normalize()

            tangential_invmag = lambda r: model_lens.m(r)/r**2 - 1.
            if tangential_invmag(eps)*tangential_invmag(rmax) < 0.:
                tcrit = brentq(tangential_invmag,eps,rmax,xtol=0.01*imerr)
                grid[i,j] = np.exp(-(tcrit - rein_obs)**2/imerr**2)

    return grid


def fit_nfw_deV_alpha_cprior_rein(lens,N=11000,burnin=1000,thin=1): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    from scipy.optimize import brentq

    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mhalo=lens.mhalo,reff_phys=lens.reff_phys,cvir=lens.cvir,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images
    xA_obs,xB_obs = lens.obs_images[0]
    rein_obs = lens.rein
    imerr = lens.obs_images[1]

    mstar_obs,mstar_err = lens.obs_lmstar
    radmagrat_obs,radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr,nsig=5.)

    eps = 0.1
    rmax = 10.*lens.reff
    model_lens.images = (rmax,-eps)

    mstar_var = pymc.Uniform('lmstar',lower=10.5,upper=12.5,value=np.log10(lens.mstar))
    alpha_var = pymc.Uniform('lalpha',lower=-0.5,upper=0.5,value=0.)
    mhalo_var = pymc.Uniform('mhalo',lower=11.,upper=14.5,value=np.log10(lens.mhalo))
    c_var = pymc.Normal('lcvir',mu=0.971 - 0.094*(mhalo_var-12.),tau=1./0.1**2,value=np.log10(lens.cvir))

    @pymc.deterministic()
    def lmsps(mstar=mstar_var,alpha=alpha_var):
        return mstar-alpha

    @pymc.deterministic()
    def rein(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var):

        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()
        
        tangential_invmag = lambda r: model_lens.m(r)/r**2 - 1.
        if tangential_invmag(eps)*tangential_invmag(rmax) >= 0.:
            return 1e10
        else:
            tcrit = brentq(tangential_invmag,eps,rmax,xtol=0.01*imerr)
            return tcrit


    rein_logp = pymc.Normal('rein_logp',mu=rein,tau=2./imerr**2,value=rein_obs,observed=True)

    msps_logp = pymc.Normal('msps_logp',mu=lmsps,tau=1./mstar_err**2,value=mstar_obs,observed=True)

    pars = [mstar_var,mhalo_var,alpha_var,c_var,rein]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mhalo_var,c_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'alpha':M.trace('lalpha')[:],'mhalo':M.trace('mhalo')[:],'lcvir':M.trace('lcvir')[:],'rein':M.trace('rein')[:]}

    return outdic


def fit_nfw_deV_alpha_cprior_noradmag(lens,N=11000,burnin=1000,thin=1): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mhalo=lens.mhalo,reff_phys=lens.reff_phys,cvir=lens.cvir,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images
    xA_obs,xB_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs,mstar_err = lens.obs_lmstar
    radmagrat_obs,radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr,nsig=5.)


    mstar_var = pymc.Uniform('lmstar',lower=10.5,upper=12.5,value=np.log10(lens.mstar))
    alpha_var = pymc.Uniform('lalpha',lower=-0.5,upper=0.5,value=0.)
    mhalo_var = pymc.Uniform('mhalo',lower=11.,upper=14.5,value=np.log10(lens.mhalo))
    c_var = pymc.Normal('lcvir',mu=0.971 - 0.094*(mhalo_var-12.),tau=1./0.1**2,value=np.log10(lens.cvir))

    @pymc.deterministic()
    def caustic(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_var = pymc.Uniform('s2',lower=0.,upper=caustic**2,value=lens.source**2)

    @pymc.deterministic()
    def lmsps(mstar=mstar_var,alpha=alpha_var):
        return mstar-alpha

    @pymc.deterministic()
    def imageA(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return np.inf
        else:
            return model_lens.images[0]


    @pymc.deterministic()
    def imageB(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return -np.inf
        else:
            return model_lens.images[1]

    @pymc.deterministic()
    def radmag_ratio(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2:
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def timedelay(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            model_lens.get_time_delay()
            return model_lens.timedelay


    imA_logp = pymc.Normal('imA_logp',mu=imageA,tau=1./imerr**2,value=xA_obs,observed=True)
    imB_logp = pymc.Normal('imB_logp',mu=imageB,tau=1./imerr**2,value=xB_obs,observed=True)

    msps_logp = pymc.Normal('msps_logp',mu=lmsps,tau=1./mstar_err**2,value=mstar_obs,observed=True)

    #radmagrat_logp = pymc.Normal('radmagrat_logp',mu=radmag_ratio,tau=1./radmagrat_err**2,value=radmagrat_obs,observed=True)

    pars = [mstar_var,mhalo_var,alpha_var,c_var,s2_var,timedelay,radmag_ratio,imageA,imageB]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mhalo_var,c_var,s2_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'alpha':M.trace('lalpha')[:],'mhalo':M.trace('mhalo')[:],'lcvir':M.trace('lcvir')[:],'timedelay':M.trace('timedelay')[:],'source':M.trace('s2')[:]**0.5,'imageA':M.trace('imageA')[:],'imageB':M.trace('imageB')[:],'radmagrat':M.trace('radmag_ratio')[:]}

    return outdic


def fit_nfw_deV_knownimf(lens,N=11000,burnin=1000,thin=1): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mhalo=lens.mhalo,reff_phys=lens.reff_phys,cvir=lens.cvir,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images
    xA_obs,xB_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs,mstar_err = lens.obs_lmstar
    radmagrat_obs,radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr,nsig=5.)


    mstar_var = pymc.Uniform('lmstar',lower=10.5,upper=12.5,value=np.log10(lens.mstar))
    mhalo_var = pymc.Uniform('mhalo',lower=11.,upper=14.5,value=np.log10(lens.mhalo))
    c_var = pymc.Uniform('lcvir',lower=0.,upper=10.,value=np.log10(lens.cvir))

    @pymc.deterministic()
    def caustic(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_var = pymc.Uniform('s2',lower=0.,upper=caustic**2,value=lens.source**2)

    @pymc.deterministic()
    def imageA(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return np.inf
        else:
            return model_lens.images[0]


    @pymc.deterministic()
    def imageB(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return -np.inf
        else:
            return model_lens.images[1]

    @pymc.deterministic()
    def radmag_ratio(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2:
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def timedelay(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            model_lens.get_time_delay()
            return model_lens.timedelay


    imA_logp = pymc.Normal('imA_logp',mu=imageA,tau=1./imerr**2,value=xA_obs,observed=True)
    imB_logp = pymc.Normal('imB_logp',mu=imageB,tau=1./imerr**2,value=xB_obs,observed=True)

    mstar_logp = pymc.Normal('mstar_logp',mu=mstar_var,tau=1./mstar_err**2,value=mstar_obs,observed=True)

    radmagrat_logp = pymc.Normal('radmagrat_logp',mu=radmag_ratio,tau=1./radmagrat_err**2,value=radmagrat_obs,observed=True)

    pars = [mstar_var,mhalo_var,c_var,s2_var,timedelay,radmag_ratio,imageA,imageB]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mhalo_var,c_var,s2_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mhalo':M.trace('mhalo')[:],'lcvir':M.trace('lcvir')[:],'timedelay':M.trace('timedelay')[:],'source':M.trace('s2')[:]**0.5,'imageA':M.trace('imageA')[:],'imageB':M.trace('imageB')[:],'radmagrat':M.trace('radmag_ratio')[:]}

    return outdic


def fit_nfw_deV_knownimf_cprior(lens,N=11000,burnin=1000,thin=1): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mhalo=lens.mhalo,reff_phys=lens.reff_phys,cvir=lens.cvir,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images
    xA_obs,xB_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs,mstar_err = lens.obs_lmstar
    radmagrat_obs,radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr,nsig=5.)


    mstar_var = pymc.Uniform('lmstar',lower=10.5,upper=12.5,value=np.log10(lens.mstar))
    mhalo_var = pymc.Uniform('mhalo',lower=11.,upper=15.,value=np.log10(lens.mhalo))
    c_var = pymc.Normal('lcvir',mu=0.971 - 0.094*(mhalo_var-12.),tau=1./0.1**2,value=np.log10(lens.cvir))

    @pymc.deterministic()
    def caustic(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_var = pymc.Uniform('s2',lower=0.,upper=caustic**2,value=lens.source**2)

    @pymc.deterministic()
    def imageA(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return np.inf
        else:
            return model_lens.images[0]


    @pymc.deterministic()
    def imageB(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return -np.inf
        else:
            return model_lens.images[1]

    @pymc.deterministic()
    def radmag_ratio(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2:
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def timedelay(mstar=mstar_var,mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            model_lens.get_time_delay()
            return model_lens.timedelay


    imA_logp = pymc.Normal('imA_logp',mu=imageA,tau=1./imerr**2,value=xA_obs,observed=True)
    imB_logp = pymc.Normal('imB_logp',mu=imageB,tau=1./imerr**2,value=xB_obs,observed=True)

    mstar_logp = pymc.Normal('mstar_logp',mu=mstar_var,tau=1./mstar_err**2,value=mstar_obs,observed=True)

    radmagrat_logp = pymc.Normal('radmagrat_logp',mu=radmag_ratio,tau=1./radmagrat_err**2,value=radmagrat_obs,observed=True)

    pars = [mstar_var,mhalo_var,c_var,s2_var,timedelay,radmag_ratio,imageA,imageB]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mhalo_var,c_var,s2_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mhalo':M.trace('mhalo')[:],'lcvir':M.trace('lcvir')[:],'timedelay':M.trace('timedelay')[:],'source':M.trace('s2')[:]**0.5,'imageA':M.trace('imageA')[:],'imageB':M.trace('imageB')[:],'radmagrat':M.trace('radmag_ratio')[:]}

    return outdic


def fit_nfw_deV_knownimf_fixedc_noradmag(lens,N=11000,burnin=1000,thin=1): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mhalo=lens.mhalo,reff_phys=lens.reff_phys,cvir=lens.cvir,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images
    xA_obs,xB_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs,mstar_err = lens.obs_lmstar
    radmagrat_obs,radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=0.3*imerr,nsig=9.)


    mstar_var = pymc.Uniform('lmstar',lower=10.5,upper=12.5,value=np.log10(lens.mstar))
    mhalo_var = pymc.Uniform('mhalo',lower=11.,upper=15.,value=np.log10(lens.mhalo))

    @pymc.deterministic()
    def caustic(mstar=mstar_var,mhalo=mhalo_var):
        lcvir = 0.971 - 0.094*(mhalo-12.)
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_var = pymc.Uniform('s2',lower=0.,upper=caustic**2,value=lens.source**2)

    @pymc.deterministic()
    def imageA(mstar=mstar_var,mhalo=mhalo_var,s2=s2_var):

        lcvir = 0.971 - 0.094*(mhalo-12.)
        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return np.inf
        else:
            return model_lens.images[0]


    @pymc.deterministic()
    def imageB(mstar=mstar_var,mhalo=mhalo_var,s2=s2_var):

        lcvir = 0.971 - 0.094*(mhalo-12.)
        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return -np.inf
        else:
            return model_lens.images[1]


    @pymc.deterministic()
    def radmag_ratio(mstar=mstar_var,mhalo=mhalo_var,s2=s2_var):

        lcvir = 0.971 - 0.094*(mhalo-12.)
        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2:
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def timedelay(mstar=mstar_var,mhalo=mhalo_var,s2=s2_var):

        lcvir = 0.971 - 0.094*(mhalo-12.)
        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            model_lens.get_time_delay()
            return model_lens.timedelay


    imA_logp = pymc.Normal('imA_logp',mu=imageA,tau=1./imerr**2,value=xA_obs,observed=True)
    imB_logp = pymc.Normal('imB_logp',mu=imageB,tau=1./imerr**2,value=xB_obs,observed=True)

    mstar_logp = pymc.Normal('mstar_logp',mu=mstar_var,tau=1./mstar_err**2,value=mstar_obs,observed=True)

    pars = [mstar_var,mhalo_var,s2_var,timedelay,imageA,imageB,radmag_ratio]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mhalo_var,s2_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mhalo':M.trace('mhalo')[:],'timedelay':M.trace('timedelay')[:],'source':M.trace('s2')[:]**0.5,'imageA':M.trace('imageA')[:],'imageB':M.trace('imageB')[:],'radmagrat':M.trace('radmag_ratio')[:]}

    return outdic


def fit_nfw_deV_knownimf_fixedc_light(lens,N=11000,burnin=1000,thin=1): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mhalo=lens.mhalo,reff_phys=lens.reff_phys,cvir=lens.cvir,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images
    xA_obs,xB_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs,mstar_err = lens.obs_lmstar
    radmagrat_obs,radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=0.3*imerr,nsig=9.)


    mstar_var = pymc.Uniform('lmstar',lower=10.5,upper=12.5,value=np.log10(lens.mstar))
    mhalo_var = pymc.Uniform('mhalo',lower=11.,upper=15.,value=np.log10(lens.mhalo))

    @pymc.deterministic()
    def caustic(mstar=mstar_var,mhalo=mhalo_var):
        lcvir = 0.971 - 0.094*(mhalo-12.)
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_var = pymc.Uniform('s2',lower=0.,upper=caustic**2,value=lens.source**2)

    @pymc.deterministic()
    def like(mstar=mstar_var,mhalo=mhalo_var,s2=s2_var):

        lcvir = 0.971 - 0.094*(mhalo-12.)
        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.get_images()
        loglike = 0.
        if len(model_lens.images) < 2:
            return -np.inf
        else:
            model_lens.get_radmag_ratio()
            loglike += -0.5*(model_lens.images[0] - xA_obs)**2/imerr**2 -0.5*(model_lens.images[1] - xB_obs)**2/imerr**2 
            loglike += -0.5*(mstar_obs - mstar)**2/mstar_err**2
            #loglike += -0.5*(radmagrat_obs - model_lens.radmag_ratio)**2/radmagrat_err**2
            return loglike

    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mstar=mstar_var,mhalo=mhalo_var,s2=s2_var):
        return like


    pars = [mstar_var,mhalo_var,s2_var,like]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mhalo_var,s2_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mhalo':M.trace('mhalo')[:],'source':M.trace('s2')[:]**0.5}

    return outdic


def fit_nfw_deV_knownimf_fixedc(lens,N=11000,burnin=1000,thin=1): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mhalo=lens.mhalo,reff_phys=lens.reff_phys,cvir=lens.cvir,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images
    xA_obs,xB_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    mstar_obs,mstar_err = lens.obs_lmstar
    radmagrat_obs,radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=0.3*imerr,nsig=9.)


    mstar_var = pymc.Uniform('lmstar',lower=10.5,upper=12.5,value=np.log10(lens.mstar))
    mhalo_var = pymc.Uniform('mhalo',lower=11.,upper=15.,value=np.log10(lens.mhalo))

    @pymc.deterministic()
    def caustic(mstar=mstar_var,mhalo=mhalo_var):
        lcvir = 0.971 - 0.094*(mhalo-12.)
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_var = pymc.Uniform('s2',lower=0.,upper=caustic**2,value=lens.source**2)

    @pymc.deterministic()
    def imageA(mstar=mstar_var,mhalo=mhalo_var,s2=s2_var):

        lcvir = 0.971 - 0.094*(mhalo-12.)
        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return np.inf
        else:
            return model_lens.images[0]


    @pymc.deterministic()
    def imageB(mstar=mstar_var,mhalo=mhalo_var,s2=s2_var):

        lcvir = 0.971 - 0.094*(mhalo-12.)
        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return -np.inf
        else:
            return model_lens.images[1]

    @pymc.deterministic()
    def radmag_ratio(mstar=mstar_var,mhalo=mhalo_var,s2=s2_var):

        lcvir = 0.971 - 0.094*(mhalo-12.)
        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2:
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def timedelay(mstar=mstar_var,mhalo=mhalo_var,s2=s2_var):

        lcvir = 0.971 - 0.094*(mhalo-12.)
        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            model_lens.get_time_delay()
            return model_lens.timedelay


    imA_logp = pymc.Normal('imA_logp',mu=imageA,tau=1./imerr**2,value=xA_obs,observed=True)
    imB_logp = pymc.Normal('imB_logp',mu=imageB,tau=1./imerr**2,value=xB_obs,observed=True)

    mstar_logp = pymc.Normal('mstar_logp',mu=mstar_var,tau=1./mstar_err**2,value=mstar_obs,observed=True)

    radmagrat_logp = pymc.Normal('radmagrat_logp',mu=radmag_ratio,tau=1./radmagrat_err**2,value=radmagrat_obs,observed=True)

    pars = [mstar_var,mhalo_var,s2_var,timedelay,radmag_ratio,imageA,imageB]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mhalo_var,s2_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mhalo':M.trace('mhalo')[:],'timedelay':M.trace('timedelay')[:],'source':M.trace('s2')[:]**0.5,'imageA':M.trace('imageA')[:],'imageB':M.trace('imageB')[:],'radmagrat':M.trace('radmag_ratio')[:]}

    return outdic



def fit_nfw_deV_knownmstar_cprior(lens,N=11000,burnin=1000,thin=1): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mhalo=lens.mhalo,reff_phys=lens.reff_phys,cvir=lens.cvir,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images
    xA_obs,xB_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    radmagrat_obs,radmagrat_err = lens.obs_radmagrat

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr,nsig=5.)


    mhalo_var = pymc.Uniform('mhalo',lower=11.,upper=15.,value=np.log10(lens.mhalo))
    c_var = pymc.Normal('lcvir',mu=0.971 - 0.094*(mhalo_var-12.),tau=1./0.1**2,value=np.log10(lens.cvir))

    @pymc.deterministic()
    def caustic(mhalo=mhalo_var,lcvir=c_var):
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_var = pymc.Uniform('s2',lower=0.,upper=caustic**2,value=lens.source**2)

    @pymc.deterministic()
    def imageA(mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return np.inf
        else:
            return model_lens.images[0]


    @pymc.deterministic()
    def imageB(mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return -np.inf
        else:
            return model_lens.images[1]

    @pymc.deterministic()
    def radmag_ratio(mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2:
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def timedelay(mhalo=mhalo_var,lcvir=c_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**lcvir

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            model_lens.get_time_delay()
            return model_lens.timedelay


    imA_logp = pymc.Normal('imA_logp',mu=imageA,tau=1./imerr**2,value=xA_obs,observed=True)
    imB_logp = pymc.Normal('imB_logp',mu=imageB,tau=1./imerr**2,value=xB_obs,observed=True)

    radmagrat_logp = pymc.Normal('radmagrat_logp',mu=radmag_ratio,tau=1./radmagrat_err**2,value=radmagrat_obs,observed=True)

    pars = [mhalo_var,c_var,s2_var,timedelay,radmag_ratio,imageA,imageB]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mhalo_var,c_var,s2_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mhalo':M.trace('mhalo')[:],'lcvir':M.trace('lcvir')[:],'timedelay':M.trace('timedelay')[:],'source':M.trace('s2')[:]**0.5,'imageA':M.trace('imageA')[:],'imageB':M.trace('imageB')[:],'radmagrat':M.trace('radmag_ratio')[:]}

    return outdic



def emcee_spherical_cow(lens,mstar_meas,radmagrat_meas,N=11000,burnin=1000,nwalkers=32,gammaup=1.8,imerr=0.1): #fits a spherical cow model to image position and stellar mass data. Does NOT fit the time-delay (that's done later in the hierarchical inference step). Sampling is not efficient and gives biased results.

    model_lens = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mdm5=lens.mdm5,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)
    model_bulge = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=1.,mdm5=0.,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)
    model_halo = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=0.,mdm5=1.,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)

    model_lens.normalize()
    model_bulge.normalize()
    model_halo.normalize()

    xA,xB = lens.images

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr)


    mstar_max = (xA - xB)/(model_bulge.alpha(xA) - model_bulge.alpha(xB))
    mdm_max = (xA - xB)/(model_halo.alpha(xA) - model_halo.alpha(xB))

    s_max = 2.*lens.caustic


    def timedelay(mstar,gamma,mdm5,s):

        model_lens.source = s**0.5*s_max
        model_lens.mstar = 10.**mstar
        model_lens.gamma = gamma
        model_lens.mdm5 = 10.**mdm5

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            model_lens.get_time_delay()
            return float(model_lens.timedelay)


    def radmag_ratio(mstar,gamma,mdm5,s):

        model_lens.s = s**0.5*s_max
        model_lens.mstar = 10.**mstar
        model_lens.gamma = gamma
        model_lens.mdm5 = 10.**mdm5

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2:
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return float(model_lens.radmag_ratio)


    def logp(pars):

        mstar,gamma,mdm5,s = pars
        model_lens.mstar = 10.**mstar
        model_lens.gamma = gamma
        model_lens.mdm5 = 10.**mdm5
        model_lens.source = s**0.5*s_max

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2 or mdm5<10. or mstar<10.5 or gamma>1.8 or gamma<0.2 or s>1. or s<0.:
            return -1e300
        else:
            imA = float(model_lens.images[0])
            imB = float(model_lens.images[1])
            loglike = -0.5*(imA - xA)**2/imerr**2 - 0.5*(imB - xB)**2/imerr**2
            return loglike -0.5*(mstar - mstar_meas[0])**2/mstar_meas[1]**2 - 0.5*(radmag_ratio(mstar,gamma,mdm5,s) - radmagrat_meas[0])**2/radmagrat_meas[1]**2


    sampler = emcee.EnsembleSampler(nwalkers,4,logp)

    start = []
    for i in range(nwalkers):
        mstar0 = np.log10(lens.mstar) + 0.03*np.random.rand(1)
        gamma0 = lens.gamma + 0.03*np.random.rand(1)
        mdm0 = np.log10(lens.mdm5) + 0.03*np.random.rand(1)
        s0 = (lens.source/s_max)**2 + 0.03*np.random.rand(1)

        start.append(np.array((mstar0,gamma0,mdm0,s0)).reshape(4,))

    print 'sampling...'
    sampler.run_mcmc(start,N)

    samples = sampler.chain[:,burnin:,:].reshape((-1,4))

    ntot = nwalkers*(N-burnin)
    #now goes through the chain and recalculates stellar masses and time delays...
    mstars = np.empty(ntot)
    timedelays = 0.*mstars

    print 'recalculating stuff...'
    for i in range(0,ntot):
        timedelays[i] = timedelay(samples[i,0],samples[i,1],samples[i,2],samples[i,3])        

    outdic = {'mstar':samples[:,0],'mdm5':samples[:,2],'gamma':samples[:,1],'timedelay':timedelays,'source':samples[:,3]}

    return outdic



def fit_sps_ang(lens,N=11000,burnin=1000,thin=1): #fits a singular power-law sphere model to image position and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    approx_rein = 0.5*(lens.images[0] - lens.images[1])

    model_lens = lens_models.sps(zd=lens.zd,zs=lens.zs,rein=approx_rein,images=lens.images)

    xA,xB = lens.images
    xA_obs,xB_obs = lens.obs_images[0]
    imerr = lens.obs_images[1]

    sA = xA - model_lens.alpha(xA)
    sB = xB - model_lens.alpha(xB)
    approx_source = 0.5*(sA + sB)

    model_lens.source = max(0.1, approx_source)

    radmagrat_obs,radmagrat_err = lens.obs_radmagrat

    rein_var = pymc.Uniform('rein',lower=approx_rein - 3*imerr,upper=approx_rein + 3.*imerr,value=approx_rein)
    gamma_var = pymc.Uniform('gamma',lower=1.5,upper=2.5,value=2.)

    s2_var = pymc.Uniform('s2',lower=0.,upper=xA**2,value=lens.source**2)


    @pymc.deterministic()
    def images(rein=rein_var, gamma=gamma_var, s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.rein = rein
        model_lens.gamma = gamma

	model_lens.get_images()

	return model_lens.images


    @pymc.deterministic()
    def radmag_ratio(rein=rein_var, gamma=gamma_var, s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.rein = rein
        model_lens.gamma = gamma

	if imgs[0] < 0.:
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def timedelay(rein=rein_var, gamma=gamma_var, s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.rein = rein
        model_lens.gamma = gamma

	if imgs[0] < 0.:
            return 0.
        else:
            model_lens.get_time_delay()
            return model_lens.timedelay

    @pymc.deterministic
    def imageA(imgs=images):
	if imgs[0] > 0:
	    return imgs[0]
	else:
	    return 1e300

    @pymc.deterministic
    def imageB(imgs=images):
	if imgs[1] < 0:
	    return imgs[1]
	else:
	    return -1e300


    imA_logp = pymc.Normal('imA_logp',mu=imageA,tau=1./imerr**2,value=xA_obs,observed=True)
    imB_logp = pymc.Normal('imB_logp',mu=imageB,tau=1./imerr**2,value=xB_obs,observed=True)

    #mstar_logp = pymc.Normal('mstar_logp',mu=mstar_var,tau=1./mstar_err**2,value=mstar_obs,observed=True)

    radmagrat_logp = pymc.Normal('radmagrat_logp',mu=radmag_ratio,tau=1./radmagrat_err**2,value=radmagrat_obs,observed=True)

    pars = [mstar_var,mhalo_var,c_var,s2_var,timedelay,radmag_ratio,imageA,imageB]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mhalo_var,c_var,s2_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mhalo':M.trace('mhalo')[:],'lcvir':M.trace('lcvir')[:],'timedelay':M.trace('timedelay')[:],'source':M.trace('s2')[:]**0.5,'imageA':M.trace('imageA')[:],'imageB':M.trace('imageB')[:],'radmagrat':M.trace('radmag_ratio')[:]}

    return outdic


