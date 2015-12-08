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


def fit_nfw_deV_knownimf_fixedc_sphcow(lens,N=11000,burnin=1000,thin=1): #fits a nfw+deV model to image position, stellar mass and radial magnification ratio data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    from allZeLenses.mass_profiles import NFW
    from allZeLenses.tools import cgsconstants as cgs
    Delta = 93.5

    mdm5 = lens.mhalo/NFW.M3d(lens.rvir,lens.rvir/lens.cvir)*NFW.M2d(5.,lens.rvir/lens.cvir/lens.arcsec2kpc)

    model_lens = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mdm5=mdm5,reff_phys=lens.reff_phys,n=4.,rs_phys=lens.rvir*lens.arcsec2kpc/lens.cvir,gamma=1.,kext=0.,images=lens.images,source=lens.source)

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
        rvir_phys = (10.**mhalo*cgs.M_Sun*3./Delta/(4.*np.pi)/lens.rhoc)**(1/3.)/cgs.kpc
        rs_phys = rvir_phys/10.**lcvir
        mdm5 = 10.**mhalo/NFW.M3d(rvir_phys,rs_phys)*NFW.M2d(5.,rs_phys)
        model_lens.mstar = 10.**mstar
        model_lens.mdm5 = mdm5
        model_lens.rs_phys = rs_phys

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_var = pymc.Uniform('s2',lower=0.,upper=caustic**2,value=lens.source**2)

    @pymc.deterministic()
    def imageA(mstar=mstar_var,mhalo=mhalo_var,s2=s2_var):

        lcvir = 0.971 - 0.094*(mhalo-12.)
        rvir_phys = (10.**mhalo*cgs.M_Sun*3./Delta/(4.*np.pi)/lens.rhoc)**(1/3.)/cgs.kpc
        rs_phys = rvir_phys/10.**lcvir
        mdm5 = 10.**mhalo/NFW.M3d(rvir_phys,rs_phys)*NFW.M2d(5.,rs_phys)
        model_lens.mstar = 10.**mstar
        model_lens.mdm5 = mdm5
        model_lens.rs_phys = rs_phys

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return np.inf
        else:
            return model_lens.images[0]


    @pymc.deterministic()
    def imageB(mstar=mstar_var,mhalo=mhalo_var,s2=s2_var):

        lcvir = 0.971 - 0.094*(mhalo-12.)
        rvir_phys = (10.**mhalo*cgs.M_Sun*3./Delta/(4.*np.pi)/lens.rhoc)**(1/3.)/cgs.kpc
        rs_phys = rvir_phys/10.**lcvir
        mdm5 = 10.**mhalo/NFW.M3d(rvir_phys,rs_phys)*NFW.M2d(5.,rs_phys)
        model_lens.mstar = 10.**mstar
        model_lens.mdm5 = mdm5
        model_lens.rs_phys = rs_phys

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return -np.inf
        else:
            return model_lens.images[1]

    @pymc.deterministic()
    def radmag_ratio(mstar=mstar_var,mhalo=mhalo_var,s2=s2_var):

        lcvir = 0.971 - 0.094*(mhalo-12.)
        rvir_phys = (10.**mhalo*cgs.M_Sun*3./Delta/(4.*np.pi)/lens.rhoc)**(1/3.)/cgs.kpc
        rs_phys = rvir_phys/10.**lcvir
        mdm5 = 10.**mhalo/NFW.M3d(rvir_phys,rs_phys)*NFW.M2d(5.,rs_phys)
        model_lens.mstar = 10.**mstar
        model_lens.mdm5 = mdm5
        model_lens.rs_phys = rs_phys

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
        rvir_phys = (10.**mhalo*cgs.M_Sun*3./Delta/(4.*np.pi)/lens.rhoc)**(1/3.)/cgs.kpc
        rs_phys = rvir_phys/10.**lcvir
        mdm5 = 10.**mhalo/NFW.M3d(rvir_phys,rs_phys)*NFW.M2d(5.,rs_phys)
        model_lens.mstar = 10.**mstar
        model_lens.mdm5 = mdm5
        model_lens.rs_phys = rs_phys

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




def fit_spherical_cow(lens,mstar_meas,radmagrat_meas,N=11000,burnin=1000,gammaup=1.8,imerr=0.1,thin=1): #fits a spherical cow model to image position and stellar mass data. Does NOT fit the time-delay (that's done later in the hierarchical inference step). Sampling is not efficient and gives biased results.

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

    mstar_var = pymc.Uniform('lmstar',lower=min(lens.mstar,mstar_meas[0]) - 5*mstar_meas[1],upper=np.log10(mstar_max)+0.1,value=np.log10(lens.mstar))
    gamma_var = pymc.Uniform('gamma',lower=0.2,upper=gammaup,value=lens.gamma)
    mdm5_var = pymc.Uniform('mdm5',lower=min(9.5,np.log10(lens.mdm5)-0.2),upper=np.log10(mdm_max)+0.1,value=np.log10(lens.mdm5))
    #s_var = pymc.Uniform('s',lower=0.,upper=2.*lens.caustic,value=lens.source)
    s_max = 2.*lens.caustic
    s_var = pymc.Uniform('s',lower=0.,upper=1.,value=(lens.source/s_max)**2)

    @pymc.deterministic()
    def images(mstar=mstar_var,gamma=gamma_var,mdm5=mdm5_var,s=s_var):

        model_lens.source = s**0.5*s_max
        model_lens.mstar = 10.**mstar
        model_lens.gamma = gamma
        model_lens.mdm5 = 10.**mdm5

        model_lens.normalize()

        model_lens.fast_images()
        return model_lens.images


    @pymc.deterministic()
    def timedelay(mstar=mstar_var,gamma=gamma_var,mdm5=mdm5_var,s=s_var):

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


    @pymc.deterministic()
    def radmag_ratio(mstar=mstar_var,mdm5=mdm5_var,gamma=gamma_var,s=s_var):

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


    @pymc.deterministic()
    def like(mstar=mstar_var,gamma=gamma_var,mdm5=mdm5_var,s=s_var):

        model_lens.mstar = 10.**mstar
        model_lens.gamma = gamma
        model_lens.mdm5 = 10.**mdm5
        model_lens.source = s**0.5*s_max

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2:
            return -1e300
        else:
            imA = float(model_lens.images[0])
            imB = float(model_lens.images[1])
            loglike = -0.5*(imA - xA)**2/imerr**2 - 0.5*(imB - xB)**2/imerr**2
            return loglike -0.5*(mstar - mstar_meas[0])**2/mstar_meas[1]**2 - 0.5*(float(radmag_ratio) - radmagrat_meas[0])**2/radmagrat_meas[1]**2


        
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mstar=mstar_var,gamma=gamma_var,mdm5=mdm5_var,s=s_var):
        return like

    pars = [mstar_var,gamma_var,mdm5_var,s_var,timedelay,like,images]

    M = pymc.MCMC(pars)
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mdm5':M.trace('mdm5')[:],'gamma':M.trace('gamma')[:],'timedelay':M.trace('timedelay')[:],'logp':M.trace('like')[:],'source':M.trace('s')[:]**0.5*s_max,'images':M.trace('images')[:]}

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




def fit_very_simple(lens,mstar_meas,radmagrat_meas,N=11000,burnin=1000,imerr=0.1,thin=1): #fits a spherical cow model to image position and stellar mass data. Does NOT fit the time-delay (that's done later in the hierarchical inference step). Sampling is not efficient and gives biased results.

    model_lens = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mdm5=lens.mdm5,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr)


    mstar_var = pymc.Uniform('lmstar',lower=10.5,upper=12.5,value=np.log10(lens.mstar))
    mdm5_var = pymc.Uniform('mdm5',lower=10.,upper=12.,value=np.log10(lens.mdm5))
    #s_var = pymc.Uniform('s',lower=0.,upper=2.*lens.caustic,value=lens.source)
    s_max = 2.*lens.caustic
    s_var = pymc.Uniform('s',lower=0.,upper=1.,value=(lens.source/s_max)**2)

    @pymc.deterministic()
    def images(mstar=mstar_var,mdm5=mdm5_var,s=s_var):

        model_lens.source = s**0.5*s_max
        model_lens.mstar = 10.**mstar
        model_lens.mdm5 = 10.**mdm5

        model_lens.normalize()

        model_lens.fast_images()
        return model_lens.images


    @pymc.deterministic()
    def timedelay(mstar=mstar_var,mdm5=mdm5_var,s=s_var):

        model_lens.source = s**0.5*s_max
        model_lens.mstar = 10.**mstar
        model_lens.mdm5 = 10.**mdm5

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            model_lens.get_time_delay()
            return float(model_lens.timedelay)


    @pymc.deterministic()
    def radmag_ratio(mstar=mstar_var,mdm5=mdm5_var,s=s_var):

        model_lens.s = s**0.5*s_max
        model_lens.mstar = 10.**mstar
        model_lens.mdm5 = 10.**mdm5

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2:
            return 0.
        else:
            model_lens.get_radmag_ratio()
            return float(model_lens.radmag_ratio)


    @pymc.deterministic()
    def like(mstar=mstar_var,mdm5=mdm5_var,s=s_var):

        model_lens.mstar = 10.**mstar
        model_lens.mdm5 = 10.**mdm5
        model_lens.source = s**0.5*s_max

        model_lens.normalize()

        model_lens.fast_images()

        if len(model_lens.images)<2:
            return -1e300
        else:
            imA = float(model_lens.images[0])
            imB = float(model_lens.images[1])
            loglike = -0.5*(imA - xA)**2/imerr**2 - 0.5*(imB - xB)**2/imerr**2
            return loglike -0.5*(mstar - mstar_meas[0])**2/mstar_meas[1]**2 - 0.5*(float(radmag_ratio) - radmagrat_meas[0])**2/radmagrat_meas[1]**2


        
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mstar=mstar_var,mdm5=mdm5_var,s=s_var):
        return like

    pars = [mstar_var,mdm5_var,s_var,timedelay,like,images]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mdm5_var,s_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mdm5':M.trace('mdm5')[:],'timedelay':M.trace('timedelay')[:],'logp':M.trace('like')[:],'source':M.trace('s')[:]**0.5*s_max,'images':M.trace('images')[:]}

    return outdic


def likelihood_grid_very_simple(lens,mstar_meas,radmagrat_meas,Ngrid=101,imerr=0.1): #evaulates the likelihood on a grid of values of mstar and mdm and maringalizing over the source position. It does NOT compute the time-delay.

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

    mstar_grid = np.linspace(10.5,12.5,Ngrid)
    mdm5_grid = np.linspace(10.,12.,Ngrid)
    s_grid = np.linspace(0.,1.,Ngrid)

    s_max = 2.*lens.caustic

    likelihood = np.zeros((Ngrid,Ngrid))

    for i in range(0,Ngrid):
        print i
        for j in range(0,Ngrid):
            likes = np.zeros(Ngrid)

            model_lens.mstar = 10.**mstar_grid[i]
            model_lens.mdm5 = 10.**mdm5_grid[j]
            model_lens.normalize()


            for k in range(0,Ngrid):

                model_lens.source = s_grid[k]**0.5*s_max

                model_lens.fast_images()
                if len(model_lens.images) == 2:
                    model_lens.get_radmag_ratio()

                    imA = float(model_lens.images[0])
                    imB = float(model_lens.images[1])
                    likes[k] = np.exp(-0.5*(imA - xA)**2/imerr**2 - 0.5*(imB - xB)**2/imerr**2 - 0.5*(lens.radmag_ratio - radmagrat_meas[0])**2/radmagrat_meas[1]**2)
            
            likelihood[i,j] = np.exp(-0.5*(mstar_grid[i] - mstar_meas[0])**2/mstar_meas[1]**2)*likes.mean()

    return likelihood



def fit_very_simple_noradmag_imAB(lens,mstar_meas,N=11000,burnin=1000,imerr=0.1,thin=1): #fits a spherical cow model to image position and stellar mass data. Does NOT fit the time-delay (that's done later in the hierarchical inference step). Sampling is not efficient and gives biased results.

    model_lens = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mdm5=lens.mdm5,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)

    model_lens.normalize()

    xA,xB = lens.images

    model_lens.get_caustic()
    model_lens.make_grids(err=imerr,nsig=5.)


    mstar_var = pymc.Uniform('lmstar',lower=10.5,upper=12.5,value=np.log10(lens.mstar))
    mdm5_var = pymc.Uniform('mdm5',lower=10.,upper=12.,value=np.log10(lens.mdm5))

    @pymc.deterministic()
    def caustic(mstar=mstar_var,mdm5=mdm5_var):
        model_lens.mstar = 10.**mstar
        model_lens.mdm5 = 10.**mdm5

        model_lens.normalize()

        model_lens.get_caustic()

        return model_lens.caustic

    s2_var = pymc.Uniform('s2',lower=0.,upper=caustic**2,value=lens.source**2)

    @pymc.deterministic()
    def imageA(mstar=mstar_var,mdm5=mdm5_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mdm5 = 10.**mdm5

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return np.inf
        else:
            return model_lens.images[0]


    @pymc.deterministic()
    def imageB(mstar=mstar_var,mdm5=mdm5_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mdm5 = 10.**mdm5

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return -np.inf
        else:
            return model_lens.images[1]


    @pymc.deterministic()
    def timedelay(mstar=mstar_var,mdm5=mdm5_var,s2=s2_var):

        model_lens.source = s2**0.5
        model_lens.mstar = 10.**mstar
        model_lens.mdm5 = 10.**mdm5

        model_lens.normalize()

        model_lens.fast_images()
        if len(model_lens.images) < 2:
            return 0.
        else:
            model_lens.get_time_delay()
            return model_lens.timedelay


    imA_logp = pymc.Normal('imA_logp',mu=imageA,tau=1./imerr**2,value=lens.images[0],observed=True)
    imB_logp = pymc.Normal('imB_logp',mu=imageB,tau=1./imerr**2,value=lens.images[1],observed=True)

    mstar_logp = pymc.Normal('mstar_logp',mu=mstar_var,tau=1./mstar_meas[1]**2,value=mstar_meas[0],observed=True)

    pars = [mstar_var,mdm5_var,s2_var,timedelay,imageA,imageB]

    M = pymc.MCMC(pars)
    M.use_step_method(pymc.AdaptiveMetropolis,[mstar_var,mdm5_var,s2_var])
    M.isample(N,burnin,thin=thin)

    outdic = {'mstar':M.trace('lmstar')[:],'mdm5':M.trace('mdm5')[:],'timedelay':M.trace('timedelay')[:],'source':M.trace('s2')[:]**0.5,'imageA':M.trace('imageA')[:],'imageB':M.trace('imageB')[:]}

    return outdic



