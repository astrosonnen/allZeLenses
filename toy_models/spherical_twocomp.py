import os,om10
from om10 import make_twocomp_lenses
import numpy as np
from mass_profiles import gNFW,sersic,lens_models
from physics.distances import Dang
from physics import cgsconstants
from scipy.optimize import brentq
from scipy.misc import derivative
import pymc




def make_sample(Nlens=1000,maglim=23.3,IQ=0.75):

    db = om10.DB(catalog=os.path.expandvars("$OM10_DIR/data/qso_mock.fits"))
    db.select_random(maglim=23.3,IQ=0.75,Nlens=1000)
    mstars,logreffs = make_twocomp_lenses.assign_stars(db)
    mdms,gammas = make_twocomp_lenses.assign_halos(db,mstars,logreffs)

    lenses = []
    for i in range(0,db.Nlenses):

        reff = 10.**logreffs[i]
        lens = lens_models.spherical_cow(zd=db.sample.ZLENS[i],zs=db.sample.ZSRC[i],mstar=mstars[i],mdm=mdms[i],reff_phys=reff,n=4.,rs_phys=10.*reff,gamma=gammas[i])
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


def fit_spherical_cow(lens,mstar_meas,N=11000,burnin=1000): #fits a spherical cow model to image position and stellar mass data. Does NOT fit the time-delay (that's done later in the hierarchical inference step).

    model_lens = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mdm=lens.mdm,reff_phys=lens.reff_phys,n=lens.n,rs_phys=lens.rs_phys,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)

    model_bulge = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=lens.mstar,mdm=-np.inf,reff_phys=lens.reff,n=lens.n,rs_phys=lens.rs,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)
    model_halo = lens_models.spherical_cow(zd=lens.zd,zs=lens.zs,mstar=-np.inf,mdm=11.,reff_phys=lens.reff,n=lens.n,rs_phys=lens.rs,gamma=lens.gamma,kext=lens.kext,images=lens.images,source=lens.source)

    xA,xB = lens.images

    mstar_var = pymc.Uniform('mstar',lower=10.5,upper=12.5,value=lens.mstar)
    gamma_var = pymc.Uniform('gamma',lower=0.2,upper=2.2,value=lens.gamma)

    model_bulge.normalize()
    model_lens.normalize()

    @pymc.deterministic()
    def mdm(mstar=mstar_var,gamma=gamma_var):

        model_bulge.mstar = mstar
        model_halo.gamma = gamma

        model_bulge.normalize()
        model_halo.normalize()

        alpha_sum_dm = (xA - xB) - model_bulge.alpha(xA) + model_bulge.alpha(xB)
        return float(np.log10(alpha_sum_dm/(model_halo.alpha(xA) - model_halo.alpha(xB))) + 11.)

    @pymc.deterministic()
    def timedelay(mstar=mstar_var,gamma=gamma_var):
        
        model_lens.mstar = mstar
        model_lens.gamma = gamma
        model_lens.mdm = mdm

        model_lens.normalize()

        model_lens.source = xA - model_lens.alpha(xA)

        model_lens.get_time_delay()
        return float(model_lens.timedelay)

    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mstar=mstar_var,gamma=gamma_var):
        if float(mdm) != float(mdm):
            return -1e300
        else:
            return -0.5*(mstar - mstar_meas[0])**2/mstar_meas[1]**2

    pars = [mstar_var,gamma_var,mdm,timedelay]

    M = pymc.MCMC(pars)
    M.isample(N,burnin)

    outdic = {'mstar':M.trace('mstar')[:],'mdm':M.trace('mdm')[:],'gamma':M.trace('gamma')[:],'timedelay':M.trace('timedelay')[:]}

    return outdic


