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


def generate_fisher_price_mock(chain,Nlens):

    ind = np.random.choice(np.arange(len(chain['mmu'])))
    mmu = chain['mmu'][ind]
    msig = chain['msig'][ind]
    mdm5_0 = chain['f'][ind]
    mdm5_sig = chain['sm'][ind]
    mdm_mstar = chain['mmstar'][ind]
    gamma_0 = chain['c'][ind]
    gamma_sig = chain['sg'][ind]
    logreff_0 = 0.46
    mstar_err = 0.1
    radmagrat_err = 0.015

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

    return (lenses,mstar_sample,radmagrat_sample)


def generate_powerlaw_mock(chain,Nlens):

    ind = np.random.choice(np.arange(len(chain['mmu'])))
    mmu = chain['mmu'][ind]
    msig = chain['msig'][ind]
    m5_0 = chain['m5_0'][ind]
    m5_sig = chain['m5_sig'][ind]
    m5_mstar = chain['m5_mstar'][ind]
    gamma_0 = chain['c'][ind]
    gamma_sig = chain['sg'][ind]
    gamma_mstar = chain['gm'][ind]
    gamma_reff = chain['gr'][ind]
    logreff_0 = 0.46
    mstar_err = 0.1
    radmagrat_err = 0.015

    zds = np.random.rand(Nlens)*0.2+0.2

    zss = statistics.general_random(lambda z: np.exp(-(np.log(z-0.4))**2),Nlens,(0.5,4.))

    mstars = np.random.normal(mmu,msig,Nlens)
    mstars_meas = mstars + np.random.normal(0.,mstar_err,Nlens)
    radmagrat_errs = np.random.normal(0.,radmagrat_err,Nlens)

    logreffs = logreff_0 + 0.59*(mstars - 11.) -0.26*(zds - 0.7)
    reffs = 10.**logreffs

    m5s = m5_0 + m5_mstar*(mstars - 11.5) + np.random.normal(0.,m5_sig,Nlens)
    gammas = gamma_0 + np.random.normal(0.,gamma_sig,Nlens) + gamma_mstar*(mstars - 11.5) + gamma_reff*(logreffs - np.log10(5.))

    lenses = []
    mstar_sample = []
    radmagrat_sample = []
    for i in range(0,Nlens):
        lens = lens_models.sps(zd=zds[i],zs=zss[i],rein=1.,m5=10.**m5s[i],gamma=gammas[i])
        lens.normalize_m5()
        lens.get_caustic()

        ysource = (np.random.rand(1))**0.5*lens.caustic

        lens.source = ysource
        lens.get_images()
        print lens.images
        lens.get_radmag_ratio()

        if lens.images is None:
            df

        lenses.append(lens)
        mstar_sample.append((mstars_meas[i],mstar_err))
        radmagrat_sample.append((lens.radmag_ratio + radmagrat_errs[i],radmagrat_err))

    return (lenses,mstar_sample,radmagrat_sample)



def powerlaw_asymmetry_distribution(chainname,Nlens):

    f = open(chainname,'r')
    chain = pickle.load(f)
    f.close()

    burnin = 0
    for par in chain:
        chain[par] = chain[par][burnin:]
    
    lenses,mstar_sample,radmagrat_sample = generate_powerlaw_mock(chain,Nlens)

    asymms = []
    for lens in lenses:
        a = (abs(lens.images[0]) - abs(lens.images[1]))/(abs(lens.images[0]) + abs(lens.images[1]))
        asymms.append(a)

    return asymms


def fisher_price_asymmetry_distribution(chainname,Nlens):

    f = open(chainname,'r')
    chain = pickle.load(f)
    f.close()

    burnin = 0
    for par in chain:
        chain[par] = chain[par][burnin:]
    
    lenses,mstar_sample,radmagrat_sample = generate_fisher_price_mock(chain,Nlens)

    asymms = []
    for lens in lenses:
        a = (abs(lens.images[0]) - abs(lens.images[1]))/(abs(lens.images[0]) + abs(lens.images[1]))
        asymms.append(a)

    return asymms


