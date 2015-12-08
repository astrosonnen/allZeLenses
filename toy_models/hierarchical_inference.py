#from allZeLenses import mass_profiles,physics
#import os,om10
#from om10 import make_twocomp_lenses
import numpy as np
#from allZeLenses.mass_profiles import gNFW,sersic,lens_models
from allZeLenses.tools.distances import Dang
from allZeLenses.tools import cgsconstants
#from scipy.optimize import brentq
#from scipy.misc import derivative
import pymc
import pickle

day = 24.*3600.

def do_simple_reality(lenses,lensname='simple_reality_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import erf

    csig = 0.1
    c0 = 0.971
    c_mhalo = 0.094

    chains = []
    t_meas = []
    reffs = []
    zd = []
    merrs = []
    msps = []

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

        samp = np.random.choice(np.arange(5000,10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

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

        msps.append(lens.obs_lmstar[0])
        merrs.append(lens.obs_lmstar[1])

    #defines the hyper-parameters

    mh_mu = pymc.Uniform('mh_mu',lower=12.0,upper=14.0,value=13.0)
    mh_sig = pymc.Uniform('mh_sig',lower=0.,upper=1.,value=0.5)

    mstar_mhalo = pymc.Uniform('mstar_mhalo',lower=0.,upper=2.,value=0.5)

    ms_mu = pymc.Uniform('ms_mu',lower=11.,upper=12.,value=11.5)
    ms_sig = pymc.Uniform('ms_sig',lower=0.,upper=2.,value=0.1)

    aimf_mu = pymc.Uniform('aimf_mu',lower=-0.2,upper=0.2,value=0.)
    aimf_sig = pymc.Uniform('aimf_sig',lower=0.,upper=0.5,value=0.05)

    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    pars = [mh_mu,mh_sig,mstar_mhalo,ms_mu,ms_sig,aimf_mu,aimf_sig,H0]


    @pymc.deterministic(name='likeall')
    def likeall(mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,aimf_mu=aimf_mu,aimf_sig=aimf_sig,H0=H0):

        totlike = 0.

        for i in range(0,Nlens):
            mglob_model = ms_mu + mstar_mhalo*(chains[i]['mhalo'] - 13.)
            cglob_model = c0 - c_mhalo*(chains[i]['mhalo'] - 12.)

            mhexp = 1./mh_sig*np.exp(-(mh_mu - chains[i]['mhalo'])**2/(2.*mh_sig**2))
            msexp = 1./ms_sig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*ms_sig**2))
            aexp = 1./(aimf_sig**2 + merrs[i]**2)**0.5*np.exp(-0.5*(aimf_mu - chains[i]['mstar'] + msps[i])**2/(aimf_sig**2 + merrs[i]**2))

            cexp = 1./csig*np.exp(-(cglob_model - chains[i]['lcvir'])**2/(2.*csig**2))

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))

            norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/ms_sig) - erf((mglob_model - 12.5)/2.**0.5/ms_sig))*0.5*(erf((mh_mu - 11.)/2.**0.5/mh_sig) - erf((mh_mu - 14.5)/2.**0.5/mh_sig))

            term = (mhexp*msexp*cexp*aexp*dtexp/norms).mean()
            totlike += np.log(term)

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,aimf_mu=aimf_mu,aimf_sig=aimf_sig,H0=H0):
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


def do_simple_reality_cprior(lenses,lensname='simple_reality_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import erf

    csig = 0.1
    c0 = 0.971
    c_mhalo = 0.094

    chains = []
    t_meas = []
    reffs = []
    zd = []
    merrs = []
    msps = []

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

        lens = lenses[i]

        samp = np.random.choice(np.arange(10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

        chain['timedelay'] /= day
        if chain['timedelay'].mean() != chain['timedelay'].mean():
            print i
            df

        chain['alpha'] = chain['mstar'] - np.random.normal(lens.obs_lmstar[0],lens.obs_lmstar[1],Nis)

        chains.append(chain)

        lens.get_time_delay()

        time_meas = (lens.timedelay/day + np.random.normal(0.,deltaT,1),deltaT)
        #time_meas = (lens.timedelay/day*(1. + np.random.normal(0.,0.05,1)),deltaT)

        t_meas.append(time_meas)

        zd.append(lens.zd)
        reffs.append(np.log10(lens.reff_phys))

        msps.append(lens.obs_lmstar[0])
        merrs.append(lens.obs_lmstar[1])

    #defines the hyper-parameters

    mh_mu = pymc.Uniform('mh_mu',lower=12.0,upper=14.0,value=13.0)
    mh_sig = pymc.Uniform('mh_sig',lower=0.,upper=1.,value=0.5)

    mstar_mhalo = pymc.Uniform('mstar_mhalo',lower=0.,upper=2.,value=0.5)

    ms_mu = pymc.Uniform('ms_mu',lower=11.,upper=12.,value=11.5)
    ms_sig = pymc.Uniform('ms_sig',lower=0.,upper=2.,value=0.1)

    aimf_mu = pymc.Uniform('aimf_mu',lower=-0.2,upper=0.2,value=0.)
    aimf_sig = pymc.Uniform('aimf_sig',lower=0.,upper=0.5,value=0.05)

    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    pars = [mh_mu,mh_sig,mstar_mhalo,ms_mu,ms_sig,aimf_mu,aimf_sig,H0]


    @pymc.deterministic(name='likeall')
    def likeall(mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,aimf_mu=aimf_mu,aimf_sig=aimf_sig,H0=H0):

        totlike = 0.

        for i in range(0,Nlens):
            mglob_model = ms_mu + mstar_mhalo*(chains[i]['mhalo'] - 13.)

            mhexp = 1./mh_sig*np.exp(-(mh_mu - chains[i]['mhalo'])**2/(2.*mh_sig**2))
            msexp = 1./ms_sig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*ms_sig**2))
            #aexp = 1./(aimf_sig**2 + merrs[i]**2)**0.5*np.exp(-0.5*(aimf_mu - chains[i]['mstar'] + msps[i])**2/(aimf_sig**2 + merrs[i]**2))
            aexp = 1./aimf_sig*np.exp(-0.5*(aimf_mu - chains[i]['alpha'])**2/aimf_sig**2)

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))

            #norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/ms_sig) - erf((mglob_model - 12.5)/2.**0.5/ms_sig))*0.5*(erf((mh_mu - 11.)/2.**0.5/mh_sig) - erf((mh_mu - 14.5)/2.**0.5/mh_sig))

            #term = (mhexp*msexp*aexp*dtexp/norms).mean()
            term = (mhexp*msexp*aexp*dtexp).mean()
            totlike += np.log(term)

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,aimf_mu=aimf_mu,aimf_sig=aimf_sig,H0=H0):
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


def do_simple_reality_cprior_nocosmo(lenses,lensname='simple_reality_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import erf

    csig = 0.1
    c0 = 0.971
    c_mhalo = 0.094

    chains = []
    t_meas = []
    reffs = []
    zd = []
    merrs = []
    msps = []

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

        lens = lenses[i]

        samp = np.random.choice(np.arange(10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

        chain['alpha'] = chain['mstar'] - np.random.normal(lens.obs_lmstar[0],lens.obs_lmstar[1],Nis)

        chains.append(chain)

        zd.append(lens.zd)
        reffs.append(np.log10(lens.reff_phys))

        msps.append(lens.obs_lmstar[0])
        merrs.append(lens.obs_lmstar[1])

    #defines the hyper-parameters

    mh_mu = pymc.Uniform('mh_mu',lower=12.0,upper=14.0,value=13.0)
    mh_sig = pymc.Uniform('mh_sig',lower=0.,upper=1.,value=0.5)

    mstar_mhalo = pymc.Uniform('mstar_mhalo',lower=0.,upper=2.,value=0.5)

    ms_mu = pymc.Uniform('ms_mu',lower=11.,upper=12.,value=11.5)
    ms_sig = pymc.Uniform('ms_sig',lower=0.,upper=2.,value=0.1)

    aimf_mu = pymc.Uniform('aimf_mu',lower=-0.2,upper=0.2,value=0.)
    aimf_sig = pymc.Uniform('aimf_sig',lower=0.,upper=0.5,value=0.05)

    pars = [mh_mu,mh_sig,mstar_mhalo,ms_mu,ms_sig,aimf_mu,aimf_sig]


    @pymc.deterministic(name='likeall')
    def likeall(mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,aimf_mu=aimf_mu,aimf_sig=aimf_sig):

        totlike = 0.

        for i in range(0,Nlens):
            mglob_model = ms_mu + mstar_mhalo*(chains[i]['mhalo'] - 13.)

            mhexp = 1./mh_sig*np.exp(-(mh_mu - chains[i]['mhalo'])**2/(2.*mh_sig**2))
            msexp = 1./ms_sig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*ms_sig**2))
            #aexp = 1./(aimf_sig**2 + merrs[i]**2)**0.5*np.exp(-0.5*(aimf_mu - chains[i]['mstar'] + msps[i])**2/(aimf_sig**2 + merrs[i]**2))
            aexp = 1./aimf_sig*np.exp(-0.5*(aimf_mu - chains[i]['alpha'])**2/aimf_sig**2)


            #norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/ms_sig) - erf((mglob_model - 12.5)/2.**0.5/ms_sig))*0.5*(erf((mh_mu - 11.)/2.**0.5/mh_sig) - erf((mh_mu - 14.5)/2.**0.5/mh_sig))

            #term = (mhexp*msexp*aexp*dtexp/norms).mean()
            term = (mhexp*msexp*aexp).mean()
            totlike += np.log(term)

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,aimf_mu=aimf_mu,aimf_sig=aimf_sig):
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


def do_simple_reality_alpha_cprior(lenses,lensname='simple_reality_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import erf

    csig = 0.1
    c0 = 0.971
    c_mhalo = 0.094

    chains = []
    t_meas = []
    reffs = []
    zd = []
    merrs = []
    msps = []

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

        lens = lenses[i]

        samp = np.random.choice(np.arange(5000,10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

        chain['timedelay'] /= day
        if chain['timedelay'].mean() != chain['timedelay'].mean():
            print i
            df

        chains.append(chain)

        lens.get_time_delay()

        time_meas = (lens.timedelay/day + np.random.normal(0.,deltaT,1),deltaT)
        #time_meas = (lens.timedelay/day*(1. + np.random.normal(0.,0.05,1)),deltaT)

        t_meas.append(time_meas)

        zd.append(lens.zd)
        reffs.append(np.log10(lens.reff_phys))

        msps.append(lens.obs_lmstar[0])
        merrs.append(lens.obs_lmstar[1])

    #defines the hyper-parameters

    mh_mu = pymc.Uniform('mh_mu',lower=12.0,upper=14.0,value=13.0)
    mh_sig = pymc.Uniform('mh_sig',lower=0.,upper=1.,value=0.5)

    mstar_mhalo = pymc.Uniform('mstar_mhalo',lower=0.,upper=2.,value=0.5)

    ms_mu = pymc.Uniform('ms_mu',lower=11.,upper=12.,value=11.5)
    ms_sig = pymc.Uniform('ms_sig',lower=0.,upper=2.,value=0.1)

    aimf_mu = pymc.Uniform('aimf_mu',lower=-0.2,upper=0.2,value=0.)
    aimf_sig = pymc.Uniform('aimf_sig',lower=0.,upper=0.5,value=0.05)

    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    pars = [mh_mu,mh_sig,mstar_mhalo,ms_mu,ms_sig,aimf_mu,aimf_sig,H0]


    @pymc.deterministic(name='likeall')
    def likeall(mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,aimf_mu=aimf_mu,aimf_sig=aimf_sig,H0=H0):

        totlike = 0.

        for i in range(0,Nlens):
            mglob_model = ms_mu + mstar_mhalo*(chains[i]['mhalo'] - 13.)

            mhexp = 1./mh_sig*np.exp(-(mh_mu - chains[i]['mhalo'])**2/(2.*mh_sig**2))
            msexp = 1./ms_sig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*ms_sig**2))
            aexp = 1./aimf_sig*np.exp(-0.5*(aimf_mu - chains[i]['alpha'])**2/aimf_sig**2)

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))

            #norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/ms_sig) - erf((mglob_model - 12.5)/2.**0.5/ms_sig))*0.5*(erf((mh_mu - 11.)/2.**0.5/mh_sig) - erf((mh_mu - 14.5)/2.**0.5/mh_sig))

            #term = (mhexp*msexp*aexp*dtexp/norms).mean()
            term = (mhexp*msexp*aexp*dtexp).mean()
            totlike += np.log(term)

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,aimf_mu=aimf_mu,aimf_sig=aimf_sig,H0=H0):
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


def do_simple_reality_alpha_cprior_rein(lenses,lensname='simple_reality_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import erf

    csig = 0.1
    c0 = 0.971
    c_mhalo = 0.094

    chains = []
    t_meas = []
    reffs = []
    zd = []
    merrs = []
    msps = []

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

        lens = lenses[i]

        samp = np.random.choice(np.arange(5000,10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

        chains.append(chain)

        zd.append(lens.zd)
        reffs.append(np.log10(lens.reff_phys))

        msps.append(lens.obs_lmstar[0])
        merrs.append(lens.obs_lmstar[1])

    #defines the hyper-parameters

    mh_mu = pymc.Uniform('mh_mu',lower=12.0,upper=14.0,value=13.0)
    mh_sig = pymc.Uniform('mh_sig',lower=0.,upper=1.,value=0.5)

    mstar_mhalo = pymc.Uniform('mstar_mhalo',lower=0.,upper=2.,value=0.5)

    ms_mu = pymc.Uniform('ms_mu',lower=11.,upper=12.,value=11.5)
    ms_sig = pymc.Uniform('ms_sig',lower=0.,upper=2.,value=0.1)

    aimf_mu = pymc.Uniform('aimf_mu',lower=-0.2,upper=0.2,value=0.)
    aimf_sig = pymc.Uniform('aimf_sig',lower=0.,upper=0.5,value=0.05)

    pars = [mh_mu,mh_sig,mstar_mhalo,ms_mu,ms_sig,aimf_mu,aimf_sig]


    @pymc.deterministic(name='likeall')
    def likeall(mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,aimf_mu=aimf_mu,aimf_sig=aimf_sig):

        totlike = 0.

        for i in range(0,Nlens):
            mglob_model = ms_mu + mstar_mhalo*(chains[i]['mhalo'] - 13.)

            mhexp = 1./mh_sig*np.exp(-(mh_mu - chains[i]['mhalo'])**2/(2.*mh_sig**2))
            msexp = 1./ms_sig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*ms_sig**2))
            aexp = 1./aimf_sig*np.exp(-0.5*(aimf_mu - chains[i]['alpha'])**2/aimf_sig**2)


            #norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/ms_sig) - erf((mglob_model - 12.5)/2.**0.5/ms_sig))*0.5*(erf((mh_mu - 11.)/2.**0.5/mh_sig) - erf((mh_mu - 14.5)/2.**0.5/mh_sig))

            #term = (mhexp*msexp*aexp*dtexp/norms).mean()
            term = (mhexp*msexp*aexp).mean()
            totlike += np.log(term)

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,aimf_mu=aimf_mu,aimf_sig=aimf_sig):
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


def do_simple_reality_cprior_rein(lenses,lensname='simple_reality_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import erf

    csig = 0.1
    c0 = 0.971
    c_mhalo = 0.094

    chains = []
    t_meas = []
    reffs = []
    zd = []
    merrs = []
    msps = []

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

        lens = lenses[i]

        samp = np.random.choice(np.arange(10000),Nis)
        #samp = np.arange(10000)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

        chain['alpha'] = chain['mstar'] - np.random.normal(lens.obs_lmstar[0],lens.obs_lmstar[1],Nis)
        chains.append(chain)

        zd.append(lens.zd)
        reffs.append(np.log10(lens.reff_phys))

        msps.append(lens.obs_lmstar[0])
        merrs.append(lens.obs_lmstar[1])

    #defines the hyper-parameters

    mh_mu = pymc.Uniform('mh_mu',lower=12.0,upper=14.0,value=13.0)
    mh_sig = pymc.Uniform('mh_sig',lower=0.,upper=1.,value=0.5)

    mstar_mhalo = pymc.Uniform('mstar_mhalo',lower=0.,upper=2.,value=0.5)

    ms_mu = pymc.Uniform('ms_mu',lower=11.,upper=12.,value=11.5)
    ms_sig = pymc.Uniform('ms_sig',lower=0.,upper=2.,value=0.1)

    aimf_mu = pymc.Uniform('aimf_mu',lower=-0.2,upper=0.2,value=0.)
    aimf_sig = pymc.Uniform('aimf_sig',lower=0.,upper=0.5,value=0.05)

    pars = [mh_mu,mh_sig,mstar_mhalo,ms_mu,ms_sig,aimf_mu,aimf_sig]


    @pymc.deterministic(name='likeall')
    def likeall(mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,aimf_mu=aimf_mu,aimf_sig=aimf_sig):

        totlike = 0.

        for i in range(0,Nlens):
            mglob_model = ms_mu + mstar_mhalo*(chains[i]['mhalo'] - 13.)

            mhexp = 1./mh_sig*np.exp(-(mh_mu - chains[i]['mhalo'])**2/(2.*mh_sig**2))
            msexp = 1./ms_sig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*ms_sig**2))
            aexp = 1./aimf_sig*np.exp(-0.5*(aimf_mu - chains[i]['alpha'])**2/aimf_sig**2)


            #norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/ms_sig) - erf((mglob_model - 12.5)/2.**0.5/ms_sig))*0.5*(erf((mh_mu - 11.)/2.**0.5/mh_sig) - erf((mh_mu - 14.5)/2.**0.5/mh_sig))

            #term = (mhexp*msexp*aexp*dtexp/norms).mean()
            term = (mhexp*msexp*aexp).mean()
            totlike += np.log(term)

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,aimf_mu=aimf_mu,aimf_sig=aimf_sig):
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


def do_simple_reality_noalpha_cprior_rein(lenses,lensname='simple_reality_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import erf

    csig = 0.1
    c0 = 0.971
    c_mhalo = 0.094

    chains = []
    t_meas = []
    reffs = []
    zd = []
    merrs = []
    msps = []

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

        lens = lenses[i]

        samp = np.random.choice(np.arange(5000,10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

        chains.append(chain)

        zd.append(lens.zd)
        reffs.append(np.log10(lens.reff_phys))

        msps.append(lens.obs_lmstar[0])
        merrs.append(lens.obs_lmstar[1])

    #defines the hyper-parameters

    mh_mu = pymc.Uniform('mh_mu',lower=12.5,upper=13.5,value=13.0)
    mh_sig = pymc.Uniform('mh_sig',lower=0.1,upper=1.,value=0.5)

    mstar_mhalo = pymc.Uniform('mstar_mhalo',lower=0.,upper=2.,value=0.5)

    ms_mu = pymc.Uniform('ms_mu',lower=11.,upper=12.,value=11.5)
    ms_sig = pymc.Uniform('ms_sig',lower=0.,upper=2.,value=0.1)

    pars = [mh_mu,mh_sig,mstar_mhalo,ms_mu,ms_sig]


    @pymc.deterministic(name='likeall')
    def likeall(mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig):

        totlike = 0.

        for i in range(0,Nlens):
            mglob_model = ms_mu + mstar_mhalo*(chains[i]['mhalo'] - 13.)

            mhexp = 1./mh_sig*np.exp(-(mh_mu - chains[i]['mhalo'])**2/(2.*mh_sig**2))
            msexp = 1./ms_sig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*ms_sig**2))


            #norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/ms_sig) - erf((mglob_model - 12.5)/2.**0.5/ms_sig))*0.5*(erf((mh_mu - 11.)/2.**0.5/mh_sig) - erf((mh_mu - 14.5)/2.**0.5/mh_sig))

            #term = (mhexp*msexp*aexp*dtexp/norms).mean()
            term = (mhexp*msexp).mean()
            totlike += np.log(term)

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig):
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


def do_simple_reality_rein_wgrid(lenses,lensname='simple_reality_lens',Ngrid=101,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import erf

    grids = []

    mstar_grid = np.linspace(10.5,12.5,Ngrid)
    mhalo_grid = np.linspace(12.,14.,Ngrid)

    MS,MH = np.meshgrid(mstar_grid,mhalo_grid)

    MS_flat = MS.flatten()
    MH_flat = MH.flatten()

    if type(Nlens) == type(1):
        lenslim = (0,Nlens)
    else:
        lenslim = Nlens
        Nlens = lenslim[1] - lenslim[0]

    print 'sampling lenses %d to %d'%lenslim

    for i in range(lenslim[0],lenslim[1]):
        f = open(chaindir+lensname+'_%03d.dat'%i,'r')
        grid = pickle.load(f)
        f.close()

        lens = lenses[i]

        grids.append(grid.flatten())

    #defines the hyper-parameters

    mh_mu = pymc.Uniform('mh_mu',lower=12.0,upper=14.0,value=13.0)
    mh_sig = pymc.Uniform('mh_sig',lower=0.1,upper=1.,value=0.5)

    mstar_mhalo = pymc.Uniform('mstar_mhalo',lower=0.,upper=2.,value=0.5)

    ms_mu = pymc.Uniform('ms_mu',lower=11.,upper=12.,value=11.5)
    ms_sig = pymc.Uniform('ms_sig',lower=0.05,upper=2.,value=0.1)

    pars = [mh_mu,mh_sig,mstar_mhalo,ms_mu,ms_sig]


    @pymc.deterministic(name='likeall')
    def likeall(mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig):

        totlike = 0.

        for i in range(0,Nlens):
            mglob_model = ms_mu + mstar_mhalo*(MH_flat - 13.)

            mhexp = 1./mh_sig*np.exp(-(mh_mu - MH_flat)**2/(2.*mh_sig**2))
            msexp = 1./ms_sig*np.exp(-(mglob_model - MS_flat)**2/(2.*ms_sig**2))


            norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/ms_sig) - erf((mglob_model - 12.5)/2.**0.5/ms_sig))*0.5*(erf((mh_mu - 12.)/2.**0.5/mh_sig) - erf((mh_mu - 14.)/2.**0.5/mh_sig))

            #term = (mhexp*msexp*aexp*dtexp/norms).mean()
            term = (mhexp*msexp*grids[i]).mean()
            #print (4./(2.*np.pi)*msexp*mhexp/norms).mean(),np.log(term),grids[i].mean()
            totlike += np.log(term)

        #print totlike
        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig):
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


def do_simple_reality_rein_superslow(lenses,Nlens=100,N=11000,burnin=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import erf
    from scipy.integrate import dblquad
    from scipy.optimize import brentq
    from allZeLenses import lens_models



    if type(Nlens) == type(1):
        lenslim = (0,Nlens)
    else:
        lenslim = Nlens
        Nlens = lenslim[1] - lenslim[0]

    print 'sampling lenses %d to %d'%lenslim

    reins = []

    for i in range(lenslim[0],lenslim[1]):
        reins.append(lenses[i].rein)        

    #defines the hyper-parameters

    mh_mu = pymc.Uniform('mh_mu',lower=12.5,upper=13.5,value=13.0)
    mh_sig = pymc.Uniform('mh_sig',lower=0.1,upper=1.,value=0.5)

    mstar_mhalo = pymc.Uniform('mstar_mhalo',lower=0.,upper=2.,value=0.5)

    ms_mu = pymc.Uniform('ms_mu',lower=11.,upper=12.,value=11.5)
    ms_sig = pymc.Uniform('ms_sig',lower=0.05,upper=2.,value=0.1)

    pars = [mh_mu,mh_sig,mstar_mhalo,ms_mu,ms_sig]

    lens = lenses[0]
    model_lens = lens_models.nfw_deV(zd=lens.zd,zs=lens.zs,reff_phys=5.)

    eps = 0.1
    rmax = 10.*model_lens.reff

    def reinf(mstar,mhalo):
        model_lens.mstar = 10.**mstar
        model_lens.mhalo = 10.**mhalo
        model_lens.cvir = 10.**(0.971-0.094*(mhalo-12.))
        model_lens.normalize()

        tang_invmag = lambda r: model_lens.m(r)/r**2-1.
        if tang_invmag(eps)*tang_invmag(rmax) < 0.:
            tcrit = brentq(tang_invmag,eps,rmax,xtol=0.001)
            return tcrit
        else:
            return -np.inf

    imerr = 0.1


    def integrand(ms,mh,ms_mu,mh_mu,ms_sig,mh_sig,mstar_mhalo,rein_obs):
        model_lens.mstar = 10.**ms
        model_lens.mhalo = 10.**mh
        model_lens.cvir = 10.**(0.971-0.094*(mh-12.))
        model_lens.normalize()

        tang_invmag = lambda r: model_lens.m(r)/r**2-1.
        if tang_invmag(eps)*tang_invmag(rmax) >= 0.:
            return 0.
        else:
            rein = brentq(tang_invmag,eps,rmax,xtol=0.001)
            return np.exp(-0.5*(mh-mh_mu)**2/mh_sig**2)*np.exp(-0.5*(ms-ms_mu-mstar_mhalo*(mh-13.))/ms_sig**2)*np.exp(-(rein-rein_obs)**2/imerr**2)


    @pymc.deterministic(name='likeall')
    def likeall(mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig):

        totlike = 0.

        for i in range(0,Nlens):

            #integral = dblquad(lambda ms,mh: np.exp(-0.5*(mh-mh_mu)**2/mh_sig**2)*np.exp(-0.5*(ms-ms_mu-mstar_mhalo*(mh-13.))/ms_sig**2)*np.exp(-(reinf(ms,mh)-reins[i])**2/imerr**2),12.,14.,lambda x: 10.5,lambda x: 12.5,epsabs=1e-4,epsrel=1e-4)[0]
            integral = dblquad(lambda ms,mh: integrand(ms,mh,ms_mu,mh_mu,ms_sig,mh_sig,mstar_mhalo,reins[i]),12.,14.,lambda x: 10.5,lambda x: 12.5,epsabs=1e-4,epsrel=1e-4)[0]
            print np.log(integral)

            totlike += np.log(integral)

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig):
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


def do_simple_reality_noalpha_cprior(lenses,lensname='simple_reality_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import erf

    csig = 0.1
    c0 = 0.971
    c_mhalo = 0.094

    chains = []
    t_meas = []
    reffs = []
    zd = []
    merrs = []
    msps = []

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

        lens = lenses[i]

        samp = np.random.choice(np.arange(5000,10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

        chain['timedelay'] /= day
        if chain['timedelay'].mean() != chain['timedelay'].mean():
            print i
            df

        chains.append(chain)

        lens.get_time_delay()

        time_meas = (lens.timedelay/day + np.random.normal(0.,deltaT,1),deltaT)
        #time_meas = (lens.timedelay/day*(1. + np.random.normal(0.,0.05,1)),deltaT)

        t_meas.append(time_meas)

        zd.append(lens.zd)
        reffs.append(np.log10(lens.reff_phys))

        msps.append(lens.obs_lmstar[0])
        merrs.append(lens.obs_lmstar[1])

    #defines the hyper-parameters

    mh_mu = pymc.Uniform('mh_mu',lower=12.0,upper=14.0,value=13.0)
    mh_sig = pymc.Uniform('mh_sig',lower=0.,upper=1.,value=0.5)

    mstar_mhalo = pymc.Uniform('mstar_mhalo',lower=0.,upper=2.,value=0.5)

    ms_mu = pymc.Uniform('ms_mu',lower=11.,upper=12.,value=11.5)
    ms_sig = pymc.Uniform('ms_sig',lower=0.,upper=2.,value=0.1)


    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    pars = [mh_mu,mh_sig,mstar_mhalo,ms_mu,ms_sig,H0]


    @pymc.deterministic(name='likeall')
    def likeall(mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,H0=H0):

        totlike = 0.

        for i in range(0,Nlens):
            mglob_model = ms_mu + mstar_mhalo*(chains[i]['mhalo'] - 13.)

            mhexp = 1./mh_sig*np.exp(-(mh_mu - chains[i]['mhalo'])**2/(2.*mh_sig**2))
            msexp = 1./ms_sig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*ms_sig**2))

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))

            #norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/ms_sig) - erf((mglob_model - 12.5)/2.**0.5/ms_sig))*0.5*(erf((mh_mu - 11.)/2.**0.5/mh_sig) - erf((mh_mu - 14.5)/2.**0.5/mh_sig))

            #term = (mhexp*msexp*dtexp/norms).mean()
            term = (mhexp*msexp*dtexp).mean()
            totlike += np.log(term)

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,H0=H0):
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


def do_simple_reality_knownimf(lenses,lensname='simple_reality_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import erf

    csig = 0.1
    c0 = 0.971
    c_mhalo = 0.094

    chains = []
    t_meas = []
    reffs = []
    zd = []
    merrs = []
    msps = []

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

        samp = np.random.choice(np.arange(5000,10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

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

        msps.append(lens.obs_lmstar[0])
        merrs.append(lens.obs_lmstar[1])

    #defines the hyper-parameters

    mh_mu = pymc.Uniform('mh_mu',lower=12.0,upper=14.0,value=13.0)
    mh_sig = pymc.Uniform('mh_sig',lower=0.,upper=1.,value=0.5)

    mstar_mhalo = pymc.Uniform('mstar_mhalo',lower=0.,upper=2.,value=0.5)

    ms_mu = pymc.Uniform('ms_mu',lower=11.,upper=12.,value=11.5)
    ms_sig = pymc.Uniform('ms_sig',lower=0.,upper=2.,value=0.1)

    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    pars = [mh_mu,mh_sig,mstar_mhalo,ms_mu,ms_sig,H0]


    @pymc.deterministic(name='likeall')
    def likeall(mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,H0=H0):

        totlike = 0.

        for i in range(0,Nlens):
            mglob_model = ms_mu + mstar_mhalo*(chains[i]['mhalo'] - 13.)
            #cglob_model = c0 - c_mhalo*(chains[i]['mhalo'] - 12.)

            mhexp = 1./mh_sig*np.exp(-(mh_mu - chains[i]['mhalo'])**2/(2.*mh_sig**2))
            msexp = 1./ms_sig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*ms_sig**2))
            #cexp = 1./csig*np.exp(-(cglob_model - chains[i]['lcvir'])**2/(2.*csig**2))

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))

            norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/ms_sig) - erf((mglob_model - 12.5)/2.**0.5/ms_sig))*0.5*(erf((mh_mu - 11.)/2.**0.5/mh_sig) - erf((mh_mu - 14.5)/2.**0.5/mh_sig))

            #term = (mhexp*msexp*cexp*dtexp/norms).mean()
            #term = (mhexp*msexp*cexp*dtexp).mean()
            term = (mhexp*msexp*dtexp).mean()
            totlike += np.log(term)

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,H0=H0):
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


def do_simple_reality_knownmstar(lenses,lensname='simple_reality_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import erf

    csig = 0.1
    c0 = 0.971
    c_mhalo = 0.094

    chains = []
    t_meas = []
    reffs = []
    zd = []
    mstars = []

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

        samp = np.random.choice(np.arange(5000,10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

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

        mstars.append(np.log10(lens.mstar))

    #defines the hyper-parameters

    mh_mu = pymc.Uniform('mh_mu',lower=12.0,upper=14.0,value=13.0)
    mh_sig = pymc.Uniform('mh_sig',lower=0.,upper=1.,value=0.5)

    mstar_mhalo = pymc.Uniform('mstar_mhalo',lower=0.,upper=2.,value=0.5)

    ms_mu = pymc.Uniform('ms_mu',lower=11.,upper=12.,value=11.5)
    ms_sig = pymc.Uniform('ms_sig',lower=0.,upper=2.,value=0.1)

    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    pars = [mh_mu,mh_sig,mstar_mhalo,ms_mu,ms_sig,H0]


    @pymc.deterministic(name='likeall')
    def likeall(mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,H0=H0):

        totlike = 0.

        for i in range(0,Nlens):
            mglob_model = ms_mu + mstar_mhalo*(chains[i]['mhalo'] - 13.)
            #cglob_model = c0 - c_mhalo*(chains[i]['mhalo'] - 12.)

            mhexp = 1./mh_sig*np.exp(-(mh_mu - chains[i]['mhalo'])**2/(2.*mh_sig**2))
            msexp = 1./ms_sig*np.exp(-(mglob_model - mstars[i])**2/(2.*ms_sig**2))
            #cexp = 1./csig*np.exp(-(cglob_model - chains[i]['lcvir'])**2/(2.*csig**2))

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))

            norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/ms_sig) - erf((mglob_model - 12.5)/2.**0.5/ms_sig))*0.5*(erf((mh_mu - 11.)/2.**0.5/mh_sig) - erf((mh_mu - 14.5)/2.**0.5/mh_sig))

            #term = (mhexp*msexp*dtexp/norms).mean()
            #term = (mhexp*msexp*cexp*dtexp).mean()
            term = (mhexp*msexp*dtexp).mean()
            totlike += np.log(term)

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig,H0=H0):
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


def do_simple_reality_knownmstar_nocosmo(lenses,lensname='simple_reality_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import erf

    csig = 0.1
    c0 = 0.971
    c_mhalo = 0.094

    chains = []
    t_meas = []
    reffs = []
    zd = []
    merrs = []
    msps = []

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

        samp = np.random.choice(np.arange(5000,10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

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

        msps.append(lens.obs_lmstar[0])
        merrs.append(lens.obs_lmstar[1])

    #defines the hyper-parameters

    mh_mu = pymc.Uniform('mh_mu',lower=12.0,upper=14.0,value=13.0)
    mh_sig = pymc.Uniform('mh_sig',lower=0.03,upper=1.,value=0.5)

    mstar_mhalo = pymc.Uniform('mstar_mhalo',lower=0.,upper=2.,value=0.5)

    ms_mu = pymc.Uniform('ms_mu',lower=11.,upper=12.,value=11.5)
    ms_sig = pymc.Uniform('ms_sig',lower=0.03,upper=2.,value=0.1)


    pars = [mh_mu,mh_sig,mstar_mhalo,ms_mu,ms_sig]


    @pymc.deterministic(name='likeall')
    def likeall(mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig):

        totlike = 0.

        for i in range(0,Nlens):
            mglob_model = ms_mu + mstar_mhalo*(chains[i]['mhalo'] - 13.)
            #cglob_model = c0 - c_mhalo*(chains[i]['mhalo'] - 12.)

            mhexp = 1./mh_sig*np.exp(-(mh_mu - chains[i]['mhalo'])**2/(2.*mh_sig**2))
            msexp = 1./ms_sig*np.exp(-(mglob_model - np.log10(lenses[i].mstar))**2/(2.*ms_sig**2))
            #cexp = 1./csig*np.exp(-(cglob_model - chains[i]['lcvir'])**2/(2.*csig**2))

            norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/ms_sig) - erf((mglob_model - 12.5)/2.**0.5/ms_sig))*0.5*(erf((mh_mu - 11.)/2.**0.5/mh_sig) - erf((mh_mu - 15.)/2.**0.5/mh_sig))

            #term = (mhexp*msexp/norms).mean()
            term = (mhexp*msexp).mean()
            totlike += np.log(term)

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mh_mu=mh_mu,mh_sig=mh_sig,mstar_mhalo=mstar_mhalo,ms_mu=ms_mu,ms_sig=ms_sig):
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


def do_simple_reality_knownimf_nomhdep(lenses,lensname='simple_reality_nomhdep_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import erf

    csig = 0.1
    c0 = 0.971
    c_mhalo = 0.094

    chains = []
    t_meas = []
    reffs = []
    zd = []
    merrs = []
    msps = []

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

        samp = np.random.choice(np.arange(5000,10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

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

        msps.append(lens.obs_lmstar[0])
        merrs.append(lens.obs_lmstar[1])

    #defines the hyper-parameters

    mh_mu = pymc.Uniform('mh_mu',lower=12.0,upper=14.0,value=13.0)
    mh_sig = pymc.Uniform('mh_sig',lower=0.,upper=1.,value=0.5)

    ms_mu = pymc.Uniform('ms_mu',lower=11.,upper=12.,value=11.5)
    ms_sig = pymc.Uniform('ms_sig',lower=0.,upper=2.,value=0.1)

    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    pars = [mh_mu,mh_sig,ms_mu,ms_sig,H0]


    @pymc.deterministic(name='likeall')
    def likeall(mh_mu=mh_mu,mh_sig=mh_sig,ms_mu=ms_mu,ms_sig=ms_sig,H0=H0):

        totlike = 0.

        for i in range(0,Nlens):
            mglob_model = ms_mu
            #cglob_model = c0 - c_mhalo*(chains[i]['mhalo'] - 12.)

            mhexp = 1./mh_sig*np.exp(-(mh_mu - chains[i]['mhalo'])**2/(2.*mh_sig**2))
            msexp = 1./ms_sig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*ms_sig**2))
            #cexp = 1./csig*np.exp(-(cglob_model - chains[i]['lcvir'])**2/(2.*csig**2))

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))

            norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/ms_sig) - erf((mglob_model - 12.5)/2.**0.5/ms_sig))*0.5*(erf((mh_mu - 11.)/2.**0.5/mh_sig) - erf((mh_mu - 14.5)/2.**0.5/mh_sig))

            term = (mhexp*msexp*dtexp/norms).mean()
            #term = (mhexp*msexp*cexp*dtexp).mean()
            #term = (mhexp*msexp*dtexp).mean()
            totlike += np.log(term)

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mh_mu=mh_mu,mh_sig=mh_sig,ms_mu=ms_mu,ms_sig=ms_sig,H0=H0):
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


def do_simple_reality_knownmstar_nomhdep(lenses,lensname='simple_reality_knownmstar_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import erf

    csig = 0.1
    c0 = 0.971
    c_mhalo = 0.094

    chains = []
    t_meas = []
    reffs = []
    zd = []
    merrs = []
    msps = []

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

        samp = np.random.choice(np.arange(5000,10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

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

        msps.append(lens.obs_lmstar[0])
        merrs.append(lens.obs_lmstar[1])

    #defines the hyper-parameters

    mh_mu = pymc.Uniform('mh_mu',lower=12.0,upper=14.0,value=13.0)
    mh_sig = pymc.Uniform('mh_sig',lower=0.,upper=1.,value=0.5)

    ms_mu = pymc.Uniform('ms_mu',lower=11.,upper=12.,value=11.5)
    ms_sig = pymc.Uniform('ms_sig',lower=0.,upper=2.,value=0.1)

    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    pars = [mh_mu,mh_sig,ms_mu,ms_sig,H0]


    @pymc.deterministic(name='likeall')
    def likeall(mh_mu=mh_mu,mh_sig=mh_sig,ms_mu=ms_mu,ms_sig=ms_sig,H0=H0):

        totlike = 0.

        for i in range(0,Nlens):
            mglob_model = ms_mu
            #cglob_model = c0 - c_mhalo*(chains[i]['mhalo'] - 12.)

            mhexp = 1./mh_sig*np.exp(-(mh_mu - chains[i]['mhalo'])**2/(2.*mh_sig**2))
            msexp = 1./ms_sig*np.exp(-(mglob_model - np.log10(lenses[i].mstar))**2/(2.*ms_sig**2))
            #cexp = 1./csig*np.exp(-(cglob_model - chains[i]['lcvir'])**2/(2.*csig**2))

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))

            norms = 0.5*(erf((mglob_model - 10.5)/2.**0.5/ms_sig) - erf((mglob_model - 12.5)/2.**0.5/ms_sig))*0.5*(erf((mh_mu - 11.)/2.**0.5/mh_sig) - erf((mh_mu - 14.5)/2.**0.5/mh_sig))

            term = (mhexp*msexp*dtexp/norms).mean()
            #term = (mhexp*msexp*cexp*dtexp).mean()
            #term = (mhexp*msexp*dtexp).mean()
            totlike += np.log(term)

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,mh_mu=mh_mu,mh_sig=mh_sig,ms_mu=ms_mu,ms_sig=ms_sig,H0=H0):
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


def do_fisher_price(lenses,lensname='fisher_price_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import gamma as gfunc,gammainc,erf


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

        samp = np.random.choice(np.arange(5000,10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]
            #chain[par] = chain[par].flatten()[-1000:]

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

    fvar = pymc.Uniform('f',lower=8.,upper=13.,value=11.0)
    mmstar = pymc.Uniform('mmstar',lower=-2.,upper=2.,value=0.0)
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
            mexp = 1./s2*np.exp(-(mmodel - chains[i]['mdm5'])**2/(2.*s2**2))
            msexp = 1./msig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*msig**2))

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))

            norms = 0.5*(erf((mmodel - 10.)/2.**0.5/s2) - erf((mmodel - 12.)/2.**0.5/s2))*0.5*(erf((gmodel - 0.2)/2.**0.5/s1) - erf((gmodel - 1.8)/2.**0.5/s1))

            totlike += np.log((gexp*mexp*msexp*dtexp*norms).mean())

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



def hierarchical_powerlaw(lenses,mstar_meas,deltaT=5.,Nlens=100,N=11000,Nis=1000,burnin=1000,outname=None,chaindir = '/home/sonnen/allZeLenses/mcmc_chains/',toyname='powerlaw'):

    from scipy.special import gamma as gfunc,gammainc


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

        samp = np.random.choice(np.arange(5000,10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

        chain['mstar'] = np.random.normal(mstar_meas[i][0],mstar_meas[i][1],len(chain['gamma']))
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

    m5_0 = pymc.Uniform('m5_0',lower=10.,upper=12.,value=11.)
    m5_mstar = pymc.Uniform('m5_mstar',lower=-2.,upper=2.,value=1.)
    m5_sig = pymc.Uniform('m5_sig',lower=0.,upper=1.,value=0.1)


    H0 = pymc.Uniform('H0',lower=40,upper=100.,value=70.)

    pars = [cvar,gmstar,greff,scatterg,mmu,msig,m5_0,m5_mstar,m5_sig,H0]


    @pymc.deterministic(name='likeall')
    def likeall(c=cvar,gm=gmstar,gr=greff,s1=scatterg,mmu=mmu,msig=msig,m5_0=m5_0,m5_mstar=m5_mstar,m5_sig=m5_sig,H0=H0):

        totlike = 0.

        for i in range(0,Nlens):
            gmodel = c + gm*(chains[i]['mstar'] - 11.5) + gr*(reffs[i] - logr0)
            mglob_model = mmu
            m5_model = m5_0 + m5_mstar*(chains[i]['mstar'] - 11.5)

            gexp = 1./s1*np.exp(-(gmodel - chains[i]['gamma'])**2/(2.*s1**2))
            msexp = 1./msig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*msig**2))
            m5exp = 1./m5_sig*np.exp(-(m5_model - chains[i]['m5kpc'])**2/(2.*m5_sig**2))

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))
            totlike += np.log((gexp*msexp*m5exp*dtexp).mean())

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    #def logp(value=0.,c=cvar,s1=scatterg,f=fvar,s2=scatterm,mmu=mmu,msig=msig,mmstar=mmstar,H0=H0,kextk=kextk_var,theta=theta_var):
    def logp(value=0.,c=cvar,gm=gmstar,gr=greff,s1=scatterg,mmu=mmu,msig=msig,m5_0=m5_0,m5_mstar=m5_mstar,m5_sig=m5_sig,H0=H0):
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



def do_very_simple(lenses,lensname='fisher_price_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import gamma as gfunc,gammainc,erf


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

        samp = np.random.choice(np.arange(5000,10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]
            #chain[par] = chain[par].flatten()[-1000:]

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

    fvar = pymc.Uniform('f',lower=10.5,upper=11.5,value=11.0)
    scatterm = pymc.Uniform('sm',lower=0.,upper=1.,value=0.1)

    mmu = pymc.Uniform('mmu',lower=11.,upper=12.,value=11.5)
    msig = pymc.Uniform('msig',lower=0.,upper=2.,value=0.3)

    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    pars = [fvar,scatterm,mmu,msig,H0]


    @pymc.deterministic(name='likeall')
    def likeall(f=fvar,s2=scatterm,mmu=mmu,msig=msig,H0=H0):

        totlike = 0.

        for i in range(0,Nlens):
            mmodel = f
            mglob_model = mmu

            mexp = 1./s2*np.exp(-(mmodel - chains[i]['mdm5'])**2/(2.*s2**2))
            msexp = 1./msig*np.exp(-(mglob_model - chains[i]['mstar'])**2/(2.*msig**2))

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))

            norms = 0.5*(erf((mmodel - 10.)/2.**0.5/s2) - erf((mmodel - 12.)/2.**0.5/s2))*0.5*(erf((mmu - 10.5)/2.**0.5/msig) - erf((mmu - 12.5)/2.**0.5/msig))

            term = (mexp*msexp*dtexp/norms).mean()
            totlike += np.log(term)
            #totlike += np.log((mexp*msexp*dtexp).mean())

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,f=fvar,s2=scatterm,mmu=mmu,msig=msig,H0=H0):
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


def do_very_simple_dmonly(lenses,lensname='fisher_price_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,Nis=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import gamma as gfunc,gammainc,erf


    chains = []

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

        samp = np.random.choice(np.arange(5000,10000),Nis)

        for par in chain:
            chain[par] = chain[par].flatten()[samp]

        chains.append(chain)

        lens = lenses[i]

    #defines the hyper-parameters

    fvar = pymc.Uniform('f',lower=10.5,upper=11.5,value=11.0)
    scatterm = pymc.Uniform('sm',lower=0.,upper=1.,value=0.1)

    pars = [fvar,scatterm]


    @pymc.deterministic(name='likeall')
    def likeall(f=fvar,s2=scatterm):

        totlike = 0.

        for i in range(0,Nlens):
            mmodel = f

            mexp = 1./s2*np.exp(-(mmodel - chains[i]['mdm5'])**2/(2.*s2**2))

            norms = 0.5*(erf((mmodel - 10.)/2.**0.5/s2) - erf((mmodel - 12.)/2.**0.5/s2))

            totlike += np.log((mexp/norms).mean())
            #totlike += np.log((mexp).mean())

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,f=fvar,s2=scatterm):
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


def do_very_simple_wgrid(lenses,lensname='very_simple_grid',deltaT=5.,Nlens=100,N=11000,burnin=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import gamma as gfunc,gammainc,erf

    mstar_grid = np.linspace(10.5,12.5,101)
    mdm_grid = np.linspace(10.,12.,101)

    MD,MS = np.meshgrid(mdm_grid,mstar_grid)
    MD_flat = MD.flatten()
    MS_flat = MS.flatten()

    grids = []

    if type(Nlens) == type(1):
        lenslim = (0,Nlens)
    else:
        lenslim = Nlens
        Nlens = lenslim[1] - lenslim[0]

    print 'sampling lenses %d to %d'%lenslim

    for i in range(lenslim[0],lenslim[1]):
        f = open(chaindir+lensname+'_%03d.dat'%i,'r')
        grid = pickle.load(f)
        f.close()

        grids.append(grid.flatten())


    #defines the hyper-parameters

    fvar = pymc.Uniform('f',lower=10.5,upper=11.5,value=11.0)
    scatterm = pymc.Uniform('sm',lower=0.05,upper=1.,value=0.1)

    mmu = pymc.Uniform('mmu',lower=11.,upper=12.,value=11.5)
    msig = pymc.Uniform('msig',lower=0.,upper=2.,value=0.3)

    pars = [fvar,scatterm,mmu,msig]


    def loglike(f,s2):
        totlike = 0.

        for i in range(0,Nlens):

            mexp = 1./s2*np.exp(-(f - MD_flat)**2/(2.*s2**2))
            #msexp = 1./msig*np.exp(-(mglob_model - MS_flat)**2/(2.*msig**2))

            norms = 0.5*(erf((f - 10.)/2.**0.5/s2) - erf((f - 12.)/2.**0.5/s2))*0.5#*(erf((mmu - 10.5)/2.**0.5/msig) - erf((mmu - 12.5)/2.**0.5/msig))

            totlike += np.log((mexp*grids[i]/norms).mean())
            #totlike += np.log((mexp*msexp*dtexp).mean())

        return totlike

    print loglike(11.,0.2)
    print loglike(10.2,0.7)


    @pymc.deterministic(name='likeall')
    def likeall(f=fvar,s2=scatterm,mmu=mmu,msig=msig):

        totlike = 0.

        for i in range(0,Nlens):
            mmodel = f
            mglob_model = mmu

            mexp = 1./s2*np.exp(-(mmodel - MD_flat)**2/(2.*s2**2))
            msexp = 1./msig*np.exp(-(mglob_model - MS_flat)**2/(2.*msig**2))

            norms = 0.5*(erf((mmodel - 10.)/2.**0.5/s2) - erf((mmodel - 12.)/2.**0.5/s2))*0.5*(erf((mmu - 10.5)/2.**0.5/msig) - erf((mmu - 12.5)/2.**0.5/msig))

            totlike += np.log((mexp*msexp*grids[i]/norms).mean())
            #totlike += np.log((mexp*msexp*dtexp).mean())

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,f=fvar,s2=scatterm,mmu=mmu,msig=msig):
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


