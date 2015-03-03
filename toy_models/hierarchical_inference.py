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



def do_om10_spherical_cows(lenses,deltaT=5.,Nlens=100,N=11000,burnin=1000,toyname='toy0',outname=None):

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


def do_wkext(lenses,deltaT=5.,Nlens=100,N=11000,burnin=1000,outname=None):

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


def hierarchical_strong_prior(lenses,deltaT=5.,Nlens=100,N=11000,burnin=1000,toyname='fisher_price',outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):
    #this assumes that we know everything about the population of ETGs from parallel studies. We just need to fit for the distribution in the independent variables (i.e. stellar mass).

    from scipy.special import gamma as gfunc,gammainc

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


def hierarchical_correct_norm(lenses,deltaT=5.,Nlens=100,N=11000,burnin=1000,outname=None):

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

    fvar = pymc.Uniform('f',lower=8.,upper=13.,value=11.)
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
        outname = chaindir+toyname+'_hierarch_correct_norm_%d-%d.dat'%(lenslim[0],lenslim[1]-1)

    f = open(outname,'w')
    pickle.dump(outdic,f)
    f.close()


def legit_hierarchical(lenses,mstar_meas,mstar_err=0.1,deltaT=5.,Nlens=100,N=11000,burnin=1000,toyname='fisher_price',outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):
    #this hierarchical inference code takes into account the truncation of the distribution in mdm, gammadm and mstar. But it's too slow to actually work.

    from scipy.special import erf
    from scipy.integrate import quad

    gup = 2.2
    gdw = 0.2
    mup = 12.5
    mdw = 9.5


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



def do_fisher_price_dumb(lenses,lensname='fisher_price_lens',deltaT=5.,Nlens=100,N=11000,burnin=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import gamma as gfunc,gammainc,erf

    toyname = 'fisher_price_dumb'

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
            chain[par] = chain[par].flatten()[-5000:]

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

    H0 = pymc.Uniform('H0',lower=60,upper=80.,value=70.)

    pars = [H0]


    @pymc.deterministic(name='likeall')
    def likeall(H0=H0):

        totlike = 0.

        for i in range(0,Nlens):

            dtexp = 1./t_meas[i][1]*np.exp(-(chains[i]['timedelay']/H0*70. - t_meas[i][0])**2/(2.*t_meas[i][1]**2))

            totlike += np.log((dtexp).mean())

        return totlike

     
    @pymc.stochastic(observed=True,name='logp')
    def logp(value=0.,H0=H0):
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

            totlike += np.log((mexp*msexp*dtexp/norms).mean())
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


def do_exactmstar_wgrid(lenses,lensname='exactmstar_grid',deltaT=5.,Nlens=100,N=11000,burnin=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

    from scipy.special import gamma as gfunc,gammainc,erf

    mdm_grid = np.linspace(10.,12.,101)

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

    pars = [fvar,scatterm]


    def loglike(f,s2):
        totlike = 0.

        for i in range(0,Nlens):

            mexp = 1./s2*np.exp(-(f - mdm_grid)**2/(2.*s2**2))

            norms = 0.5*(erf((f - 10.)/2.**0.5/s2) - erf((f - 12.)/2.**0.5/s2))*0.5

            totlike += np.log((mexp*grids[i]/norms).mean())

        return totlike

    print loglike(11.,0.2)
    print loglike(10.2,0.7)


    @pymc.deterministic(name='likeall')
    def likeall(f=fvar,s2=scatterm):

        totlike = 0.

        for i in range(0,Nlens):

            mexp = 1./s2*np.exp(-(f - mdm_grid)**2/(2.*s2**2))

            norms = 0.5*(erf((f - 10.)/2.**0.5/s2) - erf((f - 12.)/2.**0.5/s2))

            totlike += np.log((mexp*grids[i]/norms).mean())

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


def do_exactsourcepos_wgrid(lenses,lensname='exactsourcepos_grid',deltaT=5.,Nlens=100,N=11000,burnin=1000,outname=None,chaindir='/home/sonnen/allZeLenses/mcmc_chains/'):

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


