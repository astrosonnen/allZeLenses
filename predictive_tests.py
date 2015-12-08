from allZeLenses import lens_models
import pylab
import numpy as np


def very_simple_test(lenses,mstar_obs,chain,burnin=0):

    Nlens = len(lenses)

    indices = np.arange(burnin,len(chain['f']))
    ind = np.random.choice(indices)

    mstar_err = 0.1

    logreff_0=0.46

    zds = 0.3*np.ones(Nlens)
    zss = 1.5*np.ones(Nlens)

    mstars = np.random.normal(chain['mmu'][ind],chain['msig'][ind],Nlens)
    mstar_predict = mstars + np.random.normal(0.,mstar_err,Nlens)

    mdms = chain['f'][ind] + np.random.normal(0.,chain['sm'][ind],Nlens)

    logreffs = logreff_0 + 0.59*(mstars - 11.) -0.26*(zds - 0.7)
    reffs = 10.**logreffs

    rein_predict = []
    asymm_predict = []

    rein_obs = []
    asymm_obs = []

    for i in range(0,Nlens):
        lens = lens_models.spherical_cow(zd=zds[i],zs=zss[i],mstar=10.**mstars[i],mdm5=10.**mdms[i],reff_phys = reffs[i],rs_phys=10.*reffs[i],gamma=1.)
        lens.normalize()
        lens.get_caustic()

        ysource = (np.random.rand(1))**0.5*lens.caustic

        lens.source = ysource
        lens.get_images()
        lens.get_rein()
        lens.get_radmag_ratio()

        asymm = (lens.images[0] + lens.images[1])/(lens.images[0] - lens.images[1])

        if lens.images is None:
            df

        rein_predict.append(lens.rein)
        asymm_predict.append(asymm)

        rein_obs.append(lenses[i].rein)
        aobs = (lenses[i].images[0] + lenses[i].images[1])/(lenses[i].images[0] - lenses[i].images[1])
        asymm_obs.append(aobs)

    pylab.subplot(2,2,1)
    pylab.hist(mstar_obs)
    pylab.hist(mstar_predict,histtype='step')

    pylab.subplot(2,2,3)
    pylab.hist(rein_obs)
    pylab.hist(rein_predict,histtype='step')

    pylab.subplot(2,2,4)
    pylab.hist(asymm_obs)
    pylab.hist(asymm_predict,histtype='step')

    pylab.show()

