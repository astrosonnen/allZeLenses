import lens_models
import numpy as np
from allZeTools import statistics
import pickle
import pylab


day = 24.*3600.

nlens=1000

# test 1. Plots psi2 vs. reff at fixed mstar, mhalo, and rein.

mstar = 11.4
mhalo = 13.2
mhalo_sig = 0.3
cvir = 0.8

mstars = mstar * np.ones(nlens)
mhalos = np.random.normal(mhalo, mhalo_sig, nlens)
logcvirs = cvir * np.ones(nlens)

logreff_mu=0.46
logreff_sig = 0.16

h=0.7

# redshift distribution of lenses: uniform between 0.1 and 0.3 (hardcoded)
zds = np.random.rand(nlens)*0.2+0.2

# redshift distribution of sources: some sort of truncated exponential... (hardcoded)
zss = statistics.general_random(lambda z: np.exp(-(np.log(z-0.4))**2), nlens, (0.5, 4.))

logreffs = logreff_mu + 0.59*(mstar - 11.) - 0.26*(zds - 0.7) + np.random.normal(0., logreff_sig, nlens)
reffs = 10.**logreffs

eps = 1e-4

lenses = []

psi2 = np.zeros(nlens)
rein = np.zeros(nlens)
reff = np.zeros(nlens)

for i in range(nlens):
    lens = lens_models.NfwDev(zd=zds[i], zs=zss[i], mstar=10.**mstars[i], mhalo=10.**mhalos[i], \
                              reff_phys=reffs[i], cvir=10.**logcvirs[i], h=h)

    lens.normalize()
    lens.get_rein()
    rein[i] = lens.rein
    reff[i] = lens.reff

    psi2[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)

rein_mean = rein.mean()

rein_bin = (rein > rein_mean*0.97) & (rein < rein_mean*1.03)

print rein_mean, rein_bin.sum()

pylab.scatter(logreffs[rein_bin], psi2[rein_bin])
pylab.show()

reff_mean = logreffs.mean()

reff_bin = (logreffs > reff_mean*0.97) & (logreffs < reff_mean * 1.03)

print reff_mean, reff_bin.sum()

pylab.scatter(rein[reff_bin], psi2[reff_bin])
pylab.show()


