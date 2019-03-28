import numpy as np
import lens_models
import pylab
from plotters import cornerplot
import pickle
import h5py
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import brentq


mstar = 11.5
mhalo = 13.3

reff = 7.

day = 24.*3600.

lens = lens_models.NfwDev(zd=0.3, zs=1., mstar=10.**mstar, mhalo=10.**mhalo, reff_phys=reff, delta_halo=200.)

lens.normalize()
lens.get_caustic()
lens.get_rein()

lens.source = lens.rein * 1.1 - lens.alpha(lens.rein*1.1)

lens.get_images()

lens.get_radmag_ratio()

lens.get_timedelay()

xa_obs, xb_obs = lens.images

imerr = 0.01

radmagrat_obs = lens.radmag_ratio
radmagrat_err = 0.01

eps = 1e-4

f = open('/gdrive/projects/cs82_weaklensing/PyBaWL/nfw_re2_s2_grid.dat', 'r')
nfw_re2_s2_spline = pickle.load(f)
f.close()

f = open('/gdrive/projects/cs82_weaklensing/PyBaWL/deV_re2_s2.dat', 'r')
deV_re2_s2 = pickle.load(f)
f.close()

m200tomrs = (np.log(2.) - 0.5)/(np.log(1. + lens.cvir) - lens.cvir/(1. + lens.cvir))

s2_halo = lens.mhalo * m200tomrs*splev(lens.rs/lens.reff, nfw_re2_s2_spline)/reff
s2_bulge = lens.mstar * deV_re2_s2 / reff

sigma_true = (s2_halo + s2_bulge)**0.5
sigma_obs = sigma_true

psi2_true = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
psi3_true = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

a_true = psi3_true / (1. - psi2_true)

model_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, mstar=lens.mstar, mhalo=lens.mhalo, reff_phys=lens.reff_phys, delta_halo=200., cvir=lens.cvir)

model_lens.normalize()
model_lens.get_caustic()
model_lens.source = lens.source

model_lens.get_images()
model_lens.get_radmag_ratio()
model_lens.get_timedelay()
model_lens.get_rein()

nsamp = 1000
nmstar_grid = 101
nmhalo_grid = 101
ncvir_grid = 101
ns2_grid = 101

mhalo_samp = np.zeros(nsamp)
mstar_samp = np.zeros(nsamp)
cvir_samp = np.zeros(nsamp)
s2_samp = np.zeros(nsamp)

psi1_samp = np.zeros(nsamp)
psi2_samp = np.zeros(nsamp)
psi3_samp = np.zeros(nsamp)

xA_samp = np.zeros(nsamp)
xB_samp = np.zeros(nsamp)

radmagrat_samp = np.zeros(nsamp)
dt_samp = np.zeros(nsamp)

mhalo_min = 11.
mhalo_max = 15.
mstar_min = 10.
mstar_max = 12.5
cvir_min = 0.
cvir_max = 2.

xa_min = xa_obs - 5.*imerr
xa_max = xa_obs + 5.*imerr

mhalo_samp[0] = np.log10(lens.mhalo)
mstar_samp[0] = np.log10(lens.mstar)
cvir_samp[0] = np.log10(lens.cvir)
s2_samp[0] = lens.source**2

psi1_samp[0] = model_lens.rein
psi2_samp[0] = psi2_true
psi3_samp[0] = psi3_true

xA_samp[0] = model_lens.images[0]
xB_samp[0] = model_lens.images[1]

radmagrat_samp[0] = model_lens.radmag_ratio

dt_samp[0] = model_lens.timedelay/day

nostar_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, reff_phys=lens.reff_phys, mstar=0., mhalo=lens.mhalo, cvir=lens.cvir, delta_halo=200.)
nostar_lens.normalize()
nostar_lens.source = lens.source

nohalo_lens = lens_models.NfwDev(zd=lens.zd, zs=lens.zs, reff_phys=lens.reff_phys, mstar=lens.mstar, mhalo=1e-10, cvir=lens.cvir, delta_halo=200.)
nohalo_lens.normalize()
nohalo_lens.source = lens.source

for i in range(nsamp-1):


    # draws new value of stellar mass

    alpha_min_here = xa_min - nostar_lens.alpha(xa_min) - s2_samp[i]**0.5
    alpha_max_here = xa_max - nostar_lens.alpha(xa_max) - s2_samp[i]**0.5

    mstar_max_here = alpha_max_here/(nohalo_lens.alpha(xa_max)/nohalo_lens.mstar)
    mstar_min_here = alpha_min_here/(nohalo_lens.alpha(xa_min)/nohalo_lens.mstar)

    if mstar_min_here < 10.**mstar_min:
        lmstar_min_here = mstar_min
    else:
        lmstar_min_here = np.log10(mstar_min_here)

    if mstar_max_here > 10.**mstar_max:
        lmstar_max_here = mstar_max
    else:
        lmstar_max_here = np.log10(mstar_max_here)

    mstar_grid = np.linspace(lmstar_min_here, lmstar_max_here, nmstar_grid)

    logp_mstar = 0.*mstar_grid

    for j in range(nmstar_grid):
        model_lens.mstar = 10.**mstar_grid[j]
        model_lens.get_images()

        if np.isfinite(model_lens.images[0]):
            model_lens.get_radmag_ratio()

            logp_mstar[j] = -0.5*(model_lens.images[0] - xa_obs)**2/imerr**2 - 0.5*(model_lens.images[1] - xb_obs)**2/imerr**2
            logp_mstar[j] += -0.5*(model_lens.radmag_ratio - radmagrat_obs)**2/radmagrat_err**2
        else:
            logp_mstar[j] = -np.inf

    logp_mstar -= logp_mstar.max()

    p_mstar_grid = np.exp(logp_mstar)

    p_mstar_spline = splrep(mstar_grid, p_mstar_grid)

    intfunc = lambda t: splint(mstar_grid[0], t, p_mstar_spline)

    norm = intfunc(mstar_grid[-1])

    F = np.random.rand(1) * norm

    mstar_samp[i+1] = brentq(lambda t: intfunc(t) - F, mstar_grid[0], mstar_grid[-1])

    model_lens.mstar = 10.**mstar_samp[i+1]
    nohalo_lens.mstar = 10.**mstar_samp[i+1]

    # draws new value of halo mass

    alpha_min_here = xa_min - nohalo_lens.alpha(xa_min) - s2_samp[i]**0.5
    alpha_max_here = xa_max - nohalo_lens.alpha(xa_max) - s2_samp[i]**0.5

    def minmh_func(mh):
        nostar_lens.mhalo = 10.**mh
        nostar_lens.normalize()
        return nostar_lens.alpha(xa_min) - alpha_min_here

    if minmh_func(mhalo_min) * minmh_func(mhalo_max) < 0.:
        lmhalo_min_here = brentq(minmh_func, mhalo_min, mhalo_max)
    else:
        lmhalo_min_here = mhalo_min

    def maxmh_func(mh):
        nostar_lens.mhalo = 10.**mh
        nostar_lens.normalize()
        return nostar_lens.alpha(xa_max) - alpha_max_here

    if maxmh_func(mhalo_min) * maxmh_func(mhalo_max) < 0.:
        lmhalo_max_here = brentq(maxmh_func, mhalo_min, mhalo_max)
    else:
        lmhalo_max_here = mhalo_max

    mhalo_grid = np.linspace(lmhalo_min_here, lmhalo_max_here, nmhalo_grid)

    logp_mhalo = 0.*mhalo_grid

    for j in range(nmhalo_grid):
        model_lens.mhalo = 10.**mhalo_grid[j]
        model_lens.normalize()
        model_lens.get_images()

        if np.isfinite(model_lens.images[0]):
            model_lens.get_radmag_ratio()

            logp_mhalo[j] = -0.5*(model_lens.images[0] - xa_obs)**2/imerr**2 - 0.5*(model_lens.images[1] - xb_obs)**2/imerr**2
            logp_mhalo[j] += -0.5*(model_lens.radmag_ratio - radmagrat_obs)**2/radmagrat_err**2
        else:
            logp_mhalo[j] = -np.inf

    logp_mhalo -= logp_mhalo.max()

    p_mhalo_grid = np.exp(logp_mhalo)

    p_mhalo_spline = splrep(mhalo_grid, p_mhalo_grid)

    intfunc = lambda t: splint(mhalo_grid[0], t, p_mhalo_spline)

    norm = intfunc(mhalo_grid[-1])

    F = np.random.rand(1) * norm

    mhalo_samp[i+1] = brentq(lambda t: intfunc(t) - F, mhalo_grid[0], mhalo_grid[-1])

    model_lens.mhalo = 10.**mhalo_samp[i+1]
    model_lens.normalize()
    nostar_lens.mhalo = 10.**mhalo_samp[i+1]
    nostar_lens.normalize()

    # draws new value of concentration

    alpha_min_here = xa_min - nohalo_lens.alpha(xa_min) - s2_samp[i]**0.5
    alpha_max_here = xa_max - nohalo_lens.alpha(xa_max) - s2_samp[i]**0.5

    def minch_func(ch):
        nostar_lens.cvir = 10.**ch
        nostar_lens.normalize()
        return nostar_lens.alpha(xa_min) - alpha_min_here

    if minch_func(cvir_min) * minch_func(cvir_max) < 0.:
        lcvir_min_here = brentq(minch_func, cvir_min, cvir_max)
    else:
        lcvir_min_here = cvir_min

    def maxch_func(ch):
        nostar_lens.cvir = 10.**ch
        nostar_lens.normalize()
        return nostar_lens.alpha(xa_max) - alpha_max_here

    if maxch_func(cvir_min) * maxch_func(cvir_max) < 0.:
        lcvir_max_here = brentq(maxch_func, cvir_min, cvir_max)
    else:
        lcvir_max_here = cvir_max

    cvir_grid = np.linspace(lcvir_min_here, lcvir_max_here, ncvir_grid)

    logp_cvir = 0.*cvir_grid

    for j in range(ncvir_grid):
        model_lens.cvir = 10.**cvir_grid[j]
        model_lens.normalize()
        model_lens.get_images()

        if np.isfinite(model_lens.images[0]):
            model_lens.get_radmag_ratio()

            logp_cvir[j] = -0.5*(model_lens.images[0] - xa_obs)**2/imerr**2 - 0.5*(model_lens.images[1] - xb_obs)**2/imerr**2
            logp_cvir[j] += -0.5*(model_lens.radmag_ratio - radmagrat_obs)**2/radmagrat_err**2
        else:
            logp_cvir[j] = -np.inf

    logp_cvir -= logp_cvir.max()

    p_cvir_grid = np.exp(logp_cvir)

    p_cvir_spline = splrep(cvir_grid, p_cvir_grid)

    intfunc = lambda t: splint(cvir_grid[0], t, p_cvir_spline)

    norm = intfunc(cvir_grid[-1])

    F = np.random.rand(1) * norm

    cvir_samp[i+1] = brentq(lambda t: intfunc(t) - F, cvir_grid[0], cvir_grid[-1])

    model_lens.cvir = 10.**cvir_samp[i+1]
    model_lens.normalize()
    nostar_lens.cvir = 10.**cvir_samp[i+1]
    nostar_lens.normalize()

    # draws new value of source position

    s2_min_here = max(0., (xa_min - model_lens.alpha(xa_min)))**2
    s2_max_here = (xa_max - model_lens.alpha(xa_max))**2

    s2_grid = np.linspace(s2_min_here, s2_max_here, ns2_grid)

    logp_s2 = 0.*s2_grid

    for j in range(ns2_grid):
        model_lens.source = s2_grid[j]**0.5
        model_lens.get_images()

        if np.isfinite(model_lens.images[0]):
            model_lens.get_radmag_ratio()

            logp_s2[j] = -0.5*(model_lens.images[0] - xa_obs)**2/imerr**2 - 0.5*(model_lens.images[1] - xb_obs)**2/imerr**2
            logp_s2[j] += -0.5*(model_lens.radmag_ratio - radmagrat_obs)**2/radmagrat_err**2
        else:
            logp_s2[j] = -np.inf

    logp_s2 -= logp_s2.max()

    p_s2_grid = np.exp(logp_s2)

    p_s2_spline = splrep(s2_grid, p_s2_grid)

    intfunc = lambda t: splint(s2_grid[0], t, p_s2_spline)

    norm = intfunc(s2_grid[-1])

    F = np.random.rand(1) * norm

    s2_samp[i+1] = brentq(lambda t: intfunc(t) - F, s2_grid[0], s2_grid[-1])

    model_lens.source = s2_samp[i+1]**0.5
    nostar_lens.source = s2_samp[i+1]**0.5

    model_lens.get_images()
    model_lens.get_radmag_ratio()
    model_lens.get_timedelay()
    model_lens.get_rein()

    xA_samp[i+1] = model_lens.images[0]
    xB_samp[i+1] = model_lens.images[1]
    radmagrat_samp[i+1] = model_lens.radmag_ratio
    dt_samp[i+1] = model_lens.timedelay/day

    s2_samp[i+1] = model_lens.source**2

    psi1_samp[i+1] = model_lens.rein
    psi2_samp[i+1] = (model_lens.alpha(model_lens.rein + eps) - model_lens.alpha(model_lens.rein - eps))/(2.*eps)
    psi3_samp[i+1] = (model_lens.alpha(model_lens.rein + eps) - 2.*model_lens.alpha(model_lens.rein) + model_lens.alpha(model_lens.rein - eps))/eps**2

    print '%d %3.2f %3.2f %3.2f %3.2f'%(i+1, mstar_samp[i+1], mhalo_samp[i+1], cvir_samp[i+1], s2_samp[i+1]**0.5)

chain_file = h5py.File('onelens_gibbs_sample.hdf5', 'w')

truth = chain_file.create_group('truth')

truth.create_dataset('zd', data=lens.zd)
truth.create_dataset('zs', data=lens.zs)
truth.create_dataset('mstar', data=np.log10(lens.mstar))
truth.create_dataset('mhalo', data=np.log10(lens.mhalo))
truth.create_dataset('reff', data=lens.reff_phys)
truth.create_dataset('cvir', data=lens.cvir)
truth.create_dataset('xA', data=lens.images[0])
truth.create_dataset('xB', data=lens.images[1])
truth.create_dataset('radmagrat', data=lens.radmag_ratio)
truth.create_dataset('psi1', data=lens.rein)
truth.create_dataset('psi2', data=psi2_true)
truth.create_dataset('psi3', data=psi3_true)

chain_file.create_dataset('mstar', data=mstar_samp)
chain_file.create_dataset('mhalo', data=mhalo_samp)
chain_file.create_dataset('s2', data=s2_samp)
chain_file.create_dataset('cvir', data=cvir_samp)
chain_file.create_dataset('dt_model', data=dt_samp)
chain_file.create_dataset('radmagrat', data=radmagrat_samp)
chain_file.create_dataset('xA', data=xA_samp)
chain_file.create_dataset('xB', data=xB_samp)
chain_file.create_dataset('psi1', data=psi1_samp)
chain_file.create_dataset('psi2', data=psi2_samp)
chain_file.create_dataset('psi3', data=psi3_samp)

