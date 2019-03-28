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

model_lens = lens_models.broken_alpha_powerlaw(zd=lens.zd, zs=lens.zs, rein=lens.rein, gamma=2., beta=0.)

model_lens.source = lens.source

model_lens.get_images()
model_lens.get_radmag_ratio()
model_lens.get_timedelay()

nsamp = 1000

gamma_samp = np.zeros(nsamp)
rein_samp = np.zeros(nsamp)
beta_samp = np.zeros(nsamp)
s2_samp = np.zeros(nsamp)

psi1_samp = np.zeros(nsamp)
psi2_samp = np.zeros(nsamp)
psi3_samp = np.zeros(nsamp)

xA_samp = np.zeros(nsamp)
xB_samp = np.zeros(nsamp)

radmagrat_samp = np.zeros(nsamp)
dt_samp = np.zeros(nsamp)

rein_min = 0.5*(xa_obs - 5.*imerr - (xb_obs + 5.*imerr))
rein_max = 0.5*(xa_obs + 5.*imerr - (xb_obs - 5.*imerr))

rein_grid = np.arange(rein_min, rein_max, step=0.0001)
nrein_grid = len(rein_grid)

gamma_min = 1.2
gamma_max = 2.8
beta_min = -1.
beta_max = 1.

ngamma_grid = 1001
nbeta_grid = 1001
ns2_grid = 101

gamma_grid = np.linspace(gamma_min, gamma_max, ngamma_grid)
beta_grid = np.linspace(beta_min, beta_max, nbeta_grid)

xa_min = xa_obs - 5.*imerr
xa_max = xa_obs + 5.*imerr

gamma_samp[0] = model_lens.gamma
rein_samp[0] = model_lens.rein
beta_samp[0] = model_lens.beta
s2_samp[0] = model_lens.source**2

psi1_samp[0] = model_lens.rein
psi2_samp[0] = model_lens.psi2()
psi3_samp[0] = model_lens.psi3()

xA_samp[0] = model_lens.images[0]
xB_samp[0] = model_lens.images[1]

radmagrat_samp[0] = model_lens.radmag_ratio

dt_samp[0] = model_lens.timedelay/day

for i in range(nsamp-1):

    # draws new value of Einstein radius

    logp_rein = 0.*rein_grid

    for j in range(nrein_grid):
        model_lens.rein = rein_grid[j]
        model_lens.get_images()

        if np.isfinite(model_lens.images[0]):
            model_lens.get_radmag_ratio()

            logp_rein[j] = -0.5*(model_lens.images[0] - xa_obs)**2/imerr**2 - 0.5*(model_lens.images[1] - xb_obs)**2/imerr**2
            #logp_rein[j] += -0.5*(model_lens.radmag_ratio - radmagrat_obs)**2/radmagrat_err**2
        else:
            logp_rein[j] = -np.inf

    logp_rein -= logp_rein.max()

    p_rein_grid = np.exp(logp_rein)

    p_rein_spline = splrep(rein_grid, p_rein_grid)

    intfunc = lambda t: splint(rein_grid[0], t, p_rein_spline)

    norm = intfunc(rein_grid[-1])

    F = np.random.rand(1) * norm

    rein_samp[i+1] = brentq(lambda t: intfunc(t) - F, rein_grid[0], rein_grid[-1])

    model_lens.rein = rein_samp[i+1]

    # draws new value of gamma

    logp_gamma = 0.*gamma_grid

    for j in range(ngamma_grid):
        model_lens.gamma = gamma_grid[j]
        model_lens.get_images()

        if np.isfinite(model_lens.images[0]) and model_lens.images[0] == model_lens.images[0]:
            model_lens.get_radmag_ratio()

            logp_gamma[j] = -0.5*(model_lens.images[0] - xa_obs)**2/imerr**2 - 0.5*(model_lens.images[1] - xb_obs)**2/imerr**2
            logp_gamma[j] += -0.5*(model_lens.radmag_ratio - radmagrat_obs)**2/radmagrat_err**2
        else:
            logp_gamma[j] = -np.inf

    logp_gamma -= logp_gamma.max()

    p_gamma_grid = np.exp(logp_gamma)

    p_gamma_spline = splrep(gamma_grid, p_gamma_grid)

    intfunc = lambda t: splint(gamma_grid[0], t, p_gamma_spline)

    norm = intfunc(gamma_grid[-1])

    F = np.random.rand(1) * norm

    gamma_samp[i+1] = brentq(lambda t: intfunc(t) - F, gamma_grid[0], gamma_grid[-1])

    model_lens.gamma = gamma_samp[i+1]
    model_lens.get_images()

    # draws new value of beta

    logp_beta = 0.*beta_grid

    for j in range(nbeta_grid):
        model_lens.beta = beta_grid[j]
        model_lens.get_images()

        if np.isfinite(model_lens.images[0]):
            model_lens.get_radmag_ratio()

            logp_beta[j] = -0.5*(model_lens.images[0] - xa_obs)**2/imerr**2 - 0.5*(model_lens.images[1] - xb_obs)**2/imerr**2
            logp_beta[j] += -0.5*(model_lens.radmag_ratio - radmagrat_obs)**2/radmagrat_err**2
        else:
            logp_beta[j] = -np.inf

    logp_beta -= logp_beta.max()

    p_beta_grid = np.exp(logp_beta)

    p_beta_spline = splrep(beta_grid, p_beta_grid)

    intfunc = lambda t: splint(beta_grid[0], t, p_beta_spline)

    norm = intfunc(beta_grid[-1])

    F = np.random.rand(1) * norm

    beta_samp[i+1] = brentq(lambda t: intfunc(t) - F, beta_grid[0], beta_grid[-1])

    model_lens.beta = beta_samp[i+1]

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

    model_lens.get_images()
    model_lens.get_radmag_ratio()
    model_lens.get_timedelay()

    xA_samp[i+1] = model_lens.images[0]
    xB_samp[i+1] = model_lens.images[1]
    radmagrat_samp[i+1] = model_lens.radmag_ratio
    dt_samp[i+1] = model_lens.timedelay/day

    s2_samp[i+1] = model_lens.source**2

    psi1_samp[i+1] = model_lens.rein
    psi2_samp[i+1] = model_lens.psi2()
    psi3_samp[i+1] = model_lens.psi3()

    print '%d %3.2f %3.2f %3.2f %3.2f'%(i+1, rein_samp[i+1], gamma_samp[i+1], beta_samp[i+1], s2_samp[i+1]**0.5)

chain_file = h5py.File('onelens_broken_alpha_gibbs_sample.hdf5', 'w')

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

chain_file.create_dataset('rein', data=rein_samp)
chain_file.create_dataset('gamma', data=gamma_samp)
chain_file.create_dataset('beta', data=beta_samp)
chain_file.create_dataset('s2', data=s2_samp)
chain_file.create_dataset('dt_model', data=dt_samp)
chain_file.create_dataset('radmagrat', data=radmagrat_samp)
chain_file.create_dataset('xA', data=xA_samp)
chain_file.create_dataset('xB', data=xB_samp)
chain_file.create_dataset('psi1', data=psi1_samp)
chain_file.create_dataset('psi2', data=psi2_samp)
chain_file.create_dataset('psi3', data=psi3_samp)

