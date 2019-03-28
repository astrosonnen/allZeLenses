import numpy as np
import lens_models
import pylab
from plotters import cornerplot
import pymc
import pickle
from scipy.interpolate import splev


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
radmagrat_err = 0.02

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

mhalo_par = pymc.Uniform('mhalo', lower=11., upper=15., value=np.log10(lens.mhalo))
mstar_par = pymc.Uniform('mstar', lower=10., upper=13., value=np.log10(lens.mstar))

cvir_par = pymc.Uniform('cvir', lower=0., upper=2., value=np.log10(lens.cvir))

s2_par = pymc.Uniform('s2', lower=0., upper=xa_obs**2, value=model_lens.source**2)

pars = [mhalo_par, mstar_par, cvir_par, s2_par]

@pymc.deterministic()
def images(p=pars):

    mhalo, mstar, cvir, s2 = p

    model_lens.source = s2**0.5
    model_lens.mstar = 10.**mstar
    model_lens.mhalo = 10.**mhalo
    model_lens.cvir = 10.**cvir
    
    model_lens.normalize()

    model_lens.get_caustic()
    model_lens.get_images()

    if len(model_lens.images) < 2:
        return np.inf, -np.inf
    else:
        return model_lens.images
    
@pymc.deterministic()
def image_a(imgs=images):
    return imgs[0]
    
@pymc.deterministic()
def image_b(imgs=images):
    return imgs[1]
   
@pymc.deterministic()
def timedelay(p=pars, imgs=images):

    mhalo, mstar, cvir, s2 = p

    model_lens.source = s2**0.5
    model_lens.mstar = 10.**mstar
    model_lens.mhalo = 10.**mhalo
    model_lens.cvir = 10.**cvir
    
    model_lens.normalize()

    model_lens.images = imgs
    
    if not np.isfinite(imgs[0]):
        return 0.
    else:
        model_lens.get_timedelay()
        return model_lens.timedelay

@pymc.deterministic()
def rein(mstar=mstar_par, mhalo=mhalo_par, cvir=cvir_par):

    model_lens.mstar = 10.**mstar
    model_lens.mhalo = 10.**mhalo
    model_lens.cvir = 10.**cvir
    
    model_lens.normalize()

    model_lens.get_rein()

    return model_lens.rein

@pymc.deterministic()
def radmagratio(mstar=mstar_par, mhalo=mhalo_par, cvir=cvir_par, imgs=images):

    model_lens.mstar = 10.**mstar
    model_lens.mhalo = 10.**mhalo
    model_lens.cvir = 10.**cvir
    
    model_lens.normalize()

    model_lens.images = imgs

    model_lens.get_radmag_ratio()

    return model_lens.radmag_ratio

@pymc.deterministic()
def psi2(mstar=mstar_par, mhalo=mhalo_par, cvir=cvir_par, re=rein):

    model_lens.mstar = 10.**mstar
    model_lens.mhalo = 10.**mhalo
    model_lens.cvir = 10.**cvir
    
    model_lens.normalize()

    return (model_lens.alpha(re + eps) - model_lens.alpha(re - eps))/(2.*eps)

@pymc.deterministic()
def psi3(mstar=mstar_par, mhalo=mhalo_par, cvir=cvir_par, re=rein):

    model_lens.mstar = 10.**mstar
    model_lens.mhalo = 10.**mhalo
    model_lens.cvir = 10.**cvir
    
    model_lens.normalize()

    return (model_lens.alpha(re + eps) - 2.*model_lens.alpha(re) + model_lens.alpha(re - eps))/eps**2
    
ima_logp = pymc.Normal('ima_logp', mu=image_a, tau=1./imerr**2, value=xa_obs, observed=True)
imb_logp = pymc.Normal('imb_logp', mu=image_b, tau=1./imerr**2, value=xb_obs, observed=True)

radmagrat_logp = pymc.Normal('radmagrat_logp', mu=radmagratio, tau=1./radmagrat_err**2, value=radmagrat_obs, observed=True)

#sigma_logp = pymc.Normal('sigma_logp', mu=sigma, tau=1./sigma_err, value=sigma_obs, observed=True)
    
allpars = pars + [timedelay, image_a, image_b, rein, psi2, psi3, radmagratio]
    
M = pymc.MCMC(allpars)
M.use_step_method(pymc.AdaptiveMetropolis, pars)
M.sample(110000, 10000)

chain = {}
for par in allpars:
    chain[str(par)] = M.trace(par)[:].flatten()

f = open('onelens_pymc_sample.dat', 'w')
pickle.dump(chain, f)
f.close()

cp = []
cp.append({'data': chain['mhalo'], 'label': '$\log{M_h}$', 'value': np.log10(lens.mhalo)})
cp.append({'data': chain['mstar'], 'label': '$\log{M_*}$', 'value': np.log10(lens.mstar)})
cp.append({'data': chain['s2']**0.5, 'label': '$s$', 'value': lens.source})
cp.append({'data': chain['cvir'], 'label': '$\log{c_h}$', 'value': np.log10(lens.cvir)})
cp.append({'data': chain['timedelay']/day, 'label': '$\Delta t$', 'value': lens.timedelay/day})
cp.append({'data': chain['psi2'], 'label': "$\psi''$", 'value': psi2_true})
cp.append({'data': chain['psi3'], 'label': "$\psi'''$", 'value': psi3_true})

cornerplot(cp, color='g')
pylab.savefig('onelens_chain_cp_pymc.png')
#pylab.show()

