import pickle
import h5py
import os
import pymc
import lens_models


chaindir = '/Users/sonnen/allZeChains/'

mockname = 'mockI'

nstep = 20000
burnin = 10000

lensno = 6

day = 24.*3600.

f = open(mockname+'.dat', 'r')
mock = pickle.load(f)
f.close()

chainname = '%s_lens_%02d_powerlaw_highprecision.hdf5'%(mockname, lensno)

chain_file = h5py.File(chainname, 'w')

lens = mock['lenses'][lensno]
print 'sampling lens %d...'%lensno

xA, xB = lens.images
    
#xa_obs, xb_obs = lens.obs_images[0]
xa_obs, xb_obs = lens.images
imerr = lens.obs_images[1]

#radmagrat_obs, radmagrat_err = lens.obs_radmagrat
radmagrat_obs = lens.radmag_ratio
radmagrat_err = 0.010
#timedelay_obs, timedelay_err = lens.obs_timedelay
timedelay_obs = lens.timedelay
timedelay_err = 0.01 * day

rein_guess = 0.5*(xA - xB)

model_lens = lens_models.powerlaw(zd=lens.zd, zs=lens.zs, rein=rein_guess, gamma=2., images=lens.images, source=lens.source)

gamma_par = pymc.Uniform('gamma', lower=1.2, upper=2.8, value=2.)
rein_par = pymc.Uniform('rein', lower=0.2*rein_guess, upper=3.*rein_guess, value=rein_guess)
s2_par = pymc.Uniform('s2', lower=0., upper=(xa_obs)**2, value=(xA - model_lens.alpha(xA))**2)

H0_par = pymc.Uniform('H0', lower=50., upper=100., value=70.)

@pymc.deterministic()
def images(rein=rein_par, gamma=gamma_par, s2=s2_par):
    model_lens.rein = rein
    model_lens.gamma = gamma
    model_lens.source = s2**0.5
    model_lens.get_b_from_rein()
    
    model_lens.get_images()
    
    return (model_lens.images[0], model_lens.images[1])
   
@pymc.deterministic()
def imA(rein=rein_par, gamma=gamma_par, s2=s2_par):
    return float(images[0])
    
@pymc.deterministic()
def imB(rein=rein_par, gamma=gamma_par, s2=s2_par):
    return float(images[1])
    
@pymc.deterministic()
def radmagrat(rein=rein_par, gamma=gamma_par, s2=s2_par):
    model_imA, model_imB = images
    
    model_lens.rein = rein
    model_lens.gamma = gamma
    model_lens.source = s2**0.5
    
    model_lens.get_b_from_rein()
    
    return model_lens.mu_r(model_imA)/model_lens.mu_r(model_imB)
    
@pymc.deterministic()
def timedelay(rein=rein_par, gamma=gamma_par, s2=s2_par, H0=H0_par):
    model_imA, model_imB = images
    
    model_lens.rein = rein
    model_lens.gamma = gamma
    model_lens.source = s2**0.5
    
    model_lens.get_b_from_rein()
    
    model_lens.images = (model_imA, model_imB)
    
    model_lens.get_timedelay()
    
    return model_lens.timedelay * 70. / H0
    
ima_logp = pymc.Normal('ima_logp', mu=imA, tau=1./imerr**2, value=xa_obs, observed=True)
imb_logp = pymc.Normal('imb_logp', mu=imB, tau=1./imerr**2, value=xb_obs, observed=True)
    
radmag_logp = pymc.Normal('radmag_logp', mu=radmagrat, tau=1./radmagrat_err**2, value=radmagrat_obs, observed=True)

timedelay_logp = pymc.Normal('timedelay_logp', mu=timedelay, tau=1./timedelay_err**2, value=timedelay_obs, observed=True)
    
pars = [gamma_par, rein_par, s2_par, H0_par]
allpars = pars + [images, radmagrat, timedelay]
    
M = pymc.MCMC(allpars)
M.use_step_method(pymc.AdaptiveMetropolis, pars)
M.sample(nstep, burnin)
    
for par in allpars:
    chain_file.create_dataset(str(par), data=M.trace(par)[:])

