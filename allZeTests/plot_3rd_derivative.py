import numpy as np
import pylab
import lens_models
from toy_models import sample_generator


fsize = 24

ng = 101
gamma = np.linspace(1.5, 2.5, ng)

psi1 = 1.
psi2 = 2. - gamma
psi3 = (2. - gamma)*(1. - gamma)/psi1

ngal = 100
mock = sample_generator.simple_reality_sample(nlens=ngal, aimf_mu=0., mhalo_mu=13.3, mstar_mhalo=0.7)

psi1_nfwdev = np.zeros(ngal)
psi2_nfwdev = np.zeros(ngal)
psi3_nfwdev = np.zeros(ngal)

eps = 1e-4
for i in range(ngal):
    lens = mock['lenses'][i]
    lens.get_rein()
    psi1_nfwdev[i] = lens.rein
    psi2_nfwdev[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    psi3_nfwdev[i] = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

mock = sample_generator.simple_reality_sample(nlens=ngal, aimf_mu=-0.1, cvir_mu=0.877+0.3)

psi1_hicvir = np.zeros(ngal)
psi2_hicvir = np.zeros(ngal)
psi3_hicvir = np.zeros(ngal)

eps = 1e-4
for i in range(ngal):
    lens = mock['lenses'][i]
    lens.get_rein()
    psi1_hicvir[i] = lens.rein
    psi2_hicvir[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    psi3_hicvir[i] = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

# now creates broken powerlaw lenses
gmb_bpl = np.random.normal(2., 0.1, ngal)
beta_bpl = np.random.normal(0.1, 0.2, ngal)
gamma_bpl = gmb_bpl + beta_bpl

print gamma_bpl.mean(), beta_bpl.mean()

psi1_bpl = np.ones(ngal)

const_bpl = np.zeros(ngal)

bpl = lens_models.sps_ein_break()

for i in range(ngal):
    bpl.gamma = gamma_bpl[i]
    bpl.beta = beta_bpl[i]
    const_bpl[i] = bpl.const()

psi2_bpl = const_bpl/(3.-gamma_bpl)*((3.-gamma_bpl)*(2.-gamma_bpl)*2.**beta_bpl + 2.*(3.-gamma_bpl)*beta_bpl*2.**(beta_bpl-1.) + beta_bpl*(beta_bpl-1.)*2.**(beta_bpl-2.))

psi3_bpl = const_bpl/(3.-gamma_bpl)*((3.-gamma_bpl)*(2.-gamma_bpl)*(1.-gamma_bpl)*2.**beta_bpl + 3.*beta_bpl*(3.-gamma_bpl)*(2.-gamma_bpl)*2.**(beta_bpl-1.) + 3.*beta_bpl*(beta_bpl-1.)*(3.-gamma_bpl)*2.**(beta_bpl-2.) + beta_bpl*(beta_bpl-1.)*(beta_bpl-2.)*2.**(beta_bpl-3.))

fig = pylab.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twiny()

ax1.plot(psi2, psi1*psi3, label='Power-law')
xlim = ax1.set_xlim(psi2[-1], psi2[0])

ax1.scatter(psi2_nfwdev, psi1_nfwdev*psi3_nfwdev, color='r', label='deV + NFW')
#pylab.scatter(psi2_hicvir, psi1_hicvir*psi3_hicvir, color='g')
ax1.scatter(psi2_bpl, psi1_bpl*psi3_bpl, color='g', label='Broken powerlaw')
ax1.legend(loc = 'upper right', scatterpoints=1, fontsize=fsize)
ax1.set_xlabel("$\psi''$", fontsize=fsize)
ax1.set_ylabel("$\psi'\psi'''$", fontsize=fsize)
ax2.set_xlim(gamma[0], gamma[-1])
ax2.set_xlabel('$\gamma$', fontsize=fsize)
pylab.savefig('psi3_composite_vs_powerlaw.png')
pylab.show()

# now plots psi2, psi3 as a function of stuff

rein_sample = np.zeros(ngal)
reff_ang = np.zeros(ngal)
for i in range(ngal):
    rein_sample[i] = mock['lenses'][i].rein
    reff_ang[i] = mock['lenses'][i].reff

pylab.subplot(2, 2, 1)
pylab.scatter(mock['mstar_sample'] - 2.*np.log10(mock['reff_sample']), psi2_nfwdev)

pylab.subplot(2, 2, 2)
pylab.scatter(rein_sample/reff_ang, psi2_nfwdev)

pylab.subplot(2, 2, 3)
pylab.scatter(mock['mstar_sample'] - 2.*np.log10(mock['reff_sample']), psi1_nfwdev*psi3_nfwdev)

pylab.subplot(2, 2, 4)
pylab.scatter(rein_sample/reff_ang, psi1_nfwdev*psi3_nfwdev)
pylab.show()


