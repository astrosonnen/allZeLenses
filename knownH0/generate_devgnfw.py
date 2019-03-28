import numpy as np
import lens_models
import pickle
from mass_profiles import sersic as sersic_profile, gNFW as gNFW_profile
from scipy.stats import truncnorm


mockname = 'devgnfw_A'

nlens = 100

day = 24.*3600.

h = 0.7

rs2reff = 10.

minmag = 1.

mchab_mu = 11.5
mchab_sig = 0.3
mchab_err = 0.1

mstar_piv = 11.6

aimf_mu = 0.1
aimf_sig = 0.05

reff_mu = 0.85
reff_beta = 1.
reff_sig = 0.15

mdm5_mu = 11.
mdm5_sig = 0.2
mdm5_beta = 0.3

gammadm_mu = 1.3
gammadm_sig = 0.1

zd_mu = 0.4
zd_sig = 0.2
zd_min = 0.1
zd_max = 0.7

zs_mu = 1.8
zs_sig = 0.5
zs_min = 0.8

lmchab_samp = np.random.normal(mchab_mu, mchab_sig, nlens)
laimf_samp = np.random.normal(aimf_mu, aimf_sig, nlens)

lmstar_samp = lmchab_samp + laimf_samp

lreff_samp = reff_mu + reff_beta * (lmstar_samp - mstar_piv)

lmchab_obs = lmchab_samp + np.random.normal(0., mchab_err, nlens)

lmdm5_samp = mdm5_mu + mdm5_beta * (lmstar_samp - mstar_piv) + np.random.normal(0., mdm5_sig, nlens)
gammadm_samp = np.random.normal(gammadm_mu, gammadm_sig, nlens)

a, b = (zd_min - zd_mu)/zd_sig, (zd_max - zd_mu)/zd_sig
zd_samp = truncnorm.rvs(a, b, size=nlens)*zd_sig + zd_mu

a, b = (zs_min - zs_mu)/zs_sig, (np.inf - zs_mu)/zs_sig
zs_samp = truncnorm.rvs(a, b, size=nlens)*zs_sig + zs_mu

hp = {}
hp['mchab_mu'] = mchab_mu
hp['mchab_sig'] = mchab_sig
hp['aimf_mu'] = aimf_mu
hp['aimf_sig'] = aimf_sig
hp['reff_mu'] = reff_mu
hp['reff_sig'] = reff_sig
hp['reff_beta'] = reff_beta
hp['mdm5_mu'] = mdm5_mu
hp['mdm5_sig'] = mdm5_sig
hp['mdm5_beta'] = mdm5_beta
hp['gammadm_mu'] = gammadm_mu
hp['gammadm_sig'] = gammadm_sig
hp['zd_mu'] = zd_mu
hp['zd_sig'] = zd_sig
hp['zs_mu'] = zs_mu
hp['zs_sig'] = zs_sig

lenses = []

beta_samp = np.zeros(nlens)
thetaA_samp = np.zeros(nlens)
thetaB_samp = np.zeros(nlens)
timedelay_samp = np.zeros(nlens)

for i in range(nlens):

    reff = 10.**lreff_samp[i]
    rs = rs2reff * reff
    mdme = 10.**lmdm5_samp[i] / gNFW_profile.fast_M2d(5., rs, gammadm_samp[i]) * gNFW_profile.fast_M2d(reff, rs, gammadm_samp[i])

    lens = lens_models.gNfwDev(zd=zd_samp[i], zs=zs_samp[i], mstar=10.**lmstar_samp[i], mdme=mdme, \
                              reff_phys=reff, rs_phys=rs, beta=gammadm_samp[i], h=h)

    lens.normalize()
    lens.get_rein()
    lens.get_caustic()

    # source position: uniform distribution in a circle
    print i, lens.zd, lens.zs, np.log10(lens.mstar), np.log10(lens.mdme), lens.radcrit, lens.rein
    lens.get_xy_minmag(min_mag=minmag)
    ysource = (np.random.rand(1))**0.5*lens.yminmag

    lens.source = ysource
    lens.get_images()
    lens.get_timedelay()

    thetaA_samp[i] = lens.images[0]
    thetaB_samp[i] = lens.images[1]

    timedelay_samp[i] = lens.timedelay

    beta_samp[i] = lens.source[0]

    lens.obs_lmstar = (lmchab_obs[i], mchab_err)

    if lens.images is None:
        df

    lenses.append(lens)

output = {}

output['hp'] = hp

output['lmchab_samp'] = lmchab_samp
output['lmstar_samp'] = lmstar_samp
output['laimf_samp'] = laimf_samp
output['lreff_samp'] = lreff_samp
output['lmchab_obs'] = lmchab_obs
output['lmdm5_samp'] = lmdm5_samp
output['gammadm_samp'] = gammadm_samp
output['zd_samp'] = zd_samp
output['zs_samp'] = zs_samp
output['beta_samp'] = beta_samp
output['thetaA_samp'] = thetaA_samp
output['thetaB_samp'] = thetaB_samp
output['timedelay_samp'] = timedelay_samp

output['lenses'] = lenses

f = open('%s.dat'%mockname, 'w')
pickle.dump(output, f)
f.close()

   



