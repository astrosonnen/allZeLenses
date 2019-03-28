import numpy as np
from scipy.interpolate import splrep, splev, splint
from vdmodel_2013 import sigma_model
from mass_profiles import sersic as sersic_profile, gNFW as gNFW_profile, powerlaw


def get_iso_dyndic(lens, aperture):
    # calculates spherical Jeans models for a SL2S V-like spherical cow lens.
    # Returns a dictionary dic = {'powerlaw':s2_pow,'bulge':s2_bulge,'halo':s2_spline_halo}
    # with the square of the luminosity-weighted line-of-sight velocity dispersion, integrated within a circular
    # aperture of radius 'aperture'
    # s2_bulge is a float corresponding to the contribution of the bulge to sigma^2, assuming mstar=1
    # s2_spline_halo is a spline in the slope of the dark matter halo, assuming mdm5=1

    arcsec2kpc = lens.reff_phys/lens.reff

    from scipy.integrate import quad

    ng = 28
    gammadm = np.linspace(0.1, 2.8, ng)

    # defines the tracer distribution, which is the same as the mass distribution of the bulge

    # calculates 3d density for sersic profile
    print 'calculating rho(r) for Sersic profile'

    b = sersic_profile.b(lens.n)

    Nr = 1001
    radii = np.logspace(-3, 3, Nr)
    rhos = 0.*radii
    for i in range(0, Nr):
        rhos[i] = -1./np.pi*quad(lambda R : -b/lens.n*(R/lens.reff)**(1/lens.n)/R*\
                                           sersic_profile.I(R, lens.n, lens.reff)/\
                                           np.sqrt(R**2 - radii[i]**2), radii[i], np.inf)[0]

    sersic_rho_spline = splrep(radii, rhos)

    radiiw0 = np.array([0.] + list(radii))
    mprimew0 = np.array([0.] + list(4.*np.pi*rhos*radii**2))

    mprime_sersic_spline = splrep(radiiw0, mprimew0)

    m3d_ser = 0.*radii
    for i in range(0, Nr):
        m3d_ser[i] = splint(0., radii[i], mprime_sersic_spline)

    def light_profile(r, lp_pars, proj=False):

        reff, n = lp_pars

        if proj:
            return sersic_profile.I(r, n, reff)
        else:
            return splev(r, sersic_rho_spline)

    s2_bulge = sigma_model.sigma2general((radii, m3d_ser), aperture, lp_pars=(lens.reff, lens.n), seeing=None, \
                                         light_profile=light_profile)

    s2_halo = 0.*gammadm

    for i in range(0, ng):
        norm = 1./gNFW_profile.M2d(5./arcsec2kpc, lens.rs, gammadm[i])
        m3d_halo = norm*gNFW_profile.M3d(radii, lens.rs, gammadm[i])

        s2_halo[i] = sigma_model.sigma2general((radii, m3d_halo), aperture, lp_pars=(lens.reff, lens.n), seeing=None, \
                                               light_profile=light_profile)

    # s2_halo_spline = splrep(gammadm,s2_halo)

    # now does the same for a powerlaw model, normalized to have unit mass within the effective radius
    ng = 17
    gammas = np.linspace(1.2, 2.8, ng)

    s2_pow = 0.*gammas

    for i in range(0, ng):

        m3d_pow = powerlaw.M3d(radii, gammas[i])/powerlaw.M2d(lens.reff, gammas[i])

        s2_pow[i] = sigma_model.sigma2general((radii, m3d_pow), \
                                              aperture, lp_pars=(lens.reff, lens.n), seeing=None, \
                                              light_profile=light_profile)

    # s2_pow_spline = splrep(gammas, s2_pow)

    # dyndic = {'powerlaw':s2_pow_spline, 'bulge':s2_bulge, 'halo':s2_halo_spline}
    dyndic = {'powerlaw': (gammas, s2_pow), 'bulge': s2_bulge, 'halo': (gammadm, s2_halo)}

    return dyndic


