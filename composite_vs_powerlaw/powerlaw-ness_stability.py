import numpy as np
import pickle
import pylab
import h5py
import lens_models
import fit_powerlaw


# In all H0licow lenses studied so far, the H0 inferred with powerlaw is always very close to that inferred with the composite model. Why is that?
# What kind of true density profile do you need for that to happen?
# If the true density profile is a power-law, it's not obvious that you can fit it with a composite model and get out the same H0.
# Similarly, if the true density profile is composite, fitting it with a power-law will in general give you the wrong H0.
# In this code, I assume the truth to be composite. I take lenses from the existing mocks, then select the ones for which a power-law model gives the same H0 (based on the relation between psi'' and psi''').
# Then I modify lens and source redshift in order to get a different Einstein radius. Is the power-law approximation still valid?

mockname = 'mockO2019'

f = open('../paper/%s.dat'%mockname, 'rb')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

psi1_mock = np.zeros(nlens)
psi2_mock = np.zeros(nlens)
psi3_mock = np.zeros(nlens)

ind_H0 = np.zeros(nlens)

fit_file = h5py.File('../paper/%s_powerlaw_perfectobs_lensing.hdf5'%mockname, 'r')

eps = 1e-4
day = 24.*3600.
radmagrat_err = 0.020
dt_err = 1.

for i in range(nlens):
    print(i)
    lens = mock['lenses'][i]
    psi1_mock[i] = lens.rein
    psi2_mock[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    psi3_mock[i] = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

    ind_H0[i] = 70. * fit_file['timedelay'][i] / lens.timedelay 

xi = psi1_mock * psi3_mock - psi2_mock * (psi2_mock - 1.)
xo = psi3_mock/(1.-psi2_mock) - psi2_mock

psi2_diff = psi2_mock - 0.5*(1. - (1. + 4.*psi3_mock*psi1_mock)**0.5)

gamma_fit = fit_file['gamma'].value.copy()

psi2_inferred = fit_file['psi2'].value.copy()

good = (gamma_fit > 1.5) & (gamma_fit < 2.5) & (psi1_mock > 0.5)

psi2_diff = psi2_inferred - psi2_mock

pl = (abs(psi2_diff) < 0.01) & good
not_pl = (abs(psi2_diff) >= 0.01) & good

"""
pylab.scatter(psi2_inferred[not_pl] - psi2_mock[not_pl], ind_H0[not_pl], color='b')
pylab.scatter(psi2_inferred[pl] - psi2_mock[pl], ind_H0[pl], color='g')
pylab.show()
"""

npl = pl.sum()
print(npl)
colors = ['b', 'g', 'r', 'orange']
count = 0

nz = 21
zs_grid = np.linspace(1., 3., nz)

for n in range(nlens):
    if pl[n]:
        lens = mock['lenses'][n]
        psi1_fit = 0.*zs_grid
        psi2_fit = 0.*zs_grid
        psi3_fit = 0.*zs_grid

        psi1_grid = 0.*zs_grid
        psi2_grid = 0.*zs_grid
        psi3_grid = 0.*zs_grid
        H0_grid = 0.*zs_grid

        for i in range(nz):
            newlens = lens_models.NfwDev(zd=lens.zd, zs=zs_grid[i], mhalo=lens.mhalo, mstar=lens.mstar, reff_phys=lens.reff_phys, cvir=lens.cvir, source=lens.source)

            newlens.normalize()
            newlens.get_rein()
            newlens.get_caustic()

            newlens.get_images()
            newlens.get_radmag_ratio()
            newlens.get_timedelay()

            newlens.obs_images = ((newlens.images[0], newlens.images[1]), 0.01)
            newlens.obs_radmagrat = (newlens.radmag_ratio, radmagrat_err)
            newlens.obs_timedelay = (newlens.timedelay, dt_err*day)

            psi1_grid[i] = newlens.rein
            psi2_grid[i] = (newlens.alpha(newlens.rein + eps) - newlens.alpha(newlens.rein - eps))/(2.*eps)
            psi3_grid[i] = (newlens.alpha(newlens.rein + eps) - 2.*newlens.alpha(newlens.rein) + newlens.alpha(newlens.rein - eps))/eps**2

            dt_fit, gamma_fit, p1_fit, p2_fit, p3_fit = fit_powerlaw.fit_pl(newlens)

            psi2_fit[i] = p2_fit

            H0_grid[i] = 70. * dt_fit / newlens.timedelay 

        pylab.plot(psi2_fit - psi2_grid, H0_grid, color=colors[count])
        pylab.scatter(psi2_inferred[n] - psi2_mock[n], ind_H0[n], color=colors[count])
        count += 1

pylab.show()

