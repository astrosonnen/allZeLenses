import numpy as np
import pickle
import pylab


# In all H0licow lenses studied so far, the H0 inferred with powerlaw is always very close to that inferred with the composite model. Why is that?
# What kind of true density profile do you need for that to happen?
# If the true density profile is a power-law, it's not obvious that you can fit it with a composite model and get out the same H0.
# Similarly, if the true density profile is composite, fitting it with a power-law will in general give you the wrong H0.
# In this code, I assume the truth to be composite. I take lenses from the existing mocks, then select the ones for which a power-law model gives the same H0 (based on the relation between psi'' and psi''').
# Then I modify lens and source redshift in order to get a different Einstein radius. Is the power-law approximation still valid?

mockname = 'mockP'

f = open('../paper/%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

psi1_mock = np.zeros(nlens)
psi2_mock = np.zeros(nlens)
psi3_mock = np.zeros(nlens)

ind_H0 = np.zeros(nlens)

fit_file = h5py.File('../paper/%s_powerlaw_perfectobs_lensing.hdf5'%mockname, 'r')

eps = 1e-4

for i in range(nlens):
    print i
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
print psi2_diff[good].min()

pylab.scatter(psi2_inferred[good] - psi2_mock[good], ind_H0[good], color='b')

