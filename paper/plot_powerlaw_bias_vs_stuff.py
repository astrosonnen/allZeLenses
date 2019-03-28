import numpy as np
import pylab
import lens_models
import pickle
import h5py
#import pymc
#from matplotlib import rc
#rc('text', usetex=True)


mockname = 'mockP'

fsize = 20
tsize = 16

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

psi1_mock = np.zeros(nlens)
psi2_mock = np.zeros(nlens)
psi3_mock = np.zeros(nlens)

rein = np.zeros(nlens)

eps = 1e-4

taylor_true = np.zeros(nlens)
taylor_ml = np.zeros(nlens)

rein_reff = np.zeros(nlens)
rs_reff = np.zeros(nlens)

asymm = np.zeros(nlens)

ml_H0 = np.zeros(nlens)

psi1_ml = np.zeros(nlens)
psi2_ml = np.zeros(nlens)
psi3_ml = np.zeros(nlens)

fit_file = h5py.File('%s_powerlaw_perfectobs_wdyn.hdf5'%mockname, 'r')

for i in range(nlens):

    lens = mock['lenses'][i]
    psi1_mock[i] = lens.rein
    psi2_mock[i] = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
    psi3_mock[i] = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

    dtheta1 = lens.images[0] - lens.rein
    taylor_true[i] = 2.*psi1_mock[i]*(1. - psi2_mock[i]) * dtheta1 * (1. + 0.5*psi3_mock[i]/(1. - psi2_mock[i])*dtheta1)

    rein[i] = lens.rein
    rs_reff[i] = lens.rs/lens.reff

    dt_model = fit_file['timedelay'][i]

    ml_H0[i] = 70. * dt_model / lens.timedelay

    psi1_ml[i] = fit_file['psi1'][i]
    psi2_ml[i] = fit_file['psi2'][i]
    psi3_ml[i] = fit_file['psi3'][i]

    dtheta1 = lens.images[0] - fit_file['psi1'][i]
    taylor_ml[i] = 2.*psi1_ml[i]*(1. - psi2_ml[i]) * dtheta1 * (1. + 0.5*psi3_ml[i]/(1. - psi2_ml[i])*dtheta1)

    rein_reff[i] = lens.rein/lens.reff

    asymm[i] = (lens.images[0] + lens.images[1])/(lens.images[0] - lens.images[1])

    psit_true = psi3_mock[i]/(1. - psi2_mock[i])
    psit_ml = psi3_ml[i]/(1. - psi2_ml[i])
    print '%d, H0: %2.1f, rein/reff: %2.1f, psi2_off: %3.2f, psi3_off: %3.2f, psit_off: %3.2f'%(i, ml_H0[i], rein_reff[i], psi2_ml[i] - psi2_mock[i], psi1_ml[i]*psi3_ml[i] - psi3_mock[i]*psi3_mock[i], psit_ml - psit_true)

print ml_H0.mean(), ml_H0.std()

pylab.scatter(taylor_ml/taylor_true, ml_H0)
pylab.show()

pylab.scatter(rein_reff, ml_H0)
pylab.show()

pylab.scatter(rein_reff, psi2_ml - psi2_mock)
pylab.show()

pylab.scatter(psi2_ml - psi2_mock, ml_H0)
pylab.show()

