import pickle
import h5py
import numpy as np
import pylab


f = open('mockI.dat', 'r')
mock = pickle.load(f)
f.close()

gibbs = h5py.File('tmp_mockI_brokenpl_gibbs_sample_fast.hdf5', 'r')

nlens = gibbs['beta'].value.shape[0]
nlens = 10

end = 400

day = 24.*3600.

for i in range(nlens):

    pylab.subplot(3, 2, 1)

    pylab.plot(gibbs['xA_model'][i, 1:end])
    pylab.axhspan(mock['lenses'][i].obs_images[0][0] - mock['lenses'][i].obs_images[1], mock['lenses'][i].obs_images[0][0] + mock['lenses'][i].obs_images[1], color='gray', alpha=0.5)

    pylab.subplot(3, 2, 2)

    pylab.plot(gibbs['xB_model'][i, 1:end])
    pylab.axhspan(mock['lenses'][i].obs_images[0][1] - mock['lenses'][i].obs_images[1], mock['lenses'][i].obs_images[0][1] + mock['lenses'][i].obs_images[1], color='gray', alpha=0.5)

    pylab.subplot(3, 2, 3)

    pylab.plot(gibbs['radmagrat_model'][i, 1:end])
    pylab.axhspan(mock['lenses'][i].obs_radmagrat[0] - mock['lenses'][i].obs_radmagrat[1], mock['lenses'][i].obs_radmagrat[0] + mock['lenses'][i].obs_radmagrat[1], color='gray', alpha=0.5)

    pylab.subplot(3, 2, 4)

    pylab.plot(gibbs['invH0'][1:end] * gibbs['dt_model'][i, 1:end] * 70.)
    pylab.axhspan(mock['lenses'][i].obs_timedelay[0]/day - mock['lenses'][i].obs_timedelay[1]/day, mock['lenses'][i].obs_timedelay[0]/day + mock['lenses'][i].obs_timedelay[1]/day, color='gray', alpha=0.5)

    pylab.subplot(3, 2, 5)
    pylab.plot(gibbs['gamma'][i, 1:end])

    pylab.subplot(3, 2, 6)
    pylab.plot(gibbs['beta'][i, 1:end])
    #pylab.plot(1./gibbs['invH0'][1:end])

    pylab.show()



