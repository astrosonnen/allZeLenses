import h5py
import numpy as np
#from scipy.stats import histogram
import pylab
from statistics import weighted_percentile


nbins = 50
bins = np.linspace(50., 100., nbins+1)
centers = 0.5*(bins[1:] + bins[:-1])

nchains = 100

pl_hist = np.ones(nbins)
bpl_hist = np.ones(nbins)

chaindir = '/Users/sonnen/allZeChains/'

for i in range(nchains):
    pl_chain = h5py.File(chaindir+'mockJ_lens_%02d_powerlaw_wH0par.hdf5'%i, 'r')

    hist, edges = np.histogram(pl_chain['H0'].value, bins=bins)

    pl_hist *= hist

    bpl_chain = h5py.File(chaindir+'mockJ_lens_%02d_brokenpowerlaw_wH0par.hdf5'%i, 'r')

    hist, edges = np.histogram(bpl_chain['H0'].value, bins=bins)

    bpl_hist *= hist

pl_hist /= pl_hist.sum()
bpl_hist /= bpl_hist.sum()

#pylab.plot(centers, full_hist)
#pylab.show()

pylab.subplot(1, 2, 1)
pylab.hist(centers, bins=bins, weights=pl_hist, color='b')
pylab.xlim(65, 75)

pylab.subplot(1, 2, 2)
pylab.hist(centers, bins=bins, weights=bpl_hist, color='g')
pylab.xlim(65, 75)
pylab.show()

