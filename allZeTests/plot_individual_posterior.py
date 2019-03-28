import pylab
import pickle
from allZeTools.plotters import cornerplot
import numpy as np
import lens_samplers

f = open('simple_reality_100lenses_mstarmhaloprior.dat', 'r')
mock, chains = pickle.load(f)
f.close()

chain = chains[0]
lens = mock['lenses'][0]

cp = []
cp.append({'data': chain['mstar'], 'label': '$\log{M_*}$', 'value': np.log10(lens.mstar)})
cp.append({'data': chain['mhalo'], 'label': '$\log{M_h}$', 'value': np.log10(lens.mhalo)})
cp.append({'data': chain['alpha'], 'label': '$\log{\\alpha_{\mathrm{IMF}}}$', 'value': mock['aimf_sample'][0]})
cp.append({'data': chain['source'], 'label': '$s$', 'value': lens.source})
cp.append({'data': chain['timedelay'].flatten(), 'label': '$\Delta t$', 'value': lens.timedelay})
cornerplot(cp, color='g')
pylab.savefig('simple_reality_100lenses_mstarmhaloprior_lens00.png')
pylab.show()

# now fits again the same lens but with a flat prior on mhalo

chain = lens_samplers.fit_nfwdev_nocvirscat_noradmagfit_nodtfit(lens, nstep=20000, burnin=10000)

cp = []
cp.append({'data': chain['mstar'], 'label': '$\log{M_*}$', 'value': np.log10(lens.mstar)})
cp.append({'data': chain['mhalo'], 'label': '$\log{M_h}$', 'value': np.log10(lens.mhalo)})
cp.append({'data': chain['alpha'], 'label': '$\log{\\alpha_{\mathrm{IMF}}}$', 'value': mock['aimf_sample'][0]})
cp.append({'data': chain['source'], 'label': '$s$', 'value': lens.source})
cp.append({'data': chain['timedelay'].flatten(), 'label': '$\Delta t$', 'value': lens.timedelay})
cornerplot(cp, color='g')
pylab.savefig('simple_reality_100lenses_mstarmhaloprior_lens00_wflatprior.png')
pylab.show()



