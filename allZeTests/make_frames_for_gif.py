import pylab
import pickle
from allZeTools.plotters import cornerplot
import numpy as np

f = open('simple_reality_2lenses_mstarmhaloprior.dat', 'r')
mock, chains = pickle.load(f)
f.close()

nlens = len(chains)

for i in range(3):

    cp = []
    cp.append({'data': chains[i]['mstar'], 'label': '$\log{M_*}$', 'value': np.log10(mock['lenses'][i].mstar)})
    cp.append({'data': chains[i]['mhalo'], 'label': '$\log{M_h}$', 'value': np.log10(mock['lenses'][i].mhalo)})
    cp.append({'data': chains[i]['alpha'], 'label': '$\log{\\alpha_{\mathrm{IMF}}}$', 'value': mock['aimf_sample'][i]})
    cp.append({'data': chains[i]['source'], 'label': '$s$', 'value': mock['lenses'][i].source})
    cp.append({'data': chains[i]['timedelay'].flatten(), 'label': '$\Delta t$', 'value': mock['lenses'][i].timedelay})
    cornerplot(cp, color='g')
    #pylab.savefig('frames/mock_F_100lenses_wradmagfit_lens%02d.jpg'%i)
    #pylab.close()
    pylab.show()

