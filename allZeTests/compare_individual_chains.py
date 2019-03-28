import pylab
import pickle
from allZeTools.plotters import cornerplot
import numpy as np

f = open('mock_D_100lenses.dat', 'r')
mock, chains = pickle.load(f)
f.close()

f = open('mock_D_100lenses_Nind1e5.dat', 'r')
longmock, longchains = pickle.load(f)
f.close()

nlens = len(chains)

for i in range(3):

    cp = []
    cp.append({'data': [longchains[i]['mstar'], chains[i]['mstar']], 'label': '$\log{M_*}$', 'value': np.log10(mock['lenses'][i].mstar)})
    cp.append({'data': [longchains[i]['mhalo'], chains[i]['mhalo']], 'label': '$\log{M_h}$', 'value': np.log10(mock['lenses'][i].mhalo)})
    cp.append({'data': [longchains[i]['alpha'], chains[i]['alpha']], 'label': '$\log{\\alpha_{\mathrm{IMF}}}$', 'value': mock['aimf_sample'][i]})
    cp.append({'data': [longchains[i]['source'], chains[i]['source']], 'label': '$s$', 'value': mock['lenses'][i].source})
    cp.append({'data': [longchains[i]['timedelay'].flatten(), chains[i]['timedelay'].flatten()], 'label': '$\Delta t$', 'value': mock['lenses'][i].timedelay})
    cornerplot(cp, color='g')
    #pylab.savefig('frames/mock_F_100lenses_wradmagfit_lens%02d.jpg'%i)
    #pylab.close()
    pylab.show()

