import pylab
import pickle
import numpy as np
from matplotlib import rc
rc('text', usetex=True)


tsize = 16
fsize = 20

f = open('mockP.dat', 'r')
mock = pickle.load(f)
f.close()

bins = np.linspace(0., 4., 21)

nlens = len(mock['lenses'])

rein = np.zeros(nlens)

for i in range(nlens):
    rein[i] = mock['lenses'][i].rein

fig = pylab.figure()
pylab.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.98)
pylab.hist(rein, bins=bins, histtype='stepfilled', color='gray')
pylab.ylim(0., 20.)
pylab.xlabel("$\\theta_{\mathrm{E}}\,('')$", fontsize=fsize)
pylab.ylabel('$N$', fontsize=fsize)
pylab.xticks(fontsize=tsize)
pylab.yticks(fontsize=tsize)
pylab.savefig('rein_hist.eps')
pylab.show()

