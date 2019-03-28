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

bins = np.linspace(0., 1., 11)

nlens = len(mock['lenses'])

asymm = np.zeros(nlens)

for i in range(nlens):
    lens = mock['lenses'][i]
    asymm[i] = (lens.images[0] + lens.images[1])/(lens.images[0] - lens.images[1])

fig = pylab.figure()
pylab.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.98)
pylab.hist(asymm, bins=bins, histtype='stepfilled', color='gray')
pylab.ylim(0., 20.)
pylab.xlabel("$(\\theta_A + \\theta_B)/(\\theta_A - \\theta_B)$", fontsize=fsize)
pylab.ylabel('$N$', fontsize=fsize)
pylab.xticks(fontsize=tsize)
pylab.yticks(fontsize=tsize)
pylab.savefig('asymm_hist.eps')
pylab.show()

