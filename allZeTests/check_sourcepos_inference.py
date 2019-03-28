import pickle
import pylab
import numpy as np


f = open('mockF.dat', 'r')
mock = pickle.load(f)
f.close()

nlens = len(mock['lenses'])

rein = np.zeros(nlens)
source = np.zeros(nlens)

for i in range(nlens):
    rein[i] = mock['lenses'][i].rein
    source[i] = mock['lenses'][i].source

f = open('mockF_sourcepos_inference.dat', 'r')
chain = pickle.load(f)
f.close()

s2_loga = np.median(chain['s2_loga'])
s2_b = np.median(chain['s2_b'])

tpa = 2.*np.pi * 10.**(s2_loga + 1.)


pylab.hist((source/rein)**2)
xlim = pylab.xlim()
xs = np.linspace(xlim[0], xlim[1])

dist = 5.*(0.5*np.pi + np.arctan(tpa*(s2_b - xs)))
pylab.plot(xs, dist, color='r')
pylab.show()

