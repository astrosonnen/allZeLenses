import pylab
import numpy as np
import gNFW
from scipy.misc import derivative

beta = 1.5
rs = 30.

Nr = 51
r = np.linspace(0.,10.,Nr)

M2ds = gNFW.M2d(r,rs,beta)

alphas = M2ds/r/np.pi

lenspot = gNFW.lenspot(r,rs,beta)

def potfunc(r):
    return gNFW.lenspot(r,rs,beta)

gradpot = 0.*r
for i in range(0,Nr):
    gradpot[i] = derivative(potfunc,r[i],dx=0.01)

pylab.plot(r,alphas)
pylab.plot(r,gradpot)
pylab.show()

