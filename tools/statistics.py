import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import splrep,splev,splint
import scipy

def general_random(func,N,interval = (0.,1.)):
    xs = np.linspace(interval[0],interval[1],101)
    spline = splrep(xs,func(xs))
    intfunc = lambda x: splint(interval[0],x,spline)
    #    intfunc = lambda x: quad(func,interval[0],x)[0]

    norm = intfunc(interval[1])
    F = np.random.rand(N)*norm
    x = F*0.
    for i in range(0,N):
        x[i] = brentq(lambda x: intfunc(x) - F[i],interval[0],interval[1])
    return x


def percentile(sample,q=0.5):
    sample.flatten()
    samp = list(sample)
    samp.sort()
    N = len(samp)
    return samp[int(N*q)]


