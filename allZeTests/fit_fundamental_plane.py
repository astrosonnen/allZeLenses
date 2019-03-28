import numpy as np
import pickle
import pylab
import emcee


f = open('../paper/mockP.dat', 'r')
mock = pickle.load(f)
f.close()

for par in mock:
    print par

