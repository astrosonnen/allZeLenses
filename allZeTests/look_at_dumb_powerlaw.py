import numpy as np
import pickle


f = open('mockF_dumb_powerlaw_inference.dat', 'r')
chain = pickle.load(f)
f.close()

print chain['H0'].mean(), chain['H0'].std()

