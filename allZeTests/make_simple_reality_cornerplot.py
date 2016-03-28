from allZeTools.plotters import cornerplot
import pylab
import pickle

f = open('mock_E_100lenses_Nind1e5_inference.dat', 'r')
chain = pickle.load(f)
f.close()

f = open('mock_E_100lenses.dat', 'r')
mock, chains = pickle.load(f)
f.close()

labels = {'mhalo_mu': '$\mu_h$', 'mhalo_sig': '$\sigma_h$', 'mstar_mu': '$\mu_*$', 'mstar_mhalo': '$\\beta$', \
          'mstar_sig': '$\sigma_*$', 'aimf_mu': '$\mu_{\mathrm{IMF}}$', 'aimf_sig': '$\sigma_{\mathrm{IMF}}$', 'h70': '$h_{70}$'}

ticks = {'mhalo_mu': (12.6, 13., 13.4), \
	 'mhalo_sig': (0.2, 0.4), \
	 'aimf_mu': (-0.05, 0.), \
	 'mstar_mhalo': (0.4, 0.8, 1.2), \
	 'mstar_mu': (11.2, 11.4, 11.6), \
	 'aimf_sig': (0.05, 0.10, 0.15), \
	 'mstar_sig': (0.05, 0.1, 0.15), \
	 'h70': (0.9, 1.0, 1.1),}

cp = []
for par in chain:
    if par != 'logp':
	cp.append({'data': chain[par], 'label': labels[par], 'value': mock['truth'][par], 'ticks': ticks[par]})

cornerplot(cp, color='c')
#pylab.savefig('simple_reality_mockD_100lenses_nothin_cp.png')
pylab.show()


