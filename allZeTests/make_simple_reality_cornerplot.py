from allZeTools.plotters import cornerplot
import pylab
import pickle


mockname = 'mockC'

f = open('%s_wsourcepars_inference.dat'%mockname, 'r')
#f = open('%s_interimprior_inference.dat'%mockname, 'r')
chain = pickle.load(f)
f.close()

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

mock['truth']['H0'] = mock['truth']['h']*100.

labels = {'mhalo_mu': '$\mu_h$', 'mhalo_sig': '$\sigma_h$', 'mstar_mu': '$\mu_*$', 'mstar_mhalo': '$\\beta$', \
          'mstar_sig': '$\sigma_*$', 'aimf_mu': '$\mu_{\mathrm{IMF}}$', 'aimf_sig': '$\sigma_{\mathrm{IMF}}$', 'H0': '$H_0$', 'cvir_mu': '$\mu_c$', 'cvir_sig': '$\mu_\sigma$'}

ticks = {'mhalo_mu': (12.6, 13., 13.4), \
	 'mhalo_sig': (0.2, 0.4), \
	 'aimf_mu': (-0.05, 0.), \
	 'mstar_mhalo': (0.4, 0.8, 1.2), \
	 'mstar_mu': (11.2, 11.4, 11.6), \
	 'aimf_sig': (0.05, 0.10, 0.15), \
	 'mstar_sig': (0.05, 0.1, 0.15), \
	 'H0': (60., 70., 80.),}

cp = []
for par in ticks:
    if par != 'logp':
	cp.append({'data': chain[par], 'label': labels[par], 'value': mock['truth'][par], 'ticks': []})#, 'ticks': ticks[par]})

cornerplot(cp, color='c')
#pylab.savefig('simple_reality_inference_cp.png')
pylab.savefig('mockC_wsourcepars_inference_cp.png')
pylab.show()


