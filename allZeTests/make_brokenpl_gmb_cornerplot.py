from allZeTools.plotters import cornerplot
import pylab
import pickle


mockname = 'mockF'

f = open('%s_brokenpl_gmb_inference.dat'%mockname, 'r')
chain = pickle.load(f)
f.close()

f = open('%s.dat'%mockname, 'r')
mock = pickle.load(f)
f.close()

mock['truth']['H0'] = mock['truth']['h']*100.

cp = []
for par in chain:
    if par != 'logp':
	cp.append({'data': chain[par], 'label': par})

cornerplot(cp, color='c')
#pylab.savefig('simple_reality_inference_cp.png')
#pylab.savefig('mockC_wsourcepars_inference_cp.png')
pylab.show()


