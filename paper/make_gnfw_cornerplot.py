import pylab
import pickle
from plotters import probcontour
import numpy as np
import lens_models
from lensingtools import NFW
from matplotlib import rc
rc('text', usetex=True)


f = open('gnfwdev_nodyn_pymc_sample.dat', 'r')
nodyn = pickle.load(f)
f.close()

f = open('gnfwdev_wdyn_pymc_sample.dat', 'r')
wdyn = pickle.load(f)
f.close()

fsize = 10

nbins = 20

pars = ['mstar', 'mdme', 'beta', 'source', 'psi2', 'psi3', 'timedelay']

labels = ['$\log{M_*}$', '$\log{M_{\mathrm{DM},e}}$', '$\gamma_{\mathrm{DM}}$', '$\\beta_s$', "$\psi''$", "$\psi'''$", '$\Delta t$']

lims = [(11., 11.8), (10., 11.5), (0.2, 2.8), (0.05, 0.2), (-0.5, 0.3), (-0.2, 0.5), (5., 20.)]
ticks = [(11.2, 11.5), (10.5, 11.), (1., 2.), (0.1, 0.15), (-0.4, 0.), (0., 0.3), (10, 15)]

mstar = 11.5
mhalo = 13.3

reff = 7.

day = 24.*3600.

nodyn['timedelay'] = nodyn['timedelay'] / day
wdyn['timedelay'] = wdyn['timedelay'] / day

nodyn['source'] = nodyn['s2']**0.5
wdyn['source'] = wdyn['s2']**0.5

lens = lens_models.NfwDev(zd=0.3, zs=1., mstar=10.**mstar, mhalo=10.**mhalo, reff_phys=reff, delta_halo=200.)

lens.normalize()
lens.get_caustic()
lens.get_rein()

lens.source = lens.rein * 1.1 - lens.alpha(lens.rein*1.1)

lens.get_images()

lens.get_radmag_ratio()

lens.get_timedelay()

eps = 1e-4

psi2_true = (lens.alpha(lens.rein + eps) - lens.alpha(lens.rein - eps))/(2.*eps)
psi3_true = (lens.alpha(lens.rein + eps) - 2.*lens.alpha(lens.rein) + lens.alpha(lens.rein - eps))/eps**2

mdme_true = lens.mhalo / NFW.M3d(lens.rvir*lens.arcsec2kpc, lens.rs*lens.arcsec2kpc) * NFW.M2d(lens.reff_phys, lens.rs*lens.arcsec2kpc)

truth = [mstar, np.log10(mdme_true), 1., lens.source, psi2_true, psi3_true, lens.timedelay/day]

npars = len(pars)

fig = pylab.figure()
pylab.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.99, hspace=0.1, wspace=0.1)

for i in range(npars):

    ax = fig.add_subplot(npars, npars, (npars+1)*i + 1)

    bins = np.linspace(lims[i][0], lims[i][1], nbins+1)

    pylab.hist(nodyn[pars[i]], bins=bins, color='k', histtype='step', label='No vel. disp.')
    pylab.hist(wdyn[pars[i]], bins=bins, color=(1., 0., 0.), label='With vel. disp.')

    if i==0:
        ylim = pylab.ylim()
        pylab.scatter(-100., 1., marker='o', color='k', label='Truth')
        pylab.ylim(ylim[0], ylim[1])

        box = ax.get_position()
        ax.legend(loc='upper right', bbox_to_anchor=(7., 1.0), scatterpoints=1)

    ax.set_xlim((lims[i][0], lims[i][1]))
    ax.set_xticks(ticks[i])
    ax.set_yticks(())
    if i == npars-1:
        visible = True
        ax.set_xlabel(labels[i], fontsize=fsize)
    else:
        visible = False

    ax.set_xticklabels(ticks[i], fontsize=fsize, visible=visible)

for j in range(1, npars): # loops over rows
    if j == npars-1:
        xvisible = True
    else:
        xvisible = False

    for i in range(j): # loops over columns
        ax = pylab.subplot(npars, npars, npars*j+i+1)
        probcontour(nodyn[pars[i]], nodyn[pars[j]], smooth=5, style='black')
        probcontour(wdyn[pars[i]], wdyn[pars[j]], style=(1., 0., 0.))

        pylab.scatter(truth[i], truth[j], marker='o', color='k')
        pylab.axvline(truth[i], linestyle=':', color='k')
        pylab.axhline(truth[j], linestyle=':', color='k')

        ax.set_xlim(lims[i])
        ax.set_ylim(lims[j])
        ax.set_xticks(ticks[i])
        ax.set_yticks(ticks[j])

        if i == 0:
            yvisible = True
            ax.set_ylabel(labels[j], fontsize=fsize)
        else:
            yvisible = False

        if xvisible:
            ax.set_xlabel(labels[i], fontsize=fsize)

        ax.set_xticklabels(ticks[i], fontsize=fsize, visible=xvisible)
        ax.set_yticklabels(ticks[j], fontsize=fsize, visible=yvisible)

pylab.savefig('gnfw_cornerplot.eps')
#pylab.show()

