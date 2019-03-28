import numpy as np
import lens_models
import pickle
from scipy.optimize import minimize
import pylab


# N power-law lenses, image positions and time-delay known exactly. How does the inferred distribution of power-law slopes change with varying the value of H0?

gamma_mu = 2.05
gamma_sig = 0.1

nlens = 100

max_asymm = 0.5

eps = 1e-12

rein_min = 0.5
rein_max = 3.

gamma_min = 1.5
gamma_max = 2.5

rein_mu = 1.5
rein_sig = 0.1

rein_samp = np.random.normal(rein_mu, rein_sig, nlens)
gamma_samp = np.random.normal(gamma_mu, gamma_sig, nlens)

lenses = []

im_err = 0.1

rein065_samp = np.zeros(nlens)
gamma065_samp = np.zeros(nlens)
beta065_samp = np.zeros(nlens)

for i in range(nlens):

    lens = lens_models.powerlaw(rein=rein_samp[i], gamma=gamma_samp[i])
    lens.get_caustic()

    ymax = lens.rein * (1. + max_asymm) - lens.alpha(lens.rein * (1. + max_asymm))

    ysource = (np.random.rand(1))**0.5*ymax

    lens.source = ysource

    lens.get_images(xtol=1e-6)

    lens.get_timedelay()

    model_lens = lens_models.powerlaw(rein=rein_samp[i], gamma=gamma_samp[i], h=0.65)
    model_lens.source = lens.images[0] - lens.alpha(lens.images[0])
    model_lens.get_caustic()
    model_lens.get_images(xtol=1e-6)

    model_lens.get_timedelay()

    dt_err = abs(lens.timedelay - model_lens.timedelay)

    start = np.array((lens.rein, lens.gamma, lens.source))

    bounds = np.array(((rein_min, rein_max), (gamma_min, gamma_max), (0., ymax)))

    scale_free_bounds = 0.*bounds
    scale_free_bounds[:, 1] = 1.

    scale_free_guess = (start - bounds[:, 0])/(bounds[:, 1] - bounds[:, 0])

    feps = 1e-8

    minimizer_kwargs = dict(method='L-BFGS-B', bounds=scale_free_bounds, tol=eps)

    def mlogp(x):

       chi2 = 0.

       p = x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

       rein, gamma, beta = p

       model_lens.rein = rein
       model_lens.gamma = gamma
       model_lens.source = beta

       model_lens.get_b_from_rein()

       model_lens.get_images(xtol=1e-6)

       model_lens.get_timedelay()

       chi2 = 0.5*(lens.images[0] - model_lens.images[0])**2/im_err**2 + 0.5*(lens.images[1] - model_lens.images[1])**2/im_err**2
       chi2 += 0.5*(lens.timedelay - model_lens.timedelay)**2/dt_err**2

       return chi2

    res = minimize(mlogp, 0.95*scale_free_guess, bounds=scale_free_bounds)

    rein065, gamma065, source065 = res.x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    model_lens.rein = rein065
    model_lens.gamma = gamma065
    model_lens.source = source065
    model_lens.get_images()
    model_lens.get_timedelay()

    print mlogp(res.x), (model_lens.timedelay - lens.timedelay)/lens.timedelay, (model_lens.images[0] - lens.images[0])/lens.images[0]

    gamma065_samp[i] = gamma065

pylab.scatter(gamma_samp, gamma065_samp)
xlim = pylab.xlim()
xs = np.linspace(xlim[0], xlim[1])
pylab.plot(xs, xs, linestyle='--', color='k')
pylab.show()

