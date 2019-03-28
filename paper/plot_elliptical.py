import numpy as np
import pylab
import lens_models
from scipy.optimize import minimize, brentq
from matplotlib import rc
rc('text', usetex=True)


fsize = 14

q = 0.95
gamma = 2.1

lens = lens_models.powerlaw_ellpot(gamma=gamma, q=q)

tol = 1e-5
rmin = 0.1

def get_images(y, bounds_a, bounds_b):

    y1, y2 = y
    yr = (y1**2 + y2**2)**0.5

    guess_a = np.array((lens.rein * y1/yr, lens.rein * y2/yr))

    guess_b = np.array((-lens.rein * y1/yr, -lens.rein * y2/yr))

    minimizer_kwargs = dict(method='L-BFGS-B', bounds=bounds_a, tol=tol)
    
    def minfunc(x):

        alpha = lens.alpha(x)

        y1_guess = x[0] - alpha[0]
        y2_guess = x[1] - alpha[1]
        
        return (y1_guess - y1)**2 + (y2_guess - y2)**2

    res_a = minimize(minfunc, guess_a, bounds=bounds_a)
    res_b = minimize(minfunc, guess_b, bounds=bounds_b)

    return res_a.x, res_b.x

nx = 1001
x_grid = np.linspace(-2., 2., nx)

X, Y = np.meshgrid(x_grid, x_grid)

A = lens.A((X, Y))

detA = A[0, 0, :] * A[1, 1, :] - A[0, 1, :] * A[1, 0, :]

kappa = 1. - 0.5*(A[0, 0, :] + A[1, 1, :])

levels = [0.]

klevels = [0.3, 0.5, 0.7]

pylab.subplots_adjust(left=0.13, right=0.99, bottom=0.12, top=0.99)

#pylab.contour(np.flipud(X), np.flipud(Y), np.flipud(kappa), klevels, colors='k', linewidths=1)

CS = pylab.contour(np.flipud(X), np.flipud(Y), np.flipud(detA), levels, colors='k', linewidths=2)

contours = CS.collections[0].get_paths()
for contour in contours:
    v = contour.vertices
    xcont = v[:, 0]
    ycont = v[:, 1]

    alpha = lens.alpha((xcont, ycont))

    caust_x = xcont - alpha[0]
    caust_y = ycont - alpha[1]

    pylab.plot(caust_x, caust_y, color='k', linewidth=2)

def plot_images(sources, xa, xb, color='r'):
    ns = len(sources)
    for i in range(ns):
        pylab.scatter(sources[i][0], sources[i][1], color=color, marker='D')
        pylab.scatter(xa[i][0], xa[i][1], color=color, marker='o')
        pylab.scatter(xb[i][0], xb[i][1], color=color, marker='o')

sources_x = ((0.2, 0.), (0.3, 0.), (0.4, 0.), (0.5, 0.))
nx = len(sources_x)
bounds_a = np.array(((0.5*lens.rein, 2.*lens.rein), (-0.1, 0.1)))
bounds_b = np.array(((-2.*lens.rein, -0.1), (-0.1, 0.1)))

xa_x = []
xb_x = []

for i in range(nx):
    xa, xb = get_images(sources_x[i], bounds_a, bounds_b)
    xa_x.append(xa)
    xb_x.append(xb)

plot_images(sources_x, xa_x, xb_x, color='r')

sources_y = ((0., 0.2), (0., 0.3), (0., 0.4), (0., 0.5))
ny = len(sources_y)
bounds_a = np.array(((-0.1, 0.1), (0.5*lens.rein, 2.*lens.rein)))
bounds_b = np.array(((-0.1, 0.1), (-2.*lens.rein, -0.1)))

xa_y = []
xb_y = []

for i in range(ny):
    xa, xb = get_images(sources_y[i], bounds_a, bounds_b)
    xa_y.append(xa)
    xb_y.append(xb)

plot_images(sources_y, xa_y, xb_y, color='b')

def dpotdr(r):
    return lens.rein*r**(2.-lens.gamma)

def d2potdr2(r):
    return (2. - lens.gamma) * (r/lens.rein)**(1.-lens.gamma)

def d3potdr3(r):
    return (1. - lens.gamma) * (2. - lens.gamma) / lens.rein * (r/lens.rein)**(-lens.gamma)

def timedelay(xa, xb):
    alpha_a = lens.alpha(xa)
    a2_a = alpha_a[0]**2 + alpha_a[1]**2

    alpha_b = lens.alpha(xb)
    a2_b = alpha_b[0]**2 + alpha_b[1]**2

    return 0.5*(a2_b - a2_a) - lens.lenspot(xb) + lens.lenspot(xa)

xc = brentq(lambda x: dpotdr(q**0.5*x) - q**1.5 * x, 0.5*lens.rein, 1.5*lens.rein)

yc = brentq(lambda x: dpotdr(q**-0.5*x) - q**-1.5 * x, 0.5*lens.rein, 1.5*lens.rein)
pylab.xlim(-1.6, 1.6)
pylab.ylim(-1.6, 1.6)
pylab.xticks(fontsize=fsize)
pylab.yticks(fontsize=fsize)
pylab.xlabel('$\\theta_1\,\,(\\rm{arcsec})$', fontsize=fsize)
pylab.ylabel('$\\theta_2\,\,(\\rm{arcsec})$', fontsize=fsize)
pylab.axes().set_aspect('equal')

fig = pylab.gcf()
fig.set_size_inches(6, 6)
fig.savefig('ellpot_critcurve.eps')
pylab.show()

psi1x = dpotdr(q**0.5*xc)
psi2x = d2potdr2(q**0.5*xc)
psi3x = d3potdr3(q**0.5*xc)
print psi2x, psi3x

epsqx = -(q**0.5*psi1x - xc)/(1. - q*psi2x)

dtheta_A_x = np.zeros(len(sources_x))
dtheta_B_x = 0.*dtheta_A_x

dt_x = 0.*dtheta_A_x
rmur_x = 0.*dtheta_A_x

eps = 1e-4
for i in range(nx):
    dtheta_A_x[i] = xa_x[i][0] - xc
    dtheta_B_x[i] = xb_x[i][0] + xc
    dt_x[i] = timedelay(xa_x[i], xb_x[i])
    dalphadx_A = (lens.alpha((xa_x[i][0]+eps, 0.))[0] - lens.alpha((xa_x[i][0]-eps, 0.))[0])/(2.*eps)
    dalphadx_B = (lens.alpha((xb_x[i][0]+eps, 0.))[0] - lens.alpha((xb_x[i][0]-eps, 0.))[0])/(2.*eps)
    rmur_x[i] = (1. - dalphadx_B)/(1. - dalphadx_A)

dtheta_B_x_1storder = dtheta_A_x + 2.*epsqx
dt_x_1storder = 2.*q**0.5*(1. - q*psi2x)*(dtheta_A_x + epsqx)
rmur_x_1storder = 1. + 2.*q**1.5*psi3x/(1. - q*psi2x)*(dtheta_A_x + epsqx)

psi1y = dpotdr(q**-0.5*yc)
psi2y = d2potdr2(q**-0.5*yc)
psi3y = d3potdr3(q**-0.5*yc)

epsqy = -(q**-0.5*psi1y - yc)/(1. - psi2y/q)

dtheta_A_y = np.zeros(ny)
dtheta_B_y = 0.*dtheta_A_y

dt_y = 0.*dtheta_A_y
rmur_y = 0.*dtheta_A_y

for i in range(ny):
    dtheta_A_y[i] = xa_y[i][1] - yc
    dtheta_B_y[i] = xb_y[i][1] + yc
    dt_y[i] = timedelay(xa_y[i], xb_y[i])
    dalphady_A = (lens.alpha((0., xa_y[i][1]+eps))[1] - lens.alpha((0., xa_y[i][1]-eps))[1])/(2.*eps)
    dalphady_B = (lens.alpha((0., xb_y[i][1]+eps))[1] - lens.alpha((0., xb_y[i][1]-eps))[1])/(2.*eps)
    rmur_y[i] = (1. - dalphady_B)/(1. - dalphady_A)

dtheta_B_y_1storder = dtheta_A_y + 2.*epsqy
dt_y_1storder = 2.*q**-0.5*(1. - psi2y/q)*(dtheta_A_y + epsqy)

fig = pylab.figure()
pylab.subplots_adjust(left=0.15, bottom=0.12, top=0.99, right=0.99, hspace=0.)

ax1 = fig.add_subplot(3, 1, 3)

pylab.scatter(dtheta_A_x, dtheta_B_x, color='r')
pylab.scatter(dtheta_A_y, dtheta_B_y, color='b')
xlim = pylab.xlim()
xs = np.linspace(xlim[0], 0.4)
ys = np.linspace(0.2, xlim[1])
pylab.plot(xs, 2*epsqx + xs, color='r')
pylab.plot(ys, 2*epsqy + ys, color='b')
uperrx = xs + 2*epsqx + xs**2
dwerrx = xs + 2*epsqx - xs**2
#pylab.fill_between(xs, dwerr, uperr, color='r', alpha=0.5)
pylab.plot(xs, dwerrx, color='r', linestyle='--')
pylab.plot(xs, uperrx, color='r', linestyle='--')
uperry = ys + 2*epsqy + ys**2
dwerry = ys + 2*epsqy - ys**2
pylab.plot(ys, dwerry, color='b', linestyle='--')
pylab.plot(ys, uperry, color='b', linestyle='--')
pylab.axhline(2.*epsqx, color='r', linestyle=':')
pylab.axhline(2.*epsqy, color='b', linestyle=':')
pylab.xlabel('$\Delta\\theta_A\,\,(\\rm{arcsec})$', fontsize=fsize)
pylab.ylabel('$\Delta\\theta_B\,\,(\\rm{arcsec})$', fontsize=fsize)
pylab.xticks(fontsize=fsize)
pylab.yticks(fontsize=fsize)

ax2 = fig.add_subplot(3, 1, 2)
pylab.scatter(dtheta_A_x, dt_x, color='r')
pylab.scatter(dtheta_A_y, dt_y, color='b')
xlim = pylab.xlim()
pylab.plot(xs, 2.*q**0.5*psi1x*(1-q*psi2x)*(xs + epsqx), color='r')
pylab.plot(ys, 2.*q**-0.5*psi1y*(1-psi2y/q)*(ys + epsqy), color='b')
uperrx = 2.*q**0.5*(1. - q*psi2x)*(xs + epsqx) + xs**2
dwerrx = 2.*q**0.5*(1. - q*psi2x)*(xs + epsqx) - xs**2
pylab.plot(xs, dwerrx, color='r', linestyle='--')
pylab.plot(xs, uperrx, color='r', linestyle='--')
uperry = 2.*q**-0.5*(1. - psi2y/q)*(ys + epsqy) + ys**2
dwerry = 2.*q**-0.5*(1. - psi2y/q)*(ys + epsqy) - ys**2
pylab.plot(ys, dwerry, color='b', linestyle='--')
pylab.plot(ys, uperry, color='b', linestyle='--')
pylab.ylabel('$\Delta t\,\,(D_{\Delta t}\,\\rm{arcsec}^2/c)$', fontsize=fsize)
pylab.yticks(fontsize=fsize)
pylab.tick_params(axis='x', labelbottom='off')

ax3 = fig.add_subplot(3, 1, 1)
pylab.scatter(dtheta_A_x, rmur_x, color='r')
pylab.plot(xs, 1. + 2.*q**1.5*psi3x/(1. - q*psi2x)*(xs + epsqx), color='r')
uperrx = 1. + 2.*q**1.5*psi3x/(1. - q*psi2x)*(xs + epsqx) + xs**2
dwerrx = 1. + 2.*q**1.5*psi3x/(1. - q*psi2x)*(xs + epsqx) - xs**2
pylab.plot(xs, dwerrx, color='r', linestyle='--')
pylab.plot(xs, uperrx, color='r', linestyle='--')

pylab.scatter(dtheta_A_y, rmur_y, color='b')
pylab.plot(ys, 1. + 2.*q**-1.5*psi3y/(1. - q*psi2y)*(ys + epsqy), color='b')
uperry = 1. + 2.*q**-1.5*psi3y/(1. - psi2y/q)*(ys + epsqy) + ys**2
dwerry = 1. + 2.*q**-1.5*psi3x/(1. - psi2y/q)*(ys + epsqy) - ys**2
pylab.plot(ys, dwerry, color='b', linestyle='--')
pylab.plot(ys, uperry, color='b', linestyle='--')

pylab.tick_params(axis='x', labelbottom='off')
pylab.ylabel('$\mu_{r,A}/\mu_{r,B}$', fontsize=fsize)
pylab.yticks(fontsize=fsize)
fig = pylab.gcf()
fig.set_size_inches(6, 6)
fig.savefig('ellpot_plots.eps')
#pylab.savefig('ellpot_plots.eps')
pylab.show()


