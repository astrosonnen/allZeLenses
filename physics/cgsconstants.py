from numpy import *

AA = 10.**(-8)

c = 2.99792458e10

h = 6.626068e-27

hbar = h/(2.*pi)

k = 1.3806503e-16

G = 6.67300e-8

M_Sun = 1.98892e33

L_Sun = 3.839e33

R_Sun = 6.995*10**10

pc = 3.08568025e18

kpc = pc*10.**3

Mpc = pc*10.**6

AU = 149598000.*10**5

Jy = 10.e-23

rad2arcsec = 180./pi*3600.
arcsec2rad = 1./rad2arcsec

rad2deg = rad2arcsec/3600.
deg2rad = arcsec2rad*3600.

H0 = 70*100000./Mpc

#particle masses
m_p = 1.67262158*10**(-24)

m_e = 9.10938188*10**(-28)

m_n = 1.67492735*10**(-24)

N_A = 6.0221415*10**23

stefan = 2*pi**5*k**4/(15.*h**3*c**2)
sigma_SB = stefan

a_rad = 4*stefan/c

charge = 4.80320425*10**(-10)

eV = 1.602*10**(-12)

r_e = charge**2/m_e/c**2

sigma_Th = 8*pi/3*r_e**2

yr = 365*3600*24.

freff_deV = 0.4153

def Planckf(x,T,var='lambda'):
    if var=='lambda':
        return 2*h*c**2/x**5*1./(exp(h*c/(x*k*T))-1)
    elif var=='nu':
        return 2*h*x**3/c**2*1./(exp(h*x/(k*T))-1)

def Ledd(M):
    return 4*pi*G*M*m_p*c/sigma_Th
