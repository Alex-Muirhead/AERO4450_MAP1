import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from scipy import interpolate

R    = 8.314
pRef = 101.3e3


def positiveDefinite(func):
    def wrapper(t, y):
        toFix = (y < 0)
        y[toFix] = 0
        grad = func(t, y)
        grad[(grad < 0)*toFix] = 0
        return grad
    return wrapper


#  (row) species 0 :: C2H4
#                1 ::   O2
#                2 ::   CO
#                3 ::  H2O
#                4 ::  CO2

# (col) reaction 0 :: C2H4 + 2 O2 --> 2 CO + 2 H2O
#                1 :: CO + 1/2 O2 <-> CO2

# Stoichiometric coefficients
μ = np.array([
    [-1.,  0. ],
    [-2., -0.5],
    [ 2., -1. ],
    [ 2.,  0. ],
    [ 0.,  1. ]
]).T

# Experimental partial powers
ν = np.array([
    [0.5 , 0. ],
    [0.65, 0.5],
    [2.  , 1. ],
    [2.  , 0. ],
    [0.  , 1. ]
]).T

# Forward and reverse masks
maskF = np.zeros((2, 5), dtype=bool)
maskR = np.zeros((2, 5), dtype=bool)

maskF[0, (0, 1)] = True  # {C2H4,  O2}
maskR[0, (2, 3)] = True  # {  CO, H2O}

maskF[1, (1, 2)] = True  # {  CO, O2}
maskR[1, (4)]    = True  # { CO2}

chemData = []
for species in ("C2H4", "O2", "CO", "H2O", "CO2"):
    data = pd.read_csv(f"code/chemData/{species}.txt", sep="\t", skiprows=1)
    chemData.append(data[1:])  # Skip T=0K

Kfuncs = []
for data in chemData:
    T     = data["T(K)"].values.astype(float)
    logKf = data["log Kf"].values.astype(float)
    Kfuncs.append(interpolate.interp1d(T, pow(10, logKf)))


def Kp(T, p):
    Kf_i    = np.array([f(T) for f in Kfuncs])(p/pRef)
    forward = pow(Kf_i, maskF*ν)
    reverse = pow(Kf_i, maskR*ν)
    return np.prod(reverse, axis=1) / np.prod(forward, axis=1)


def arrhenius(T):
    return np.array([
        1.739e+09 * math.exp(-1.485e+05 / (R*T)),
        6.235e+07 * math.exp(-5.021e+04 / (R*T))
    ])


@positiveDefinite
def gradient(t, χ):
    T = 1400 + 7e5*t
    p = 70e3 + 35e6*t
    kf = arrhenius(T)
    kr = kf / Kp(T, p)
    kr[0] = 0  # One way reaction
    forward = kf * np.prod(pow(χ, maskF*ν), axis=1)
    reverse = kr * np.prod(pow(χ, maskR*ν), axis=1)
    return μ.T @ forward - μ.T @ reverse


χ0  = np.array([0.065, 0.196, 0.0, 0.0, 0.0])
sol = integrate.solve_ivp(gradient, (0, 2e-3), χ0, method="LSODA")

formula = ("C2H4", "O2", "CO", "H2O", "CO2")
[plt.plot(sol.t, sol.y[i], label=formula[i]) for i in range(5)]
plt.legend()
plt.show()
