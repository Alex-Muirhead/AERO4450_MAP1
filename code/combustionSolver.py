import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from scipy import interpolate

Ru   = 8.314  # kJ/kmol.K
pRef = 100    # kPa

plt.style.use("PaperDoubleFig.mplstyle")


def lin(lower, upper, deltaX):
    deltaY = upper - lower
    grad = deltaY / deltaX

    def inner(x):
        return lower + grad*x

    return inner


def vectorInterface(lengths):
    L = [0, *np.cumsum(lengths)]

    def wrapper(func):
        def inner(t, args):
            splitArgs = [args[l:r] for l, r in zip(L[:-1], L[1:])]
            output = func(t, *splitArgs)
            return np.hstack([*output])
        return inner
    return wrapper


#  (row) species 0 :: C2H4
#                1 ::   O2
#                2 ::   CO
#                3 ::  H2O
#                4 ::  CO2

# (col) reaction 0 :: C2H4 + 2 O2 --> 2 CO + 2 H2O
#                1 :: CO + 1/2 O2 <-> CO2

# Stoichiometric coefficients
ν = np.array([
    [-1.,  0. ],
    [-2., -0.5],
    [ 2., -1. ],
    [ 2.,  0. ],
    [ 0.,  1. ]
]).T

# Experimental partial powers
νExp = np.array([
    [0.5 , 0. ],
    [0.65, 0.5],
    [2.  , 1. ],
    [2.  , 0. ],
    [0.  , 1. ]
]).T

# Forward and reverse masks
maskF = np.zeros_like(ν, dtype=bool)
maskR = np.zeros_like(ν, dtype=bool)
maskF[ν < 0.] = True
maskR[ν > 0.] = True

chemData = []
for species in ("C2H4", "O2", "CO", "H2O", "CO2"):
    data = pd.read_csv(f"chemData/{species}.txt", sep="\t", skiprows=1)
    chemData.append(data[1:])  # Skip T=0K

logKfuncs, deltaHfuncs = [], []
for data in chemData:
    T      = data["T(K)"].values.astype(float)
    logKf  = data["log Kf"].values.astype(float)
    deltaH = data["delta-f H"].values.astype(float) * 1e+03  # kJ/mol->kJ/kmol
    logKfuncs.append(interpolate.interp1d(T, logKf, kind="quadratic"))
    deltaHfuncs.append(interpolate.interp1d(T, deltaH, kind="quadratic"))


def Kc(T, p):
    """Kc = Kp * pow(pRef/Ru*T, νExp+...)"""
    # NOTE: Account for partial pressures
    Kf_i    = np.array([pow(10, Kf(T)) for Kf in logKfuncs]) * (pRef/(Ru*T))
    forward = pow(Kf_i, maskF*νExp)
    reverse = pow(Kf_i, maskR*νExp)
    return np.prod(reverse, axis=1) / np.prod(forward, axis=1)


def arrhenius(T):
    return np.array([
        1.739e+09 * math.exp(-1.485e+05 / (Ru*T)),
        6.324e+07 * math.exp(-5.021e+04 / (Ru*T))
    ])


ΔT   = 0.1e-03
temp = lin(1400, 2800, ΔT)  # K
pres = lin(70, 140, ΔT)     # kPa


@vectorInterface((5, 1))
def gradient(t, χ, h):
    limit = (χ < 0)
    χ[limit] = 0

    # Would normally calculate T from h = \int cp(T) dT
    T , p = temp(t), pres(t)
    kf    = arrhenius(T)
    kr    = kf / Kc(T, p)
    kr[0] = 0  # One way reaction

    forward = kf * np.prod(pow(χ, maskF*νExp), axis=1)
    reverse = kr * np.prod(pow(χ, maskR*νExp), axis=1)
    χGrad   = ν.T @ forward - ν.T @ reverse
    χGrad[(χGrad < 0)*limit] = 0

    hGrad = -sum([dχ_i*h_i(T) for dχ_i, h_i in zip(χGrad, deltaHfuncs)])

    return χGrad, hGrad


n = 1 + 3*(1 + 3.76)
χ0  = np.array(
    [1/n, 3/n, 0.0, 0.0, 0.0]
) * 70 / (Ru * 1400)
sol = integrate.solve_ivp(
    gradient, (0, ΔT), np.append(χ0, 0.),
    method="LSODA", events=None,
    atol=1e-10, rtol=1e-10
)

t, y = sol.t, sol.y
print(f"The heat released is {y[-1][-1]*1e-03:.3f} MJ/m^3")
print(np.array([1/n, 3/n, 0.0, 0.0, 0.0]))

fig, ax = plt.subplots()
formula = ("C$_2$H$_4$", "O$_2$", "CO", "H$_2$O", "CO$_2$")
[ax.plot(t*1e+06, y[i]*1e+03, label=formula[i]) for i in range(5)]
ax.legend()
ax.set_xlim([0, 100])
plt.xlabel(r"Time [$\mu$s]")
plt.ylabel("Concentration [mol/m$^3$]")
plt.title("Concentration of species over combustion")
plt.savefig("../images/concentration.pdf")

fig, ax = plt.subplots()
ax.plot(sol.t*1e+06, sol.y[-1]*1e-03, "k-", label="Net heat")
ax.legend()
ax.set_xlim([0, 100])
ax.set_ylim([0, 0.5])
plt.xlabel(r"Time [$\mu$s]")
plt.ylabel("Net heat [MJ/m$^3$]")
plt.title("Net heat release from combustion")
plt.savefig("../images/netHeat.pdf")

fig, ax = plt.subplots()
ax.plot(
    sol.t*1e+06,
    np.gradient(sol.y[-1], sol.t)*1e-06,
    "k-", label="Heat rate"
)
ax.legend()
ax.set_xlim([0, 100])
ax.set_ylim([-5, 15])
plt.xlabel(r"Time [$\mu$s]")
plt.ylabel("Rate of heat [GW/m$^3$]")
plt.title("Rate of heat of combustion")
plt.savefig("../images/heatRate.pdf")
plt.show()
