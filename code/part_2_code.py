"""
AERO4450 Major Assignment Part 2
Authors: Alex Muirhead and Robert Watt
Purpose: Simulate the combustion in a variable area scramjet combustor
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy import integrate
import pandas as pd

# ----- Define combuster inlet conditions (in burner gas constant terms) -----

Ru = 8.314                             # Universal gas constant [J/mol/K]
pRef = 101.3                           # reference pressure [kPa]

# ------------------------- Combuster gas properties -------------------------

yb  = 1.3205                           # gamma
Rb  = 288.45                           # Gas constant [J/kg/K]
cpb = Rb / (1 - 1/yb) / 1000           # J/g/K specific heat constant pressure

# ------------------------ Combuster inlet properties ------------------------

M3b  = 3.814                           # mach number
p3b  = 70.09                           # static pressure [kPa]
T3b  = 1237.63                         # temperature [K]
T3b  = 1380
Tt3b = T3b * (1 + 0.5*(yb-1) * M3b**2)  # stagnation temperature
# combined mass flow rate of stoichiometric mixture of ethylene and air [kg/s]
mdot = 31.1186
rho3b = p3b * 1e3 / (Rb * T3b)         # kg/m^3
V3b = M3b * np.sqrt(yb * Rb * T3b)     # m/s
A3 = mdot / (rho3b*V3b)                # m^2
YN2 = 0.7168                           # mass fraction of nitrogen

# calculate initial concentrations
n = 1 + 3*(1 + 3.76)
MW = np.array([28, 32, 28, 18, 44])    # kg/kmol
X3 = np.array(
    [1/n, 3/n, 0.0, 0.0, 0.0]
) * p3b / (Ru * T3b)

# --------------------------- Combustor properties ---------------------------

combustor_length = 0.5                 # m
Cf = 0.002                             # skin friction coefficient

# calculate the area for each point along the combustor


def A(x, A3, Length=0.5):
    return A3 * (1 + 3*x/Length)


def dAonA(x, A3, length=0.5):
    """Calculate dA/A for each point along the combustor"""
    return 3 * A3 / (length * A(x, A3))


def vectorInterface(lengths):
    """Decorator to re-order vector into multiple variables."""
    L = [0, *np.cumsum(lengths)]

    def wrapper(func):
        def inner(t, args):
            splitArgs = [args[l:r] for l, r in zip(L[:-1], L[1:])]
            output = func(t, *splitArgs)
            return np.hstack([*output])
        return inner
    return wrapper

# ------------------------- Set up chemical reactions -------------------------

#  (row) species 0 :: C2H4
#                1 ::   O2
#                2 ::   CO
#                3 ::  H2O
#                4 ::  CO2

# (col) reaction 0 :: C2H4 + 2 O2 --> 2 CO + 2 H2O
#                1 :: CO + 1/2 O2 <-> CO2


# Stoichiometric coefficients
μ = np.array([
    [-1.,  0.],
    [-2., -0.5],
    [2., -1.],
    [2.,  0.],
    [0.,  1.]
]).T

# Experimental partial powers
ν = np.array([
    [0.5, 0.],
    [0.65, 0.5],
    [2., 1.],
    [2., 0.],
    [0., 1.]
]).T

# Forward and reverse masks
maskF = np.zeros((2, 5), dtype=bool)
maskR = np.zeros((2, 5), dtype=bool)

maskF[0, (0, 1)] = True  # {C2H4,  O2}
maskR[0, (2, 3)] = True  # {CO, H2O}

maskF[1, (1, 2)] = True  # {CO, O2}
maskR[1, (4)] = True  # {CO2}


# --------------------------- Import Chemical Data ---------------------------


chemData = []
for species in ("C2H4", "O2", "CO", "H2O", "CO2"):
    data = pd.read_csv(f"code/chemData/{species}.txt", sep="\t", skiprows=1)
    chemData.append(data[1:])  # Skip T=0K

logKfuncs, deltaHfuncs = [], []
for data in chemData:
    T = data["T(K)"].values.astype(float)
    logKf = data["log Kf"].values.astype(float)
    # kJ/mol->kJ/kmol = J/mol
    deltaH = data["delta-f H"].values.astype(float) * 1e+03
    logKfuncs.append(interpolate.interp1d(T, logKf, kind="quadratic"))
    deltaHfuncs.append(interpolate.interp1d(T, deltaH, kind="quadratic"))


def dTtdx(X, M, Tt, x, T):
    """Calculate spatial gradient of total temperature."""
    h0f = np.array([np.float64(deltaHfuncs[i](T))
                    for i in range(5)])  # kJ/kmol
    h0f = h0f / MW  # kJ/kg
    temp_gradient = -1/cpb * np.sum(dYdx(X, M, Tt, x, T) * h0f)
    return temp_gradient

# calculate dM^2


def dM2(M, X, x, Tt, T):
    Deff = 2 * np.sqrt(A(x, A3) / np.pi)
    return M**2 * ((1 + 0.5*(yb - 1)*M**2) / (1 - M**2)) * (
        -2 * dAonA(x, A3)  # area change
        + (1 + yb*M**2)*dTtdx(X, M, Tt, x, T)/Tt  # total temperature change
        + yb*M**2 * 4 * Cf / Deff  # friction
    )


def arrhenius(T):
    return np.array([
        1.739e+09 * np.exp(-1.485e+05 / (Ru*T)),
        6.324e+07 * np.exp(-5.021e+04 / (Ru*T))
    ])


def Kc(T):
    """Kc = Kp * pow(pRef/p, ν+...)"""
    # NOTE: Account for partial pressures
    Kf_i = np.array([pow(10, f(np.float64(T)))
                     for f in logKfuncs]) * (pRef/(Ru*T))**(-1)
    forward = pow(Kf_i, maskF*ν)
    reverse = pow(Kf_i, maskR*ν)
    return np.prod(reverse, axis=1) / np.prod(forward, axis=1)


def concentration_gradient(χ, M, Tt, T):
    """Return the gradient of the concentrations in time."""
    limit = (χ < 0)
    χ[limit] = 0

    kf = arrhenius(T)
    kr = kf / Kc(T)
    kr[0] = 0  # One way reaction

    forward = kf * np.prod(pow(χ, maskF*ν), axis=1)
    reverse = kr * np.prod(pow(χ, maskR*ν), axis=1)
    χGrad = μ.T @ forward - μ.T @ reverse

    χGrad[(χGrad < 0)*limit] = 0

    #  hGrad = -sum([dχ_i*h_i(T) for dχ_i, h_i in zip(χGrad, deltaHfuncs)])
    return χGrad


# ---------- Calculate the spatial derivative of the concentrations ----------

def dXdx(M, Tt, X, T):
    """Rate of change of concentration w.r.t space."""
    v = M * np.sqrt(yb * Rb * T)
    return concentration_gradient(X, M, Tt, T) / v


def dYdx(X, M, Tt, x, T):
    """Rate of change of mass fraction w.r.t space."""
    reacting_sum = np.sum(X * MW)
    return MW * (1 - YN2) * (
        dXdx(M, Tt, X, T)/reacting_sum
        - X * np.sum(MW * dXdx(M, Tt, X, T)) / reacting_sum**2
    )


@vectorInterface((5, 1, 1))
def gradient(x, X, Tt, M2):
    """Return the gradient of all variables in a single vector"""
    x = np.float64(x)
    Tt = np.float64(Tt)
    M = np.sqrt(np.float64(M2))
    T = Tt * (1 + 0.5*(yb - 1) * M**2)**(-1)
    return dXdx(M, Tt, X, T), dTtdx(X, M, Tt, x, T), dM2(M, X, x, Tt, T)


def massFraction(X):
    """Convert concentration into mass fraction"""
    return X * MW / (np.sum(X * MW) * (1 - YN2))


# create initial conditions vector
init_conds = np.append(X3, [Tt3b, M3b**2])

# integrate IV
sol = integrate.solve_ivp(
    gradient, (0, 0.5),
    init_conds,
    method="LSODA",
    atol=1e-10, rtol=1e-10
)
# extract variables from integrator
x, X, Tt, M = sol.t, sol.y[0:5], sol.y[5], np.sqrt(sol.y[6])


# calculate the static temperature from stagnation temperature and Mach number
T = Tt * (1 + 0.5*(yb - 1) * M**2)**(-1)


fig, ax = plt.subplots()
formula = ("C$_2$H$_4$", "O$_2$", "CO", "H$_2$O", "CO$_2$")
[ax.plot(x, X[i]*1e+03, label=formula[i]) for i in range(5)]
ax.legend()
plt.xlabel("distance along combustor [m]")
plt.ylabel("Concentration [mol/m$^3$]")
plt.title("Concentration over combustion")

fig, ax = plt.subplots()
ax.plot(x, Tt, label="Tt")
ax.plot(x, [1.15*Tt3b for i in x], label="ignition temp")
plt.xlabel("distance along combustor [m]")
plt.ylabel("$T_t$ [K]")
ax.legend()

fig, ax = plt.subplots()
ax.plot(x, T, label="T")
plt.xlabel("distance along combustor [m]")
plt.ylabel("T [K]")
ax.legend()

fig, ax = plt.subplots()
ax.plot(x, M, label="M")
plt.xlabel("distance along combustor [m]")
plt.ylabel("M")
ax.legend()

plt.show()
