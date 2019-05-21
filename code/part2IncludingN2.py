"""
AERO4450 Major Assignment Part 2
Authors: Alex Muirhead and Robert Watt
Purpose: Simulate the combustion and calculate thrust produced by a scramjet
"""

from math import pi, sqrt

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import integrate, interpolate

plt.style.use("PaperDoubleFig.mplstyle")


def vectorInterface(argLengths):
    """Decorator to re-order vector into multiple variables."""
    start, slices = 0, []
    for length in argLengths:
        if length == 1:
            slices.append(start)
        else:
            slices.append(slice(start, start+length))
        start += length

    def wrapper(func):
        def inner(t, args):
            splitArgs = [args[s] for s in slices]
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
#                5 ::   N2

# (col) reaction 0 :: C2H4 + 2 O2 --> 2 CO + 2 H2O
#                1 :: CO + 1/2 O2 <-> CO2


# Stoichiometric coefficients
nu = np.array([
    [-1.,  0. ],
    [-2., -0.5],
    [ 2., -1. ],
    [ 2.,  0. ],
    [ 0.,  1. ],
    [ 0.,  0. ]
]).T

# Experimental partial powers
nuExp = np.array([
    [0.5,  0. ],
    [0.65, 0.5],
    [2.,   1. ],
    [2.,   0. ],
    [0.,   1. ],
    [0.,   0. ]
]).T

# Forward and reverse masks
maskF = np.zeros_like(nuExp, dtype=bool)
maskR = np.zeros_like(nuExp, dtype=bool)

maskF[0, (0, 1)] = True  # {C2H4,  O2}
maskR[0, (2, 3)] = True  # {CO, H2O}

maskF[1, (1, 2)] = True  # {CO, O2}
maskR[1, (4)]    = True  # {CO2}

alpha = maskF * nuExp
beta  = maskR * nuExp


# --------------------------- Import Chemical Data ----------------------------

allSpecies, chemData = ("C2H4", "O2", "CO", "H2O", "CO2", "N2"), []
for species in allSpecies:
    data = pd.read_csv(f"chemData/{species}.txt", sep="\t", skiprows=1)
    chemData.append(data[1:])  # Skip T=0K

logKfuncs, deltaHfuncs = [], []
for data in chemData:
    T = data["T(K)"].values.astype(float)
    logKf = data["log Kf"].values.astype(float)
    # kJ/mol -> kJ/kmol = J/mol
    deltaH = data["delta-f H"].values.astype(float) * 1e+03
    logKfuncs.append(interpolate.interp1d(T, logKf, kind="quadratic"))
    deltaHfuncs.append(interpolate.interp1d(T, deltaH, kind="quadratic"))


# ---------------------- Define Reaction Rate Functions -----------------------

def arrhenius(T):
    return np.array([
        1.739e+09 * np.exp(-1.485e+05 / (Ru*T)),
        6.324e+07 * np.exp(-5.021e+04 / (Ru*T))
    ])


def Kc(T):
    """Kc = Kp * pow(pRef/p, nu+...)"""
    # NOTE: Account for partial pressures
    Kf_i = np.array([pow(10, f(T)) for f in logKfuncs])/(Ru*T/pRef)
    forward = pow(Kf_i, alpha)
    reverse = pow(Kf_i, beta)
    return np.prod(reverse, axis=1) / np.prod(forward, axis=1)


# --------------------- Define Spatial Gradient Functions ---------------------

def area(x):
    """Cross sectional area of combustor"""
    return A3 * (1 + 3 * x/combustorLength)


def wallGradient(x):
    """Gradient of wall along the combuster"""
    return 3 * A3 / combustorLength


def concentrationGradient(X, Msqr, T):
    """Return the gradient of the concentrations in space."""
    limit = (X < 0)
    X[limit] = 0

    kf = arrhenius(T)
    kr = kf / Kc(T)
    kr[0] = 0  # One way reaction

    forward = kf * np.prod(pow(X, alpha), axis=1)
    reverse = kr * np.prod(pow(X, beta), axis=1)
    XRate = nu.T @ forward - nu.T @ reverse

    XRate[(XRate < 0)*limit] = 0

    XGrad = XRate / sqrt(Msqr*(k*R*T))
    return XGrad


def massFractionGradient(X, dXdx):
    """Rate of change of mass fraction w.r.t space."""
    avgMWeight = np.sum(X*mWeights)
    return mWeights * (
        dXdx / avgMWeight - X*np.sum(dXdx*mWeights) / avgMWeight**2
    )


def totalTempGradient(T, dYdx):
    """Calculate spatial gradient of total temperature."""
    deltaHForm = np.array([deltaHf(T) for deltaHf in deltaHfuncs])  # kJ/kmol
    hGrad = -np.sum(dYdx*deltaHForm/mWeights)
    return hGrad / cp


def machGradient(x, Msqr, Tt, dTtdx, A, dAdx):
    """Calculate spatial derivative of the square of the mach number"""
    diamEff = 2 * sqrt(A/pi)
    return Msqr * ((1 + 0.5*(k-1)*Msqr) / (1 - Msqr)) * (
        - 2*dAdx/A                     # area change
        + dTtdx/Tt * (1 + k*Msqr)      # total temperature change
        + 4*Cf/diamEff * k*Msqr        # friction
    )


@vectorInterface((6, 1, 1))
def gradient(x, X, Tt, Msqr):
    """Return the gradient of all variables in a single vector"""
    T = Tt * (1 + 0.5*(k-1) * Msqr)**(-1)
    A = area(x)

    dAdx    = wallGradient(x)
    dXdx    = concentrationGradient(X, Msqr, T)
    dYdx    = massFractionGradient(X, dXdx)
    dTtdx   = totalTempGradient(T, dYdx)
    dMsqrdx = machGradient(x, Msqr, Tt, dTtdx, A, dAdx)

    return dXdx, dTtdx, dMsqrdx


# ---------------------------- Universal constants ----------------------------

Ru = 8.314                             # Universal gas constant [kJ/kmol/K]
pRef = 101.3                           # Reference pressure [kPa]

# ------------------------- Combuster gas properties --------------------------

k = 1.3205                             # Ratio of specific heats
R = 288.45                             # Gas constant [J/kg/K]
cp = R * k/(k-1) / 1000                # J/g/K specific heat constant pressure

# ------------------------ Combuster inlet properties -------------------------

M3b  = 3.814                           # mach number
p3b  = 70.09                           # static pressure [kPa]
T3b  = 1237.63                         # temperature [K]
#T3b  = 1400
Tt3b = T3b * (1 + 0.5*(k-1) * M3b**2)  # stagnation temperature
# combined mass flow rate of stoichiometric mixture of ethylene and air [kg/s]
mdot  = 31.1186
rho3b = p3b * 1e3 / (R * T3b)          # kg/m^3
V3b   = M3b * sqrt(k * R * T3b)        # m/s
A3    = mdot / (rho3b*V3b)             # m^2

# -------------------------- Combustor Calculations ---------------------------

# calculate initial concentrations
n = 1 + 3*(1 + 3.76)
mWeights = np.array([
    28.054,  # C2H4
    31.998,  #   O2
    28.01,   #   CO
    18.015,  #  H2O
    44.009,  #  CO2
    28.014   #   N2
])           # kg/kmol
X3 = np.array(
    [1/n, 3/n, 0.0, 0.0, 0.0, 3*3.76/n]
) * p3b / (Ru * T3b)

# --------------------------- Combustor properties ----------------------------

combustorLength = 0.5                  # m
Cf = 0.002                             # skin friction coefficient


# create initial conditions vector
inititalConditions = np.append(X3, [Tt3b, M3b**2])

# integrate IV
sol = integrate.solve_ivp(
    gradient, (0, combustorLength),
    inititalConditions,
    method="LSODA",
    atol=1e-10, rtol=1e-10
)
# extract variables from integrator

x, X, Tt, Msqr = sol.t, sol.y[0:6], sol.y[6], sol.y[7]
M = np.sqrt(Msqr)


def massFraction(X):
    """Convert concentration into mass fraction"""
    Y = X * mWeights / np.sum(X*mWeights)
    if abs(np.sum(Y) - 1) > 0.001:
        print(f"total mass fraction = {np.sum(Y):.3f}")
    return Y


# calculate static temperature and mass fraction over combustion
T   = Tt * (1 + 0.5*(k-1) * Msqr)**(-1)
Y   = np.array([massFraction(X[:, i]) for i in range(len(x))]).T
v   = np.sqrt(Msqr*(k*R*T))
rho = mdot / (v*area(x))
P   = rho * R * T


# calculate combustor exit conditions:
X4  = X[:, -1]
Y4  = Y[:, -1]
Tt4 = Tt[-1]
M4  = M[-1]
T4  = T[-1]
P4  = P[-1]


# ------------------------------- Nozzle solver -------------------------------

fuelRatio = Y[0, 0] / sum(Y[0, :])
mdotAir = mdot * (1 - fuelRatio)

# Note there is an error in calculating something here
q0   = 50e+03
T0   = 220
M0   = 10
v0   = M0 * sqrt(1.4 * 287 * T0)
rho0 = 2 * q0 / v0**2
P0   = rho0 * 287 * T0
A0   = mdotAir / (rho0 * v0)

P10  = 3 * P0
M10_ = sqrt( 2/(k-1) * ( pow(P4/P10, (k-1)/k) * (1 + 0.5*(k-1)*M4**2) - 1 ) )
Tt10 = Tt4
T10_ = Tt10 * (1 + 0.5*(k-1) * M10_**2)**(-1)
v10_ = M10_ * sqrt(k*R*T10_)
v10  = 0.95 * v10_
T10  = Tt10 - 0.5*v10**2 / (cp*1E+03)
M10  = v10 / sqrt(k*R*T10)

rho10 = P10 / (R*T10)  # Ideal gas law
A10   = mdot / (rho10*v10)


# --------------------- Calculate performance of scramjet ---------------------

thrustPerEngine = mdot * (v10 - v0) + (P10 - P0) * A10
specificThrust  = thrustPerEngine / mdot

craftDrag = q0 * (9.25E-03) * (62.77)  # = q * Cd * Aref
netForce  = 4 * thrustPerEngine - craftDrag


# ------------------------------ Display results ------------------------------


def kPa(Pa):
    return Pa * 1E-03


print(f"\n{' Inlet conditions ':-^53}\n")
print(f"{'Mach Number':>25} = {M0:.2f}")
print(f"{'Temperature':>25} = {T0:.2f} K")
print(f"{'Pressure':>25} = {kPa(P0):.2f} kPa")
print(f"{'Inlet Area':>25} = {A0:.2f} m^2")

print(f"\n{' Combustor Exit Conditions ':-^53}\n")
print(f"{'Mach Number':>25} = {M4:.2f}")
print(f"{'Temperature':>25} = {T4:.2f} K")
print(f"{'Pressure':>25} = {kPa(P4):.2f} kPa")
print(f"{'Total Temperature':>25} = {Tt4:.2f} K")
# print(f"{'Total Pressure':>25} = {kPa(Pt4):.2f} kPa")

print("")
for species, Xs, Ys in zip(allSpecies, X4, Y4):
    print(
        f"{species:>10} = {abs(Xs):.5f} kmol/m^3 = {abs(Ys):.5f} kg/kg"
    )

print(f"\n{' Nozzle exit conditions ':-^53}\n")
print(f"{'Mach Number':>25} = {M10:.2f}")
print(f"{'Temperature':>25} = {T4:.2f} K")
print(f"{'Pressure':>25} = {kPa(P10):.2f} kPa")
print(f"{'Exit Area':>25} = {A10:.2f} m^2")
print(f"{'Capture Area Ratio':>25} = {A10/A0:.2f}")

print(f"\n{' Scramjet performance ':-^53}\n")
print(f"{'Thrust per Engine':>25} = {thrustPerEngine:.2f} N")
print(f"{'Specfic Thrust':>25} = {specificThrust:.2f} N.s/kg")
print(f"{'Drag':>25} = {craftDrag:.2f} N")
print(f"{'Net force':>25} = {netForce:.2f} N")

print("\n" + '-'*53)


# ========================== LaTeX Formatted Tables ==========================

rowEnd = r" \\" + "\n"
concentration = ["kmol", "per", "m", "cubed"]
area = ["m", "squared"]


def SI(value, *units, style="2f"):
    unitCommands = "".join(["\\"+unit for unit in units])
    formattedValue = f"{value:.{style}}"
    if unitCommands:
        return "\\SI{"+formattedValue+"}{"+unitCommands+"}"
    else:
        return "\\num{"+formattedValue+"}"


lines = [
    f"{'Variable':^17} & {'Value':^16}",                rowEnd,
    r"\midrule",                                       "\n",
    f"{'Mach Number':>17} & {M4:>16.2f}",               rowEnd,
    f"{'Temperature':>17} & {SI(T4, 'K'):>16}",         rowEnd,
    f"{'Pressure':>17} & {SI(kPa(P4), 'kPa'):>16}",     rowEnd,
    f"{'Total Temperature':>17} & {SI(Tt4, 'K'):>16}",  rowEnd,
    r"\bottomrule",                                    "\n",
    "\n",
    f"{'Variable':^11} & {'Value':^21}",                rowEnd,
    r"\midrule",                                       "\n",
    f"{'Mach Number':>11} & {M10:>21.2f}",              rowEnd,
    f"{'Temperature':>11} & {SI(T10_, 'K'):>21}",       rowEnd,
    f"{'Pressure':>11} & {SI(kPa(P10), 'kPa'):>21}",    rowEnd,
    f"{'Exit Area':>11} & {SI(A10, *area):>21}",        rowEnd,
    f"{'Thrust':>11} & {SI(thrustPerEngine, 'N'):>21}", rowEnd,
    r"\bottomrule",                                    "\n",
    "\n"
]

with open("LaTeX.txt", "w") as output:
    for line in lines:
        output.write(line)

    output.write(f"  Species & {'Concentration':^31} & {'Mass Fraction':^24}")
    output.write(rowEnd)

    output.write(r"\midrule" + "\n")
    for s, Xs, Ys in zip(allSpecies, X4, Y4):
        output.write(r"\ce{"f"{s:>4}"r"}")
        output.write(" & " + SI(abs(Xs), *concentration, style="5f"))
        output.write(" & " + SI(abs(Ys), "kg", "per", "kg", style="5f"))
        output.write(rowEnd)
    output.write(r"\bottomrule" + "\n")


# =================================== Plots ===================================

if T3b % 1 == 0:
    temp_label = T3b
else:
    temp_label = round(T3b)

loc = "{3b}"
fig, ax = plt.subplots()
formula = ("C$_2$H$_4$", "O$_2$", "CO", "H$_2$O", "CO$_2$", "N$_2$")
[ax.plot(x, X[i]*1e+03, label=formula[i]) for i in range(5)]
ax.legend()
plt.xlabel("distance along combustor [m]")
plt.ylabel("Concentration [kmol/m$^3$]")
plt.title(f"concentration over combustion at $T_{loc}$ = {T3b} K")
plt.grid()
plt.savefig(f"../part_2_img/concentration_{temp_label}.pdf")

fig, ax = plt.subplots()
[ax.plot(x, Y[i], label=formula[i]) for i in range(5)]
ax.legend()
plt.xlabel("distance along combustor [m]")
plt.ylabel("Mass fraction")
plt.title(f"Mass fraction over combustion at $T_{loc}$ = {T3b} K")
plt.grid()
plt.savefig(f"../part_2_img/mass_fraction_{temp_label}.pdf")

fig, ax = plt.subplots()
ax.plot(x, Tt, label="Tt")
ax.plot(x, [1.15*Tt3b for i in x], label="Ignition temperature")
plt.xlabel("distance along combustor [m]")
plt.ylabel("$T_0$ [K]")
plt.title(f"Stagnation temperature over combustion at $T_{loc}$ = {T3b} K")
ax.legend()
plt.grid()
plt.savefig(f"../part_2_img/stag_temp_{temp_label}.pdf")

fig, ax = plt.subplots()
ax.plot(x, T, label="T")
plt.xlabel("distance along combustor [m]")
plt.ylabel("T [K]")
ax.legend()
plt.title(f"Static temperature over combustion at $T_{loc}$ = {T3b} K")
plt.grid()
plt.savefig(f"../part_2_img/static_temp_{temp_label}.pdf")

fig, ax = plt.subplots()
ax.plot(x, M, label="M")
plt.xlabel("distance along combustor [m]")
plt.ylabel("M")
plt.title(f"Mach number over combustion at $T_{loc}$ = {T3b} K")
ax.legend()
plt.grid()
plt.savefig(f"../part_2_img/mach_{temp_label}.pdf")

fig, ax = plt.subplots()
ax.plot(x, kPa(P), label="pressure")
plt.xlabel("distance along combustor [m]")
plt.ylabel("Pressure [kPa]")
plt.title(f"Pressure over combustion at $T_{loc}$ = {T3b} K")
ax.legend()
plt.grid()
plt.savefig(f"../part_2_img/pressure_{temp_label}.pdf")

#plt.show()
