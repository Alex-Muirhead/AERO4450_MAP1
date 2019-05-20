"""
AERO4450 Major Assignment Part 2
Authors: Alex Muirhead and Robert Watt
Purpose: Simulate the combustion and calculate thrust produced by a scramjet
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy import integrate
import pandas as pd

plt.style.use("code/PaperDoubleFig.mplstyle")


# --------------------------- Universal constants ----------------------------

Ru = 8.314                             # Universal gas constant [J/mol/K]
pRef = 101.3                           # reference pressure [kPa]

# ------------------------- Combuster gas properties -------------------------

yb  = 1.3205                           # gamma
Rb  = 288.45                           # Gas constant [J/kg/K]
cpb = Rb * yb/(yb-1) / 1000            # J/g/K specific heat constant pressure

# ------------------------ Combuster inlet properties ------------------------

M3b  = 3.814                           # mach number
p3b  = 70.09                           # static pressure [kPa]
T3b  = 1237.63                         # temperature [K]
T3b  = 1400
Tt3b = T3b * (1 + 0.5*(yb-1) * M3b**2) # stagnation temperature
# combined mass flow rate of stoichiometric mixture of ethylene and air [kg/s]
mdot = 31.1186
rho3b = p3b * 1e3 / (Rb * T3b)         # kg/m^3
V3b = M3b * np.sqrt(yb * Rb * T3b)     # m/s
A3 = mdot / (rho3b*V3b)                # m^2


# --------------------- Combustor Calculations ---------------------------

# calculate initial concentrations
n = 1 + 3*(1 + 3.76)
MW = np.array([28, 32, 28, 18, 44])    # kg/kmol
X3 = np.array(
    [1/n, 3/n, 0.0, 0.0, 0.0]
) * p3b / (Ru * T3b)

# calculate mass fraction of N2 at inlet
X_N2 = (3 * 3.76 / n) * p3b / (Ru * T3b)
YN2 = X_N2 * 28 / (np.sum(X3 * MW) + X_N2 * 28)

# --------------------------- Combustor properties ---------------------------

combustor_length = 0.5                 # m
Cf = 0.002                             # skin friction coefficient


# calculate the area for each point along the combustor
def A(x, A3, Length=0.5):
    """calculate cross sectional area of combustor"""
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
    h0f = np.array([deltaHfuncs[i](T) for i in range(5)])  # kJ/kmol
    h0f = h0f / MW  # kJ/kg
    temp_gradient = -1/cpb * np.sum(dYdx(X, M, Tt, x, T) * h0f)
    return temp_gradient


def dM2(M, X, x, Tt, T):
    """Calculate spatial derivative of the square of the mach number"""
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
    Kf_i = np.array([pow(10, f(np.float64(T))) for f in logKfuncs])/(Ru*T/pRef)
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
    Y = X * MW / np.sum(X*MW) * (1 - YN2)
    if 0.999 > np.sum(Y) + YN2 or np.sum(Y) + YN2 > 1.001:
        print("total mass fraction = ", np.sum(Y) + YN2)
    return Y


# create initial conditions vector
init_conds = np.append(X3, [Tt3b, M3b**2])

# integrate IV
sol = integrate.solve_ivp(
    gradient, (0, combustor_length),
    init_conds,
    method="LSODA",
    atol=1e-10, rtol=1e-10
)
# extract variables from integrator
x, X, Tt, M = sol.t, sol.y[0:5], sol.y[5], np.sqrt(sol.y[6])

# calculate static temperature and mass fraction over combustion
T = Tt * (1 + 0.5*(yb - 1) * M**2)**(-1)
Y = np.array([massFraction(X[:, i]) for i in range(len(x))]).T

# calculate velocity as function of position
vel = M * np.sqrt(yb * Rb * T)

# calculate density as function of position
density = mdot / (vel * A(x, A3))

# calculate pressure as function of position P = rho*R*T
pressure = density * Rb * T


# calculate combustor exit conditions:
X4 = X[:, -1]
Tt4 = Tt[-1]
M4 = M[-1]
T4 = T[-1]
Y4 = Y[:, -1]
P4 = pressure[-1]


# ------------------------------- Nozzle solver -------------------------------

def AonAstar(M):
    power = (yb + 1) / (2*(yb-1))
    a = (yb + 1)/2
    b = 1 + 0.5*(yb - 1) * M**2
    return a**(-power) * b**power / M

def A10onA0(M0, M10, P0, P10):
    return P0 * M0 / (P10 * M10) * np.sqrt(1.4 * 287 * 220) / np.sqrt(yb * Rb * T10_dash)

# Sam's thrust 42.2 kN

# Note there is an error in calculating something here
v0 = 10 * np.sqrt(1.4 * 287 * 220)
rho0 = 2 * 50.e3 / v0**2
P0 = rho0*287.*220.
A0 = mdot / (rho0 * v0)
P10 = 3 * P0

M10_dash = np.sqrt((2/(yb -1))*((P4/P10)**((yb-1)/yb) * (1 + 0.5*(yb-1)*M4**4) - 1 ))
T10_dash = Tt4 * (1 + 0.5*(yb - 1) * M10_dash**2)**(-1)
v10_dash = M10_dash * np.sqrt(yb * Rb * T10_dash)
v10 = 0.95 * v10_dash
M10 = v10 / np.sqrt(yb * Rb * T10_dash)
A0 = mdot / (rho0 * v0)
A10 =  A0 * (P0 * 10 / (P10 * M10_dash) * np.sqrt(1.4 * Rb * T10_dash / ( yb * 287 * 220 )))


# Calculate performance of scramjet
thrust = mdot * (v10 - v0) + (P10 - P0) * A10
SF = thrust / mdot


# ------------------------------ Display results ------------------------------


def kPa(Pa):
    return Pa * 1E-03


print(f"\n{' Combustor Exit Conditions ':-^61}\n")
print(f"{'Mach Number':>30} & {M4:.2f}")
print(f"{'Temperature':>30} & {T4:.2f} K")
print(f"{'Pressure':>30} & {kPa(P4):.2f} kPa")
print(f"{'Total Temperature':>30} & {Tt4:.2f} K")
print(f"{'Total Pressure':>30} & {kPa(Pt4):.2f} kPa")

print("")
for species, Xs, Ys in zip(("C2H4", "O2", "CO", "H2O", "CO2"), X4, Y4):
    print(
        f"{species:>12} = {abs(Xs):.5f} kmol/m^3 = {abs(Ys):.5f} kg/kg"
    )

print(f"\n{' Nozzle exit conditions ':-^61}\n")

print(f"{'Mach Number':>30} = {M10:.2f}")
print(f"{'Temperature':>30} = {T4:.2f} K")
print(f"{'Pressure':>30} = {kPa(P4):.2f} kPa")
print(f"{'Exit Area':>30} = {A10:.2f} m^2")
print(f"{'Area Ratio':>30} = {A10onA4:.2f}")
print(f"{'Thrust':>30} = {thrust:.2f} N")


# ========================== LaTeX Formatted Tables ==========================

print(f"\n{'LaTeX formatted tables':-^61}\n")
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


print(f"{'Variable':^17} & {'Value':^16}", end=rowEnd)
print(r"\midrule")
print(f"{'Mach Number':>17} & {M4:>16.2f}", end=rowEnd)
print(f"{'Temperature':>17} & {SI(T4, 'K'):>16}", end=rowEnd)
print(f"{'Pressure':>17} & {SI(kPa(P4), 'kPa'):>16}", end=rowEnd)
print(f"{'Total Temperature':>17} & {SI(Tt4, 'K'):>16}", end=rowEnd)
print(f"{'Total Pressure':>17} & {SI(kPa(Pt4), 'K'):>16}", end=rowEnd)
print(r"\bottomrule""\n")

print(f"{'Variable':^11} & {'Value':^21}", end=rowEnd)
print(r"\midrule")
print(f"{'Mach Number':>11} & {M10:>21.2f}", end=rowEnd)
print(f"{'Temperature':>11} & {SI(T10, 'K'):>21}", end=rowEnd)
print(f"{'Pressure':>11} & {SI(kPa(P10), 'kPa'):>21}", end=rowEnd)
print(f"{'Exit Area':>11} & {SI(A10, *area):>21}", end=rowEnd)
print(f"{'Area Ratio':>11} & {SI(A10onA4):>21}", end=rowEnd)
print(f"{'Thrust':>11} & {SI(thrust, 'N'):>21}", end=rowEnd)
print(r"\bottomrule""\n")

print(f"  Species & {'Concentration':^31} & {'Mass Fraction':^24}", end=rowEnd)
print(r"\midrule")
for s, Xs, Ys in zip(("C2H4", "O2", "CO", "H2O", "CO2"), X4, Y4):
    print(
        r"\ce{"f"{s:>4}"r"}",
        "&", SI(abs(Xs), *concentration, style="5f"),
        "&", SI(abs(Ys), "kg", "per", "kg", style="5f"),
        end=rowEnd
    )
print(r"\bottomrule""\n")


# =================================== Plots ===================================

loc = "{3b}"
fig, ax = plt.subplots()
formula = ("C$_2$H$_4$", "O$_2$", "CO", "H$_2$O", "CO$_2$")
[ax.plot(x, X[i]*1e+03, label=formula[i]) for i in range(5)]
ax.legend()
plt.xlabel("distance along combustor [m]")
plt.ylabel("Concentration [kmol/m$^3$]")
plt.title(f"concentration over combustion at $T_{loc}$ = {T3b} K")
plt.grid()

fig, ax = plt.subplots()
formula = ("C$_2$H$_4$", "O$_2$", "CO", "H$_2$O", "CO$_2$")
[ax.plot(x, Y[i], label=formula[i]) for i in range(5)]
ax.legend()
plt.xlabel("distance along combustor [m]")
plt.ylabel("Mass fraction")
plt.title(f"Mass fraction over combustion at $T_{loc}$ = {T3b} K")
plt.grid()

fig, ax = plt.subplots()
ax.plot(x, Tt, label="Tt")
ax.plot(x, [1.15*Tt3b for i in x], label="Ignition temperature")
plt.xlabel("distance along combustor [m]")
plt.ylabel("$T_0$ [K]")
plt.title(f"Stagnation temperature over combustion at $T_{loc}$ = {T3b} K")
ax.legend()
plt.grid()

fig, ax = plt.subplots()
ax.plot(x, T, label="T")
plt.xlabel("distance along combustor [m]")
plt.ylabel("T [K]")
ax.legend()
plt.title(f"Static temperature over combustion at $T_{loc}$ = {T3b} K")
plt.grid()

fig, ax = plt.subplots()
ax.plot(x, M, label="M")
plt.xlabel("distance along combustor [m]")
plt.ylabel("M")
plt.title(f"Mach number over combustion at $T_{loc}$ = {T3b} K")
ax.legend()
plt.grid()

fig, ax = plt.subplots()
ax.plot(x, pressure/1000, label="pressure")
plt.xlabel("distance along combustor [m]")
plt.ylabel("Pressure [kPa]")
plt.title(f"Pressure over combustion at $T_{loc}$ = {T3b} K")
ax.legend()
plt.grid()

#plt.show()
