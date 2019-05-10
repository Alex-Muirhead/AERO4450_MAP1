"""

"""

import numpy as np 
from matplotlib import pyplot as plt 
from scipy import interpolate
from scipy import integrate
import pandas as pd

#Define combuster inlet conditions (in burner gas constant terms)
Ru   = 8.314
pRef = 101.3e3

yb = 1.3205 # gamma
Rb = 188.45 # Gas constant [J/kg K]
M3b = 3.814 # mach number
p3b = 70.09 # static pressure [kPa]
pt3b = p3b / (1 - 0.5*(yb - 1)*M3b**2)**(-yb/(yb-1)) #stagnation pressure
T3b = 1237.63 # temperature [K]
Tt3b = T3b * (1 + 0.5*(yb - 1) * M3b**2) # stagnation temperature
mdot = 31.1186 # combined mass flow rate of stoichiometric mixture of ethylene and air [kg/s]
cpb = Rb / (1 - 1/yb) # specific heat at constant pressure
rho3b = p3b / (Rb * T3b)
V3b = M3b * np.sqrt(yb * Rb * T3b)
A3 = mdot / rho3b*V3b
combustor_length = 0.5 # m

increments = 1000
dx = combustor_length/increments

Cf = 0.002 # skin friction coefficient

MW = np.array([28, 32, 28, 18, 44])



#read chemical data
MW = np.array([28, 32, 28, 18, 44])
chemData = []
for species in ("C2H4", "O2", "CO", "H2O", "CO2"):
    data = pd.read_csv(f"code/chemData/{species}.txt", sep="\t", skiprows=1)
    chemData.append(data[1:])  # Skip T=0K

logKfuncs, deltaHfuncs = [], []
for data in chemData:
    T      = data["T(K)"].values.astype(float)
    logKf  = data["log Kf"].values.astype(float)
    deltaH = data["delta-f H"].values.astype(float) * 1e+03  # kJ/mol->kJ/kmol
    logKfuncs.append(interpolate.interp1d(T, logKf, kind="quadratic"))
    deltaHfuncs.append(interpolate.interp1d(T, deltaH, kind="quadratic"))


def A(x, A3, Length=0.5):
    return A3 * (1 + 3*x/Length)

def dAonA(x, A3, Length=0.5):
    return 3 * A3 / (Length * A(x, A3))

def arrhenius(T):
    return np.array([
        1.739e+09 * np.exp(-1.485e+05 / (Ru*T)),
        6.324e+07 * np.exp(-5.021e+04 / (Ru*T))
    ])

def Y(X):
    return X * MW * (1 - X[5]*MW[5]) / ( np.sum(X[0:5] * MW[0:5]) )

def vectorInterface(lengths):
    L = [0, *np.cumsum(lengths)]

    def wrapper(func):
        def inner(t, args):
            splitArgs = [args[l:r] for l, r in zip(L[:-1], L[1:])]
            output = func(t, *splitArgs)
            return np.hstack([*output])
        return inner
    return wrapper

def dXdx(M, Tt, X):
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

    def Kc(T, p):
        """Kc = Kp * pow(pRef/p, ν+...)"""
        # NOTE: Account for partial pressures
        Kf_i    = np.array([pow(10, f(np.float64(T))) for f in logKfuncs]) * (pRef/p)
        forward = pow(Kf_i, maskF*ν)
        reverse = pow(Kf_i, maskR*ν)
        return np.prod(reverse, axis=1) / np.prod(forward, axis=1)

    #@vectorInterface((5,1,1))
    def concentration_gradient(χ, M, T):
        limit = (χ < 0)
        χ[limit] = 0

        p = pt3b * (1 + 0.5*(yb-1)*M**2)**(-yb/(yb-1))

        kf    = arrhenius(T)
        kr    = kf / Kc(T, p)
        kr[0] = 0  # One way reaction

        forward = kf * np.prod(pow(χ, maskF*ν), axis=1)
        reverse = kr * np.prod(pow(χ, maskR*ν), axis=1)
        χGrad   = μ.T @ forward - μ.T @ reverse

        χGrad[(χGrad < 0)*limit] = 0

        
        #hGrad = -sum([dχ_i*h_i(T) for dχ_i, h_i in zip(χGrad, deltaHfuncs)])

        #return np.append(χGrad, 0.0) #hGrad
        return χGrad

    T = Tt * 1/(1 + 0.5*(yb - 1) * M**2)
    v = M * np.sqrt(yb * Rb * T)
    return concentration_gradient(X, M, T) / v

def new_Tt(X, M, Tt, gamma = yb):
    T = Tt * 1/(1 + 0.5*(gamma - 1) * M**2)
    h0fi = [deltaHfuncs[i](T) for i in range(5)]
    h03fi = [deltaHfuncs[i](T3b) for i in range(5)]
    return T3b - (1/cpb) * ( np.sum(Y(X[-1])) * h0fi - np.sum(Y(X3)) * h03fi )

def dYdx(X):
    reacting_sum = np.sum(X[0:5] * MW[0:5])
    return MW * (1 - Y(X)[6]) * ( 1/reacting_sum * dXdx - X/reacting_sum**2 * np.sum(MW[0:5] * dXdx))

def dTtdx(X):
    return -1/cpb * np.sum(dYdx(X))

def dM2dx(M, X, x, Tt, gamma = yb):
    Deff = 2 * np.sqrt(A(x, A3) / np.pi)
    return M**2 / (dx) * ((1 + 0.5*(gamma - 1)*M**2) / (1 - M**2)) * (-2 * dAonA(x, A3) + (1 + gamma*M**2) * dTtdx(X)/Tt + gamma*M**2 * 4 * Cf * dx / Deff)

@vectorInterface((5,1,1))
def gradient(x, X, Tt, M):
    return [dXdx(M, Tt, X), dTtdx(X), dM2dx(M, X, x, Tt)]

n = 1 + 3*(1 + 3.76)
X3 = np.array(
        [1/n, 3/n, 0.0, 0.0, 0.0]
    ) * 70e+03 / (Ru * T3b) * 1e-03

init_conds = np.append(X3, [Tt3b, M3b])
sol = (integrate.solve_ivp(gradient, (0, 0.5), init_conds, method="LSODA", events=None, atol=1e-10, rtol=1e-10))



