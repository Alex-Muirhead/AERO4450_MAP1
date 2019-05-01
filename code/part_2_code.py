"""

"""

import numpy as np 
from matplotlib import pyplot as plt 

#Define combuster inlet conditions (in burner gas constant terms)
yb = 1.3205 # gamma
Rb = 188.45 # Gas constant [J/kg K]
p3b = 70.09 # static pressure [kPa]
M3b = 3.814 # mach number
T3b = 1237.63 # temperature [K]
mdot = 31.1186 # combined mass flow rate of stoichiometric mixture of ethylene and air [hg/s]

Cf = 0.002 # skin friction coefficient


def A(x, A3, L=0.5):
    return A3 * (1 + 3*x/L)

def dAonA(x, A3, L=0.5):
    return 3 * A3 / (L * A(x, A3))