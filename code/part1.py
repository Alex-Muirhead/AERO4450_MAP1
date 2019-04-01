import math

import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.integrate import solve_ivp

R    = 8.314    # kJ/kmol
pRef = 101.3e3  # Pa


class Blank:
    """Blank class.

    Returns `None` when iterated or indexed.
    Acts as `False` within `if` statement.

    """
    def __getitem__(self, key):
        return None

    def __iter__(self):
        return self

    def __len__(self):
        return 0

    def __next__(self):
        return None


def decode(string, override=None):
    """Decode a string of the form 'a X'

    Determines the stoichiometric coefficient and species formula
    from the given string. Provides an option to override the
    coefficient when calculating reaction rates from empiral global
    reactions.

    Parameters
    ----------
    string (str):
        The string to decode, in the form 'a X'. Can only contain
        a single species and coefficient.
    override (float, optional):
        An optional parameter to override the stoichiometric
        coefficient when calculating reaction rates.

    Returns
    -------
    float:
        Stoichiometric coefficient
    float:
        Species power for reaction rate
    str:
        Species chemical formula

    """
    stoichRe  = re.search(r"(?:\A|[^a-zA-Z])([\d./]+)", string)
    speciesRe = re.search(r"([a-zA-Z]\w*)", string)

    stoich = eval(stoichRe.group()) if stoichRe else 1
    power = override if override else stoich

    if not speciesRe:
        raise ValueError("Invalid formula")
    species = speciesRe.group()

    return stoich, power, species


initialConcentrations = {
    "C2H4": 0.0,
    "O2"  : 0.8,
    "CO"  : 0.0,
    "H2O" : 0.0,
    "CO2" : 0.2
}

reactions = [
    {
        "equation": "C2H4 + 2 O2 --> 2 CO + 2 H2O",
        "override": [0.5, 0.65, None, None],
        "forward" : {
            "A" : 1.739e+09,
            "Ea": 1.485e+05
        }
    },
    {
        "equation": "CO + 0.5 O2 <-> CO2",
        "forward" : {
            "A" : 6.235e+07,
            "Ea": 5.021e+04
        }
    }
]


store = {}
μ = []
ν = []
d = []

for i, reaction in enumerate(reactions):
    eq = reaction["equation"]
    override = iter(reaction.get("override", Blank()))

    direction = re.search(r"(-->|<->)", eq)
    if not direction:
        raise ValueError("Reaction direction not defined")
    direction = direction.group()
    d.append(direction)

    left, right = eq.split(direction)
    for string in left.split("+"):
        stoich, power, species = decode(string, next(override))

        if species not in store:
            store[species] = len(store)
            μ.append([0]*len(reactions))
            ν.append([0]*len(reactions))

        μ[store[species]][i] = -stoich
        ν[store[species]][i] = power

    for string in right.split("+"):
        stoich, power, species = decode(string, next(override))

        if species not in store:
            store[species] = len(store)
            μ.append([0]*len(reactions))
            ν.append([0]*len(reactions))

        μ[store[species]][i] = +stoich
        ν[store[species]][i] = power

μ = np.array(μ).T
ν = np.array(ν).T

mask = (np.array([-μ, μ]) > 0)

νForward, νReverse = ν * mask
νTotal = np.sum(νForward-νReverse, axis=-1)


def arrhenius(T, A, Ea):
    return A * math.exp(-Ea / (R*T))


def rateConstants(T, p):
    n = len(reactions)
    constants = np.zeros((n, 2, 2))
    for i, r in enumerate(reactions):
        kf, kr = 0, 0
        if "forward" in r:
            kf = arrhenius(T, **r["forward"])
            constants[i, 0, 0] = kf

        if "reverse" in r:
            kr = arrhenius(T, **r["reverse"])
            constants[i, 1, 1] = -kr

        if "<-" in d[i] and kf and not kr:
            constants[i, 1, 0] = -kf * pow(p/pRef, νTotal[i])
        elif "->" in d[i] and kr and not kf:
            constants[i, 0, 1] = kr * pow(p/pRef, -νTotal[i])

        if not (kf or kr):
            raise ValueError(f"No defined reaction rate for reaction {r}!")

    return constants


def gradient(t, χ):
    # HACK: Stop negative concentrations
    χ[χ < 0] = 0.

    T = 1400 + 7e5*t
    p = 70e3 + 35e6*t
    k = rateConstants(T, p)
    Π = np.prod(pow(χ, ν*mask), axis=-1)
    rate = np.sum(np.squeeze(k @ Π[..., None]), axis=-1)
    deriv = μ.T @ rate
    print(f"Time {t}s")
    print("\t".join([f"{n:.3e}" for n in χ]))
    print("\t".join([f"{n:.3e}" for n in deriv]))
    print("-"*70)
    return deriv


χ0 = np.array([initialConcentrations[s] for s in store])
solution = solve_ivp(gradient, (0, 1.3e-3), χ0, method="LSODA")

for s, y in zip(store.keys(), solution.y):
    plt.plot(solution.t, y, label=s)
    plt.legend()
plt.show()
