import numpy as np
import re

concentrations = {
    "C2H4": 1.,
    "O2"  : 1.,
    "CO"  : 1.,
    "H2O" : 1.,
    "CO2" : 1.
}

reactions = [
    {
        "equation": "C2H4 + 2 O2 --> 2 CO + 2 H2O",
        "override": [0.5, 0.65, None, None],
        "A"       : 1.739e+09,
        "Ea"      : 1.485e-01
    },
    {
        "equation": "CO + 0.5 O2 <-> CO2",
        "A"       : 6.235e+07,
        "Ea"      : 5.021e-02
    }
]


def exponents(half, override=None):
    for i, X in enumerate(half.split("+")):
        expRe     = re.search(r"(?:\A|[^a-zA-Z])([\d./]+)", X)
        speciesRe = re.search(r"([a-zA-Z]\w*)", X)

        if override and override[i]:
            exp = override[i]
        elif expRe:
            exp = eval(expRe.group())
        else:
            exp = 1

        if not speciesRe:
            raise ValueError("Invalid formula")
        species = speciesRe.group()

        print(f"[{species}]^{exp}")


for reaction in reactions:
    eq = reaction["equation"]
    direction = re.search(r"(-->|<->)", eq)
    if not direction:
        raise ValueError("Reaction direction not defined")
    direction = direction.group()

    reactants, products = eq.split(direction)
    exponents(reactants)
    if direction == "<->":
        exponents(products)


def rates(X):
    # Calculate Arhennius rates here
    k1f, k2f, k2r = ...
    return np.array([
        k1f * pow(X["C2H4"], 0.5) * pow(X["O2"], 0.65),
        k2f * pow(X["CO"], 1) * pow(X["O2"], 0.65)
    ])
