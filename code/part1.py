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
    "C2H4 + 2 O2 --> 2 CO + 2 H2O",
    "CO + 0.5 O2 <-> CO2"
]

for reaction in reactions:
    direction = re.search(r"(-->|<->)", reaction)
    if not direction:
        raise ValueError("Reaction direction not defined")

    reactants, products = reaction.split(direction.group())
    test = reactants.split("+")
    print(products.split("+"))


def rates(X):
    # Calculate Arhennius rates here
    k1f, k2f, k2r = ...
    return np.array([
        k1f * pow(X["C2H4"], 0.5) * pow(X["O2"], 0.65),
        k2f * pow(X["CO"], 1) * pow(X["O2"], 0.65),

    ])
