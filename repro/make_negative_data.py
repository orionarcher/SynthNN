import random

import numpy as np
from pymatgen.core import Composition


def make_negative_data(
    num_examples, max_atoms=5, max_coefficient=11, seed=3, weighted=False
):
    # play around with max_coefficient values!!

    output_array = []
    element_names_array = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
    ]
    # element_sum = np.loadtxt("GNN_icsd_mpids_unique_element_proportions.txt")
    # dummy element sum only for testing
    element_sum = np.ones(94)
    random.seed(seed)
    np.random.seed(seed)
    while len(output_array) < num_examples:
        if len(output_array) % 10000 == 0:
            pass
        num_atoms = np.random.randint(2, max_atoms, 1)[
            0
        ]  # number of atom types (binary, tern,  quat)
        coeffs = np.random.randint(
            1, max_coefficient, num_atoms
        )  # coeffs for each atom
        if weighted:
            atomic_numbers = (
                np.random.choice(94, num_atoms, p=np.reshape(element_sum, [94])) + 1
            )  # add one to give between 1,95
        else:
            atomic_numbers = np.random.randint(
                1, 95, num_atoms
            )  # goes up to atomic number 94

        output = ""
        for i in range(num_atoms):
            output += element_names_array[atomic_numbers[i] - 1]
            output += str(coeffs[i])
        if (
            Composition(output).alphabetical_formula.replace(" ", "")
            not in output_array
        ):
            output_array.append(
                Composition(output).alphabetical_formula.replace(" ", "")
            )
    return output_array


# %%
neg_samples = make_negative_data(10_000)

import pandas as pd

pd.Series(neg_samples, name="formula").to_csv("negative_samples.txt", header=None, index=False, sep=' ')
