from quantumml.featurizers.utils import *
import numpy as np
from pymatgen.core import Structure

elements = ["Al", "Ni"]
expected_list_rdf = [["Al", "Al"], ["Al", "Ni"], ["Ni", "Ni"]]
calculated_list_rdf = calc_rdf_tup(elements)

expected_list_adf = [
    ["Al", "Al", "Al"],
    ["Al", "Al", "Ni"],
    ["Al", "Ni", "Al"],
    ["Ni", "Al", "Ni"],
    ["Ni", "Ni", "Al"],
    ["Ni", "Ni", "Ni"],
]
calculated_list_adf = calc_adf_tup(elements)


poscar_dict = {
    "@module": "pymatgen.core.structure",
    "@class": "Structure",
    "charge": None,
    "lattice": {
        "matrix": [
            [5.905482, 0.0, 0.0],
            [-0.0, 3.58454, 0.0],
            [-2.464177, -0.0, 4.578934],
        ],
        "a": 5.905482,
        "b": 3.58454,
        "c": 5.199885081776808,
        "alpha": 90.0,
        "beta": 118.28714589396515,
        "gamma": 90.0,
        "volume": 96.92887337986855,
    },
    "sites": [
        {
            "species": [{"element": "Ni", "occu": 1}],
            "abc": [0.595702, 0.176304, 0.228367],
            "xyz": [2.955170729405, 0.6319687401599999, 1.045677420778],
            "label": "Ni",
            "properties": {},
        },
        {
            "species": [{"element": "Ni", "occu": 1}],
            "abc": [0.341657, 0.676304, 0.352839],
            "xyz": [1.148191515171, 2.4242387401600003, 1.615626493626],
            "label": "Ni",
            "properties": {},
        },
        {
            "species": [{"element": "Al", "occu": 1}],
            "abc": [0.824438, 0.676304, 0.576332],
            "xyz": [3.4485197103520004, 2.4242387401600003, 2.638986190088],
            "label": "Al",
            "properties": {},
        },
        {
            "species": [{"element": "Al", "occu": 1}],
            "abc": [0.580691, 0.176304, 0.738274],
            "xyz": [1.6100224375640002, 0.6319687401599999, 3.380507919916],
            "label": "Al",
            "properties": {},
        },
        {
            "species": [{"element": "Al", "occu": 1}],
            "abc": [0.077944, 0.176304, 0.446586],
            "xyz": [-0.6401700607139998, 0.6319687401599999, 2.044887819324],
            "label": "Al",
            "properties": {},
        },
        {
            "species": [{"element": "Ni", "occu": 1}],
            "abc": [0.336293, 0.676304, 0.842711],
            "xyz": [-0.09061680562099994, 2.4242387401600003, 3.858718050074],
            "label": "Ni",
            "properties": {},
        },
        {
            "species": [{"element": "Al", "occu": 1}],
            "abc": [0.126352, 0.176304, 0.986085],
            "xyz": [-1.683718515381, 0.6319687401599999, 4.51521813339],
            "label": "Al",
            "properties": {},
        },
        {
            "species": [{"element": "Ni", "occu": 1}],
            "abc": [0.842499, 0.676304, 0.08286],
            "xyz": [4.771180973298, 2.4242387401600003, 0.37941047124000005],
            "label": "Ni",
            "properties": {},
        },
    ],
}
structure = Structure.from_dict(poscar_dict)
expect_molarFrac_list = [("Al", 0.5), ("Ni", 0.5)]
calculated_molarFrac_list = calc_mol_frac(elements, structure)

specs_ab = get_elements(elements)
expected_indices = [[2, 3, 4, 6], [0, 1, 5, 7]]


def test_calc_rdf_tup_len():
    assert len(expected_list_rdf) == len(calculated_list_rdf)


def test_calc_rdf_tup_element():
    for tup in calculated_list_rdf:
        assert tup in expected_list_rdf


def test_calc_adf_tup_len():
    assert len(expected_list_adf) == len(calculated_list_adf)


def test_calc_adf_tup_element():
    for tup in calculated_list_adf:
        assert tup in expected_list_adf


def test_calc_mol_frac_len():
    assert len(expect_molarFrac_list) == len(calculated_molarFrac_list)


def test_calc_mol_frac_element():
    for tup in calculated_molarFrac_list:
        assert tup in expect_molarFrac_list


def test_elements():
    for el in specs_ab:
        assert type(el) == type(Element("Al"))


def test_get_indices():
    calculated_indices = get_indices(structure.sites, specs_ab)
    assert calculated_indices == expected_indices


# def test_get_neighbor_atoms():
#     neighbors = structure.get_all_neighbors(10.1)
#     neighbor_atoms = get_neighbor_atoms(neighbors, expected_indices[0])
#     assert isinstance(neighbor_atoms, List)

def test_get_neighbor_distribution_list():
    neighbors = structure.get_all_neighbors(10.1)
    ndl = get_neighbor_distribution_list(neighbors,expected_indices[0], specs_ab[-1])
    assert isinstance(ndl, List)