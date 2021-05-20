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

from pymatgen.core import Structure
from quantumml.featurizers.rdf_adf import Global_RDF
import numpy as np
elements = ['Al', 'Ni']

def test_global_rdf_class():
    grdf = Global_RDF(elements)
    structure = Structure.from_dict(poscar_dict)
    features = grdf.featurize([structure])
    assert isinstance(features, np.ndarray)