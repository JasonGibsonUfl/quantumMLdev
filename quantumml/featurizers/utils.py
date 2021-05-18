import itertools
from typing import List,Any
from pymatgen.core.periodic_table import Element

PymatgenStructure = Any

def calc_adf_tup(elements: List) -> List:
    """
    compute the tuple list required for ADF featurizer

    Parameters
    ----------
    elements : list
        list of str elements

    Returns
    -------
    adf_tup : list
        list of adf tuples
    """
    if len(elements) != 2:
        raise ValueError('Element must be of length 2')

    adf_tup = [list(p) for p in itertools.product(elements, repeat=3)]
    del adf_tup[3]
    del adf_tup[3]
    return adf_tup

def calc_rdf_tup(elements: List) -> List:
    """
    compute the tuple list required for RDF featurizer

    Parameters
    ----------
    elements : list
        list of str elements

    Returns
    -------
    rdf_tup : list
        list of adf tuples
    """
    if len(elements) != 2:
        raise ValueError('Element must be of length 2')
    return [list(p)for p in itertools.combinations_with_replacement(elements, 2)]

def calc_mol_frac(elements: List ,structure: PymatgenStructure) -> List:
    """
    Calculate the molar fraction of elements from pymatgen structure.
    Parameters
    ----------
    struct: pymatgen.Structure
        A periodic crystal composed of a lattice and a sequence of atomic
        sites with 3D coordinates and elements.

    Returns
    -------
    molarFrac: list
        list of molar fraction for each element
    """
    molarFrac = []
    numElements = len(elements)
    for i in range(0, numElements):
        elem = Element(elements[i])
        elemPerc = structure.composition.get_atomic_fraction(elem)
        molarFrac.append((elements[i], elemPerc))
    return molarFrac

def get_specs_ab(pair: List) -> List:
    """
    get elements a and b in the rdf tup

    Parameters
    ----------
    pair : list
        pair of elements

    Returns
    -------
    specs_ab : list
        list of element objects for elements a and b
    """
    return [Element(pair[0]), Element(pair[1])]

def get_sites_ab(sites, specs_ab):
    """
    get the indices corresponding to elements a and b in the rdf tup

    Parameters
    ----------
    sites : list
        list of PyMatGen periodic sites

    Returns
    -------
    indices_ab : list
        list of indices corresponding to elements a and b in the rdf tup
    """
    indices_ab = [[j[0] for j in enumerate(sites) if j[1].specie == spec] for spec in specs_ab]
    return indices_ab

