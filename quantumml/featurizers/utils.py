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

def calc_mol_frac(elements,structure: PymatgenStructure) -> List:
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