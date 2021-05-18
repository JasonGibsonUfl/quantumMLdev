import itertools
from typing import List, Any
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
        raise ValueError("Element must be of length 2")

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
        raise ValueError("Element must be of length 2")
    return [list(p) for p in itertools.combinations_with_replacement(elements, 2)]


def calc_mol_frac(elements: List, structure: PymatgenStructure) -> List:
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


def get_indices_ab(sites: List, specs_ab: List) -> List:
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
    indices_ab = [
        [j[0] for j in enumerate(sites) if j[1].specie == spec] for spec in specs_ab
    ]
    return indices_ab


def conatians_ab(indices_ab: List):
    """
    checks to see if structure contains both elements a and b

    Parameters
    ----------
    indices_ab : list
        2D list of the indices of atom A and B

    Returns
    -------

    """
    return 0 in [len(site) for site in sites_ab]


def get_neighbor_atoms(neighbors: List, indices: List) -> List:
    """
    returns list of pymatgen periodic sites of the neighbor atoms of element

    Parameters
    ----------
    neighbors : List
        list of neighbors within a cutoff radius using pymatgen Strcture.get_neighbor_atoms
    indices : List
        list of indices of the site of the desired element A or B
    Returns
    -------
    Neighbors : List
        list of neighbors of element A or B
    """
    Neighbors = [neighbors[i] for i in indices]  # Get all neighbors of alphaSpec
    return Neighbors

def get_neighbor_distribution_list(Neighbors_ab, Spec_ba):
    NeighborDistList = []
    for aN in Neighbors_ab:
        tempNeighborList = [neighbor for neighbor in aN if neighbor[0].specie==Spec_ba]# Neighbors of alphaSpec that are betaSpec
        NeighborDist = [j[1][1] for j in enumerate(tempNeighborList)]
        NeighborDistList.append(NeighborDist) # Add the neighbor distances of all such neighbors to a list
    return NeighborDistList
