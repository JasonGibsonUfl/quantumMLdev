import itertools
from typing import List, Any
from pymatgen.core.periodic_table import Element

PymatgenStructure = Any


def calc_adf_tup(elements: List) -> List:
    """
    compute the tuple list required for ADF featurizer

    Parameters
    ----------
    elements : List
        list of str elements

    Returns
    -------
    adf_tup : List
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
    molarFrac: List
        list of molar fraction for each element
    """
    molarFrac = []
    numElements = len(elements)
    for i in range(0, numElements):
        elem = Element(elements[i])
        elemPerc = structure.composition.get_atomic_fraction(elem)
        molarFrac.append((elements[i], elemPerc))
    return molarFrac


def get_elements(tup: List) -> List:
    """
    get elements in tup

    Parameters
    ----------
    tup : List
        tuple of elements

    Returns
    -------
    elements : List
        list of element objects
    """
    return [Element(element) for element in tup]


def get_indices(sites: List, specs: List) -> List:
    """
    get the indices corresponding to elements in tup

    Parameters
    ----------
    sites : List
        list of PyMatGen periodic sites

    Returns
    -------
    indices_ab : List
        list of indices corresponding to elements
    """
    indices = [
        [j[0] for j in enumerate(sites) if j[1].specie == spec] for spec in specs
    ]
    return indices


def contains_ab(indices_ab: List):
    """
    checks to see if structure contains both elements a and b

    Parameters
    ----------
    indices_ab : List
        2D list of the indices of atom A and B

    Returns
    -------

    """
    return 0 in [len(site) for site in indices_ab]


def get_neighbor_list(Neighbors: List, spec1, spec2=None) -> List:
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

    NeighborList1 = []
    NeighborList2 = []
    for bN in Neighbors:
        tempNeighborList1 = []
        tempNeighborList2 = []
        for neighbor in bN:
            if neighbor[0].specie == spec1:
                tempNeighborList1.append(neighbor)
            elif neighbor[0].species == spec2:
                tempNeighborList2.append(neighbor)
        NeighborList1.append(tempNeighborList1)
        NeighborList2.append(tempNeighborList2)
    return NeighborList1, NeighborList2


def get_neighbor_distribution_list(Neighbors: List, indices_ab, Spec_ba) -> List:
    """
    get neighbor distribution of elemnet B/A with respect to element A/B
    Parameters
    ----------
    Neighbors_ab : List
        list of neighbor atoms for element A/B
    Spec_ba : Element
        element B/A
    Returns : List
        list of the of neighbor distribution
    """

    neighbors = [Neighbors[i] for i in indices_ab]
    NeighborDistList = []
    for aN in neighbors:
        tempNeighborList = [
            neighbor for neighbor in aN if neighbor[0].specie == Spec_ba
        ]  # Neighbors of alphaSpec that are betaSpec
        NeighborDist = [j[1][1] for j in enumerate(tempNeighborList)]
        NeighborDistList.append(
            NeighborDist
        )  # Add the neighbor distances of all such neighbors to a list
    return NeighborDistList
