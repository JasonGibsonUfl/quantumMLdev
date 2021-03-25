"""
Module for calculating various ML descriptors for inputs of ML models
"""
from ase.io import vasp
import numpy as np
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from pymatgen.io.ase import AseAtomsAdaptor


def compute_descriptor(descriptor, structure):
    """
    compute the descriptor for a structure for a pretrained model

    Parameters
    ----------
    descriptor : descriptor object
        descriptor object returned from mlModels.MLModel.rebuild_descriptor
    structure : pymatgen.Structure
        pymatgen structure of the material to compute descriptor
    Returns
    -------
    descriptor : np.ndarray
        returns array of descriptor
    """

    adaptor = AseAtomsAdaptor()
    entry = adaptor.get_atoms(structure)
    descriptor_raw = descriptor.create(entry)
    return descriptor_raw


def get_soap(file, rcut=6, nmax=6, lmax=8, normalize=True):
    """Initialize the SOAP module from dscribe package and calculate the soap descriptor
    Parameters
    ----------
    file : str
        Path to POSCAR file
    rcut : float
        A cutoff for local region in angstroms. Should be bigger than 1 angstrom
    nmac : int
        The number of radial basis functions.
    lmax : int
        The maximum degree of spherical harmonics
    normalize : bool
        set to False to return unnormalized descriptor
    Returns
    -------
    descriptor : np.ndarray
        returns array of SOAP descriptors

    todo: Normalize is not functioning as desired
    """
    structure = vasp.read_vasp(file)
    descriptor_gen = SOAP(
        periodic=True,
        species=np.unique(structure.get_atomic_numbers()),
        rcut=rcut,
        nmax=nmax,
        lmax=lmax,
        rbf="gto",
        sigma=0.125,
        average="inner",
    )
    descriptor = descriptor_gen.create(structure)
    if normalize:
        descriptor = StandardScaler().fit_transform(descriptor)

    return descriptor
