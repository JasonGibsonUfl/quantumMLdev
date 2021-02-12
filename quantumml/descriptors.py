"""
Module for calculating various ML descriptors for inputs of ML models
"""
from ase.io import vasp
import numpy as np
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler


def get_soap(file, rcut=6, nmax=8, lmax=6, normalize=True):
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
