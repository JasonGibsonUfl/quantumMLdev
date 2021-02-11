"""
Module for calculating various ML descriptors for inputs of ML models
"""
from ase.io import vasp
import numpy as np
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler

def get_soap(file, rcut = 8, nmax = 6, lmax = 8, normalize = True):
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
    """
    structure = vasp.read_vasp(file)
    descriptor_gen = SOAP(
        periodic=True,
        species=np.unique(structure.get_atomic_numbers()),
        rcut=rcut,
        nmax=nmax,
        lmax=lmax,
        rbf='gto',
        sigma=0.125,
        average='inner'
    )
    descriptor = descriptor_gen.create(structure)
    if normalize:
        scaler = StandardScaler().fit(descriptor)
        descriptor = scaler.transform(descriptor)

    return descriptor

