from quantumml.featurizers import MaterialStructureFeaturizer
from typing import Any
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

PymatgenStructure = Any

class SOAP(MaterialStructureFeaturizer):
    """
    Calculate sine Coulomb matrix for crystals.
    A variant of Coulomb matrix for periodic crystals.
    The sine Coulomb matrix is identical to the Coulomb matrix, except
    that the inverse distance function is replaced by the inverse of
    sin**2 of the vector between sites which are periodic in the
    dimensions of the crystal lattice.
    Features are flattened into a vector of matrix eigenvalues by default
    for ML-readiness. To ensure that all feature vectors are equal
    length, the maximum number of atoms (eigenvalues) in the input
    dataset must be specified.
    This featurizer requires the optional dependencies pymatgen and
    matminer. It may be useful when crystal structures with 3D coordinates
    are available.
    See [1]_ for more details.
    References
    ----------
    .. [1] Faber et al. Inter. J. Quantum Chem. 115, 16, 2015.
    Examples
    --------
    >>> import pymatgen as mg
    >>> lattice = mg.Lattice.cubic(4.2)
    >>> structure = mg.Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> featurizer = SineCoulombMatrix(max_atoms=2)
    >>> features = featurizer.featurize([structure])
    Note
    ----
    This class requires matminer and Pymatgen to be installed.
    """

    def __init__(
        self,
        species: list,
        rcut: int = 6,
        nmax: int = 6,
        lmax: int = 8,
        rbf: str = "gto",
        sigma: float = 0.125,
        average: str = "inner",
        periodic: bool = True,
    ):
        """
        Parameters
        ----------
        max_atoms: int (default 100)
          Maximum number of atoms for any crystal in the dataset. Used to
          pad the Coulomb matrix.
        flatten: bool (default True)
          Return flattened vector of matrix eigenvalues.
        """
        self.species = species
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.soap: Any = None
        self.rbf = rbf
        self.sigma = sigma
        self.average = average
        self.periodic = periodic

    def _featurize(self, struct: PymatgenStructure) -> np.ndarray:
        """
        Calculate sine Coulomb matrix from pymatgen structure.
        Parameters
        ----------
        struct: pymatgen.Structure
        A periodic crystal composed of a lattice and a sequence of atomic
        sites with 3D coordinates and elements.
        Returns
        -------
        features: np.ndarray
        2D sine Coulomb matrix with shape (max_atoms, max_atoms),
        or 1D matrix eigenvalues with shape (max_atoms,).
        """
        if self.soap is None:
            try:
                from dscribe.descriptors import SOAP

                self.soap = SOAP(
                    periodic=self.periodic,
                    species=self.species,
                    rcut=self.rcut,
                    nmax=self.nmax,
                    lmax=self.lmax,
                    rbf=self.rbf,
                    sigma=self.sigma,
                    average=self.average,
                )
            except ModuleNotFoundError:
                raise ImportError("This class requires matminer to be installed.")
        adaptor = AseAtomsAdaptor()
        struct = adaptor.get_atoms(struct)
        features = self.soap.create(struct)

        features = np.asarray(features)

        return features
