from quantumml.featurizers import MaterialStructureFeaturizer
from typing import Any
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.base import TransformerMixin
from dscribe.descriptors import SOAP

PymatgenStructure = Any


class SoapTransformer(MaterialStructureFeaturizer,TransformerMixin):
    """
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
        species: list =[],
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
        TransformerMixin.__init__(self)
        self.species = species
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.soap: Any = None
        self.rbf = rbf
        self.sigma = sigma
        self.average = average
        self.periodic = periodic

    def __str__(self):
        return 'Soap'

    def _featurize(self, struct: PymatgenStructure) -> np.ndarray:
        #
        # if self.soap is None:
        #     try:
        #         from dscribe.descriptors import SOAP
        #     except ModuleNotFoundError:
        #         raise ImportError("This class requires dscribe to be installed.")
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

        soap = SOAP(
            periodic=self.periodic,
            species=self.species,
            rcut=self.rcut,
            nmax=self.nmax,
            lmax=self.lmax,
            rbf=self.rbf,
            sigma=self.sigma,
            average=self.average,
        )

        adaptor = AseAtomsAdaptor()
        struct = adaptor.get_atoms(struct)
        features = soap.create(struct)

        features = np.asarray(features)
        return features

    def fit(self, x, y=None):
        self.adaptor = AseAtomsAdaptor()

        self.soap = SOAP(species=self.species,
                         periodic=self.periodic,
                         rcut=self.rcut,
                         nmax=self.nmax,
                         lmax=self.lmax,
                         rbf=self.rbf,
                         sigma=self.sigma,
                         average=self.average)
        flattened_entry_list = [self.adaptor.get_atoms(struct) for struct in x]
        self.soap_raw = self.soap.create(flattened_entry_list)
        return self

    def transform(self, x, y=None):

        flattened_entry_list = [self.adaptor.get_atoms(struct) for struct in x]
        self.soap_raw = self.soap.create(flattened_entry_list)
        return self.soap_raw

    def set_params(
        self,
        species: list =[],
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
