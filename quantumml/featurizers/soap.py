"""
soap.py
A short description of the project.

"""


from quantumml.featurizers import MaterialStructureFeaturizer
from typing import Any
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.base import TransformerMixin
from dscribe.descriptors import SOAP

PymatgenStructure = Any


class SoapTransformer(MaterialStructureFeaturizer, TransformerMixin):
    """ SoapTransformer class 
    
    Class for generating a partial power spectrum from Smooth Overlap of
    Atomic Orbitals (SOAP). This implementation uses real (tesseral) spherical
    harmonics as the angular basis set and provides two orthonormalized
    alternatives for the radial basis functions: spherical primitive gaussian
    type orbitals ("gto") or the polynomial basis set ("polynomial").

    For reference, see:

    "On representing chemical environments, Albert P. Bartók, Risi Kondor, and
    Gábor Csányi, Phys. Rev. B 87, 184115, (2013),
    https://doi.org/10.1103/PhysRevB.87.184115

    "Comparing molecules and solids across structural and alchemical space",
    Sandip De, Albert P. Bartók, Gábor Csányi and Michele Ceriotti, Phys.
    Chem. Chem. Phys. 18, 13754 (2016), https://doi.org/10.1039/c6cp00415f

    "Machine learning hydrogen adsorption on nanoclusters through structural
    descriptors", Marc O. J. Jäger, Eiaki V. Morooka, Filippo Federici Canova,
    Lauri Himanen & Adam S. Foster, npj Comput. Mater., 4, 37 (2018),
    https://doi.org/10.1038/s41524-018-0096-5

    Examples
    --------
    >>> import pymatgen
    >>> lattice = pymatgen.Lattice.cubic(4.2)
    >>> structure = pymatgen.Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    >>> featurizer = SoapTransformer(species=["Cs",'Cl'])
    >>> features = featurizer.featurize([structure])
    """

    def __init__(
        self,
        species: list = [],
        rcut: int = 6,
        nmax: int = 6,
        lmax: int = 8,
        rbf: str = "gto",
        sigma: float = 0.125,
        average: str = "inner",
        periodic: bool = True,
        ):
        """
        Initiallize class
        
        Parameters
        ----------
        rcut : float
            A cutoff for local region in angstroms. Should be bigger than 1 angstrom
        nmax : int
            The number of radial basis functions.
        lmax : int
            The maximum degree of spherical harmonics.
        species : List
            list of elements
        sigma : float
            The standard deviation of the gaussians used to expand the atomic density.
        rbf : str
            The radial basis functions to use. The available options are:            
                * "gto": Spherical gaussian type orbitals defined as :math:`g_{nl}(r) = \sum_{n'=1}^{n_\mathrm{max}}\,\\beta_{nn'l} r^l e^{-\\alpha_{n'l}r^2}`                
                * "polynomial": Polynomial basis defined as :math:`g_{n}(r) = \sum_{n'=1}^{n_\mathrm{max}}\,\\beta_{nn'} (r-r_\mathrm{cut})^{n'+2}`                
        periodic : bool
            Set to true if you want the descriptor output to respect the periodicity of the atomic systems (see the
            pbc-parameter in the constructor of ase.Atoms).
        average : str
            The averaging mode over the centers of interest.
            Valid options are:
                * "off": No averaging.
                * "inner": Averaging over sites before summing up the magnetic quantum numbers: :math:`p_{nn'l}^{Z_1,Z_2} \sim \sum_m (\\frac{1}{n} \sum_i c_{nlm}^{i, Z_1})^{*} (\\frac{1}{n} \sum_i c_{n'lm}^{i, Z_2})`
                * "outer": Averaging over the power spectrum of different sites: :math:`p_{nn'l}^{Z_1,Z_2} \sim \\frac{1}{n} \sum_i \sum_m (c_{nlm}^{i, Z_1})^{*} (c_{n'lm}^{i, Z_2})`        
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
        return "Soap"

    def _featurize(self, struct: PymatgenStructure) -> np.ndarray:
        """Calculate SOAP descriptor from pymatgen structure.
        
        Parameters
        ----------
        struct: pymatgen.Structure
            A periodic crystal composed of a lattice and a sequence of atomic
            sites with 3D coordinates and elements.
            
        Returns
        -------
        features: np.ndarray
            soap descriptor
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

        self.soap = SOAP(
            species=self.species,
            periodic=self.periodic,
            rcut=self.rcut,
            nmax=self.nmax,
            lmax=self.lmax,
            rbf=self.rbf,
            sigma=self.sigma,
            average=self.average,
        )
        #flattened_entry_list = [self.adaptor.get_atoms(struct) for struct in x]
        #self.soap_raw = self.soap.create(flattened_entry_list)
        return self

    def transform(self, x, y=None):

        flattened_entry_list = [self.adaptor.get_atoms(struct) for struct in x]
        self.soap_raw = self.soap.create(flattened_entry_list)
        return self.soap_raw

    def set_params(
        self,
        species: list = [],
        rcut: int = 6,
        nmax: int = 6,
        lmax: int = 8,
        rbf: str = "gto",
        sigma: float = 0.125,
        average: str = "inner",
        periodic: bool = True,
    ):
        """Sets the featurizer parameters
        
        Parameters
        ----------
        rcut : float
            A cutoff for local region in angstroms. Should be bigger than 1 angstrom.
        nmax : int
            The number of radial basis functions.
        lmax : int
            The maximum degree of spherical harmonics.
        species : List
            list of elements
        sigma : float
            The standard deviation of the gaussians used to expand the atomic density.
        rbf : str
            The radial basis functions to use.
        periodic : bool
            Set to true if you want the descriptor output to respect the periodicity of the atomic systems
        average : str
            The averaging mode over the centers of interest.
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
