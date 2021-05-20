from quantumml.featurizers import MaterialStructureFeaturizer
from typing import Any
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
import pandas as pd
import os.path
import pickle
import scipy
from pymatgen.io.vasp import Poscar
from pymatgen.core.periodic_table import Element
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import RandomizedSearchCV as CV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .utils import *
from typing import List

PymatgenStructure = Any


class Global_RDF(MaterialStructureFeaturizer):
    """
    Calculate the global radial distribution function feature vector for crystals.

    References
    ----------
    .. [1] ref

    Examples
    --------
    """

    def __init__(
        self,
        elements: List,
        rcut: float = 10.1,
        stepSize: float = 0.1,
        sigma: float = 0.2,
    ):
        """
        Parameters : list
            list of elements symbols
        """
        self.elements = elements
        self.rdf_tup = calc_rdf_tup(elements)
        self.rcut = rcut
        self.stepSize = stepSize
        self.sigma = sigma
        self.binRad = np.arange(0.1, self.rcut, self.stepSize)
        self.numBins = len(self.binRad)
        self.numPairs = len(self.rdf_tup)


    def _get_ind(self, dist):
        inds = int(dist / self.stepSize)
        lowerInd = (inds - 5) if (inds - 5) > 0 else 0
        upperInd = (inds + 5) if (inds + 5) < self.numBins else self.numBins - 1
        return range(lowerInd, upperInd)

    def _calculate_rdf(self, ind, dist):
        evalRad = self.binRad[ind]
        exp_Arg = .5 * ((np.subtract(evalRad, dist) / (self.sigma)) ** 2)  # Calculate RDF value for each bin
        rad2 = np.multiply(evalRad, evalRad)  # Add a 1/r^2 normalization term, check paper for descripton
        return np.divide(np.exp(-exp_Arg), rad2)

    def _apply_gaussian_broadening(self, NeighborDistList):
        # Apply gaussian broadening to the neigbor distances,
        # so the effect of having a neighbor at distance x is spread out over few bins around x
        hist = np.zeros(self.numBins)
        for aND in NeighborDistList:
            for dist in aND:
                ind = self._get_ind(dist)
                hist[ind] += self._calculate_rdf(ind, dist)
        return hist

    def _featurize(self, structure: PymatgenStructure) -> np.ndarray:

        vec = np.zeros((self.numPairs, self.numBins))  # Create a vector of zeros (dimension: numPairs*numBins)
        neighbors = structure.get_all_neighbors(self.rcut)
        sites = structure.sites
        for index, tup in enumerate(self.rdf_tup):
            specs_ab = get_elements(tup)
            indices_ab = get_indices(sites, specs_ab)
            if contains_ab(indices_ab):
                vec[index] = np.zeros(self.numBins)
                continue

            alphaNeighborDistList = get_neighbor_distribution_list(neighbors, indices_ab[0], specs_ab[-1])

            # Apply gaussian broadening to the neigbor distances,
            # so the effect of having a neighbor at distance x is spread out over few bins around x
            hist = self._apply_gaussian_broadening(alphaNeighborDistList)
            tempHist = hist / len(
                indices_ab[0])  # Divide by number of AlphaSpec atoms in the unit cell to give the final partial RDF
            vec[index] = tempHist

        vec = np.row_stack((vec[0], vec[1], vec[2]))  # Combine all vectors to get RDFMatrix
        return vec


class Global_ADF(MaterialStructureFeaturizer):
    """
    Calculate the global radial distribution function feature vector for crystals.

    References
    ----------
    .. [1] ref

    Examples
    --------
    """

    def __init__(
        self,
        elements: List,
        rcut: float = 5,
        stepSize: float = 0.1,
        sigma: float = 0.2,
    ):
        """
        Parameters : list
            list of elements symbols
        """
        self.elements = elements
        self.rdf_tup = calc_rdf_tup(elements)
        self.rcut = rcut
        self.stepSize = stepSize
        self.sigma = sigma
        self.binRad = np.arange(0.1, self.rcut, self.stepSize)
        self.numBins = len(self.binRad)
        self.numPairs = len(self.rdf_tup)


    def _get_ind(self, ang):
        inds = int(ang / self.stepSize)
        lowerInd = (inds +8) if (inds +8) > 0 else 0
        upperInd = (inds + 12) if (inds + 12) <= self.numBins else self.numBins
        return range(lowerInd, upperInd)

    def _calculate_adf(self, ind, dist):
        evalRad = self.binRad[ind]
        exp_Arg = .5 * ((np.subtract(evalRad, dist) / (self.sigma)) ** 2)  # Calculate RDF value for each bin
        return np.exp(-exp_Arg)*f

    def _apply_gaussian_broadening(self, NeighborDistList):
        # Apply gaussian broadening to the neigbor distances,
        # so the effect of having a neighbor at distance x is spread out over few bins around x
        hist = np.zeros(self.numBins)
        for aND in NeighborDistList:
            for dist in aND:
                ind = self._get_ind(dist)
                hist[ind] += self._calculate_rdf(ind, dist)
        return hist

    def _featurize(self, structure: PymatgenStructure) -> np.ndarray:

        vec = np.zeros((self.numPairs, self.numBins))  # Create a vector of zeros (dimension: numPairs*numBins)
        neighbors = structure.get_all_neighbors(self.rcut)
        sites = structure.sites
        for index, tup in enumerate(self.rdf_tup):
            specs_ab = get_elements(tup)
            indices_ab = get_indices_ab(sites, specs_ab)
            if contains_ab(indices_ab):
                vec[index] = np.zeros(self.numBins)
                continue
            alphaNeighbors = [neighbors[i] for i in indices_ab[0]]

            alphaNeighborDistList = get_neighbor_distribution_list(neighbors, indices_ab[0], specs_ab[-1])

            # Apply gaussian broadening to the neigbor distances,
            # so the effect of having a neighbor at distance x is spread out over few bins around x
            hist = self._apply_gaussian_broadening(alphaNeighborDistList)
            tempHist = hist / len(
                indices_ab[0])  # Divide by number of AlphaSpec atoms in the unit cell to give the final partial RDF
            vec[index] = tempHist

        vec = np.row_stack((vec[0], vec[1], vec[2]))  # Combine all vectors to get RDFMatrix
        return vec
