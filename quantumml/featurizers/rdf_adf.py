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

        print(f'numPairs {numPairs}, numBins {numBins}')
        vec = np.zeros((self.numPairs, self.numBins))  # Create a vector of zeros (dimension: numPairs*numBins)
        neighbors = structure.get_all_neighbors(self.rcut)
        sites = structure.sites
        for index, pair in enumerate(self.rdf_tup):
            specs_ab = get_specs_ab(pair)
            indices_ab = get_indices_ab(sites, specs_ab)
            if contains_ab(indices_ab):
                vec[index] = np.zeros(self.numBins)
                continue
            alphaNeighbors = get_neighbor_atoms(neighbors, indices_ab[0])

            alphaNeighborDistList = get_neighbor_distribution_list(alphaNeighbors, specs_ab[-1])

            # Apply gaussian broadening to the neigbor distances,
            # so the effect of having a neighbor at distance x is spread out over few bins around x
            hist = self._apply_gaussian_broadening(alphaNeighborDistList)
            tempHist = hist / len(
                indices_ab[0])  # Divide by number of AlphaSpec atoms in the unit cell to give the final partial RDF
            vec[index] = tempHist

        vec = np.row_stack((vec[0], vec[1], vec[2]))  # Combine all vectors to get RDFMatrix
        return vec


class ADF(MaterialStructureFeaturizer):
    """

    """

    def __init__(
        self,
        species: list,
        ADF_Tup: list,
        rcut: float = 5,
        stepSize: float = 0.1,
        sigma: float = 0.2,
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
        self.ADF_Tup = ADF_Tup
        self.stepSize = stepSize
        self.sigma = sigma

    def _featurize(self, struct: PymatgenStructure) -> np.ndarray:
        """
            Calculates the ADF for every structure.

            Args:
                struct: input structure.

                ADF_Tup: list of all element triplets for which the ADF is calculated.

                cutOffRad: max. distance up to which atom-atom intereactions are considered.

                sigma: width of the Gaussian, used for broadening

                stepSize: bin width, binning transforms the ADF into a discrete representation.

        """

        ADF_Tup = self.ADF_Tup
        rcut = self.rcut
        sigma = self.sigma
        stepSize = self.stepSize

        binRad = np.arange(-1, 1, stepSize)  # Make bins based on stepSize
        numBins = len(binRad)
        numTriplets = len(ADF_Tup)
        vec = np.zeros(
            (numTriplets, numBins)
        )  # Create a vector of zeros (dimension: numTriplets*numBins)

        # Get all neighboring atoms within cutOffRad for alphaSpec, betaSpec, and gammaSpec
        # alphaSpec, betaSpec, and gammSpec are the three elements from ADF_Tup
        for index, triplet in enumerate(ADF_Tup):
            alphaSpec = Element(triplet[0])
            betaSpec = Element(triplet[1])
            gammaSpec = Element(triplet[2])
            hist = np.zeros(numBins)
            neighbors = struct.get_all_neighbors(cutOffRad)

            sites = struct.sites  # All sites in the structue
            indicesA = [
                j[0] for j in enumerate(sites) if j[1].specie == alphaSpec
            ]  # Get all alphaSpec sites in the structure
            numAlphaSites = len(indicesA)
            indicesB = [
                j[0] for j in enumerate(sites) if j[1].specie == betaSpec
            ]  # Get all betaSpec sites in the structure
            numBetaSites = len(indicesB)
            indicesC = [
                j[0] for j in enumerate(sites) if j[1].specie == gammaSpec
            ]  # Get all gammaSpec sites in the structure
            numGammaSites = len(indicesC)

            # If no alphaSpec or betaSpec or gammsSpec atoms, RDF vector is zero
            if numAlphaSites == 0 or numBetaSites == 0 or numGammaSites == 0:
                vec[index] = hist
                continue

            betaNeighbors = [
                neighbors[i] for i in indicesB
            ]  # Neighbors of betaSpec only

            alphaNeighborList = []
            for bN in betaNeighbors:
                tempalphaNeighborList = [
                    neighbor for neighbor in bN if neighbor[0].specie == alphaSpec
                ]  # Neighbors of betaSpec that are alphaSpec
                alphaNeighborList.append(
                    tempalphaNeighborList
                )  # Add all such neighbors to a list

            gammaNeighborList = []
            for bN in betaNeighbors:
                tempgammaNeighborList = [
                    neighbor for neighbor in bN if neighbor[0].specie == gammaSpec
                ]  # Neighbors of betaSpec that are gammaSpec
                gammaNeighborList.append(
                    tempgammaNeighborList
                )  # Add all such neighbors to a list

            # Calculate cosines for every angle ABC using side lengths AB, BC, AC
            cosines = []
            f_AB = []
            f_BC = []
            for B_i, aN in enumerate(alphaNeighborList):
                for i in range(len(aN)):
                    for j in range(len(gammaNeighborList[B_i])):
                        AB = aN[i][1]
                        BC = gammaNeighborList[B_i][j][1]
                        AC = np.linalg.norm(
                            aN[i][0].coords - gammaNeighborList[B_i][j][0].coords
                        )
                        if AC != 0:
                            cos_angle = np.divide(
                                ((BC * BC) + (AB * AB) - (AC * AC)), 2 * BC * AB
                            )
                        else:
                            continue
                        # Use a logistic cutoff that decays sharply, check paper for details [d_k=3, k=2.5]
                        AB_transform = 2.5 * (3 - AB)
                        f_AB.append(np.exp(AB_transform) / (np.exp(AB_transform) + 1))
                        BC_transform = 2.5 * (3 - BC)
                        f_BC.append(np.exp(BC_transform) / (np.exp(BC_transform) + 1))
                        cosines.append(cos_angle)

            # Apply gaussian broadening to the neigbor distances,
            # so the effect of having a neighbor at distance x is spread out over few bins around x
            for r, ang in enumerate(cosines):
                inds = ang / stepSize
                inds = int(inds)
                lowerInd = inds - 2 + 10
                if lowerInd < 0:
                    while lowerInd < 0:
                        lowerInd = lowerInd + 1
                upperInd = inds + 2 + 10
                if upperInd > numBins:
                    while upperInd > numBins:
                        upperInd = upperInd - 1
                ind = range(lowerInd, upperInd)
                evalRad = binRad[ind]
                exp_Arg = 0.5 * (
                    (np.subtract(evalRad, ang) / (sigma)) ** 2
                )  # Calculate ADF value for each bin
                hist[ind] += np.exp(-exp_Arg)
                hist[ind] += np.exp(-exp_Arg) * f_AB[r] * f_BC[r]

            vec[index] = hist

        vec = np.row_stack(
            (vec[0], vec[1], vec[2], vec[3], vec[4], vec[5])
        )  # Combine all vectors to get ADFMatrix
        return vec
