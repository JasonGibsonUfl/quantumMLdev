from uf3.util import json_io
import os
from uf3.util import user_config
from quantumml.rest import MLRester

from uf3.data.composition import ChemicalSystem
from uf3.regression.least_squares import WeightedLinearModel
from uf3.representation.bspline import BSplineConfig
from uf3.forcefield.calculator import UFCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from uf3.data import geometry
import ast
import numpy as np


class UFCC(UFCalculator):
    def _structure_to_ase_atoms(self, structure):
        adaptor = AseAtomsAdaptor()
        return adaptor.get_atoms(structure)

    def get_potential_energy(self, structure, force_consistent=None):
        """Evaluate the total energy of a configuration.""" ""
        atoms = self._structure_to_ase_atoms(structure)
        energy = super().get_potential_energy(atoms)
        return energy

    def get_forces(self, atoms=None):
        atoms = self._structure_to_ase_atoms(structure)
        forces = super().get_forces(atoms)
        return forces

    @staticmethod
    def rebuild(element_list):

        # try:
        with MLRester() as mlr:
            query_results = (mlr.get_uf3(element_list))[-1]
        # except:
        # print('model not found')

        def str_to_tuple(dictionary):
            new_dict = {}
            for key in dictionary:
                if len(key) > 2:
                    new_key = tuple(
                        key.strip(")")
                        .strip("(")
                        .replace(",", "")
                        .replace("'", "")
                        .split()
                    )
                    new_dict[new_key] = np.array(dictionary[key])
                else:
                    new_dict[key] = dictionary[key]
            return new_dict

        degree = query_results["degree"]
        model_data = ast.literal_eval(query_results["model_data"])
        knots_map = str_to_tuple(model_data["knots"])
        coefficients = str_to_tuple(model_data["coefficients"])

        chemical_system = ChemicalSystem(element_list=element_list, degree=degree)
        bspline_config = BSplineConfig(
            chemical_system, knots_map=knots_map, knot_spacing="custom"
        )

        model = WeightedLinearModel(bspline_config)
        model.load(coefficients)

        return model
