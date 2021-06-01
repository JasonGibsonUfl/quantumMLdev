"""
rest.py
quantumML is 

Handles the primary functions
"""
from urllib.request import urlopen
import json
import urllib
import os
import io
import math
from pymatgen.core.structure import Structure
from ase.io import vasp
from dscribe.descriptors import SOAP
from pymatgen.io.vasp import Xdatcar, Oszicar
from sklearn.cluster import KMeans
import numpy as np

# from .mlModels import MLModel
import onnx
from quantumml.featurizers import SoapTransformer
import base64
import ast
from quantumml.models.sklearnModel import MLModel
import pickle as pkl

featurizers = {"SOAP": SoapTransformer()}


class MLRester(object):
    """
    A class to conveniently interface with the MaterialsWeb REST
    interface. The recommended way to use MLRester is with the "with" context
    manager to ensure that sessions are properly closed after
    usage::
        with MLRester() as mlr:
            do_something
    MLRester uses the "requests" package, which provides for HTTP connection
    pooling. All connections are made via https for security.
    """

    results = {}

    def __init__(self, api_key=None, endpoint="http://127.0.0.1:8000/rest/"):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = ""
        self.preamble = endpoint
        import requests

        self.session = requests.Session()
        self.session.headers = {"x-api-key": self.api_key}

    def __str__(self):
        return "%s" % self.results

    def __enter__(self):
        """
        Support for "with" context.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Support for "with" context.
        """
        self.session.close()

    def _make_request(self, sub_url, payload=None, method="GET"):
        url = self.preamble + sub_url  # + "/" + self.api_key
        # print(url)
        x = urlopen(url)

        response = self.session.get(url, verify=True)
        data = json.loads(response.text)
        return data

    def get_model(self, target_property="", elements=[""], doi=""):
        """
        Parameters
        ----------
        target_property : str
            Property for model to predict
        elements : list
            list of elements in string form e.g. ['Cd', 'Te']
        doi : str
            doi of model
        Returns
        -------
        model : quantumml.sklearnModel.MLModel
            returns model containing all training information
        """
        suburl = "MachineLearningModel/?"

        alpha = ["A", "B", "C"]
        for i, el in enumerate(elements):
            suburl += f"element_{alpha[i]}={el}&"
        suburl += f"DOI={doi}"
        self.results = self._make_request(suburl)[0]
        training_data_url = self.results["training_data_url"]
        onnx_binary = base64.b64decode(self.results["model_onnx_binary"])
        model_onnx = onnx.load_from_string(onnx_binary)
        pipeline_binary = base64.b64decode(self.results["pipeline_binary"])
        pipeline_model = pkl.loads(pipeline_binary)
        featurizer = featurizers[self.results["descriptor_name"]]
        param = self.results["featurizer_params"]
        params = ast.literal_eval(param)
        featurizer.set_params(**params)
        pipeline = self.results["pipeline"]
        learning_curve = self.results["learning_curve"]
        model = MLModel(
            elements=elements,
            featurizer=featurizer,
            training_data_path=training_data_url,
            model=model_onnx,
            pipeline=pipeline,
            full_pipeline=pipeline_model,
            learning_curve=learning_curve,
        )
        return model

    def get_uf3(self, elements):
        """
        gets uf3 pair potentials
        """
        suburl = "UFCC/?element1=" + elements[0] + "&element2=" + elements[1]
        self.results = self._make_request(suburl)
        return self.results


class MWRester(object):
    """"""

    results = {}

    def __init__(
        self, api_key=None, endpoint="http://materialsweb.org/rest/calculation/?"
    ):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = ""
        self.preamble = endpoint
        import requests

        self.session = requests.Session()
        self.session.headers = {"x-api-key": self.api_key}

    def __str__(self):
        return "%s" % self.results

    def __enter__(self):
        """
        Support for "with" context.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Support for "with" context.
        """
        self.session.close()

    def _make_request(self, sub_url, payload=None, method="GET", mp_decode=True):
        url = self.preamble + sub_url + "/" + self.api_key
        x = urlopen(url)

        response = self.session.get(url, verify=True)
        data = json.loads(response.text)
        return data

    def get_calculation(
        self,
        band_gap_range=None,
        formation_energy_range=None,
        elements=[],
        space_group_number=None,
        dimension=None,
        crystal_system=None,
        name=None,
    ):
        """
        todo: add support for elements and name
        Method to that queries materialsweb database and returns a list of dictionaries of calculations that
        fit the querries parameters. Additionally
        Parameters
        ----------
        band_gap_range : list
        List of band gap range e.g. [min, max]
        formation_energy_range : list
            List of formation energy range e.g. [min, max]
        elements : list
        List of str of elements
        space_group_number : int
            space group number
        dimension : int
            dimension as int e.g. 1 2 3
        crystal_system : str
            name of crystal system
        name : str
            name of material e.g. MoS2
        Returns
        -------
        results : list
            List of results matching query parameters
        """

        suburl = ""
        if band_gap_range != None:
            suburl += (
                "band_gap_min="
                + str(band_gap_range[0])
                + "&band_gap_max="
                + str(band_gap_range[-1])
                + "&"
            )
        if formation_energy_range != None:
            suburl += (
                "formation_energy_min="
                + str(formation_energy_range[0])
                + "&formation_ener_max="
                + str(formation_energy_range[-1])
                + "&"
            )

        if space_group_number != None:
            suburl += "spacegroup_number=" + str(space_group_number) + "&"

        if dimension != None:
            suburl += "dimension=" + str(dimension) + "&"

        if crystal_system != None:
            "lattice_system=" + str(crystal_system) + "&"
        self.results = self._make_request(suburl)["results"]
        return self.results

    def as_pymatgen_struc(self):
        """
        Method that converts results to list of pymatgen strucutures

        Returns
        -------
        struc : list
            List of pymatgen structures
        """
        struc = []
        for c in self.results:
            urlp = "http://" + c["path"][9:21] + ".org/" + c["path"][22:] + "/POSCAR"
            file = urllib.request.urlopen(urlp)
            poscar = ""
            for line in file:
                poscar += line.decode("utf-8")

            s = Structure.from_str(poscar, fmt="poscar")
            struc.append(s)

        return struc

    def write(self, index=0):
        """
        Writes INCAR, KPOINTS, POSCAR of entry to current directory

        Parameters
        ----------
        index : int
            Index of entry to write files for
        todo: add unit test
        """
        self.write_poscar(index=index)
        self.write_incar(index=index)
        self.write_kpoints(index=index)

    def write_all(self=0):
        """
        Creates a directory named by composition for every entry in results. Then, Writes INCAR, KPOINTS,
        POSCAR of entry to respective directory
        todo: add unit test
        """
        for index in range(0, len(self.results)):
            dir_name = (
                self.results[index]["composition"].split("/")[-2].replace("%", "")
            )
            os.mkdir(dir_name)
            os.chdir(dir_name)
            self.write_poscar(index=index)
            self.write_incar(index=index)
            self.write_kpoints(index=index)
            os.chdir("..")

    def write_poscar(self, index=0):
        """
        Writes POSCAR of entry to current directory

        Parameters
        ----------
        index : int
            Index of entry to write POSCAR for
        todo: add unit test
        """
        urlp = "http://materialsweb.org/" + self.results[index]["path"][22:] + "/POSCAR"
        file = urllib.request.urlopen(urlp)
        with open("POSCAR", "a") as poscar:
            for line in file:
                decoded_line = line.decode("utf-8")
                poscar.write(decoded_line)

    def write_kpoints(self, index=0):
        """
        Writes KPOINTS of entry to current directory

        Parameters
        ----------
        index : int
            Index of entry to write KPOINT for
        todo: add unit test
        """
        urlp = (
            "http://materialsweb.org/" + self.results[index]["path"][22:] + "/KPOINTS"
        )
        file = urllib.request.urlopen(urlp)
        with open("KPOINTS", "a") as poscar:
            for line in file:
                decoded_line = line.decode("utf-8")
                poscar.write(decoded_line)

    def write_incar(self, index=0):
        """
        Writes INCAR of entry to current directory

        Parameters
        ----------
        index : int
            Index of entry to write INCAR for
        todo: add unit test
        """
        urlp = "http://materialsweb.org/" + self.results[index]["path"][22:] + "/INCAR"
        file = urllib.request.urlopen(urlp)
        with open("INCAR", "a") as poscar:
            for line in file:
                decoded_line = line.decode("utf-8")
                poscar.write(decoded_line)


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(MWRester())
