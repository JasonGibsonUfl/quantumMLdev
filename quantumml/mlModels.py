"""
Module for retrieving machine learning model
"""
import pickle
import urllib
from sklearn.svm import SVR
import numpy as np
import ast
import base64
from dscribe.descriptors import SOAP
from pymatgen.core import Structure


class MLModel:
    """
    Base class for pre-trained machine learning model
    todo: Alot needs to be implemented here.
    todo: Should the MLModel inherit from sklearn?
    todo: 1. mysql database needs to be set up for each mlmodel and the structure need to be determined
    todo: 2. I feel this should be a class but I am not sure how to implement this while using sklearn
    """

    def __init__(self, evaluation, target, DOI):
        self.evaluation = evaluation
        self.target = target
        self.DOI = DOI

    @staticmethod
    def read_numpy_field(value, dtype=np.float64):
        asnp = ast.literal_eval(value)
        return np.asarray(asnp, dtype=dtype, order="C")

    @staticmethod
    def read_svr_parameters(query_results):
        """
        method to read the parameters of query results into proper format
        Parameters
        ----------
        query_results : dict
            results from get_SVR method
        Returns
        -------
        model_params : dict
            list of all needed components of SVR in proper format
        """
        params = ast.literal_eval(query_results["parameters"])
        intercept = MLModel.read_numpy_field(query_results["intercept"])
        dual_coef = MLModel.read_numpy_field(query_results["dual_coef"])
        sparse = bool(query_results["sparse"])
        shape_fit = ast.literal_eval(query_results["shape_fit"])
        support = MLModel.read_numpy_field(query_results["support"], dtype=np.int32)
        support_vectors = MLModel.read_numpy_field(query_results["support_vectors"])
        n_support = MLModel.read_numpy_field(query_results["n_support"], dtype=np.int32)
        probA = MLModel.read_numpy_field("[]")
        probB = MLModel.read_numpy_field("[]")
        gamma = float(query_results["gamma"])

        model_params = {
            "params": params,
            "intercept": intercept,
            "dual_coef": dual_coef,
            "sparse": sparse,
            "shape_fit": shape_fit,
            "support": support,
            "support_vectors": support_vectors,
            "n_support": n_support,
            "probA": probA,
            "probB": probB,
            "gamma": gamma,
        }
        return model_params

    @staticmethod
    def rebuild_SVR(query_results):
        """
        rebuild sklearn SVR model from query results
        Parameters
        ----------
        query_results : dict
            results from get_SVR method
        Returns
        -------
        model : sklearn.SVR
            returns the trained svr model
        """
        model = SVR()
        model_params = MLModel.read_svr_parameters(query_results)
        model.set_params(**model_params["params"])
        model._intercept_ = model_params["intercept"]
        model._dual_coef_ = model_params["dual_coef"]
        model._sparse = model_params["sparse"]
        model.shape_fit_ = model_params["shape_fit"]
        model.support_ = model_params["support"]
        model.support_vectors_ = model_params["support_vectors"]
        model._n_support = model_params["n_support"]
        model.probA_ = model_params["probA"]
        model.probB_ = model_params["probB"]
        model._gamma = model_params["gamma"]
        print("NEWNEW")
        # model = pickle.loads(base64.b64decode(query_results['pickle_str']))
        return model

    @staticmethod
    def rebuild_transformer(query_results):
        """
        rebuild data transformer from query results
        Parameters
        ----------
        query_results : dict
            results from get_SVR method
        Returns
        -------
        scaler : sklearn.StandardScaler
            returns the fitted sklearn standardscaler model
        """
        scaler = pickle.loads(base64.b64decode(query_results["pickle_str_transformer"]))
        return scaler

    @staticmethod
    def rebuild_structure_list(query_results):
        """
        rebuild list of pymatgen structures used for training
        Parameters
        ----------
        query_results : dict
            results from get_SVR method
        Returns
        -------
        structure_list : List
            list of pymatgen structures used to train the model
        """
        structure_list = []
        for i in range(100000):
            try:
                structure_list.append(
                    Structure.from_dict(ast.literal_eval(query_results[i]["structure"]))
                )
            except:
                return structure_list

    @staticmethod
    def rebuild_training_targets(query_results, target):
        """
        rebuild list of target properties for training
        Parameters
        ----------
        query_results : dict
            results from get_SVR method
        Returns
        -------
        target_list : List
            list of target values used to train the model
        """
        target_list = []
        print("here")
        for i in range(100000):
            try:
                print(f"i = {i}\t target = {target}")
                target_list.append(query_results[i][target])
            except:
                print(f"i = {i}\t target = {target}")
                return target_list

    @staticmethod
    def rebuild_descriptor(model_params):
        """
        method to reconsruct the descriptor object from query_results
        Parameters
        ----------
        model_params : dict
            results from get_SCR method
        Returns
        -------
        descriptor : onbject
            object that can compute the descriptor given a pymatgen structure
        """
        descriptor_params = ast.literal_eval(model_params["descriptor_parameters"])
        if descriptor_params["name"] == "soap":
            descriptor = SOAP(
                species=descriptor_params["species"],
                periodic=descriptor_params["periodic"],
                rcut=descriptor_params["rcut"],
                nmax=descriptor_params["nmax"],
                lmax=descriptor_params["lmax"],
                rbf=descriptor_params["rbf"],
                sigma=descriptor_params["sigma"],
                average=descriptor_params["average"],
            )
        return descriptor

    @staticmethod
    def get_ml_model(system):
        """
        Method to allow easy access to all pre-trainned kernal ridge regresion machine learning models of GASP runs
        Parameters
        ----------
        system : str
            The name of the desired material system
        Returns
        -------
        model : sklearn.svm._classes.SVR
            The pretrained model

        todo: This will change from retrieving a pickle string to rebuilding from json responce
        """
        urlm = "http://materialsweb.org/static/models/" + system + ".sav"
        model = pickle.load(urllib.request.urlopen(urlm))
