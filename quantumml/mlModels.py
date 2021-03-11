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
from pymatgen import Structure
class MLModel:
    """
    Base class for pre-trained machine learning model
    todo: Alot needs to be implemented here.
    todo: Should the MLModel inherit from sklearn?
    todo: 1. mysql database needs to be set up for each mlmodel and the structure need to be determined
    todo: 2. I feel this should be a class but I am not sure how to implement this while using sklearn
    """
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
        model = pickle.loads(base64.b64decode(query_results['pickle_str'])) 
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
        scaler = pickle.loads(base64.b64decode(query_results['pickle_str_transformer']))
        return scaler

    @staticmethod
    def rebuild_structure_list(query_results):
        structure_list = []
        for i in range(100000):
            try:
                structure_list.append(Structure.from_dict(ast.literal_eval(query_results[i]['structure'])))
            except:
                return structure_list


    @staticmethod
    def rebuild_training_targets(query_results, target):
        def rebuild_structure_list(query_results):
            target_list = []
            for i in range(100000):
                try:
                    target_list.append(Structure.from_dict(ast.literal_eval(query_results[i][target])))
                except:
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
        descriptor_params = ast.literal_eval(model_params['descriptor_parameters'])
        if descriptor_params['name'] == 'soap':
            descriptor = SOAP(
                species = descriptor_params['species'],
                periodic = descriptor_params['periodic'],
                rcut = descriptor_params['rcut'],
                nmax = descriptor_params['nmax'],
                lmax = descriptor_params['lmax'],
                rbf = descriptor_params['rbf'],
                sigma = descriptor_params['sigma'],
                average = descriptor_params['average']
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
