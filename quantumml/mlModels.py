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
        '''
        model = SVR()
        model.fit(np.array([[1, 1], [1, 2]]), np.array([1, 1]))
        params = ast.literal_eval(model_params['parameters'])
        model.set_params(**params)
        model._intercept_ = np.array(ast.literal_eval(model_params['intercept']))
        model._dual_coef_ = np.array(ast.literal_eval(model_params['dual_coef']))
        model._sparse = bool(model_params['sparse'])
        model.shape_fit_ = ast.literal_eval(model_params['shape_fit'])
        model.support_ = np.array(ast.literal_eval(model_params['support']))
        model.support_vectors_ = np.array(ast.literal_eval(model_params['support_vectors']))
        model._n_support =np.array(ast.literal_eval(model_params['n_support']))
        model.probA_ = np.array([])
        model.probB_ = np.array([])
        model._gamma = float(model_params['gamma'])
        '''
        return model

    @staticmethod
    def rebuild_descriptor(model_params):
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
        return model
