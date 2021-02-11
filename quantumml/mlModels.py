"""
Module for retrieving machine learning model
"""
import pickle
import urllib

class MLModel:
    """
    Base class for pre-trained machine learning model
    todo: Alot needs to be implemented here.
    todo: 1. mysql database needs to be set up for each mlmodel and the structure need to be determined
    todo: 2. I feel this should be a class but I am not sure how to implement this while using sklearn
    """
    
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
        urlm ='http://materialsweb.org/static/models/'+system+'.sav'
        model = pickle.load(urllib.request.urlopen(urlm))
        return model
    