import onnxruntime as rt
import numpy as np
import urllib.request, json

from PIL import Image
import requests

from pymatgen.core import Structure
class MLModel:

    """
    Class to hold all components used to train a ML model

    Attributes
    ----------
    elemnts: list
        list of elements in the models training set
    featurizer: quantumml.featurizers.Featurizer
        class used to transform poscar file to descriptor
    training_data_path: str
        string of the path to the training_data json file
    model: onnx.onnx_ml_pb2.ModelProto
        ONNX format trained model
    pipeline: str
        path of pipeline image
    full_pipeline: sklearn.pipeline.Pipeline
        full training pipeline of model
    learning_curve: str
        path to learning curve image
    """

    def __init__(
        self,
        elements,
        featurizer,
        training_data_path,
        model,
        pipeline,
        full_pipeline,
        learning_curve,
    ):
        self.elements = elements
        self.featurizer = featurizer
        self.training_data_path = training_data_path
        self.model = model
        self.pipeline = pipeline
        self.full_pipeline = full_pipeline
        self.learning_curve = learning_curve

    def predict(self, X):
        """
        Makes prediction of target property given a list of pymatgen.structures

        Parameters
        ----------
        X : list or np.ndarray
            list or array of pymatgen.structure
        Returns
        -------
        pred : np.ndarray
            array of target predictions
        Examples
        --------
        >>> with MLRester() as mlr:
        >>>     model = mlr.get_model(['Al','Ni'],doi='1.2.5555')
        >>> train_in, train_out, test_in, test_out = model.get_training_data(split=True)
        >>> model.predict(test_in[0:3])
        array([-0.03378353,  0.13828443, -0.30566216], dtype=float32)
        """
        x_features = self.featurizer.featurize(X)
        sess = rt.InferenceSession(self.model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred = sess.run([label_name], {input_name: x_features.astype(np.float32)})
        pred = pred[0].reshape(len(x_features))
        return pred

    def plot_pipeline(self):
        """
        display a dot graph of the pipeline used to predict the target value

        Returns
        -------
        pipeline : PIL.PngImagePlugin.PngImageFile
            pipeline for prediction
        Examples
        --------
        >>> with MLRester() as mlr:
        >>>     model = mlr.get_model(['Al','Ni'],doi='1.2.5555')        
        >>> model.plot_pipeline()
        """
        pipeline = Image.open(requests.get(self.pipeline, stream=True).raw)
        return pipeline

    def get_training_data(self, split = False):
        """returns the data used to train the model

        Parameters
        ----------
        split : bool
            If True return the train/test split dataset used to train model. If False return list of input structure and target properties
        Returns
        -------
        data_list: List
             If split==False, list of dictionary's with keys as 'structure' 'TARGET'
            'structure' will return the structure as a dict that can be reconstructed as a pymatgen.structure
        train_in, test_in : List
            If split==True, list of pymatgen.Structure used to train/test model
        train_out, test_out : List
            If split==True, list of target values used to train/test model
        Examples
        --------
        >>> with MLRester() as mlr:
        >>>     model = mlr.get_model(['Al','Ni'],doi='1.2.5555')
        >>> data = model.get_training_data()
        >>> input = []
        >>> target = []
        >>> for data_point in data[0:3]:
        >>>     input.append(Structure.from_dict(data_point['structure']))
        >>>     target.append(data_point['Formation_Energy'])
        >>> print(f'len(test_in)  = {len(test_in)} element_type = {str(type(test_in[0]))[8:-2]}')
        >>> print(f'len(test_out) = {len(test_out)} element_type = {str(type(test_out[0]))[8:-2]}')
        len(test_in)  = 3 element_type = pymatgen.core.structure.Structure
        len(test_out) = 3 element_type = float
        """
        with urllib.request.urlopen(self.training_data_path) as url:
            data = json.loads(url.read().decode())
        train = data["training"]
        test = data["testing"]
        if split:
            test_in = []
            test_out = []
            train_in = []
            train_out = []
            for ind in range(len(train)):
                if ind < len(test):
                    data_point = test[str(ind)]
                    test_in.append(Structure.from_dict(data_point['structure']))
                    test_out.append(data_point['Formation_Energy'])
                data_point = train[str(ind)]
                train_in.append(Structure.from_dict(data_point['structure']))
                train_out.append(data_point['Formation_Energy'])
            return train_in, train_out, test_in, test_out
        else:
            data_list = [data[k][key] for k in data.keys() for key in data[k].keys()]
            return data_list

    def get_featurizer(self):
        """
        Returns the feturizer used to transform crystal structure into descriptor

        Returns
        -------
        featurizer : quantumml.featurizer
            returns featurizer

        """
        featurizer = self.featurizer
        return featurizer

    def get_onnx_model(self):
        """
        Returns the trained model in an onnx format

        Returns
        -------
        model : onnx.onnx_ml_pb2.ModelProto
            ONNX model

        """
        model = self.model
        return model

    def get_full_pipeline(self):
        """
        Returns the full sklearn pipeline used to train model to allow users to modify the model

        Returns
        -------
        full_pipeline : sklearn.pipeline.Pipeline
            full pipeline used to create model

        """
        full_pipeline = self.full_pipeline
        return full_pipeline

    def plot_learning_curve(self):
        """
        display a dot graph of the learning curve produced in model training

        Returns
        -------
        learning_curve : PIL.PngImagePlugin.PngImageFile
            learning curve plot
        Examples
        --------
        >>> self.plot_learning_curve()
        """
        learning_curve = Image.open(requests.get(self.learning_curve, stream=True).raw)
        return learning_curve
