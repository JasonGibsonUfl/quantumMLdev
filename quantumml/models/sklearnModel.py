
import onnxruntime as rt
import numpy as np
import urllib.request, json

from PIL import Image
import requests
class MLModel():

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

    def __init__(self, elements, featurizer, training_data_path, model,pipeline, full_pipeline, learning_curve):
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
        >>> train, test = self.get_training_data()
        >>> test_in = []
        >>> for ind in range(3):
        >>>     test_in.append(Structure.from_dict(test[str(ind)]['structure']))
        >>> self.predict(test_in)
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
        pipeline = Image.open(requests.get(self.pipeline, stream=True).raw)
        return pipeline

    def get_training_data(self):
        with urllib.request.urlopen(self.training_data_path) as url:
            data = json.loads(url.read().decode())
        train = data['training']
        test = data['testing']
        return train, test

    def get_full_pipeline(self):
        return self.full_pipeline

    def plot_learning_curve(self):
        return Image.open(requests.get(self.learning_curve, stream=True).raw)



