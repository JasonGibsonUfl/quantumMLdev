"""
Contains an abstract base class that supports data transformations.
"""
import os
import logging
import time
import warnings
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy


logger = logging.getLogger(__name__)


class Transformer(object):
    """
    Abstract base class for different data transformation techniques.
    """

    __module__ = os.path.splitext(os.path.basename(__file__))[0]

    def __init__(self):
        if self.__class__.__name__ == "Transformer":
            raise ValueError(
                "Transformer is an abstract superclass and cannot be directly instantiated. You probably want to instantiate a concrete subclass instead."
            )

    def transform_array(self, X: np.ndarray) -> np.ndarray:
        """Transform the data in a set of X to array.
        Parameters
        ----------
        X: np.ndarray
          Array of features
        Returns
        -------
        Xtrans: np.ndarray
          Transformed array of features
        """
        raise NotImplementedError(
            "Each Transformer is responsible for its own transform_array method."
        )

    def untransform(self, transformed):
        """
        Parameters
        ----------
        transformed: np.ndarray
          Array which was previously transformed by this class.
        """
        raise NotImplementedError(
            "Each Transformer is responsible for its own untransform method."
        )


class MinMaxTransformer(Transformer):
    """Ensure each value rests between 0 and 1 by using the min and max.
    Examples
    --------
    >>> transformer = MinMaxTransformer(x_min = 5, x_max = 30)
    >>> dataset = transformer.transform_array(X=np.array([5,15,30,29,23]))
    """

    def __init__(self, X: np.ndarray = None, x_max: float = None, x_min: float = None):
        """
        Parameters
        ----------
        X: np.ndarray
          Array of features
        x_max: float
          Max value of x
        x_min: float
          min value of x
        """
        self.X = X
        if x_min is not None and x_max is not None:
            self.X_max = x_max
            self.X_min = x_min
        else:
            self.X_min = np.min(self.X, axis=0)
            self.X_max = np.max(self.X, axis=0)

        super(MinMaxTransformer, self).__init__()

    def transform_array(self, X: np.ndarray = None) -> np.ndarray:
        """
        Parameters
        ----------
        X: np.ndarray
          Array of features
        Returns
        -------
        Xtrans: np.ndarray
          Transformed array of features
        """
        # Handle division by zero
        if X is not None:
            self.X = X
        denominator = np.where(
            (self.X_max - self.X_min) > 0,
            (self.X_max - self.X_min),
            np.ones_like(self.X_max - self.X_min),
        )
        Xtrans = np.nan_to_num((X - self.X_min) / denominator)

        return Xtrans
