"""
module for unit testing descriptors
"""
import numpy as np
import quantumml

from .test_data import *
def test_get_soap():

    assert np.allclose(quantumml.decsriptors.get_soap('quantumml/tests/test_data/POSCAR', normalize = False),soap_output)

    #assert np.allclose(quantumml.decsriptors.get_soap('quantumml/tests/test_data/POSCAR'),soap_output_norm)


