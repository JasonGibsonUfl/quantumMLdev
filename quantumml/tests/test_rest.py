"""
Module for testing rest.py
"""
from quantumml import MWRester

#def test_get_calculation():
    #mwr = MWRester()
    #mwr.get_calculation(name='MoS2')

def test_as_pymatgen_struc():
    with MWRester() as mwr:
        assert mwr.preamble == "http://materialsweb.org/rest/calculation/?"

