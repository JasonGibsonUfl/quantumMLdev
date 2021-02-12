"""
Module for testing rest.py
"""
from quantumml import MWRester

def test_get_calculation():
    band_gap_range = [0.5, 1.5]

    with MWRester() as mwr:
        mwr.get_calculation(band_gap_range=[0.5,1.5])
        results = mwr.results
        for calc in results:
            assert(band_gap_range[0] <= calc['band_gap'] and calc['band_gap'] <= band_gap_range[-1])

def test_as_pymatgen_struc():
    with MWRester() as mwr:
        assert mwr.preamble == "http://materialsweb.org/rest/calculation/?"

