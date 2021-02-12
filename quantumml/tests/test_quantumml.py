"""
Unit and regression test for the quantumml package.
"""

# Import package, test suite, and other packages as needed
import quantumml
import pytest
import sys


def test_quantumml_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "quantumml" in sys.modules
