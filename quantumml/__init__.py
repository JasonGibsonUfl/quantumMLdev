"""
quantumML
quantumML is 
"""

# Add imports here
from .rest import *
from .decsriptors import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
