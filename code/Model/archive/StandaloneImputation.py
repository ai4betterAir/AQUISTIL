"""
ARCHIVE STUB: Monolithic StandaloneImputation implementation

This module has been archived and replaced by modular implementations
in `Model/Standalone_<method>.py`. The original full implementation
was moved to `Model/archive/StandaloneImputation.py` for reference.

Importing this module will raise an informative error to avoid accidental use.
"""

import os

def __getattr__(name):
    raise ImportError(
        "The monolithic 'Model.StandaloneImputation' is archived. "
        "Use the modular 'Model/Standalone_<method>.py' implementations or "
        "see 'Model/archive/StandaloneImputation.py' for the original code."
    )