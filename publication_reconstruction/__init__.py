"""Explicitly versioned, isolated reconstruction of public 2022 HetNet code.

This is not an identification of the unpublished model-producing checkout.
Run from the repository root with ``python -m publication_reconstruction``.
"""

UPSTREAM_REFERENCE = "d57da0717d5564027df7e8ba75614feca8006960"
SCAFFOLD_REFERENCE = "0ee9ceb38b6133866e9c1db40bed2c6caa3b842d"
SCHEMA_VERSION = 2

# Establish these before importing numerical libraries, including on login nodes.
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
