"""
This package implements the Volume-Preserving (and non-preserving) MBO scheme for data clustering and classification.
"""

# Main API
from volumembo import datasets
from volumembo.plot import SimplexPlotter
from volumembo.mbo import MBO

__all__ = [
    "datasets",
    "MBO",
    "SimplexPlotter",
]
