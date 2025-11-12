"""
This package implements the Volume-Preserving (and non-preserving) MBO scheme for data clustering and classification.
"""

# Main API
from volumembo import datasets
from volumembo.mbo import MBO
from volumembo.plot import SimplexPlotter

__all__ = [
    "datasets",
    "MBO",
    "SimplexPlotter",
]
